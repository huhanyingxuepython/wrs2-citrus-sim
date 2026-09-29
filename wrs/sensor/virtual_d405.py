"""D405 camera with compatible legacy effects and opt-in stateful raw-like noise."""

import copy
from dataclasses import replace
from time import perf_counter

import numpy as np

from .camera_model import CameraModel
from .d405_noise_config import D405NoiseConfig, load_noise_profile
from .d405_noise import D405NoiseModel
from .depth_noise import StereoDepthNoise
from .virtual_depth_camera import VirtualDepthCamera


class VirtualD405(VirtualDepthCamera):
    """Passive-stereo D405 approximation, anchored at the left optical frame.

    Defaults: 640x480, nominal 87x58 degree FOV, 18 mm baseline, 7--50 cm
    operating range. FOV-derived intrinsics are approximate and mode dependent;
    supply camera_model from the active real device profile for calibration.
    The 0.1 mm depth_scale is a selectable simulation encoding, not a claim
    about firmware defaults or measurement accuracy. Random noise is off by
    default; d405_fast geometric occlusion is on. Set mode='ideal' for truth.

    RGB and depth share the left optical view. No projector, separate RGB
    camera, or D4 firmware matcher is simulated. Other keywords are forwarded
    to VirtualDepthCamera (intrinsics, distortion, pose, noise, seed, near/far,
    fps, T_mount_camera, mode, rgb_lighting, rgb_background, name).

    Set noise_config='default'/'agriculture_foliage'/'harsh'/'ideal', or supply
    D405NoiseConfig, to enable stateful analytic hole/value corruption. Existing
    calls without noise_config retain the legacy GPU behavior and 0.5 m limit.
    New profiles default to a 0.5 m preferred range with degradation up to a
    separate 2 m cutoff. These are simulation settings, not factory accuracy.
    noise_profile accepts a JSON from tools/analyze_d405_noise.py. Explicit
    constructor calibration/range/scale overrides loaded profile settings.
    """

    def __init__(self, camera_model=None, *, width=None, height=None,
                 min_depth=None, max_depth=None, depth_scale=None,
                 baseline_m=None, noise_config=None, noise_profile=None,
                 noise_model=None, **kwargs):
        profile = None
        if noise_profile is not None:
            if noise_config is not None or noise_model is not None:
                raise ValueError('supply noise_profile OR noise_config/noise_model')
            profile = load_noise_profile(noise_profile)
            noise_config = D405NoiseConfig.from_dict(profile['config'])
            calibration = profile.get('calibration', {})
            if baseline_m is None:
                baseline_m = calibration.get('baseline_m')
            if camera_model is None and width is None and height is None and not any(
                    key in kwargs for key in ('fx', 'fy', 'cx', 'cy', 'distortion_model', 'distortion_coeffs')):
                intrinsics = calibration.get('camera_model')
                if intrinsics is not None:
                    camera_model = CameraModel(**intrinsics)
        if noise_model is not None:
            if noise_config is not None or not isinstance(noise_model, D405NoiseModel):
                raise ValueError('supply a D405NoiseModel OR noise_config')
            noise_config = noise_model.config
        if isinstance(noise_config, str):
            noise_config = D405NoiseConfig.preset(noise_config)
        if isinstance(noise_config, dict):
            noise_config = D405NoiseConfig.from_dict(noise_config)
        if noise_config is not None and not isinstance(noise_config, D405NoiseConfig):
            raise TypeError('noise_config must be a preset name or D405NoiseConfig')
        defaults = noise_config or D405NoiseConfig()
        min_depth = defaults.min_depth_m if min_depth is None else min_depth
        max_depth = (defaults.max_depth_m if noise_config is not None else .5) if max_depth is None else max_depth
        depth_scale = defaults.depth_unit_m if depth_scale is None else depth_scale
        baseline_m = .018 if baseline_m is None else baseline_m
        self._d405_noise_model = None
        if noise_config is not None:
            if kwargs.get('noise') is not None:
                raise ValueError('legacy noise and noise_config cannot be combined')
            noise_config = replace(noise_config, min_depth_m=min_depth, max_depth_m=max_depth,
                preferred_max_depth_m=min(noise_config.preferred_max_depth_m, max_depth),
                depth_unit_m=depth_scale,
                enabled=noise_config.enabled and kwargs.get('mode', 'd405_fast') != 'ideal')
            if noise_model is not None and noise_model.config != noise_config:
                raise ValueError('injected noise_model calibration conflicts with camera settings')
            self._d405_noise_model = (noise_model if noise_model is not None else
                                      D405NoiseModel(noise_config, seed=kwargs.get('seed')))
        if camera_model is None and 'fx' not in kwargs and 'fy' not in kwargs:
            camera_model = CameraModel.from_fov(640 if width is None else width,
                                               480 if height is None else height, 87.0, 58.0)
            width = height = None
        kwargs.setdefault('name', 'virtual_d405')
        kwargs.setdefault('mode', 'd405_fast')
        kwargs.setdefault('far', max_depth)
        super().__init__(camera_model, width=width, height=height,
                         min_depth=min_depth, max_depth=max_depth,
                         depth_scale=depth_scale, baseline_m=baseline_m, **kwargs)

    @property
    def noise_model(self):
        """Analytic model/state, or None for the compatible legacy GPU path."""
        return self._d405_noise_model

    @property
    def noise_config(self):
        return None if self.noise_model is None else self.noise_model.config

    @VirtualDepthCamera.noise.setter
    def noise(self, value):
        if self.noise_model is not None and value is not None:
            raise ValueError('legacy noise cannot be set while noise_config is active')
        VirtualDepthCamera.noise.fset(self, value)

    def capture(self, scene, *, T_world_mount=None, exclude=(), debug=False,
                normals=None, semantic_ids=None, material_difficulty=None, motion_px=None):
        """Capture raw-like RGB-D; optional diagnostics are frame.noise_debug.

        Additional cue maps must already match the output image calibration.
        Normals use optical axes. No semantic or material channel is fabricated
        when the renderer does not supply one. Distorted images use horizontal
        output-row donors as an approximation, not a rectified stereo matcher.
        """
        started = perf_counter()
        frame = super().capture(scene, T_world_mount=T_world_mount, exclude=exclude)
        if self.noise_model is None:
            if debug or any(value is not None for value in (normals, semantic_ids, material_difficulty, motion_px)):
                raise ValueError('debug/additional noise cues require noise_config')
            return frame
        if not self.noise_config.enabled and not debug and all(
                value is None for value in (normals, semantic_ids, material_difficulty, motion_px)):
            # The existing GPU pass already produced ideal Z16 and XYZ. Reuse
            # them when corruption is off, without CPU cues or a second XYZ pass.
            timings = dict(frame.timings_ms, noise_model_cpu=0., noise_pointcloud_cpu=0.)
            timings['capture_total'] = (perf_counter()-started)*1000
            return replace(frame, timings_ms=timings)
        noise_started = perf_counter()
        result = self.noise_model.apply(frame.depth_gt, self.camera_model, self.baseline_m,
            rgb=frame.rgb, camera_tf=frame.T_world_camera, normals=normals, semantic_ids=semantic_ids,
            material_difficulty=material_difficulty, motion_px=motion_px, debug=debug)
        point_started = perf_counter()
        camera_points = self.camera_model.deproject(result.depth)
        tf = frame.T_world_camera
        world_points = (camera_points @ tf[:3, :3].astype(np.float32).T + tf[:3, 3].astype(np.float32))
        world_points[result.depth_raw == 0] = 0
        timings = dict(frame.timings_ms)
        timings['noise_model_cpu'] = (point_started-noise_started)*1000
        timings['noise_pointcloud_cpu'] = (perf_counter()-point_started)*1000
        timings['capture_total'] = (perf_counter()-started)*1000
        return replace(frame, depth=result.depth, depth_raw=result.depth_raw,
            points_cam_image=camera_points, points_world_image=world_points,
            confidence=np.where(result.depth_raw > 0, frame.confidence, 0),
            timings_ms=timings, noise_debug=result.debug)

    def process_depth(self, depth_gt, rgb=None, *, debug=False, **cues):
        """Process externally rendered depth; legacy tuple return is preserved.

        debug=True returns D405NoiseResult for analytic profiles. Each call is
        a new temporal sample, including calls interleaved with capture().
        """
        if self.noise_model is None:
            if debug or cues:
                raise ValueError('debug/additional noise cues require noise_config')
            return super().process_depth(depth_gt, rgb)
        result = self.noise_model.apply(depth_gt, self.camera_model, self.baseline_m,
                                       rgb=rgb, debug=debug, **cues)
        return result if debug else (result.depth, result.depth_raw)

    def reset_noise(self, seed=None):
        """Start a new analytic episode; no seed repeats the initial sequence."""
        if self.noise_model is None:
            raise ValueError('reset_noise requires noise_config')
        self.noise_model.reset(seed)

    def clone(self, postfix='(clone)'):
        new = super().clone(postfix)
        new._d405_noise_model = copy.deepcopy(self.noise_model)
        return new
