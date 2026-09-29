"""Scene-mounted virtual depth camera with separate rendering and sensor stages."""

import copy
import math
from dataclasses import dataclass
from functools import cached_property
from numbers import Integral
from time import perf_counter

import numpy as np

from wrs.scene.scene_object import SceneObject
from wrs.utils.math import ensure_tf
from .camera_model import CameraModel
from .depth_noise import StereoDepthNoise
from .rgb_lighting import RGBLighting


@dataclass(frozen=True)
class DepthFrame:
    """One capture with calibration and world-from-camera pose snapshots.

    depth_gt and depth are float32 (H, W) Z images in meters. depth_gt is the
    geometric render within the render clipping planes, before sensor range,
    noise and quantization; depth is the measurement. Zero denotes invalid
    depth. depth_raw is uint16 with depth = depth_raw * depth_scale. rgb is
    uint8 (H, W, 3), registered to this camera, optionally matte-lit.
    """

    camera_model: CameraModel
    camera_tf: np.ndarray
    depth_scale: float
    depth_gt: np.ndarray
    depth: np.ndarray
    depth_raw: np.ndarray
    rgb: np.ndarray
    timings_ms: dict
    points_cam_image: np.ndarray
    points_world_image: np.ndarray
    confidence: np.ndarray
    noise_debug: dict | None = None

    @property
    def depth_m(self):
        return self.depth

    @property
    def intrinsics(self):
        return self.camera_model.matrix

    @property
    def T_world_camera(self):
        return self.camera_tf.copy()

    @cached_property
    def points_cam(self):
        return self.points_cam_image.reshape(-1, 3)[self._valid_indices]

    @cached_property
    def points_world(self):
        return self.points_world_image.reshape(-1, 3)[self._valid_indices]

    @cached_property
    def valid_mask(self):
        return self.depth > 0

    @cached_property
    def _valid_indices(self):
        return np.flatnonzero(self.valid_mask)

    def get_point_cloud(self, *, world_frame=False, stride=1):
        """Return valid (N, 3) points in meters and float32 RGB in [0, 1].

        Row-major pixels are subsampled with ``[::stride, ::stride]`` before
        invalid samples are removed. World points use the captured pose even
        if the camera has moved since capture. An empty frame returns (0, 3)
        arrays. The camera frame is +X right, +Y down, +Z forward.
        """
        if isinstance(stride, bool) or not isinstance(stride, Integral) or stride <= 0:
            raise ValueError('stride must be a positive integer')
        keep = self.valid_mask[::stride, ::stride]
        image = self.points_world_image if world_frame else self.points_cam_image
        points = image[::stride, ::stride][keep]
        colors = self.rgb[::stride, ::stride][keep].astype(np.float32) / 255
        return points, colors


class VirtualDepthCamera(SceneObject):
    """Render a WRS scene on the GPU without a viewer or external simulator.

    Parameters
    ----------
    camera_model : CameraModel, optional
        Intrinsics and distortion. Alternatively supply width/height/fx/fy/cx/cy
        and distortion_model/distortion_coeffs. Calibration is immutable.
    fps : float
        Requested scheduling rate; capture is synchronous and does not sleep.
    near, far : float
        Geometric clipping planes in meters. Keep near below min_depth so
        too-close surfaces still occlude objects behind them.
    T_mount_camera : array-like, shape (4, 4), optional
        Mount-from-optical transform used when capture receives T_world_mount.
    mode : str
        'ideal' or 'd405_fast'. The latter requires a positive baseline_m.
    min_depth, max_depth : float
        Inclusive measurement range in meters; invalid samples become zero.
    depth_scale : float
        Meters per uint16 unit. Quantization is independent of sensor accuracy.
    baseline_m : float or None
        Stereo baseline in meters, required in d405_fast mode.
    noise : StereoDepthNoise or None
        Optional single-view sensor approximation; defaults to no noise.
    seed : int or None
        Seed for an independent numpy random generator per camera.
    rgb_lighting : RGBLighting or None
        Optional camera-relative matte lighting. None retains unlit RGB.
    rgb_background : tuple
        Background sRGB in [0, 1]. Independent of invalid depth (always zero).
    pos, rotmat : array-like, optional
        Optical frame pose in the world (+X right, +Y down, +Z forward).

    Notes
    -----
    Inherits SceneObject for ``robot.mount(camera, link, loc_tf)``. The camera
    has no body mesh by default and excludes itself during capture. Meshes
    with alpha > 0 are opaque sensor surfaces; transparent optics, textures,
    cast shadows and actual stereo matching are not simulated. Brown-Conrady
    distortion uses a cached inverse-ray LUT and nearest GPU sampling.
    """

    def __init__(self, camera_model=None, *, width=None, height=None,
                 fx=None, fy=None, cx=None, cy=None, fps=30, near=0.001, far=5.0,
                 distortion_model='none', distortion_coeffs=(0, 0, 0, 0, 0),
                 T_mount_camera=None, min_depth=0.05, max_depth=5.0,
                 depth_scale=0.001, baseline_m=None, noise=None, seed=None,
                 rgb_lighting=None, rgb_background=(0., 0., 0.),
                 mode='ideal', pos=None, rotmat=None, name='virtual_depth_camera'):
        if camera_model is None:
            width = 640 if width is None else width
            height = 480 if height is None else height
            if fx is None or fy is None:
                raise ValueError('supply camera_model or explicit fx and fy')
            camera_model = CameraModel(width, height, fx, fy,
                                       (width-1)/2 if cx is None else cx,
                                       (height-1)/2 if cy is None else cy,
                                       distortion_model, tuple(distortion_coeffs))
        elif (any(value is not None for value in (width, height, fx, fy, cx, cy))
              or distortion_model != 'none' or any(distortion_coeffs)):
            raise ValueError('supply calibration through camera_model OR individual parameters')
        if not isinstance(camera_model, CameraModel):
            raise TypeError('camera_model must be a CameraModel')
        if not np.all(np.isfinite([near, far, min_depth, max_depth, depth_scale, fps])):
            raise ValueError('depth limits and depth_scale must be finite')
        if not 0 < near <= min_depth < max_depth <= far or depth_scale <= 0 or fps <= 0:
            raise ValueError('require 0 < near <= min_depth < max_depth <= far; scale and fps > 0')
        if np.nextafter(max_depth / depth_scale, -np.inf) > np.iinfo(np.uint16).max:
            raise ValueError('max_depth is not representable with this uint16 depth_scale')
        if baseline_m is not None and (not np.isfinite(baseline_m) or baseline_m <= 0):
            raise ValueError('baseline_m must be positive and finite')
        if mode not in ('ideal', 'd405_fast'):
            raise ValueError('supported modes are ideal and d405_fast; d405_stereo is not implemented')
        if mode == 'd405_fast' and baseline_m is None:
            raise ValueError('d405_fast requires a positive baseline_m')
        background = np.asarray(rgb_background, dtype=float)
        if background.shape != (3,) or not np.isfinite(background).all() or np.any((background < 0) | (background > 1)):
            raise ValueError('rgb_background must contain three finite sRGB values in [0, 1]')
        super().__init__(name=name)
        self.set_pos_rotmat(pos, rotmat)
        self._camera_model = camera_model
        self._min_depth, self._max_depth = float(min_depth), float(max_depth)
        self._depth_scale, self._baseline_m = float(depth_scale), baseline_m
        # Integer bounds avoid rejecting exact range endpoints because decimal
        # depth scales (e.g. 1e-4) are not exactly representable in float32.
        self._raw_min = max(1, math.ceil(np.nextafter(min_depth/depth_scale, -np.inf)))
        self._raw_max = math.floor(np.nextafter(max_depth/depth_scale, np.inf))
        if self._raw_min > self._raw_max:
            raise ValueError('measurement range contains no representable nonzero depth unit')
        self._near, self._far, self._fps = float(near), float(far), float(fps)
        self._mode = mode
        self._T_mount_camera = _rigid_tf(T_mount_camera)
        self.noise = noise
        self.rgb_lighting = rgb_lighting
        self._rgb_background = tuple(background)
        self._rng = np.random.default_rng(seed)
        self._renderer = None

    @property
    def T_mount_camera(self):
        return self._T_mount_camera.copy()

    @property
    def T_world_camera(self):
        return self.tf

    @property
    def near(self):
        return self._near

    @property
    def far(self):
        return self._far

    @property
    def fps(self):
        return self._fps

    @property
    def mode(self):
        return self._mode

    @property
    def camera_model(self):
        return self._camera_model

    @property
    def min_depth(self):
        return self._min_depth

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def depth_scale(self):
        return self._depth_scale

    @property
    def baseline_m(self):
        return self._baseline_m

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, value):
        """Change measurement effects for the next capture without rebuilding GPU resources."""
        value = StereoDepthNoise() if value is None else value
        if not isinstance(value, StereoDepthNoise):
            raise TypeError('noise must be a StereoDepthNoise')
        if (value.disparity_std_px or value.disparity_step_px) and self.baseline_m is None:
            raise ValueError('disparity effects require baseline_m')
        if self.mode == 'ideal' and value != StereoDepthNoise():
            raise ValueError('enable mode="d405_fast" to use sensor noise')
        self._noise = value

    @property
    def rgb_lighting(self):
        return self._rgb_lighting

    @rgb_lighting.setter
    def rgb_lighting(self, value):
        if value is not None and not isinstance(value, RGBLighting):
            raise TypeError('rgb_lighting must be RGBLighting or None')
        self._rgb_lighting = value

    @property
    def rgb_background(self):
        return self._rgb_background

    def capture(self, scene, *, T_world_mount=None, exclude=()):
        """Synchronously capture a Scene or iterable of SceneObjects/robot links.

        Only visual triangle meshes are rendered, using owner.tf @ model.loc_tf.
        ``exclude`` contains objects to omit, e.g. debug frames. Each capture
        reads current poses and scene membership; geometry buffers are cached.
        The first capture lazily creates GPU resources. Returns a DepthFrame.
        With T_world_mount, updates the optical pose to T_world_mount @
        T_mount_camera. Without it, uses the current SceneObject pose (including
        poses set by robot.mount). Do not combine the two mounting mechanisms.
        """
        started = perf_counter()
        if T_world_mount is not None:
            if self.mounted_by is not None:
                raise ValueError('use either robot.mount or T_world_mount, not both')
            self.tf = _rigid_tf(T_world_mount) @ self.T_mount_camera
        excluded = set(exclude)
        excluded.add(self)
        objects = [obj for obj in scene if obj not in excluded]
        if not all(isinstance(obj, SceneObject) for obj in objects):
            raise TypeError('scene must contain SceneObjects (or runtime robot links)')
        camera_tf = _rigid_tf(self.tf)
        if self._renderer is None:
            from .offscreen_renderer import _OffscreenRenderer
            self._renderer = _OffscreenRenderer(self.camera_model)
        result = self._renderer.render(objects, camera_tf, self.near, self.far, self)
        timings = dict(self._renderer.timings_ms)
        timings['capture_total'] = (perf_counter() - started) * 1000
        return DepthFrame(self.camera_model, camera_tf, self.depth_scale,
                          timings_ms=timings, **result)

    def process_depth(self, depth_gt, rgb=None):
        """Apply sensor range, optional disparity effects and uint16 encoding.

        Accepts (H, W) metric Z depth and returns (float32 depth, uint16 raw).
        Input is never modified. Unrepresentable/invalid samples become zero,
        never wrap or saturate to a false surface. Usable independently of GPU
        rendering, for rectified inputs only. This CPU reference uses NumPy
        randomness; capture uses a GPU hash generator with the same seed
        reproducibility contract, not an identical random sequence. For texture
        confidence, rgb must be an (H, W, 3) uint8 RGB image.
        """
        depth = np.array(depth_gt, dtype=np.float32, copy=True)
        if self.camera_model.distortion_model != 'none':
            raise ValueError('CPU process_depth expects rectified input; use capture for distortion')
        if depth.shape != (self.camera_model.height, self.camera_model.width):
            raise ValueError('depth shape must match camera height and width')
        tolerance = 4 * np.finfo(np.float32).eps * self.max_depth
        valid = (np.isfinite(depth) & (depth > 0)
                 & (depth >= self.min_depth-tolerance) & (depth <= self.max_depth+tolerance))
        # Retain too-close geometry during reprojection: it can occlude surfaces
        # in the sensor range, even though its own measurement is invalid.
        depth[~np.isfinite(depth) | (depth <= 0)] = 0
        if self.mode == 'd405_fast':
            depth = self.noise.apply(depth, self.camera_model.fx, self.baseline_m, self._rng, rgb=rgb)
        valid &= (np.isfinite(depth) & (depth > 0)
                  & (depth >= self.min_depth-tolerance) & (depth <= self.max_depth+tolerance))
        units = np.zeros(depth.shape, dtype=np.float64)
        units[valid] = np.rint(depth[valid].astype(np.float64) / self.depth_scale)
        valid &= (units >= self._raw_min) & (units <= self._raw_max)
        raw = np.where(valid, units, 0).astype(np.uint16)
        return (raw.astype(np.float64) * self.depth_scale).astype(np.float32), raw

    def clone(self, postfix='(clone)'):
        """Clone pose, calibration and RNG state, without sharing GPU resources."""
        new = copy.copy(self)
        for attr in ('_pos', '_rotmat', '_tf', '_inrtmat', '_com', '_T_mount_camera'):
            value = getattr(self, attr)
            setattr(new, attr, None if value is None else value.copy())
        new.visuals = [model.clone() for model in self.visuals]
        new.collisions = [shape.clone() for shape in self.collisions]
        new._mounted_by = None
        new._rng = copy.deepcopy(self._rng)
        new._renderer = None
        new.name = None if self.name is None else self.name + postfix
        return new

    def close(self):
        """Release this camera's GPU resources; a later capture recreates them."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _rigid_tf(value):
    """Validate a world/optical rigid transform before using its inverse."""
    if value is not None and np.shape(value) != (4, 4):
        raise ValueError('camera transforms must have shape (4, 4)')
    tf = ensure_tf(value).copy()
    if (not np.all(np.isfinite(tf)) or not np.allclose(tf[3], [0, 0, 0, 1])
            or not np.allclose(tf[:3, :3].T @ tf[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(tf[:3, :3]), 1, atol=1e-5)):
        raise ValueError('camera transforms must be finite rigid 4x4 transforms')
    return tf
