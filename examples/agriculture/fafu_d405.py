"""Calibrated flange-mounted D405 and live RGB-D for the citrus example."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from wrs import wssop
from wrs.sensor import VirtualD405, RGBLighting, D405NoiseConfig
from wrs.viewer.web_ui import colorize_depth
from agriculture.config import CONFIG_DIR


def load_handeye(path):
    """Read the user's metre-valued T_flange_cam; reject ambiguous frames."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('frame') != 'T_flange_cam':
        raise ValueError('hand-eye frame must be T_flange_cam, not T_tcp_cam')
    convention = 'p_base = T_base_flange @ T_flange_cam @ p_cam'
    if data.get('convention') != convention:
        raise ValueError('unsupported hand-eye transform convention')
    optical = np.asarray(data['handeye_mat'], dtype=float)
    if (optical.shape != (4, 4) or not np.isfinite(optical).all()
            or not np.allclose(optical[3], (0, 0, 0, 1))
            or not np.allclose(optical[:3, :3].T @ optical[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(optical[:3, :3]), 1, atol=1e-6)):
        raise ValueError('T_flange_cam must be a finite rigid transform')
    if 'affine_mat' in data:
        duplicate = np.asarray(data['affine_mat'], dtype=float)
        if duplicate.shape != (4, 4) or not np.allclose(duplicate, optical, atol=1e-9, rtol=0):
            raise ValueError('affine_mat and handeye_mat disagree')
    return optical


def camera_mount_transform(settings):
    """Return T_flange_cam directly; no TCP offset or URDF axis conversion.

    Relative calibration paths resolve under agriculture/configs, independent
    of the working directory. Absolute paths allow replacement calibrations.
    """
    if settings['wrs_parent_link'] != 'flange':
        raise ValueError('T_flange_cam must be mounted on flange')
    return load_handeye(CONFIG_DIR / settings['handeye_file'])


class MountedD405:
    """Own one mounted sensor, its latest frame, and at most one display cloud."""

    def __init__(self, robot, scene, settings, *, exclude=()):
        self.scene, self.settings = scene, settings
        # Keep the collection live: the example can toggle debug mounts later.
        self.excluded_objects = exclude
        self.mount_tf = camera_mount_transform(settings)
        self.parent_link = next(link for link in robot.runtime_lnks if link.name == settings['wrs_parent_link'])
        noise = settings['depth_noise']
        profile = D405NoiseConfig.preset(noise['preset'], **noise.get('overrides', {}))
        self.camera = VirtualD405(rgb_lighting=RGBLighting(**settings['rgb_lighting']),
            rgb_background=settings['rgb_background'], far=settings['render_far_m'],
            noise_config=replace(profile, enabled=bool(noise['enabled'])),
            **{key: settings[key] for key in ('width', 'height', 'fps', 'mode', 'seed', 'min_depth', 'max_depth')})
        self.noise_profile = self.camera.noise_config
        self.noise_enabled = self.noise_profile.enabled
        self.depth_color_range = (self.camera.min_depth, settings['depth_color_max_m'])
        self._add_body_visuals()
        # capture() uses this native mounting pose; do not apply T_world_mount again.
        robot.mount(self.camera, self.parent_link, self.mount_tf, update=True)
        self.last_frame = None
        self.frame_count = 0
        self.cloud = None
        self.show_cloud = False
        self.show_clean_depth = False
        self.panel = self.rgb_view = self.depth_view = None
        self.simulation_time = None

    def _add_body_visuals(self):
        p, camera = self.settings, self.camera
        size = np.asarray(p['body_size_m'])
        # Schematic body with two lenses. The sensor origin is the left optical
        # centre, so the body centre is half a stereo baseline to its right.
        parts = [wssop.box(pos=(camera.baseline_m / 2, 0, -size[2] / 2),
                           xyz_lengths=size, rgb=p['body_color'])]
        for x in (0, camera.baseline_m):
            parts.append(wssop.cylinder(spos=(x, 0, -p['lens_length_m'] / 2),
                epos=(x, 0, p['lens_length_m'] / 2), radius=p['lens_radius_m'], segments=16,
                rgb=p['lens_color']))
        for part in parts:
            for visual in part.visuals:
                local = part.tf @ visual.loc_tf
                visual.set_pos_rotmat(local[:3, 3], local[:3, :3])
                camera.add_visual(visual, auto_make_collision=False)
        # No collider/mass/DOF: this is an observation sensor with a display shell.
        # VirtualD405 excludes itself, including this shell, during capture.

    def capture(self, simulation_time=None):
        excluded = list(self.excluded_objects)
        excluded.extend(getattr(self.scene, '_force_arrows', ()))
        excluded.extend(getattr(self.scene, '_contact_spheres', ()))
        if self.cloud is not None:
            excluded.append(self.cloud)
        frame = self.camera.capture(self.scene, exclude=excluded)
        self.last_frame = frame
        self.frame_count += 1
        self.simulation_time = simulation_time
        if self.show_cloud:
            self._replace_cloud()
        if self.panel is not None:
            self.rgb_view.update(frame.rgb)
            self._update_depth_view()
            info = self.statistics()
            self.panel.set_value('capture', f'Frame {self.frame_count} · {info["capture_ms"]:.1f} ms total\n'
                                  f'Noise {info["noise_ms"]:.1f} ms · XYZ {info["noise_pointcloud_ms"]:.1f} ms')
            self.panel.set_value('points', f'{info["valid_pixels"]:,} valid pixels ({info["valid_fraction"]:.1%})\n'
                                  f'{info["sampled_points"]:,} sampled points · world XYZ (m)\n'
                                  f'Fill on in-range geometry: {info["geometry_fill_rate"]:.1%}')
            self.panel.set_value('camera_xyz', ' / '.join(f'{value:+.3f}' for value in info['camera_xyz_m']))
            depth = info['depth_range_m']
            self.panel.set_value('depth_range', 'No valid depth' if depth is None else
                                  f'{depth[0]:.3f} – {depth[1]:.3f} m')
            bounds = info['point_bbox_world_m']
            self.panel.set_value('cloud_bounds', 'Empty' if bounds is None else '\n'.join(
                f'{axis}: {lo:.3f} … {hi:.3f}' for axis, lo, hi in zip('XYZ', *bounds)))
        return frame

    def statistics(self):
        frame = self.last_frame
        if frame is None:
            return dict(frame_count=0)
        depth = frame.depth_m[frame.valid_mask]
        points, _ = frame.get_point_cloud(world_frame=True, stride=self.settings['point_stride'])
        support = (frame.depth_gt >= self.camera.min_depth) & (frame.depth_gt <= self.camera.max_depth)
        sigma = self.noise_profile.disparity_sigma_px if self.noise_enabled else 0.
        return dict(frame_count=self.frame_count, simulation_time=self.simulation_time,
                    resolution=[frame.camera_model.width, frame.camera_model.height],
                    valid_pixels=int(depth.size), valid_fraction=float(depth.size / frame.depth_m.size),
                    sampled_points=len(points), capture_ms=frame.timings_ms['capture_total'],
                    depth_noise_enabled=self.noise_enabled,
                    noise_preset=self.settings['depth_noise']['preset'],
                    disparity_sigma_px=sigma, disparity_std_px=sigma,
                    noise_ms=frame.timings_ms.get('noise_model_cpu', 0.),
                    noise_pointcloud_ms=frame.timings_ms.get('noise_pointcloud_cpu', 0.),
                    geometry_fill_rate=float(np.count_nonzero(frame.valid_mask & support) / max(1, support.sum())),
                    camera_xyz_m=frame.T_world_camera[:3, 3].tolist(),
                    depth_range_m=None if not depth.size else [float(depth.min()), float(depth.max())],
                    point_bbox_world_m=None if not len(points) else [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                    T_world_optical=frame.T_world_camera.tolist())

    def _remove_cloud(self):
        if self.cloud is not None:
            self.cloud.remove_from_scene(self.scene)
            self.cloud = None

    def _replace_cloud(self):
        self._remove_cloud()
        if self.last_frame is None:
            return
        points, colors = self.last_frame.get_point_cloud(world_frame=True, stride=self.settings['point_stride'])
        if len(points):
            # Cyan separates measured surfaces from the original green/orange
            # meshes. Registered RGB is still available in last_frame's cloud.
            self.cloud = wssop.point_cloud(points, np.full_like(colors, self.settings['cloud_color']))
            self.cloud.add_to_scene(self.scene)

    def set_show_cloud(self, checked):
        self.show_cloud = checked
        if checked:
            self._replace_cloud()
        else:
            self._remove_cloud()

    def toggle_depth_view(self):
        """Switch the displayed channel of the latest frame without recapturing."""
        self.show_clean_depth = not self.show_clean_depth
        self._update_depth_view()

    def _update_depth_view(self):
        if self.panel is None:
            return
        self.panel.set_value('depth_mode', 'Clean' if self.show_clean_depth else 'Noise')
        frame = self.last_frame
        if frame is None:
            return
        if self.show_clean_depth:
            depth = frame.depth_gt
            valid = (depth >= self.camera.min_depth) & (depth <= self.camera.max_depth)
        else:
            depth, valid = frame.depth_m, frame.valid_mask
        self.depth_view.update(colorize_depth(depth, value_range=self.depth_color_range, valid_mask=valid))

    def set_noise_enabled(self, checked):
        self.noise_enabled = bool(checked)
        self.camera.noise_model.config = replace(self.noise_profile, enabled=self.noise_enabled)
        self.invalidate()
        self._update_noise_label()

    def set_noise_sigma(self, value):
        if not np.isfinite(value) or not 0 <= value <= self.settings['depth_noise']['slider_max_px']:
            raise ValueError('disparity noise is outside the configured slider range')
        self.noise_profile = replace(self.noise_profile, disparity_sigma_px=value)
        self.camera.noise_model.config = replace(self.noise_profile, enabled=self.noise_enabled)
        if self.noise_enabled:
            self.invalidate()
        self._update_noise_label()

    def _update_noise_label(self):
        if self.panel is None:
            return
        if self.noise_enabled:
            fb = self.camera.camera_model.fx * self.camera.baseline_m
            estimate = ' / '.join(f'{z:.2f} m: ~{1000*z*z*self.noise_profile.disparity_sigma_px/fb:.2f} mm'
                                 for z in self.settings['depth_noise']['reference_depths_m'])
            self.panel.set_value('noise_info', self.settings['depth_noise']['preset'] + '\n'
                'Correlated holes + edge mismatch + flicker\n' + estimate + '\nPrecision only · heuristic parameters')
        else:
            self.panel.set_value('noise_info', 'Noise off · clean geometry + range + Z16')

    def invalidate(self):
        """Drop a pre-reset measurement so it cannot be mistaken for current data."""
        self.last_frame = None
        self.simulation_time = None
        self.camera.reset_noise()
        self._remove_cloud()
        if self.panel is not None:
            self.rgb_view.clear()
            self.depth_view.clear()
            for key in ('capture', 'points', 'depth_range', 'cloud_bounds', 'camera_xyz'):
                self.panel.set_value(key, 'Waiting for capture')

    def add_controls(self, panel):
        self.panel = panel
        self.rgb_view = panel.add_image('rgb', label='D405 RGB', format='jpeg', max_fps=self.camera.fps)
        panel.add_label('depth_mode', label='Depth display', value='Clean' if self.show_clean_depth else 'Noise')
        self.depth_view = panel.add_image('depth',
            label=f'D405 depth · Jet {self.depth_color_range[0]:.2f}–{self.depth_color_range[1]:.2f} m',
            max_fps=self.camera.fps)
        panel.add_button('depth_toggle', label='Switch Clean / Noise', on_click=self.toggle_depth_view)
        panel.add_checkbox('point_cloud', label='Show world point cloud (cyan)', on_change=self.set_show_cloud)
        panel.add_checkbox('depth_noise', label='D405 holes / mismatch / temporal noise',
                           value=self.noise_enabled, on_change=self.set_noise_enabled)
        panel.add_slider('noise_sigma', label='Disparity noise sigma', unit='px', min_value=0,
            max_value=self.settings['depth_noise']['slider_max_px'],
            step=self.settings['depth_noise']['slider_step_px'], value=self.noise_profile.disparity_sigma_px,
            on_change=self.set_noise_sigma)
        panel.add_label('noise_info', label='Noise model · sigma changes precision only')
        self._update_noise_label()
        panel.add_label('capture', label='Capture · calibrated mount / nominal intrinsics')
        panel.add_label('camera_xyz', label='Camera optical origin · X / Y / Z metres')
        panel.add_label('points', label='Point cloud')
        panel.add_label('depth_range', label='Measured Z range')
        panel.add_label('cloud_bounds', label='Sampled cloud bounds · world metres')

    def close(self):
        self._remove_cloud()
        self.camera.close()
