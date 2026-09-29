"""Adjust a VirtualD405 and compare world-frame points with source meshes.

Run: python -m examples.virtual_d405_model_overlay
Use the left panel for XYZ / roll-pitch-yaw, resolution, focal scale and mode.
Click the 3-D view for W/S (+/-Z), A/D (-/+X), Q/E (-/+Y); R resets pose.
Keys translate in world coordinates; mouse dragging orbits only the viewer.
Resolution changes scale focal lengths uniformly to preserve square pixels;
wider profiles extend the horizontal field of view. Previews use letterboxing.
Camera body and optical axes are display-only, as in virtual_d405_rgbd.
"""
import time

import numpy as np

from wrs import key, wss, wssop, wum, wvw
from wrs.sensor import VirtualD405
from wrs.viewer.web_ui import Anchor, colorize_depth


class OverlayDemo:
    def __init__(self):
        # Capture opaque originals; show translucent copies in the viewer.
        self.scene = wss.Scene()
        wssop.icosphere(pos=(-.055, -.015, .29), radius=.045, subdivisions=3,
                       rgb=(1, .4, .03)).add_to_scene(self.scene)
        wssop.box(pos=(.06, .025, .37), xyz_lengths=(.08, .07, .08),
                  rgb=(.15, .55, .9)).add_to_scene(self.scene)
        wssop.cylinder(spos=(-.12, .065, .40), epos=(.12, .065, .40), radius=.008,
                       rgb=(.35, .18, .06)).add_to_scene(self.scene)
        self.base = wvw.World(cam_pos=(.55, -.65, .35), cam_lookat_pos=(0, 0, .18))
        self.base.set_caption('VirtualD405 / adjustable model overlay')
        for obj in self.scene:
            ghost = obj.clone()
            ghost.alpha = .22
            ghost.add_to_scene(self.base.scene)

        self.home_pos = np.array((.025, -.015, 0.))
        self.angles = np.zeros(3)
        self.settings = dict(resolution='640x480', focal_scale=1., mode='ideal', fps=20, stride=6)
        self.camera = self._make_camera(self.settings, self.home_pos, np.eye(3))
        self.cloud = None
        self.frame = None
        self.dirty = True
        self.last_capture = -np.inf

        body = wssop.box(pos=(0, 0, -.016), xyz_lengths=(.045, .026, .024),
                         rgb=(.25, .28, .32))
        lens = wssop.cylinder(spos=(0, 0, -.004), epos=(0, 0, .003), radius=.008,
                              segments=24, rgb=(.05, .5, .65))
        axes = wssop.frame(length_scale=.35, radius_scale=.2)
        self.markers = [(part, part.tf) for part in (body, lens, axes)]
        for part, _ in self.markers:
            part.add_to_scene(self.base.scene)

        self.controls = self.base.ui.add_panel(
            'camera', title='VirtualD405 controls', anchor=Anchor.TOP_LEFT,
            width=280, offset=16, movable=True,
            description='Cyan: measured points. Translucent: source meshes. '
                        'Click the scene for WASD / QE; R resets pose.')
        for axis, value in enumerate(self.home_pos):
            self.controls.add_slider(
                'xyz'[axis], label=f'{"XYZ"[axis]} / world', unit='m',
                min_value=-.3, max_value=.5, step=.001, value=float(value),
                continuous=True, update_hz=15, group='Optical pose',
                on_change=lambda value, axis=axis: self.set_position(axis, value))
        for axis, name in enumerate(('roll', 'pitch', 'yaw')):
            self.controls.add_slider(
                name, label=name.title(), unit='deg', min_value=-180, max_value=180,
                step=1, continuous=True, update_hz=15, group='Optical pose',
                on_change=lambda value, axis=axis: self.set_angle(axis, value))
        self.controls.add_button('reset', label='Reset pose', on_click=self.reset_pose)
        self.controls.add_select(
            'resolution', label='Resolution', group='Capture',
            options=['320x240', '640x480', '848x480', '1280x720'], value='640x480',
            on_change=lambda value: self.configure_camera(resolution=value))
        self.controls.add_slider(
            'focal_scale', label='Focal length scale', min_value=.5, max_value=2,
            step=.05, value=1, group='Capture',
            on_change=lambda value: self.configure_camera(focal_scale=value))
        self.controls.add_select(
            'mode', label='Depth mode', options=['ideal', 'd405_fast'], group='Capture',
            on_change=lambda value: self.configure_camera(mode=value))
        self.controls.add_select(
            'fps', label='Max capture rate / Hz', options=['5', '10', '20', '30'],
            value='20', group='Capture', on_change=self.set_rate)
        self.controls.add_slider(
            'stride', label='Point sampling stride', min_value=1, max_value=12,
            step=1, value=6, group='Capture', on_change=self.set_stride)
        self.controls.add_button('capture', label='Capture again', on_click=self.request_capture)

        self.preview = self.base.ui.add_panel(
            'preview', title='RGB / Depth', anchor=Anchor.TOP_RIGHT,
            width=320, offset=16, movable=True)
        self.rgb_view = self.preview.add_image('rgb', label='RGB', format='jpeg', max_fps=30)
        self.depth_view = self.preview.add_image('depth', label='Depth Z / 0.07–0.50 m', max_fps=30)
        self.preview.add_label('pose', label='Optical origin / world (m)')
        self.preview.add_label('calibration', label='Example intrinsics / pixels')
        self.preview.add_label('status', label='Last capture')
        self.preview.add_label('axes', label='Optical axes',
                               value='Red X: right · Green Y: down · Blue Z: forward')
        self._sync_pose()
        self._sync_calibration()

    @staticmethod
    def _make_camera(settings, pos, rotmat):
        width, height = map(int, settings['resolution'].split('x'))
        # Scale both focal lengths together: independent X/Y scaling would
        # stretch objects when switching between 4:3 and widescreen profiles.
        # These are example values, not a real D405 calibration/profile.
        focal = 600 * min(width / 640, height / 480) * settings['focal_scale']
        return VirtualD405(width=width, height=height,
                           fx=focal, fy=focal,
                           fps=settings['fps'], mode=settings['mode'], pos=pos, rotmat=rotmat)

    def configure_camera(self, **changes):
        settings = {**self.settings, **changes}
        # Calibration/render targets are immutable. Capture the replacement
        # before releasing the old camera, so a failed change leaves it usable.
        replacement = self._make_camera(settings, self.camera.pos, self.camera.rotmat)
        try:
            frame = replacement.capture(self.scene)
        except Exception:
            replacement.close()
            raise
        old, self.camera = self.camera, replacement
        self.settings = settings
        old.close()
        self._sync_calibration()
        self._show_frame(frame)

    def set_rate(self, value):
        # The example's scheduler owns the sampling rate; no GPU rebuild needed.
        self.settings['fps'] = int(value)

    def set_stride(self, value):
        self.settings['stride'] = int(value)
        self.dirty = True

    def set_position(self, axis, value):
        pos = self.camera.pos
        pos[axis] = value
        self.camera.pos = pos
        self._sync_pose()
        self.dirty = True

    def set_angle(self, axis, value):
        self.angles[axis] = value
        self.camera.rotmat = wum.rotmat_from_euler(*np.deg2rad(self.angles))
        self._sync_pose()
        self.dirty = True

    def reset_pose(self):
        self.angles[:] = 0
        self.camera.set_pos_rotmat(self.home_pos, np.eye(3))
        for name in ('roll', 'pitch', 'yaw'):
            self.controls.set_value(name, 0)
        self._sync_pose()
        self.dirty = True

    def request_capture(self):
        self.dirty = True

    def _sync_pose(self):
        for part, local_tf in self.markers:
            part.tf = self.camera.tf @ local_tf
        for axis, value in zip('xyz', self.camera.pos):
            self.controls.set_value(axis, float(value))
        self.preview.set_value('pose', ' / '.join(f'{v:+.4f}' for v in self.camera.pos))

    def _sync_calibration(self):
        model = self.camera.camera_model
        self.preview.set_value('calibration',
                               f'{model.width} × {model.height}\n'
                               f'fx={model.fx:.1f}, fy={model.fy:.1f}\n'
                               f'cx={model.cx:.1f}, cy={model.cy:.1f}')

    def _show_frame(self, frame):
        points, colors = frame.get_point_cloud(world_frame=True, stride=self.settings['stride'])
        # New scene membership announces the new geometry to the viewer.
        # An empty capture removes stale points without creating a zero buffer.
        if self.cloud is not None:
            self.cloud.remove_from_scene(self.base.scene)
            self.cloud = None
        if len(points):
            self.cloud = wssop.point_cloud(points, np.full_like(colors, (0, .9, 1)))
            self.cloud.add_to_scene(self.base.scene)
        self.frame = frame
        self.rgb_view.update(frame.rgb)
        self.depth_view.update(colorize_depth(frame.depth_m,
                                              value_range=(self.camera.min_depth, self.camera.max_depth),
                                              valid_mask=frame.valid_mask))
        self.preview.set_value('status', f'{len(points):,} points · {self.settings["mode"]}\n'
                               f'Capture {frame.timings_ms["capture_total"]:.1f} ms')
        self.last_capture = time.monotonic()
        self.dirty = False

    def update(self, dt):
        direction = np.array([int(self.base.is_key_pressed(key.D)) - int(self.base.is_key_pressed(key.A)),
                              int(self.base.is_key_pressed(key.E)) - int(self.base.is_key_pressed(key.Q)),
                              int(self.base.is_key_pressed(key.W)) - int(self.base.is_key_pressed(key.S))])
        if self.base.is_key_pressed_edge(key.R):
            self.reset_pose()
        elif np.any(direction):
            self.camera.pos = np.clip(self.camera.pos + direction / np.linalg.norm(direction)
                                      * .08 * min(dt, .1), -.3, .5)
            self._sync_pose()
            self.dirty = True
        # Static scene: capture only after pose/parameter changes or a request.
        if self.dirty and time.monotonic() - self.last_capture >= 1 / self.settings['fps']:
            self._show_frame(self.camera.capture(self.scene))

    def run(self):
        try:
            self.update(0)
            self.base.schedule_interval(self.update, interval=1 / 60)
            self.base.run()
        finally:
            self.camera.close()


if __name__ == '__main__':
    OverlayDemo().run()
