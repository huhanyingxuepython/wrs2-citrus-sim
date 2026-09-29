"""Move a VirtualD405 and view live RGB/depth in the WRS browser.

Run from the repository root: python -m examples.virtual_d405_rgbd
Click the 3D view, then hold W/S, A/D or Q/E; R resets the optical pose.
"""

import numpy as np

from wrs import key, wss, wssop, wvw
from wrs.sensor import VirtualD405
from wrs.viewer.web_ui import Anchor, colorize_depth


# Only these objects participate in sensor capture.
scene = wss.Scene()
wssop.icosphere(pos=(-.055, -.015, .29), radius=.045, subdivisions=3,
               rgb=(1, .4, .03)).add_to_scene(scene)
wssop.box(pos=(.06, .025, .37), xyz_lengths=(.08, .07, .08),
          rgb=(.15, .55, .9)).add_to_scene(scene)
wssop.cylinder(spos=(-.12, .065, .40), epos=(.12, .065, .40), radius=.008,
               rgb=(.35, .18, .06)).add_to_scene(scene)

# Example intrinsics, not a real D405 calibration. Optical +Z points forward.
camera = VirtualD405(width=640, height=480, fx=600, fy=600, mode='ideal', fps=20)
home_pos = camera.pos.copy()
move_speed = .08  # Meters per second while a key is held.
button_step = .01  # Meters per button click.

base = wvw.World(cam_pos=(.55, -.65, .35), cam_lookat_pos=(0, 0, .18))
base.set_caption('VirtualD405 / live RGB-D')
for obj in scene:
    obj.add_to_scene(base.scene)

# A schematic body behind the optical origin, a lens, and 7 cm RGB axes.
# Display geometry stays out of the capture scene, so it cannot hide objects.
body = wssop.box(pos=(0, 0, -.016), xyz_lengths=(.045, .026, .024),
                 rgb=(.25, .28, .32))
lens = wssop.cylinder(spos=(0, 0, -.004), epos=(0, 0, .003), radius=.008,
                      segments=24, rgb=(.05, .5, .65))
axes = wssop.frame(length_scale=.35, radius_scale=.2)
markers = [(part, part.tf) for part in (body, lens, axes)]
for part, local_tf in markers:
    part.add_to_scene(base.scene)

base.ui.configure(title='Live RGB / Depth', anchor=Anchor.TOP_RIGHT, width=340,
                  offset=16, movable=True,
                  description='20 Hz target. Jet: blue = near, red = far, black = invalid.')
rgb_view = base.ui.add_image('rgb', label='RGB', max_fps=camera.fps)
depth_view = base.ui.add_image('depth', label='Depth Z / Jet / 0.07-0.50 m', max_fps=camera.fps)

controls = base.ui.add_panel('camera', title='VirtualD405 position',
                             anchor=Anchor.TOP_LEFT, width=250, offset=16, movable=True,
                             description='Click the 3D view to use keys. Mouse drag orbits the viewer.')
for axis in 'xyz':
    controls.add_label(axis, label=f'{axis.upper()} / world (m)')
controls.add_label('axes', label='Optical axes',
                    value='Red X: right\nGreen Y: down\nBlue Z: forward')


def move_camera(delta):
    camera.pos = camera.pos + np.asarray(delta)


def reset_camera():
    camera.pos = home_pos


# This example keeps rotation fixed, so optical axes also match world axes.
controls.add_button('left', label='A / left: X - 1 cm',
                     on_click=lambda: move_camera((-button_step, 0, 0)))
controls.add_button('right', label='D / right: X + 1 cm',
                     on_click=lambda: move_camera((button_step, 0, 0)))
controls.add_button('up', label='Q / up: Y - 1 cm',
                     on_click=lambda: move_camera((0, -button_step, 0)))
controls.add_button('down', label='E / down: Y + 1 cm',
                     on_click=lambda: move_camera((0, button_step, 0)))
controls.add_button('forward', label='W / forward: Z + 1 cm',
                     on_click=lambda: move_camera((0, 0, button_step)))
controls.add_button('back', label='S / back: Z - 1 cm',
                     on_click=lambda: move_camera((0, 0, -button_step)))
controls.add_button('reset', label='R / reset position', on_click=reset_camera)


def update(dt):
    direction = np.array([int(base.is_key_pressed(key.D)) - int(base.is_key_pressed(key.A)),
                          int(base.is_key_pressed(key.E)) - int(base.is_key_pressed(key.Q)),
                          int(base.is_key_pressed(key.W)) - int(base.is_key_pressed(key.S))])
    if base.is_key_pressed_edge(key.R):
        reset_camera()
    elif np.any(direction):
        # Normalize diagonals and limit jumps after a stalled frame.
        move_camera(direction / np.linalg.norm(direction) * move_speed * min(dt, .1))

    frame = camera.capture(scene)
    rgb_view.update(frame.rgb)
    depth_view.update(colorize_depth(frame.depth_m,
                                     value_range=(camera.min_depth, camera.max_depth),
                                     valid_mask=frame.valid_mask))
    for part, local_tf in markers:
        part.tf = frame.T_world_camera @ local_tf
    for axis, value in zip('xyz', frame.T_world_camera[:3, 3]):
        controls.set_value(axis, f'{value:+.4f} m')


if __name__ == '__main__':
    try:
        update(0)
        base.schedule_interval(update, interval=1 / camera.fps)
        base.run()
    finally:
        camera.close()
