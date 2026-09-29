"""Display a VirtualD405 colored point cloud in the WRS viewer.

Run from the repository root: python -m examples.virtual_d405_point_cloud
"""

from wrs import wss, wssop, wvw
from wrs.sensor import VirtualD405


scene = wss.Scene()
wssop.icosphere(pos=(-.055, -.015, .29), radius=.045, subdivisions=3,
               rgb=(1, .4, .03)).add_to_scene(scene)
wssop.box(pos=(.06, .025, .37), xyz_lengths=(.08, .07, .08),
          rgb=(.15, .55, .9)).add_to_scene(scene)
wssop.cylinder(spos=(-.12, .065, .40), epos=(.12, .065, .40), radius=.008,
               rgb=(.35, .18, .06)).add_to_scene(scene)

# Example intrinsics, not a real D405 calibration.
with VirtualD405(width=640, height=480, fx=600, fy=600, mode='ideal') as camera:
    frame = camera.capture(scene)

# Subsample pixels before removing invalid depth; XYZ and RGB stay paired.
points, colors = frame.get_point_cloud(world_frame=True, stride=3)
print(f'{len(points)} points; XYZ in meters, RGB in [0, 1]')

base = wvw.World(cam_pos=(.4, -.45, .12), cam_lookat_pos=(0, 0, .32))
base.set_caption('VirtualD405 - colored point cloud')
wssop.point_cloud(points, colors).add_to_scene(base.scene)
base.run()
