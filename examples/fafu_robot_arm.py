"""Display the migrated FAFU arm with its original two-finger gripper."""
import numpy as np
from wrs import wvw, wssop
from wrs.robots.manipulators.fafu import fafu_with_gripper


def main():
    world = wvw.World(cam_pos=(1.2, 1.2, 0.8), cam_lookat_pos=(0, 0, 0.25))
    wssop.frame().add_to_scene(world.scene)
    arm, gripper = fafu_with_gripper(jaw_width=0.05)
    arm.fk(qs=np.array([0, 1.1, 1.5, -0.4, 0.3, 0.2]))
    arm.add_to_scene(world.scene)
    arm.toggle_tcp('flange')
    gripper.toggle_tcp('grasp_center')
    print('Grasp center:', gripper.tcp('grasp_center').pos)
    world.run()


if __name__ == '__main__':
    main()
