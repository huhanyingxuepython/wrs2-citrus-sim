"""Look at the Bunker cart with the FAFU arm on the cabinet. No tree, no physics."""
from pathlib import Path
import argparse
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from wrs import wss, wssop, wuc, wum, wvw
from wrs.robots.manipulators.fafu import FAFURobotArm
from wrs.robots.end_effectors.fafu_gripper import FAFUGripper
from agriculture.bunker_cart import add_bunker_cart
from agriculture.config import CONFIG_DIR, load_config

DEFAULT_CONFIG = CONFIG_DIR / 'presets/lab_citrus_robot.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    config = load_config(args.config)
    p = config['robot_demo']
    scene = wss.Scene()
    size = np.asarray(p['ground_size'])
    ground = wssop.box(pos=(*p['ground_center_xy'], -size[2] / 2), xyz_lengths=size,
                       rgb=p['ground_color'], collision_type=wuc.CollisionType.AABB,
                       name='ground')
    ground.add_to_scene(scene)
    cart = add_bunker_cart(scene, p)
    if cart is None:
        raise SystemExit('cart mesh missing: agriculture/assets/bunker_cart/cart_viz.stl')
    robot = FAFURobotArm(
        pos=p['base_position'],
        rotmat=wum.rotmat_from_axangle((0, 0, 1), np.deg2rad(p['base_yaw_deg'])))
    gripper = FAFUGripper()
    gripper.set_opening(p['gripper']['opening_m'])
    robot.mount(gripper, robot.tcp('flange').parent_lnk, update=True)
    robot.fk(np.deg2rad(p['ik_seed_deg']))
    robot.add_to_scene(scene)
    base = wvw.World(
        cam_pos=(1.8, -2.3, 1.5),
        cam_lookat_pos=(0.0, -0.70, 0.45),
        port=args.port)
    base.set_scene(scene)
    base.set_caption('Bunker cart / FAFU arm')
    print(f'cart at plate z={p["base_position"][2]:.3f} m  http://127.0.0.1:{args.port}/',
          flush=True)
    base.run()


if __name__ == '__main__':
    main()
