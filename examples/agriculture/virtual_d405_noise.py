"""Live raw-like D405 on the existing citrus tree; no command-line options.

python -m examples.agriculture.virtual_d405_noise
"""

import numpy as np

from wrs import wss, wssop, wvw
from wrs.sensor import CameraModel, VirtualD405, RGBLighting
from wrs.viewer.web_ui import Anchor, colorize_depth
from examples.agriculture.citrus_tree.generator import load_config, generate_tree
from examples.agriculture.citrus_tree.builder import build_tree


def make_scene(width=320, height=240, preset='agriculture_foliage'):
    """Look toward a fruit from 30 cm, with neighboring leaves/twigs in view."""
    config = load_config()
    tree = build_tree(generate_tree(config, seed=42), config)
    scene = wss.Scene()
    tree.add_to_scene(scene)
    target = tree.fruit_objects[0].pos
    position = target + np.array([0, -.30, .035])
    forward = target-position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 0, 1))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    model = CameraModel(width, height, width*.85, width*.85, (width-1)/2, (height-1)/2)
    camera = VirtualD405(model, noise_config=preset, seed=42, fps=15,
                         rgb_lighting=RGBLighting(), pos=position,
                         rotmat=np.column_stack((right, down, forward)))
    return scene, tree, camera


def main():
    scene, tree, camera = make_scene()
    base = wvw.World(cam_pos=(.9, -1.2, .9), cam_lookat_pos=(0, 0, .65))
    base.set_caption('Citrus / raw-like VirtualD405')
    for obj in scene:
        ghost = obj.clone()
        ghost.alpha = .18
        ghost.add_to_scene(base.scene)
    axes = wssop.frame_from_tf(camera.tf, length_scale=.3, radius_scale=.2)
    axes.add_to_scene(base.scene)
    base.ui.configure(title='Raw-like D405 / agriculture', width=330,
                      description='Blue: near. Red: far. Black: missing. No hole filling.')
    rgb = base.ui.add_image('rgb', label='RGB', max_fps=15)
    noisy = base.ui.add_image('noisy', label='Raw-like depth / Jet', max_fps=15)
    diagnostics = base.ui.add_panel('diagnostics', title='Simulation diagnostics',
                                    anchor=Anchor.TOP_LEFT, width=300)
    clean = diagnostics.add_image('clean', label='Clean geometry / not a policy input', max_fps=15)
    diagnostics.add_label('stats', label='Capture')
    display = {'cloud': None, 'motion': False, 'time': 0.}
    initial = camera.pos
    diagnostics.add_checkbox('motion', label='Small wrist-like translation',
                              on_change=lambda value: display.update(motion=value))

    def refresh(dt):
        display['time'] += dt
        camera.pos = initial + camera.rotmat[:, 0] * (.008*np.sin(display['time']) if display['motion'] else 0)
        frame = camera.capture(scene)
        axes.tf = frame.T_world_camera
        rgb.update(frame.rgb)
        limits = (camera.min_depth, .8)
        noisy.update(colorize_depth(frame.depth_m, value_range=limits))
        clean.update(colorize_depth(frame.depth_gt, value_range=limits))
        if display['cloud'] is not None:
            display['cloud'].remove_from_scene(base.scene)
        points, colors = frame.get_point_cloud(world_frame=True, stride=3)
        display['cloud'] = None
        if len(points):
            display['cloud'] = wssop.point_cloud(points, colors)
            display['cloud'].add_to_scene(base.scene)
        support = (frame.depth_gt >= camera.min_depth) & (frame.depth_gt <= camera.max_depth)
        fill = frame.valid_mask.sum()/max(1, support.sum())
        diagnostics.set_value('stats', f'Fill on geometry: {fill:.1%}\n'
            f'Camera XYZ (m): {camera.pos[0]:+.3f}, {camera.pos[1]:+.3f}, {camera.pos[2]:+.3f}\n'
            f'Noise: {frame.timings_ms["noise_model_cpu"]:.1f} ms\n'
            f'Capture: {frame.timings_ms["capture_total"]:.1f} ms')

    try:
        refresh(0)
        base.schedule_interval(refresh, 1/camera.fps)
        base.run()
    finally:
        camera.close()


if __name__ == '__main__':
    main()
