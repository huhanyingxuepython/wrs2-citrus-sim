"""Live RGB/depth panels and an on-demand snapshot in the WRS browser UI.

python -m examples.viewer_images          # animated NumPy images, no camera GPU
python -m examples.viewer_images --camera # VirtualD405 RGB and metric depth
"""
import argparse
import time

import numpy as np

from wrs import wssop, wvw
from wrs.viewer.web_ui import Anchor, colorize_depth


def build_demo(use_camera=False):
    base = wvw.World(cam_pos=(.7, -.6, .6), cam_lookat_pos=(0, 0, .3))
    base.set_caption('WRS / Image controls')
    block = wssop.box(pos=(.05, 0, .35), xyz_lengths=(.07, .07, .07), rgb=(.2, .5, .9))
    block.add_to_scene(base.scene)
    wssop.icosphere(pos=(-.06, 0, .27), radius=.04, rgb=(1, .4, .05)).add_to_scene(base.scene)
    camera = None
    if use_camera:
        from wrs.sensor import VirtualD405
        camera = VirtualD405(width=640, height=480, fx=600, fy=600, mode='ideal')

    panel = base.ui.add_panel('camera', title='Live images', anchor=Anchor.TOP_LEFT,
                              width=360, offset=16, movable=True)
    rgb_view = panel.add_image('rgb', label='RGB', format='jpeg', max_fps=20)
    depth_view = panel.add_image('depth', label='Depth · 0.07–0.5 m', max_fps=20)
    result = base.ui.add_panel('result', title='Snapshot', width=360, movable=True)
    snapshot = result.add_image('snapshot', label='Last captured image')
    latest = [None]
    result.add_button('capture', label='Take snapshot',
                      on_click=lambda: snapshot.update(latest[0]) if latest[0] is not None else None)
    result.add_button('clear', label='Clear snapshot', on_click=snapshot.clear)
    result.add_slider('x', label='Block X', min_value=-.1, max_value=.1, value=.05,
                      step=.001, continuous=True,
                      on_change=lambda x: setattr(block, 'pos', (x, 0, .35)))
    y, x = np.mgrid[:480, :640]

    def refresh(dt):
        if camera is not None:
            frame = camera.capture(base.scene)
            rgb, depth = frame.rgb, frame.depth_m
        else:
            phase = time.monotonic()
            rgb = np.stack((np.broadcast_to((x / 639 * 255), x.shape),
                            y / 479 * 255, 127 + 127 * np.sin(x / 60 + phase)), axis=-1).astype(np.uint8)
            depth = .07 + .43 * (1 + np.sin(x / 90 + y / 70 + phase)) / 2
        latest[0] = rgb
        rgb_view.update(rgb)
        depth_view.update(colorize_depth(depth, value_range=(.07, .5)))

    base.schedule_interval(refresh, interval=1 / 20)
    return base, camera


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', action='store_true')
    base, camera = build_demo(parser.parse_args().camera)
    try:
        base.run()
    finally:
        if camera is not None:
            camera.close()
