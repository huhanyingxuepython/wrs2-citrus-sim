"""Headless capture/benchmark; add --show to display the measured WRS cloud.

    python -m examples.virtual_depth_camera --benchmark 100
    python -m examples.virtual_depth_camera --mode d405_fast --benchmark 100
    python -m examples.virtual_depth_camera --distortion --show
"""

import argparse
from time import perf_counter

import numpy as np

from wrs import wss, wssop
from wrs.sensor import CameraModel, StereoDepthNoise, VirtualD405
from wrs.utils.gpu_device import get_adapter


def make_scene():
    """Simple fruit-like geometry, entirely in meters in the optical frame."""
    scene = wss.Scene()
    for pos, radius in (((-.07, -.035, .29), .035),
                        ((.06, .025, .34), .045),
                        ((-.01, .09, .40), .03)):
        scene.add(wssop.icosphere(pos=pos, radius=radius, subdivisions=3, rgb=(1, .4, .03)))
    scene.add(wssop.cylinder(spos=(-.12, -.1, .42), epos=(.12, .1, .42),
                             radius=.008, segments=16, rgb=(.3, .15, .04)))
    scene.add(wssop.box(pos=(0, 0, .49), xyz_lengths=(.8, .6, .02), rgb=(.12, .3, .12)))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('ideal', 'd405_fast'), default='ideal')
    parser.add_argument('--benchmark', type=int, default=0, metavar='FRAMES')
    parser.add_argument('--distortion', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.benchmark < 0:
        parser.error('--benchmark must be nonnegative')
    scene = make_scene()
    # Example calibration, intentionally not presented as real D405 intrinsics.
    model = CameraModel(640, 480, 430, 435, 317.2, 241.1,
                        'brown_conrady' if args.distortion else 'none',
                        (-.08, .01, .001, -.001, 0) if args.distortion else (0, 0, 0, 0, 0))
    noise = (StereoDepthNoise(disparity_std_px=.08, disparity_step_px=1/32,
                              dropout_rate=.005, edge_dropout_strength=.1)
             if args.mode == 'd405_fast' else None)
    with VirtualD405(model, mode=args.mode, noise=noise, seed=42, fps=30) as camera:
        frame = camera.capture(scene, T_world_mount=np.eye(4))
        print(f'{args.mode}: {model.width}x{model.height}, '
              f'{frame.valid_mask.sum()} valid pixels, depth_scale={camera.depth_scale:g} m')
        print(f'GPU: {get_adapter().info["device"]}')
        triangles = sum(len(m.geom.fs) for obj in scene for m in obj.visuals)
        print(f'Scene: {len(scene.sobjs)} objects, {triangles} triangles')
        if args.benchmark:
            for _ in range(5):
                camera.capture(scene)
            rows = []
            for _ in range(args.benchmark):
                started = perf_counter()
                frame = camera.capture(scene)
                compact_started = perf_counter()
                # Dense camera/world XYZ are already computed on the GPU.
                # Access both to include the final CPU compaction into Nx3 arrays.
                _ = frame.points_cam, frame.points_world
                row = dict(frame.timings_ms)
                row['pointcloud_compact'] = (perf_counter()-compact_started)*1000
                row['total_with_points'] = (perf_counter()-started)*1000
                rows.append(row)
            for key in rows[0]:
                values = np.array([row[key] for row in rows])
                print(f'{key:28s} mean={values.mean():7.3f} ms  p95={np.percentile(values, 95):7.3f} ms')
            average = np.mean([row['total_with_points'] for row in rows])
            print(f'Average FPS including both point clouds: {1000/average:.1f}')
            print('Encode/submit are CPU wall times; readback_wait includes GPU execution, '
                  'synchronization, transfer and unpacking. GPU stages are not separately timestamped.')
        if args.show:
            from wrs import wvw
            world = wvw.World(cam_pos=(.45, -.45, .1), cam_lookat_pos=(0, 0, .33))
            points, colors = frame.get_point_cloud(world_frame=True, stride=2)
            wssop.point_cloud(points, colors).add_to_scene(world.scene)
            wssop.frame().add_to_scene(world.scene)
            world.run()


if __name__ == '__main__':
    main()
