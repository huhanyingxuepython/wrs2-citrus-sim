"""Headless citrus validation, artifacts, timings and optional real/sim comparison.

python -m tools.validate_d405_noise --frames 40 --output build/d405_noise
"""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy
from PIL import Image, ImageDraw

from examples.agriculture.virtual_d405_noise import make_scene
from wrs.sensor import VirtualD405, D405NoiseModel
from wrs.sensor.d405_noise_analysis import analyze_depth_sequence, load_depth_recording, temporal_reference, write_colored_ply
from wrs.viewer.web_ui import colorize_depth


def labeled_image(array, title):
    image = Image.new('RGB', (array.shape[1], array.shape[0]+30), (245, 245, 245))
    image.paste(Image.fromarray(array), (0, 30))
    ImageDraw.Draw(image).text((8, 8), title, fill=(20, 20, 20))
    return image


def run_validation(output, *, frames=40, width=640, height=480, preset='agriculture_foliage',
                   real_recording=None, real_depth_unit_m=None, real_fx=None, real_fy=None,
                   real_baseline_m=None, real_reference=None, real_mask=None):
    if frames < 2:
        raise ValueError('at least two frames are required for temporal metrics')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scene, tree, camera = make_scene(width, height, preset)
    depths, timings, render_timings = [], [], []
    try:
        for _ in range(3):
            camera.capture(scene)
        # One geometric reference is valid because this sequence is static.
        first = camera.capture(scene, debug=True)
        for _ in range(frames):
            frame = camera.capture(scene)
            depths.append(frame.depth_m)
            timings.append(frame.timings_ms)
        depth = np.stack(depths)
        support = (first.depth_gt >= camera.min_depth) & (first.depth_gt <= camera.max_depth)
        metrics = analyze_depth_sequence(depth, reference=first.depth_gt, evaluation_mask=support,
            edge_threshold_m=camera.noise_config.depth_edge_threshold_m,
            edge_band_width_px=camera.noise_config.edge_band_width_px)
        with VirtualD405(camera.camera_model, mode='ideal', min_depth=camera.min_depth,
                max_depth=camera.max_depth, depth_scale=camera.depth_scale, far=camera.far,
                pos=camera.pos, rotmat=camera.rotmat, rgb_lighting=camera.rgb_lighting) as ideal:
            for _ in range(3):
                clean_frame = ideal.capture(scene)
            for _ in range(frames):
                clean_frame = ideal.capture(scene)
                render_timings.append(clean_frame.timings_ms['capture_total'])
        for name, cloud_frame in (('clean', clean_frame), ('noisy', first)):
            points, colors = cloud_frame.get_point_cloud(world_frame=True, stride=1)
            write_colored_ply(output/f'{name}.ply', points, colors)
        limits = (camera.min_depth, .8)
        Image.fromarray(colorize_depth(first.depth_gt, value_range=limits)).save(output/'clean_depth.png')
        Image.fromarray(colorize_depth(first.depth_m, value_range=limits)).save(output/'noisy_depth.png')
        panels = [labeled_image(first.rgb, 'RGB / citrus'),
                  labeled_image(colorize_depth(first.depth_gt, value_range=limits), 'Clean Z / Jet / 0.07-0.8 m'),
                  labeled_image(colorize_depth(first.depth_m, value_range=limits), 'Raw-like D405 / Jet / same scale')]
        montage = Image.new('RGB', (width*3, height+30))
        for index, panel in enumerate(panels):
            montage.paste(panel, (index*width, 0))
        montage.save(output/'comparison.png')
        diagnostics = first.noise_debug
        reason_colors = np.array([[85, 85, 85], [0, 0, 0], [255, 255, 255],
                                   [255, 60, 30], [0, 190, 255], [220, 0, 220]], dtype=np.uint8)
        Image.fromarray(reason_colors[diagnostics['corruption_reason']]).save(output/'corruption_reason.png')
        # An animated raw stream makes persistence and flicker directly inspectable.
        animation = [Image.fromarray(colorize_depth(z, value_range=limits)) for z in depth[:24]]
        animation[0].save(output/'temporal.gif', save_all=True, append_images=animation[1:], duration=100, loop=0)
        np.savez_compressed(output/'sequence.npz', depth_m=depth, clean_depth=first.depth_gt,
                            rgb=first.rgb, depth_unit_m=camera.depth_scale,
                            T_world_camera=first.T_world_camera, intrinsics=first.intrinsics)
        np.savez_compressed(output/'debug.npz', **diagnostics)
        runtime = {key: {'mean_ms': float(np.mean([row[key] for row in timings])),
                         'p95_ms': float(np.percentile([row[key] for row in timings], 95))}
                   for key in ('noise_model_cpu', 'noise_pointcloud_cpu', 'capture_total')}
        runtime['ideal_capture'] = dict(mean_ms=float(np.mean(render_timings)), p95_ms=float(np.percentile(render_timings, 95)))
        environment = dict(platform=platform.platform(), python=platform.python_version(),
                           processor=os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
                           numpy=np.__version__, scipy=scipy.__version__)
        report = dict(environment=environment, scene=tree.summary(), preset=preset, config=camera.noise_config.to_dict(),
                      camera_model=asdict(camera.camera_model), baseline_m=camera.baseline_m,
                      T_world_camera=first.T_world_camera.tolist(), metrics=metrics, runtime=runtime)
        # Exercise motion separately; static metrics above remain interpretable.
        camera.reset_noise(43)
        home = camera.pos
        motion = []
        for index in range(12):
            camera.pos = home + camera.rotmat[:, 0]*(.008*np.sin(index/11*np.pi))
            moved = camera.capture(scene, debug=True)
            motion.append(dict(motion_px=moved.noise_debug['motion_px'],
                               invalid_pixels=int((moved.depth_raw == 0).sum())))
        report['motion_smoke'] = motion
        if real_recording is not None:
            if real_fx is None or real_baseline_m is None:
                raise ValueError('real comparison requires --real-fx and --real-baseline-m from the stream')
            real, rgb = load_depth_recording(real_recording, depth_unit_m=real_depth_unit_m)
            reference = temporal_reference(real) if real_reference is None else np.load(real_reference, allow_pickle=False)
            mask = None if real_mask is None else np.load(real_mask, allow_pickle=False)
            from wrs.sensor import CameraModel
            h, w = real.shape[1:]
            model = CameraModel(w, h, real_fx, real_fx if real_fy is None else real_fy, (w-1)/2, (h-1)/2)
            unit = real_depth_unit_m
            if unit is None and Path(real_recording).suffix.lower() == '.npz':
                with np.load(real_recording, allow_pickle=False) as recording:
                    if 'depth_unit_m' in recording:
                        unit = float(recording['depth_unit_m'])
            comparison_config = (camera.noise_config if unit is None else
                                 replace(camera.noise_config, depth_unit_m=unit))
            simulator = D405NoiseModel(comparison_config, seed=42)
            simulated = np.stack([simulator.apply(reference if reference.ndim == 2 else reference[i], model,
                real_baseline_m, rgb=rgb if rgb is None or rgb.ndim == 3 else rgb[i]).depth for i in range(len(real))])
            supported = reference > 0 if mask is None else mask & (reference > 0)
            report['real_comparison'] = dict(
                reference_note='Temporal median cannot reveal permanent holes or fixed bias; use supplied reference/ROI.',
                depth_unit_m=comparison_config.depth_unit_m,
                depth_unit_source='reported' if unit is not None else 'simulation fallback; recording omitted Z16 unit',
                real=analyze_depth_sequence(real, reference=reference, evaluation_mask=supported),
                simulated=analyze_depth_sequence(simulated, reference=reference, evaluation_mask=supported))
        camera.noise_config.save_profile(output/'d405_noise_profile.json',
            calibration={'camera_model': asdict(camera.camera_model), 'baseline_m': camera.baseline_m},
            statistics=metrics, provenance='heuristic citrus validation; not fitted to real D405')
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        return report
    finally:
        camera.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/d405_noise'))
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--preset', choices=('ideal', 'default', 'agriculture_foliage', 'harsh'), default='agriculture_foliage')
    parser.add_argument('--real-recording', type=Path)
    parser.add_argument('--real-depth-unit-m', type=float)
    parser.add_argument('--real-fx', type=float)
    parser.add_argument('--real-fy', type=float)
    parser.add_argument('--real-baseline-m', type=float)
    parser.add_argument('--real-reference', type=Path)
    parser.add_argument('--real-mask', type=Path)
    report = run_validation(**vars(parser.parse_args()))
    print(json.dumps({'metrics': report['metrics'], 'runtime': report['runtime']}, indent=2))


if __name__ == '__main__':
    main()
