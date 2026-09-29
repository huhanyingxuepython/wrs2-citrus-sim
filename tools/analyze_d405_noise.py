"""Analyze a STATIC real D405 sequence and write a loadable heuristic profile.

python -m tools.analyze_d405_noise recording.npz --depth-unit-m .0001 \
    --fx 600 --fy 600 --baseline-m .018 --known-depth-m .3
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from wrs.sensor import CameraModel, D405NoiseConfig
from wrs.sensor.d405_noise_analysis import load_depth_recording, fit_noise_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path, default=Path('d405_noise_profile.json'))
    parser.add_argument('--depth-unit-m', type=float)
    parser.add_argument('--known-depth-m', type=float)
    parser.add_argument('--reference', type=Path, help='Metric NPY reference (H,W) or (T,H,W)')
    parser.add_argument('--mask', type=Path, help='Boolean NPY evaluation/board ROI')
    parser.add_argument('--fx', type=float)
    parser.add_argument('--fy', type=float)
    parser.add_argument('--cx', type=float)
    parser.add_argument('--cy', type=float)
    parser.add_argument('--baseline-m', type=float)
    parser.add_argument('--min-depth-m', type=float, default=.07)
    parser.add_argument('--preferred-max-depth-m', type=float, default=.5)
    parser.add_argument('--max-depth-m', type=float, default=2.)
    args = parser.parse_args()
    depth, rgb = load_depth_recording(args.recording, depth_unit_m=args.depth_unit_m)
    unit = args.depth_unit_m
    if unit is None and args.recording.suffix.lower() == '.npz':
        with np.load(args.recording, allow_pickle=False) as archive:
            if 'depth_unit_m' in archive:
                unit = float(archive['depth_unit_m'])
    config = D405NoiseConfig(depth_unit_m=.0001 if unit is None else unit,
        min_depth_m=args.min_depth_m, preferred_max_depth_m=args.preferred_max_depth_m,
        max_depth_m=args.max_depth_m)
    reference = None if args.reference is None else np.load(args.reference, allow_pickle=False)
    mask = None if args.mask is None else np.load(args.mask, allow_pickle=False)
    config, statistics = fit_noise_config(depth, fx=args.fx, baseline_m=args.baseline_m,
        reference=reference, known_depth_m=args.known_depth_m, evaluation_mask=mask, base_config=config)
    calibration = {}
    if args.fx is not None:
        height, width = depth.shape[1:]
        model = CameraModel(width, height, args.fx, args.fx if args.fy is None else args.fy,
                            (width-1)/2 if args.cx is None else args.cx,
                            (height-1)/2 if args.cy is None else args.cy)
        calibration['camera_model'] = asdict(model)
    if args.baseline_m is not None:
        calibration['baseline_m'] = args.baseline_m
    statistics['rgb_available'] = rgb is not None
    statistics['depth_unit_source'] = 'reported' if unit is not None else 'simulation fallback; metric recording did not report Z16 unit'
    config.save_profile(args.output, calibration=calibration, statistics=statistics,
                        provenance='static recording initial fit; unestimated fields remain heuristic')
    print(json.dumps(statistics, indent=2, allow_nan=False))
    print(f'Profile: {args.output.resolve()}')


if __name__ == '__main__':
    main()
