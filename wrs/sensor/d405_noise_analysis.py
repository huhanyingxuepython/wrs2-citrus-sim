"""Offline metrics and conservative initial fitting for real/simulated depth sequences."""

from dataclasses import replace
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .d405_noise import depth_edges
from .d405_noise_config import D405NoiseConfig


def load_depth_recording(path, *, depth_unit_m=None):
    """Read metric NPY/NPZ, or sorted Z16 PNG files in a directory.

    NPZ keys: depth_m OR depth_raw, optional depth_unit_m and rgb (RGB uint8).
    Integer inputs always require an explicit unit, never a guessed D400 scale.
    Returned arrays are (T,H,W) float32 meters and optional (T,H,W,3) RGB.
    """
    path = Path(path)
    rgb = None
    if path.is_dir():
        files = sorted(path.glob('*.png'))
        if not files:
            raise ValueError('directory must contain depth PNG frames')
        depth = np.stack([np.asarray(Image.open(file)) for file in files])
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as recording:
            key = 'depth_m' if 'depth_m' in recording else 'depth_raw'
            depth = recording[key].copy()
            if 'rgb' in recording:
                rgb = recording['rgb'].copy()
            if depth_unit_m is None and 'depth_unit_m' in recording:
                depth_unit_m = float(recording['depth_unit_m'])
            if key == 'depth_raw' and depth.dtype.kind not in 'ui':
                raise ValueError('depth_raw must be integer Z16 units')
    else:
        depth = np.load(path, allow_pickle=False)
    if depth.ndim == 2:
        depth = depth[None]
    if depth.ndim != 3 or 0 in depth.shape or depth.dtype.kind not in 'uif':
        raise ValueError('depth sequence must have nonempty shape (T,H,W)')
    if depth.dtype.kind in 'ui':
        if depth_unit_m is None or not np.isfinite(depth_unit_m) or depth_unit_m <= 0:
            raise ValueError('integer recordings require their reported positive depth_unit_m')
        depth = depth.astype(np.float32) * depth_unit_m
    depth = np.asarray(depth, dtype=np.float32)
    if rgb is not None and (rgb.shape not in ((*depth.shape, 3), (*depth.shape[1:], 3)) or rgb.dtype != np.uint8):
        raise ValueError('RGB must be uint8 (T,H,W,3) or a static (H,W,3) image')
    return depth, rgb


def temporal_reference(depth):
    """Static-scene temporal median; never-observed pixels remain zero."""
    samples = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        median = np.nanmedian(samples, axis=0)
    return np.nan_to_num(median, nan=0).astype(np.float32)


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _quantiles(values):
    values = np.asarray(values)
    return {f'p{q}': None if not values.size else float(np.percentile(values, q))
            for q in (10, 50, 90, 95, 99, 100)}


def _temporal_std(values, valid):
    count = valid.sum(axis=0)
    values = np.where(valid, values, 0).astype(np.float64)
    mean = values.sum(axis=0)/np.maximum(count, 1)
    variance = np.where(valid, (values-mean)**2, 0).sum(axis=0)/np.maximum(count, 1)
    return np.sqrt(variance)[count >= 2]


def analyze_depth_sequence(depth, *, reference=None, known_depth_m=None, evaluation_mask=None,
                           edge_threshold_m=.005, edge_band_width_px=3, outlier_threshold_m=.01):
    """Compute quality/morphology metrics with explicit denominator and reference metadata.

    reference may be (H,W) or (T,H,W). Temporal-median references require a
    stationary scene/camera and cannot reveal fixed biases or permanent holes.
    Fill/invalid rates use the evaluation ROI; error metrics require reference
    support too. Outlier rates use supported pixels, including missing ones.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 3 or 0 in depth.shape:
        raise ValueError('depth must be nonempty (T,H,W)')
    if reference is not None and known_depth_m is not None:
        raise ValueError('supply reference OR known_depth_m')
    if not np.isfinite(outlier_threshold_m) or outlier_threshold_m <= 0:
        raise ValueError('outlier_threshold_m must be positive')
    if known_depth_m is not None:
        if not np.isfinite(known_depth_m) or known_depth_m <= 0:
            raise ValueError('known_depth_m must be positive')
        reference = np.full(depth.shape[1:], known_depth_m, np.float32)
        source = 'known frontoparallel board Z (apply evaluation_mask for board ROI)'
    elif reference is None:
        reference = temporal_reference(depth)
        source = 'static temporal median; fixed bias/permanently missing surfaces unobservable'
    else:
        source = 'supplied reference'
    reference = np.asarray(reference, dtype=np.float32)
    if reference.shape not in (depth.shape, depth.shape[1:]):
        raise ValueError('reference must be (H,W) or (T,H,W)')
    ref = np.broadcast_to(reference, depth.shape)
    if evaluation_mask is None:
        roi = np.ones(depth.shape, bool)
    else:
        mask = np.asarray(evaluation_mask)
        if mask.dtype != bool or mask.shape not in (depth.shape, depth.shape[1:]):
            raise ValueError('evaluation_mask must be bool (H,W) or (T,H,W)')
        roi = np.broadcast_to(mask, depth.shape)
    finite = np.isfinite(depth) & (depth > 0)
    valid = finite & roi
    support = (ref > 0) & np.isfinite(ref) & roi
    measured = valid & support
    errors = np.where(measured, depth-ref, 0)
    magnitude = np.abs(errors)
    gross = measured & (magnitude > outlier_threshold_m)
    invalid = ~finite & roi
    if reference.ndim == 2:
        _, _, distance = depth_edges(reference, edge_threshold_m)
        edge = np.broadcast_to(distance <= edge_band_width_px, depth.shape) & support
    else:
        edge = np.stack([depth_edges(frame, edge_threshold_m)[2] <= edge_band_width_px for frame in ref]) & support
    areas = []
    for frame in invalid:
        components, count = ndi.label(frame, structure=np.ones((3, 3)))
        if count:
            areas.extend(np.bincount(components.ravel())[1:].tolist())
    areas = np.asarray(areas)
    temporal = _temporal_std(depth, valid)
    residual_std = _temporal_std(errors, measured)
    inlier_std = _temporal_std(errors, measured & ~gross)
    previous_invalid = invalid[:-1] & roi[1:]
    interior = support & ~edge
    values = magnitude[measured]
    bins = [1, 2, 5, 10, 25, 100, 500, 2000, np.inf]
    histogram = np.histogram(areas, bins=bins)[0]
    return dict(frames=int(depth.shape[0]), resolution=[int(depth.shape[2]), int(depth.shape[1])],
        reference=source, reference_coverage=_ratio(support.sum(), roi.sum()),
        fill_rate=_ratio(valid.sum(), roi.sum()), full_frame_fill_rate=float(finite.mean()),
        invalid_rate=_ratio(invalid.sum(), roi.sum()),
        valid_pixel_depth_MAE_m=None if not values.size else float(values.mean()),
        valid_pixel_depth_RMSE_m=None if not values.size else float(np.sqrt(np.mean(values.astype(float)**2))),
        temporal_depth_std_m=None if not temporal.size else float(temporal.mean()),
        temporal_residual_std_m=None if not residual_std.size else float(residual_std.mean()),
        temporal_inlier_residual_std_m=None if not inlier_std.size else float(inlier_std.mean()),
        temporal_std_quantiles_m=_quantiles(temporal), gross_outlier_rate=_ratio(gross.sum(), support.sum()),
        outlier_threshold_m=float(outlier_threshold_m), invalid_persistence=_ratio(
            (previous_invalid & invalid[1:]).sum(), previous_invalid.sum()),
        edge_invalid_rate=_ratio((invalid & edge).sum(), edge.sum()),
        edge_outlier_rate=_ratio((gross & edge).sum(), edge.sum()),
        interior_invalid_rate=_ratio((invalid & interior).sum(), interior.sum()),
        interior_outlier_rate=_ratio((gross & interior).sum(), interior.sum()),
        hole_components=dict(count=int(areas.size), quantiles_px=_quantiles(areas),
            histogram={f'{bins[i]}:{bins[i+1]}': int(value) for i, value in enumerate(histogram)}),
        outlier_magnitude_quantiles_m=_quantiles(magnitude[gross]),
        error_magnitude_quantiles_m=_quantiles(values))


def fit_noise_config(depth, *, fx=None, baseline_m=None, known_depth_m=None, reference=None,
                     evaluation_mask=None, base_config=None):
    """Estimate initial parameters, retaining unidentifiable gains as heuristics.

    This is an initialization for multi-scene fitting, not a complete system
    identification. Sigma/bias/rho require the actual fx and stereo baseline.
    Absolute bias requires a known/supplied reference, not temporal medians.
    """
    c = D405NoiseConfig() if base_config is None else base_config
    depth = np.asarray(depth, dtype=np.float32)
    stats = analyze_depth_sequence(depth, reference=reference, known_depth_m=known_depth_m,
        evaluation_mask=evaluation_mask, edge_threshold_m=c.depth_edge_threshold_m,
        edge_band_width_px=c.edge_band_width_px)
    ref = (np.full(depth.shape[1:], known_depth_m, np.float32) if known_depth_m is not None else
           temporal_reference(depth) if reference is None else np.asarray(reference, dtype=np.float32))
    ref = np.broadcast_to(ref, depth.shape)
    changes = {}
    interior = stats['interior_invalid_rate']
    if interior is not None:
        changes['base_invalid_prob'] = min(interior, .95)
    if stats['edge_invalid_rate'] is not None and interior is not None:
        changes['edge_invalid_gain'] = max(0., stats['edge_invalid_rate'] - interior)
    if stats['interior_outlier_rate'] is not None:
        changes['base_outlier_prob'] = stats['interior_outlier_rate']
    if stats['edge_outlier_rate'] is not None and stats['interior_outlier_rate'] is not None:
        changes['edge_outlier_gain'] = max(0., stats['edge_outlier_rate'] - stats['interior_outlier_rate'])
    p, stay = stats['invalid_rate'], stats['invalid_persistence']
    if p is not None and stay is not None and p < 1:
        changes['invalid_persistence'] = float(np.clip((stay-p)/(1-p), 0, .98))
    if fx is not None and baseline_m is not None:
        if not np.isfinite([fx, baseline_m]).all() or min(fx, baseline_m) <= 0:
            raise ValueError('fx and baseline_m must be positive and finite')
        valid = (depth > 0) & np.isfinite(depth) & (ref > 0) & np.isfinite(ref)
        valid &= np.abs(depth-ref) <= stats['outlier_threshold_m']
        if evaluation_mask is not None:
            valid &= evaluation_mask
        residual = (np.divide(fx*baseline_m, depth, out=np.zeros_like(depth), where=valid)
                    - np.divide(fx*baseline_m, ref, out=np.zeros_like(depth), where=valid))
        samples = residual[valid]
        if samples.size:
            center = float(np.median(samples))
            changes['disparity_sigma_px'] = float(1.4826*np.median(np.abs(samples-center)))
            if known_depth_m is not None or reference is not None:
                changes['disparity_bias_px'] = center
        paired = valid[1:] & valid[:-1]
        if paired.sum() > 2:
            a, b = residual[:-1][paired].astype(float), residual[1:][paired].astype(float)
            a -= a.mean()
            b -= b.mean()
            norm = np.linalg.norm(a)*np.linalg.norm(b)
            if norm > 0:
                changes['temporal_rho'] = float(np.clip(np.dot(a, b)/norm, 0, .98))
    stats['fit'] = dict(estimated_fields=sorted(changes), absolute_bias_observable=known_depth_m is not None or reference is not None,
                       status='initial estimates; coupled cue gains, component sizes and outcome weights need multi-scene tuning')
    return replace(c, **changes), stats


def write_colored_ply(path, points, colors):
    """Write binary little-endian XYZ/RGB PLY without an extra point-cloud library."""
    points, colors = np.asarray(points), np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError('points and colors must be (N,3)')
    records = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    rgb = np.rint(np.clip(colors, 0, 1)*255).astype(np.uint8)
    for index, key in enumerate(('x', 'y', 'z')):
        records[key] = points[:, index]
    for index, key in enumerate(('red', 'green', 'blue')):
        records[key] = rgb[:, index]
    header = ('ply\nformat binary_little_endian 1.0\n' + f'element vertex {len(points)}\n'
              + 'property float x\nproperty float y\nproperty float z\n'
              + 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(records.tobytes())
