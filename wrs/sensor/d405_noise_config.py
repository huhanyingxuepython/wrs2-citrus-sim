"""Serializable analytic D405 profiles; values are simulation defaults, not factory data."""

from dataclasses import asdict, dataclass, field, fields, replace
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class D405NoiseConfig:
    """Raw-like stereo corruption controls (meters and image pixels).

    See docs/tutorials/d405_noise.md for every default and its interpretation.
    Range and depth units are profile-specific; explicit VirtualD405 constructor
    calibration overrides these values. Semantic keys are stringified IDs/names.
    """

    enabled: bool = True
    disparity_sigma_px: float = .06
    disparity_bias_px: float = 0.
    spatial_correlation_px: float = .6
    temporal_rho: float = .75
    base_invalid_prob: float = .002
    edge_invalid_gain: float = .16
    thin_structure_invalid_gain: float = .16
    low_texture_invalid_gain: float = .025
    grazing_invalid_gain: float = .06
    near_range_invalid_gain: float = .3
    far_range_invalid_gain: float = .25
    lighting_invalid_gain: float = .025
    material_invalid_gain: float = .08
    occlusion_invalid_gain: float = .2
    base_outlier_prob: float = .0005
    edge_outlier_gain: float = .09
    thin_structure_outlier_gain: float = .08
    low_texture_outlier_gain: float = .008
    grazing_outlier_gain: float = .02
    lighting_outlier_gain: float = .01
    material_outlier_gain: float = .05
    far_range_outlier_gain: float = .03
    background_takeover_weight: float = .45
    foreground_bleed_weight: float = .20
    wrong_disparity_weight: float = .20
    reject_to_invalid_weight: float = .15
    depth_edge_threshold_m: float = .005
    edge_band_width_px: int = 3
    thin_structure_width_px: int = 4
    epipolar_search_px: int = 12
    horizontal_edge_weight: float = .75
    hole_correlation_px: float = 1.4
    failure_field_stride: int = 2
    epipolar_correlation_ratio: float = 1.8
    texture_window_px: int = 5
    texture_threshold: float = .035
    texture_gain: float = 1.
    dark_threshold: float = .08
    saturated_threshold: float = .97
    grazing_cos_threshold: float = .35
    invalid_persistence: float = .70
    outlier_persistence: float = .55
    temporal_depth_threshold_m: float = .01
    temporal_reset_motion_px: float = 4.
    motion_invalid_gain: float = 0.
    motion_outlier_gain: float = 0.
    motion_reference_px: float = 2.
    near_degradation_width_m: float = .025
    far_degradation_width_m: float = .5
    far_disparity_noise_gain: float = 1.5
    occlusion_tolerance_m: float = .002
    depth_unit_m: float = .0001
    min_depth_m: float = .07
    preferred_max_depth_m: float = .5
    max_depth_m: float = 2.
    semantic_multipliers: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be bool')
        for item in fields(self):
            name, value = item.name, getattr(self, item.name)
            if name in ('enabled', 'semantic_multipliers'):
                continue
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f'{name} must be finite numeric')
            if name != 'disparity_bias_px' and value < 0:
                raise ValueError(f'{name} must be nonnegative')
            if name.endswith('_prob') or name in ('temporal_rho', 'invalid_persistence',
                    'outlier_persistence', 'horizontal_edge_weight', 'grazing_cos_threshold'):
                if not 0 <= value <= 1:
                    raise ValueError(f'{name} must be in [0, 1]')
        for name in ('edge_band_width_px', 'thin_structure_width_px', 'epipolar_search_px',
                     'texture_window_px', 'failure_field_stride'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.texture_window_px % 2 != 1:
            raise ValueError('texture_window_px must be odd')
        for name in ('hole_correlation_px', 'epipolar_correlation_ratio', 'texture_threshold',
                     'depth_edge_threshold_m', 'near_degradation_width_m', 'far_degradation_width_m',
                     'motion_reference_px', 'temporal_reset_motion_px', 'depth_unit_m'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0 < self.min_depth_m < self.preferred_max_depth_m <= self.max_depth_m:
            raise ValueError('require 0 < min_depth_m < preferred_max_depth_m <= max_depth_m')
        if not 0 <= self.dark_threshold < self.saturated_threshold <= 1:
            raise ValueError('require 0 <= dark_threshold < saturated_threshold <= 1')
        if self.max_depth_m / self.depth_unit_m > 65535 + 1e-6:
            raise ValueError('max_depth_m does not fit Z16 at depth_unit_m')
        if sum(getattr(self, name) for name in ('background_takeover_weight',
                'foreground_bleed_weight', 'wrong_disparity_weight', 'reject_to_invalid_weight')) <= 0:
            raise ValueError('at least one mismatch outcome must have positive weight')
        multipliers = {str(key): float(value) for key, value in self.semantic_multipliers.items()}
        if any(not np.isfinite(value) or value < 0 for value in multipliers.values()):
            raise ValueError('semantic multipliers must be finite and nonnegative')
        object.__setattr__(self, 'semantic_multipliers', multipliers)

    @classmethod
    def preset(cls, name='default', **overrides):
        """Build ideal/default/agriculture_foliage/harsh heuristic presets."""
        if name == 'default':
            config = cls()
        elif name == 'ideal':
            settings = {f.name: 0. for f in fields(cls)
                        if f.name.endswith('_gain') or f.name.endswith('_prob')}
            config = cls(**(settings | {'disparity_sigma_px': .025}))
        elif name == 'agriculture_foliage':
            config = cls(edge_invalid_gain=.23, thin_structure_invalid_gain=.24,
                         edge_outlier_gain=.14, thin_structure_outlier_gain=.13,
                         motion_invalid_gain=.035, motion_outlier_gain=.02,
                         semantic_multipliers={'FOLIAGE': 1.15, 'TWIG': 1.3})
        elif name == 'harsh':
            config = cls(disparity_sigma_px=.12, base_invalid_prob=.01,
                         edge_invalid_gain=.35, thin_structure_invalid_gain=.3,
                         low_texture_invalid_gain=.08, edge_outlier_gain=.22,
                         thin_structure_outlier_gain=.18, far_range_invalid_gain=.4,
                         motion_invalid_gain=.07, motion_outlier_gain=.04)
        else:
            raise ValueError(f'unknown D405 preset: {name}')
        return replace(config, **overrides)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        """Reject misspelled/unknown parameters instead of silently dropping them."""
        return cls(**values)

    @classmethod
    def from_profile(cls, path):
        return cls.from_dict(load_noise_profile(path)['config'])

    def save_profile(self, path, *, calibration=None, statistics=None, provenance='heuristic'):
        """Write JSON usable by VirtualD405(noise_profile=path)."""
        profile = dict(schema_version=1, config=self.to_dict(), calibration=calibration or {},
                       statistics=statistics or {}, provenance=provenance)
        Path(path).write_text(json.dumps(profile, indent=2, allow_nan=False), encoding='utf-8')


def load_noise_profile(path):
    """Load the small versioned profile format, without executable objects."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('unsupported D405 profile schema_version')
    D405NoiseConfig.from_dict(data['config'])
    return data
