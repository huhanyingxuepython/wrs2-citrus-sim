"""Small camera-relative matte light rig for virtual RGB observations."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBLighting:
    """Two directional lights plus ambient, evaluated in linear RGB.

    Directions point from surface toward light in optical axes (+X right,
    +Y down, +Z forward). The rig follows the camera. Intensities are linear;
    key_power shapes the half-Lambert falloff, as in the WRS viewer.
    This supplies surface shading, not shadows or calibrated illumination.
    """
    key_direction: tuple = (-.4, -.5, -1.)
    fill_direction: tuple = (.6, .2, -1.)
    key_intensity: float = 1.
    fill_intensity: float = .15
    ambient: float = .04
    key_power: float = 4.

    def __post_init__(self):
        for field in ('key_direction', 'fill_direction'):
            value = np.asarray(getattr(self, field), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
                raise ValueError('light directions must be finite nonzero 3-vectors')
            object.__setattr__(self, field, tuple(value / np.linalg.norm(value)))
        values = (self.key_intensity, self.fill_intensity, self.ambient, self.key_power)
        if not np.isfinite(values).all() or min(values) < 0 or self.key_power <= 0:
            raise ValueError('light intensities must be nonnegative; key_power must be positive')
