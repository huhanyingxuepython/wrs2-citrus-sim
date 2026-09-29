"""Optional single-view approximations of stereo depth errors."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoDepthNoise:
    """Configurable disparity noise, quantization, and independent dropout.

    Random effects default to off; geometric stereo occlusion is enabled in
    d405_fast. Parameters are experimental controls, not a calibrated D405
    profile. Single-view reprojection cannot discover right-only geometry.

    Parameters
    ----------
    disparity_std_px : float
        Standard deviation of zero-mean Gaussian disparity error in pixels.
    disparity_step_px : float
        Disparity quantization step in pixels; zero disables quantization.
    dropout_rate : float
        Independent probability of invalidating a valid pixel, in [0, 1].
    stereo_occlusion : bool
        Reject pixels outside the synthetic right FOV or behind closer samples.
    confidence_threshold, texture_dropout_strength : float
        Sobel luminance confidence cutoff and dropout probability multiplier,
        respectively, in [0, 1]. Disabled by default for WRS's unlit colors.
    edge_dropout_strength : float
        Dropout probability at four-neighbor Z jumps above edge_threshold_m.
    edge_threshold_m, occlusion_tolerance_m : float
        Depth discontinuity threshold and right-view visibility tolerance.
    """

    disparity_std_px: float = 0.0
    disparity_step_px: float = 0.0
    dropout_rate: float = 0.0
    stereo_occlusion: bool = True
    confidence_threshold: float = 0.0
    edge_dropout_strength: float = 0.0
    texture_dropout_strength: float = 0.0
    edge_threshold_m: float = 0.005
    occlusion_tolerance_m: float = 0.001

    def __post_init__(self):
        probabilities = [self.dropout_rate, self.confidence_threshold,
                         self.edge_dropout_strength, self.texture_dropout_strength]
        values = [self.disparity_std_px, self.disparity_step_px, self.edge_threshold_m,
                  self.occlusion_tolerance_m, *probabilities]
        if (not np.all(np.isfinite(values)) or min(values) < 0
                or max(probabilities) > 1 or not isinstance(self.stereo_occlusion, bool)):
            raise ValueError('noise values must be finite/nonnegative; strengths and probabilities <= 1')

    def apply(self, depth, fx, baseline_m, rng, rgb=None):
        """Return perturbed float32 Z depth; leave the input unchanged.

        ``depth`` has shape (H, W) in meters, with zero for invalid pixels.
        Disparity is d = fx * baseline_m / Z; invalid disparity becomes zero
        depth. ``rng`` is a numpy Generator. Range/uint16 conversion follows
        this stage in VirtualDepthCamera.
        """
        out = np.array(depth, dtype=np.float32, copy=True)
        valid = np.isfinite(out) & (out > 0)
        out[~valid] = 0
        if self.stereo_occlusion or self.disparity_std_px or self.disparity_step_px:
            if baseline_m is None or not np.isfinite(baseline_m) or baseline_m <= 0:
                raise ValueError('disparity effects require a positive baseline_m')
            if not np.isfinite(fx) or fx <= 0:
                raise ValueError('fx must be positive and finite')
            fb = fx * baseline_m
            if self.stereo_occlusion:
                v, u = np.nonzero(valid)
                right_u = np.floor(u - fb/out[v, u] + .5).astype(np.int64)
                inside = (right_u >= 0) & (right_u < out.shape[1])
                indices = v[inside]*out.shape[1] + right_u[inside]
                right_z = np.full(out.size, np.inf)
                np.minimum.at(right_z, indices, out[v[inside], u[inside]])
                visible = np.zeros_like(inside)
                visible[inside] = out[v[inside], u[inside]] <= right_z[indices]+self.occlusion_tolerance_m
                out[v[~visible], u[~visible]] = 0
                valid = out > 0
            disparity = fb / out[valid].astype(np.float64)
            if self.disparity_step_px:
                disparity = np.rint(disparity / self.disparity_step_px) * self.disparity_step_px
            if self.disparity_std_px:
                disparity += rng.normal(0, self.disparity_std_px, disparity.shape)
            z = np.zeros_like(disparity)
            np.divide(fb, disparity, out=z, where=np.isfinite(disparity) & (disparity > 0))
            out[valid] = z
        if self.edge_dropout_strength:
            padded = np.pad(np.asarray(depth), 1, mode='edge')
            jump = np.maximum.reduce([np.abs(padded[1:-1, 1:-1]-neighbor) for neighbor in
                                      (padded[:-2, 1:-1], padded[2:, 1:-1],
                                       padded[1:-1, :-2], padded[1:-1, 2:])])
            out[(jump > self.edge_threshold_m)
                & (rng.random(out.shape) < self.edge_dropout_strength)] = 0
        if self.confidence_threshold or self.texture_dropout_strength:
            if rgb is None or np.shape(rgb) != (*out.shape, 3):
                raise ValueError('texture confidence requires RGB with shape (H, W, 3)')
            from scipy.ndimage import sobel
            luminance = np.asarray(rgb, dtype=np.float32) @ [.2126, .7152, .0722] / 255
            confidence = np.clip(np.hypot(sobel(luminance, axis=1, mode='nearest'),
                                          sobel(luminance, axis=0, mode='nearest')), 0, 1)
            out[confidence < self.confidence_threshold] = 0
            out[rng.random(out.shape) < self.texture_dropout_strength*(1-confidence)] = 0
        if self.dropout_rate:
            out[rng.random(out.shape) < self.dropout_rate] = 0
        return out
