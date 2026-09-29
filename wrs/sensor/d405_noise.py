"""Vectorized, stateful hole/value stereo corruption with no learned dependency."""

import copy
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr

from .d405_noise_config import D405NoiseConfig


class CorruptionReason(IntEnum):
    VALID = 0
    OUT_OF_RANGE = 1
    HOLE = 2
    BACKGROUND_BLEED = 3
    FOREGROUND_BLEED = 4
    WRONG_DISPARITY = 5


@dataclass
class D405NoiseState:
    """Sensor-grid history; disparity history is a unit-normal AR field.

    get_state/set_state also preserve RNG state for exact continuation.
    """

    previous_invalid_state: np.ndarray | None = None
    previous_outlier_state: np.ndarray | None = None
    previous_disparity_noise: np.ndarray | None = None
    previous_outlier_depth: np.ndarray | None = None
    previous_reason: np.ndarray | None = None
    previous_clean_depth: np.ndarray | None = None
    previous_camera_tf: np.ndarray | None = None
    rng_state: dict | None = None
    frame_index: int = 0


@dataclass
class D405NoiseResult:
    depth: np.ndarray
    depth_raw: np.ndarray
    debug: dict | None = None


@lru_cache(maxsize=32)
def _filter_std(sigma_y, sigma_x):
    """L2 norm of the separable Gaussian kernel, for unit-variance random fields."""
    norm = 1.
    for sigma in (sigma_y, sigma_x):
        if sigma > 0:
            radius = int(3 * sigma + .5)
            weights = np.exp(-.5 * (np.arange(-radius, radius + 1) / sigma)**2)
            weights /= weights.sum()
            norm *= float(np.sqrt(np.sum(weights**2)))
    return norm


def _normal_field(shape, rng, sigma_y, sigma_x):
    values = rng.standard_normal(shape, dtype=np.float32)
    if sigma_y > 0 or sigma_x > 0:
        values = ndi.gaussian_filter(values, (sigma_y, sigma_x), mode='wrap', truncate=3.)
        values /= _filter_std(sigma_y, sigma_x)
    return values


def _uniform_field(shape, rng, config):
    # Gaussian CDF preserves uniform marginal probabilities after correlation.
    # Failure fields can use small blocks; metric depth and edge cues never
    # undergo downsampling/interpolation. Random phase avoids a fixed block grid.
    stride = config.failure_field_stride
    small_shape = tuple((size+stride-1)//stride + (stride > 1) for size in shape)
    values = ndtr(_normal_field(small_shape, rng, config.hole_correlation_px/stride,
                                config.hole_correlation_px*config.epipolar_correlation_ratio/stride))
    if stride == 1:
        return values
    y, x = rng.integers(0, stride, 2)
    return values.repeat(stride, 0).repeat(stride, 1)[y:y+shape[0], x:x+shape[1]]


def depth_edges(depth, threshold_m=.005):
    """Return edge mask, horizontal-edge mask, and Chebyshev distance in pixels.

    Both sides of discontinuities count, including boundaries against missing
    geometry. An entirely smooth image has infinite distance everywhere.
    """
    depth = np.asarray(depth, dtype=np.float32)
    good = np.isfinite(depth) & (depth > 0)
    z = np.where(good, depth, 0)
    horizontal = np.zeros(z.shape, bool)
    vertical = np.zeros(z.shape, bool)
    dx = (np.abs(z[:, 1:] - z[:, :-1]) > threshold_m) | (good[:, 1:] != good[:, :-1])
    dy = (np.abs(z[1:] - z[:-1]) > threshold_m) | (good[1:] != good[:-1])
    horizontal[:, 1:] |= dx
    horizontal[:, :-1] |= dx
    vertical[1:] |= dy
    vertical[:-1] |= dy
    edge = horizontal | vertical
    distance = (ndi.distance_transform_cdt(~edge, metric='chessboard').astype(np.float32)
                if edge.any() else np.full(z.shape, np.inf, np.float32))
    return edge, horizontal, distance


def _thin_foreground(z, valid, config):
    """Foreground pixels with deeper/empty neighbors on opposite sides."""
    reference = np.where(valid, z, config.max_depth_m + config.depth_edge_threshold_m)
    size = config.thin_structure_width_px + 1
    opposite = np.zeros(z.shape, bool)
    for axis in (0, 1):
        lo = ndi.maximum_filter1d(reference, size, axis=axis, origin=(size - 1)//2, mode='nearest')
        hi = ndi.maximum_filter1d(reference, size, axis=axis, origin=-(size//2), mode='nearest')
        opposite |= (lo > z + config.depth_edge_threshold_m) & (hi > z + config.depth_edge_threshold_m)
    return opposite & valid


def _occlusion(z, valid, fb, tolerance):
    v, u = np.nonzero(valid)
    right_u = np.rint(u - fb / z[v, u]).astype(np.int64)
    inside = (right_u >= 0) & (right_u < z.shape[1])
    index = v[inside] * z.shape[1] + right_u[inside]
    right_z = np.full(z.size, np.inf, np.float32)
    np.minimum.at(right_z, index, z[v[inside], u[inside]])
    hidden = ~inside
    hidden[inside] = z[v[inside], u[inside]] > right_z[index] + tolerance
    result = np.zeros(z.shape, bool)
    result[v, u] = hidden
    return result


def build_noise_cues(clean_depth, config, camera_model, baseline_m, *, rgb=None,
                     normals=None, semantic_ids=None, material_difficulty=None, motion_px=0.):
    """Build reusable geometry/appearance features, all in the input image grid.

    normals are optical XYZ (H,W,3). material_difficulty is an optional [0,1]
    map, e.g. a renderer's gloss/material annotation. Neither is inferred from
    semantic names. RGB is optional uint8 RGB; absence disables texture/light cues.
    """
    z = np.asarray(clean_depth, dtype=np.float32)
    if z.shape != (camera_model.height, camera_model.width):
        raise ValueError('clean depth must match CameraModel resolution')
    if not np.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError('baseline_m must be positive and finite')
    if not np.isfinite(motion_px) or motion_px < 0:
        raise ValueError('motion_px must be finite and nonnegative')
    geometric = np.isfinite(z) & (z > 0)
    z = np.where(geometric, z, 0)
    tolerance = 4 * np.finfo(np.float32).eps * config.max_depth_m
    valid = geometric & (z >= config.min_depth_m - tolerance) & (z <= config.max_depth_m + tolerance)
    edge, horizontal, distance = depth_edges(z, config.depth_edge_threshold_m)
    horizontal_band = ndi.maximum_filter(horizontal, size=(1, 2*config.edge_band_width_px+1))
    edge_score = np.maximum(1 - distance / (config.edge_band_width_px + 1), 0)
    edge_score *= (1 - config.horizontal_edge_weight) + config.horizontal_edge_weight * horizontal_band
    texture = np.ones(z.shape, np.float32)
    lighting = np.zeros(z.shape, np.float32)
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.shape != (*z.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError('rgb must be uint8 (H,W,3) in RGB order')
        luminance = (rgb.astype(np.float32) @ np.array([.2126, .7152, .0722], np.float32)) / 255
        mean = ndi.uniform_filter(luminance, config.texture_window_px, mode='nearest')
        variance = np.maximum(ndi.uniform_filter(luminance*luminance, config.texture_window_px,
                                                 mode='nearest') - mean*mean, 0)
        texture = np.clip(np.sqrt(variance) / config.texture_threshold * config.texture_gain, 0, 1)
        lighting = ((luminance < config.dark_threshold) | (luminance > config.saturated_threshold)).astype(np.float32)
    grazing = np.zeros(z.shape, np.float32)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.shape != (*z.shape, 3) or not np.isfinite(normals).all():
            raise ValueError('normals must be finite optical (H,W,3)')
        length = np.linalg.norm(normals, axis=-1)
        rays = camera_model.ray_lut
        cosine = np.abs(np.sum(normals*rays, axis=-1)) / np.maximum(length*np.linalg.norm(rays, axis=-1), 1e-12)
        grazing = np.where(length > 0, np.clip((config.grazing_cos_threshold - cosine)
                           / max(config.grazing_cos_threshold, 1e-12), 0, 1), 0)
    semantic = np.ones(z.shape, np.float32)
    if semantic_ids is not None:
        ids = np.asarray(semantic_ids)
        if ids.shape != z.shape:
            raise ValueError('semantic_ids must have shape (H,W)')
        for key, multiplier in config.semantic_multipliers.items():
            match = ids == (int(key) if ids.dtype.kind in 'iu' and key.lstrip('-').isdigit() else key)
            semantic[match] = multiplier
    material = np.zeros(z.shape, np.float32)
    if material_difficulty is not None:
        material = np.asarray(material_difficulty, dtype=np.float32)
        if material.shape != z.shape or not np.isfinite(material).all() or np.any((material < 0) | (material > 1)):
            raise ValueError('material_difficulty must be finite (H,W) in [0,1]')
    far = np.clip((z - config.preferred_max_depth_m) / config.far_degradation_width_m, 0, 1)
    near = np.exp(-np.maximum(z-config.min_depth_m, 0) / config.near_degradation_width_m)
    occlusion = (_occlusion(z, geometric, camera_model.fx*baseline_m, config.occlusion_tolerance_m)
                 if config.occlusion_invalid_gain else np.zeros(z.shape, bool))
    return dict(clean=z, valid=valid, edge_mask=edge, distance_to_depth_edge=distance,
                edge=edge_score, thin=_thin_foreground(z, geometric, config).astype(np.float32),
                texture_confidence=texture, low_texture=1-texture, lighting=lighting, grazing=grazing,
                semantic=semantic, material=material, far=far, near=near, occlusion=occlusion,
                motion=min(motion_px/config.motion_reference_px, 1.))


class D405HoleNoiseModel:
    """Replaceable probability model: probability(cues, config) -> (H,W) [0,1]."""

    def probability(self, cues, config):
        c, q = config, cues
        difficulty = (c.edge_invalid_gain*q['edge'] + c.thin_structure_invalid_gain*q['thin']
                      + c.low_texture_invalid_gain*q['low_texture'] + c.grazing_invalid_gain*q['grazing']
                      + c.lighting_invalid_gain*q['lighting'] + c.material_invalid_gain*q['material']
                      + c.motion_invalid_gain*q['motion']*np.maximum(q['thin'], q['edge']))
        return np.clip(c.base_invalid_prob + q['semantic']*difficulty
                       + c.near_range_invalid_gain*q['near'] + c.far_range_invalid_gain*q['far'], 0, 1)


class D405ValueNoiseModel:
    """Replaceable value model; small precision and gross donor errors stay separate."""

    def precision(self, cues, config, fb, rng, previous, stable):
        c, z = config, cues['clean']
        fresh = _normal_field(z.shape, rng, c.spatial_correlation_px, c.spatial_correlation_px)
        if previous is not None:
            innovation_scale = np.float32(np.sqrt(1-c.temporal_rho**2))
            fresh = np.where(stable, c.temporal_rho*previous + innovation_scale*fresh, fresh)
        perturbation = c.disparity_bias_px + c.disparity_sigma_px * (1+c.far_disparity_noise_gain*cues['far']) * fresh
        disparity = np.divide(fb, z, out=np.zeros_like(z), where=z > 0) + perturbation
        noisy = np.divide(fb, disparity, out=np.zeros_like(z), where=disparity > 0)
        return noisy, fresh, perturbation

    def mismatch_probability(self, cues, config):
        c, q = config, cues
        difficulty = (c.edge_outlier_gain*q['edge'] + c.thin_structure_outlier_gain*q['thin']
                      + c.low_texture_outlier_gain*q['low_texture'] + c.grazing_outlier_gain*q['grazing']
                      + c.lighting_outlier_gain*q['lighting'] + c.material_outlier_gain*q['material']
                      + c.motion_outlier_gain*q['motion']*np.maximum(q['thin'], q['edge']))
        return np.clip(c.base_outlier_prob + q['semantic']*difficulty + c.far_range_outlier_gain*q['far'], 0, 1)

    def mismatch(self, cues, config, rng):
        """Select real depths on the SAME row; no image wrapping or invented Z spikes."""
        c, z, valid = config, cues['clean'], cues['valid']
        radius = c.epipolar_search_px
        foreground = ndi.minimum_filter1d(np.where(valid, z, np.inf), 2*radius+1, axis=1,
                                            mode='constant', cval=np.inf)
        background = ndi.maximum_filter1d(np.where(valid, z, 0), 2*radius+1, axis=1,
                                             mode='constant', cval=0)
        weights = np.array([c.background_takeover_weight, c.foreground_bleed_weight,
                            c.wrong_disparity_weight, c.reject_to_invalid_weight])
        outcome = np.searchsorted(np.cumsum(weights)/weights.sum(), _uniform_field(z.shape, rng, c), side='right')
        offsets = rng.integers(-radius, radius+1, z.shape)
        columns = np.arange(z.shape[1])[None, :] + offsets
        inside = (columns >= 0) & (columns < z.shape[1])
        donor = np.take_along_axis(z, np.clip(columns, 0, z.shape[1]-1), axis=1)
        donor_valid = np.take_along_axis(valid, np.clip(columns, 0, z.shape[1]-1), axis=1) & inside
        donor = np.where(donor_valid, donor, z)
        value = np.where(outcome == 0, background, np.where(outcome == 1, foreground, donor))
        reason = np.where(outcome == 0, CorruptionReason.BACKGROUND_BLEED,
                          np.where(outcome == 1, CorruptionReason.FOREGROUND_BLEED,
                                   CorruptionReason.WRONG_DISPARITY)).astype(np.uint8)
        reject = outcome == 3
        effective = np.isfinite(value) & (value > 0) & (np.abs(value-z) > c.depth_edge_threshold_m)
        # An unavailable donor does not manufacture a failure on a reliable plane.
        reason[~effective] = CorruptionReason.VALID
        reason[reject] = CorruptionReason.HOLE
        value[reject] = 0
        return value, reason


class D405NoiseModel:
    """Compose replaceable hole/value models with spatial and temporal sampling.

    apply() consumes an independent Generator, never numpy's global RNG.
    State is sensor-grid based, reset locally at changed surfaces and globally
    for large camera motion; it does not claim optical-flow tracking.
    """

    def __init__(self, config=None, *, seed=None, rng=None, hole_model=None, value_model=None):
        self.config = D405NoiseConfig() if config is None else config
        if not isinstance(self.config, D405NoiseConfig):
            raise TypeError('config must be D405NoiseConfig')
        if rng is not None and seed is not None:
            raise ValueError('supply seed OR rng')
        self.rng = np.random.default_rng(seed) if rng is None else rng
        if not isinstance(self.rng, np.random.Generator):
            raise TypeError('rng must be numpy.random.Generator')
        self._initial_rng = copy.deepcopy(self.rng.bit_generator.state)
        self.state = D405NoiseState()
        self.hole_model = D405HoleNoiseModel() if hole_model is None else hole_model
        self.value_model = D405ValueNoiseModel() if value_model is None else value_model

    def reset(self, seed=None):
        """Clear history; no seed replays this model's original RNG sequence."""
        self.state = D405NoiseState()
        if seed is None:
            self.rng.bit_generator.state = copy.deepcopy(self._initial_rng)
        else:
            self.rng = np.random.default_rng(seed)
            self._initial_rng = copy.deepcopy(self.rng.bit_generator.state)

    def get_state(self):
        snapshot = copy.deepcopy(self.state)
        snapshot.rng_state = copy.deepcopy(self.rng.bit_generator.state)
        return snapshot

    def set_state(self, state):
        if not isinstance(state, D405NoiseState) or state.rng_state is None:
            raise ValueError('state must come from get_state()')
        self.state = copy.deepcopy(state)
        self.rng.bit_generator.state = copy.deepcopy(state.rng_state)

    def apply(self, clean_depth, camera_model, baseline_m, *, rgb=None, normals=None,
              semantic_ids=None, material_difficulty=None, camera_tf=None, motion_px=None, debug=False):
        """Return metric/Z16 raw-like depth and advance the sensor's temporal state.

        Parameters
        ----------
        clean_depth : array-like, shape (H, W)
            Optical Z in meters; nonpositive/nonfinite values have no geometry.
        camera_model : CameraModel
            Intrinsics and output resolution for this stream.
        baseline_m : float
            Positive calibrated stereo baseline, in meters.
        rgb : ndarray of uint8, shape (H, W, 3), optional
            Registered RGB for texture/luminance cues.
        normals : ndarray, shape (H, W, 3), optional
            Surface normals in optical coordinates; zero means unavailable.
        semantic_ids : ndarray, shape (H, W), optional
            Names or integer IDs matched against configured semantic multipliers.
        material_difficulty : ndarray, shape (H, W), optional
            Finite values in [0, 1]; a caller-provided difficulty proxy.
        camera_tf : ndarray, shape (4, 4), optional
            World-from-optical pose. Used to estimate motion when motion_px is absent.
        motion_px : float, optional
            Nonnegative frame-to-frame pixel motion; overrides pose estimation.
        debug : bool
            Allocate diagnostic maps only when requested, outside policy inputs.

        Returns
        -------
        D405NoiseResult
            float32 metric depth, uint16 Z16, and optional diagnostic dictionary.
            Invalid depth is exactly zero; no depth smoothing or hole filling.

        Notes
        -----
        Estimated motion is fx * (translation / median Z + rotation angle).
        Calls mutate temporal history and the owned RNG. Use get_state/set_state
        for reproducible continuation, or reset at episode boundaries.
        """
        c, state, rng = self.config, self.state, self.rng
        shape = np.shape(clean_depth)
        if state.previous_clean_depth is not None and state.previous_clean_depth.shape != shape:
            self.state = state = D405NoiseState(frame_index=state.frame_index)
        tf = None if camera_tf is None else np.asarray(camera_tf, dtype=float)
        if tf is not None and (tf.shape != (4, 4) or not np.isfinite(tf).all()):
            raise ValueError('camera_tf must be finite (4,4)')
        if motion_px is None:
            motion_px = 0.
            if tf is not None and state.previous_camera_tf is not None:
                previous_tf = state.previous_camera_tf
                # Rotation difference avoids spurious motion from float32
                # poses whose R.T @ R is only approximately identity.
                rotation_difference = np.linalg.norm(tf[:3, :3]-previous_tf[:3, :3])
                angle = 2*np.arcsin(np.clip(rotation_difference/(2*np.sqrt(2)), 0, 1))
                positive = np.asarray(clean_depth)[np.isfinite(clean_depth) & (np.asarray(clean_depth) > 0)]
                distance = float(np.median(positive)) if positive.size else c.preferred_max_depth_m
                motion_px = camera_model.fx*(np.linalg.norm(tf[:3, 3]-previous_tf[:3, 3])/distance + angle)
        q = build_noise_cues(clean_depth, c, camera_model, baseline_m, rgb=rgb, normals=normals,
                             semantic_ids=semantic_ids, material_difficulty=material_difficulty, motion_px=motion_px)
        z, valid = q['clean'], q['valid']
        stable = np.zeros(z.shape, bool)
        if state.previous_clean_depth is not None and motion_px < c.temporal_reset_motion_px:
            stable = (np.abs(z-state.previous_clean_depth) <= c.temporal_depth_threshold_m) & valid
        reason = np.full(z.shape, CorruptionReason.VALID, np.uint8)
        normal_map = np.zeros(z.shape, np.float32)
        standardized = np.zeros(z.shape, np.float32)
        measured = z.copy()
        invalid = np.zeros(z.shape, bool)
        outlier = np.zeros(z.shape, bool)
        if c.enabled:
            measured, standardized, normal_map = self.value_model.precision(
                q, c, camera_model.fx*baseline_m, rng, state.previous_disparity_noise, stable)
            # Range -> precision -> conditioned holes -> occlusion/edge rejection.
            probability = self.hole_model.probability(q, c)
            probability = 1 - (1-probability)*(1-c.occlusion_invalid_gain*q['occlusion'].astype(np.float32))
            probability = np.clip(probability, 0, 1)
            invalid = _uniform_field(z.shape, rng, c) < probability
            donor, candidate_reason = self.value_model.mismatch(q, c, rng)
            candidate = (_uniform_field(z.shape, rng, c) < self.value_model.mismatch_probability(q, c)) & valid
            candidate &= candidate_reason != CorruptionReason.VALID
            invalid |= candidate & (candidate_reason == CorruptionReason.HOLE)
            # Markov retention mixes old state with a fresh correlated proposal.
            # Both failure types share a correlated retention field. Proposals
            # use independent fields so holes remain blobs, not contour strips.
            retention = _uniform_field(z.shape, rng, c)
            retain_hole = (retention < c.invalid_persistence) & stable
            retain_outlier = (retention < c.outlier_persistence) & stable
            if state.previous_invalid_state is not None:
                invalid = np.where(retain_hole, state.previous_invalid_state, invalid)
            outlier = candidate & (candidate_reason >= CorruptionReason.BACKGROUND_BLEED)
            if state.previous_outlier_state is not None:
                outlier = np.where(retain_outlier, state.previous_outlier_state, outlier)
                donor = np.where(retain_outlier & outlier, state.previous_outlier_depth, donor)
                candidate_reason = np.where(retain_outlier & outlier, state.previous_reason, candidate_reason)
            invalid &= valid
            outlier &= valid & ~invalid
            measured[outlier] = donor[outlier]
            reason[outlier] = candidate_reason[outlier]
            reason[invalid] = CorruptionReason.HOLE
            measured[invalid] = 0
        reason[~valid] = CorruptionReason.OUT_OF_RANGE
        measured[~valid] = 0
        # Only here are measurements quantized. Zero never becomes a valid bin.
        tolerance = 4*np.finfo(np.float32).eps*c.max_depth_m
        keep = valid & np.isfinite(measured) & (measured >= c.min_depth_m-tolerance) & (measured <= c.max_depth_m+tolerance)
        units = np.rint(np.where(keep, measured, 0).astype(np.float64)/c.depth_unit_m)
        raw_min = max(1, int(np.ceil(np.nextafter(c.min_depth_m/c.depth_unit_m, -np.inf))))
        raw_max = int(np.floor(np.nextafter(c.max_depth_m/c.depth_unit_m, np.inf)))
        keep &= (units >= raw_min) & (units <= raw_max) & (units <= 65535)
        raw = np.where(keep, units, 0).astype(np.uint16)
        depth = (raw.astype(np.float64)*c.depth_unit_m).astype(np.float32)
        reason[valid & ~keep & ~invalid] = CorruptionReason.OUT_OF_RANGE
        outlier &= keep
        self.state = D405NoiseState(previous_invalid_state=valid & ~keep,
            previous_outlier_state=outlier.copy(), previous_disparity_noise=standardized.copy(),
            previous_outlier_depth=depth.copy(), previous_reason=reason.copy(),
            previous_clean_depth=z.copy(), previous_camera_tf=None if tf is None else tf.copy(),
            frame_index=state.frame_index+1)
        diagnostics = None
        if debug:
            diagnostics = dict(clean_depth=z.copy(), noisy_depth=depth.copy(), valid_mask=keep.copy(),
                invalid_mask=~keep, normal_noise_map=normal_map.copy(), outlier_mask=outlier.copy(),
                edge_mask=q['edge_mask'], distance_to_depth_edge=q['distance_to_depth_edge'],
                thin_structure_mask=q['thin'] > 0, texture_confidence=q['texture_confidence'],
                corruption_reason=reason, motion_px=float(motion_px))
        return D405NoiseResult(depth, raw, diagnostics)
