"""Reference-fitted macro segments plus deterministic procedural micro foliage."""
from copy import deepcopy

import numpy as np
from .skeleton import PlantSkeleton, unit_direction

from .compat import BranchSegment, LeafPlacement, TreeSpec
from .spec import LeafShape
from .foliage import place_leaves, _intersects_foliage_gap
from .fruit import place_fruits
from .config import load_config

def _range(config, key, *, lower=None, upper=None):
    value = config[key]
    if (len(value) != 2 or not np.isfinite(value).all() or value[0] > value[1]
            or (lower is not None and value[0] < lower)
            or (upper is not None and value[1] > upper)):
        raise ValueError(f"invalid range {key}: {value}")
    return value


def _growth_parameters(config, profile_name=None):
    """Merge optional named overrides; no species/site-specific profile names."""
    profile = config.get("growth_profiles", {}).get(profile_name, {})
    if set(profile) - {"twig", "leaf"}:
        raise ValueError(f"unknown growth profile sections: {profile_name}")
    twig = config["twig_generation"] | profile.get("twig", {})
    foliage = config["leaf_generation"] | profile.get("leaf", {})
    for override, base in ((profile.get("twig", {}), config["twig_generation"]),
                           (profile.get("leaf", {}), config["leaf_generation"])):
        if set(override) - set(base):
            raise ValueError(f"unknown growth parameter in profile {profile_name}")
    for value in (twig["per_parent"], foliage["per_branch"]):
        if type(value) is not int or value < 0:
            raise ValueError("twig/leaf counts must be nonnegative integers")
    for section in (twig, foliage):
        _range(section, "attachment_range", lower=0, upper=1)
        if not np.isfinite(section["azimuth_step_deg"]):
            raise ValueError("azimuth_step_deg must be finite")
        if not 0 <= section["attachment_jitter"] <= 0.5:
            raise ValueError("attachment_jitter must be in [0, .5]")
    for value in (twig["min_parent_order"], foliage["min_branch_order"]):
        if type(value) is not int or value < 0:
            raise ValueError("minimum branch orders must be nonnegative integers")
    _range(twig, "length_range", lower=np.finfo(float).eps)
    _range(twig, "cone_angle_deg", lower=0, upper=90)
    _range(foliage, "pitch_deg", lower=-80, upper=80)
    if not 0 <= foliage["size_jitter"] < 1:
        raise ValueError("size_jitter must be in [0, 1)")
    for section, keys in ((twig, ("radius_start", "radius_end")),
                          (foliage, ("length", "width", "max_branch_radius"))):
        for key in keys:
            if not np.isfinite(section[key]) or section[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
    for section, keys in ((twig, ("outward_bias", "upward_bias")),
                          (foliage, ("outward_bias", "axial_bias", "min_radial_distance", "roll_deg"))):
        for key in keys:
            if not np.isfinite(section[key]) or section[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
    return twig, foliage


def generate_tree(config=None, *, seed=None, skeleton=None):
    """Generate a pure TreeSpec, without constructing any WRS scene objects.

    skeleton optionally replaces config['macro_branches'] with hand-fitted
    BranchSegments (or their dicts). twig_generation.parent_ids is explicit;
    set it to null to select terminal macro branches by min_parent_order.
    All random draws use a local NumPy Generator; input objects are not mutated.
    """
    config = deepcopy(load_config() if config is None else config)
    if not isinstance(config, dict) or config.get("schema_version") != 1 or config.get("units") != "metre":
        raise ValueError("expected config schema_version=1 and units='metre'")
    seed = config["seed"] if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    source = list(config["macro_branches"] if skeleton is None else skeleton)
    branches = [deepcopy(b) if isinstance(b, BranchSegment) else BranchSegment(**b) for b in source]
    shape = LeafShape(**config["leaf_shape"])
    spec = TreeSpec(name=config["name"], branches=branches, leaf_shape=shape)
    spec.validate()
    threshold = config["collision"]["branch_radius_threshold"]
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("branch_radius_threshold must be finite and nonnegative")
    for branch in branches:
        branch.collidable = branch.collidable and max(branch.radius_start, branch.radius_end) >= threshold
    base_growth = _growth_parameters(config)
    twig, foliage = base_growth
    profiles = {name: _growth_parameters(config, name) for name in config.get("growth_profiles", {})}
    if len(config["visual"]["leaf_colors"]) < 3:
        raise ValueError("provide at least three leaf color shades")
    by_id = {b.id: b for b in branches}
    assignments = config.get("branch_profiles", {})
    if any(key not in by_id or value not in profiles for key, value in assignments.items()):
        raise ValueError("branch_profiles must map existing macro branch IDs to named growth_profiles")
    inherited = {}
    for branch in branches:
        current = branch
        while current.id not in assignments and current.parent_id is not None:
            current = by_id[current.parent_id]
        inherited[branch.id] = assignments.get(current.id)
    gaps = config.get("foliage_exclusions", [])
    for gap in gaps:
        for key in ("center", "radii"):
            if np.shape(gap[key]) != (3,) or not np.isfinite(gap[key]).all():
                raise ValueError(f"foliage exclusion {key} must be a finite 3-vector")
        if np.any(np.asarray(gap["radii"]) <= 0):
            raise ValueError("foliage exclusion radii must be positive")
    has_children = {b.parent_id for b in branches}
    host_ids = twig["parent_ids"]
    if host_ids is None:
        hosts = [b for b in branches if b.id not in has_children and b.order >= twig["min_parent_order"]]
    else:
        if len(set(host_ids)) != len(host_ids) or any(i not in by_id for i in host_ids):
            raise ValueError("twig parent_ids must be unique existing macro branch IDs")
        hosts = [by_id[i] for i in host_ids]
    for parent in hosts:
        twig, _ = profiles.get(inherited[parent.id], base_growth)
        count = twig["per_parent"]
        axis = unit_direction(np.subtract(parent.end, parent.start))
        frame = parent.frame
        phase = rng.uniform(0, 2 * np.pi)
        for i in range(count):
            lo, hi = twig["attachment_range"]
            t = lo + (hi - lo) * (i + 0.5 + rng.uniform(-twig["attachment_jitter"], twig["attachment_jitter"])) / count
            start = parent.point_at(t)
            angle = phase + np.deg2rad(twig["azimuth_step_deg"]) * i
            radial = frame[:, 0] * np.cos(angle) + frame[:, 1] * np.sin(angle)
            cone = np.deg2rad(rng.uniform(*twig["cone_angle_deg"]))
            outward = unit_direction([start[0], start[1], 0])
            direction = unit_direction(axis * np.cos(cone) + radial * np.sin(cone)
                                     + outward * twig["outward_bias"]
                                     + np.array([0, 0, twig["upward_bias"]]))
            end = start + direction * rng.uniform(*twig["length_range"])
            radius = min(twig["radius_start"], (1 - t) * parent.radius_start + t * parent.radius_end)
            branch_id = f"twig_{parent.id}_{i:02d}"
            if branch_id in by_id:
                raise ValueError(f"generated ID conflicts with skeleton: {branch_id}")
            branch = BranchSegment(branch_id, parent.id, tuple(start), tuple(end),
                                   radius, min(radius, twig["radius_end"]), parent.order + 1,
                                   radius >= threshold, float(t))
            branches.append(branch)
            by_id[branch.id] = branch
            inherited[branch.id] = inherited[parent.id]
    # Persist a shared rest frame (3x3, plant-local) for every generated segment.
    for branch in branches:
        if branch.rotmat is None:
            branch.rotmat = tuple(tuple(float(v) for v in row) for row in branch.frame)
    statistics = {}
    spec.leaves = place_leaves(PlantSkeleton(branches), foliage, shape, len(config['visual']['leaf_colors']),
                              rng=rng, per_segment={b.id: profiles.get(inherited[b.id], base_growth)[1] for b in branches},
                              gaps=gaps, statistics=statistics)
    excluded_leaf_count = statistics['excluded_leaf_count']
    profile_leaf_counts = {}
    for leaf in spec.leaves:
        name = inherited[leaf.parent_segment] or 'default'
        profile_leaf_counts[name] = profile_leaf_counts.get(name, 0) + 1
    # A few explicitly authored attachments can preserve salient occluding leaves.
    # These deliberate placements are not removed by procedural foliage gaps.
    spec.leaves.extend(LeafPlacement(**deepcopy(leaf)) for leaf in config.get("leaf_placements", []))
    spec.fruits = place_fruits(PlantSkeleton(branches), config['fruit_placements'], config['fruit_generation'])
    spec.metadata = {"schema_version": 1, "units": "metre", "seed": seed,
                     "reference_status": config["reference_status"],
                     "macro_branch_ids": [b.id if isinstance(b, BranchSegment) else b["id"] for b in source],
                     "leaf_counts_by_profile": profile_leaf_counts,
                     "excluded_leaf_count": excluded_leaf_count,
                     "authored_leaf_count": len(config.get("leaf_placements", [])),
                     "reference": deepcopy(config.get("reference", {})),
                     "generation": "hand-fitted macro skeleton + procedural attached twigs/leaves"}
    spec.validate()
    factor = config.get("scale_multiplier", 1.0)
    return spec.scaled(factor) if factor != 1 else spec
