"""Pure skeleton -> leaf placements; mesh/rendering/physics are separate."""
import numpy as np
from .spec import LeafPlacement
from .skeleton import unit_direction
from .geometry import leaf_mesh

def _intersects_foliage_gap(leaf, shape, gaps):
    """Visual opening heuristic, NOT a certified swept-volume clearance test.

    Test blade vertices and triangle centroids against nominal ellipsoids.
    Parent branches remain intact; no unattached leaf points are introduced.
    """
    if not gaps:
        return False
    vs, fs = leaf_mesh(leaf.length, leaf.width, shape)
    vs = vs @ np.asarray(leaf.rotmat).T + leaf.position
    samples = np.concatenate([vs, vs[fs].mean(axis=1)])
    return any(np.any(np.sum(((samples - gap["center"]) / gap["radii"]) ** 2, axis=1) < 1)
               for gap in gaps)



def place_leaves(skeleton, parameters, shape, shade_count=3, *, seed=0, rng=None,
                 per_segment=None, gaps=(), statistics=None):
    """Explicit RNG can share a generator stream; standalone calls use seed."""
    skeleton.validate()
    if rng is None:
        rng = np.random.default_rng(seed)
    has_children = {b.parent_id for b in skeleton.segments}
    leaves = []
    excluded_leaf_count = 0
    for branch in skeleton.segments:
        foliage = (per_segment or {}).get(branch.id, parameters)
        density = foliage.get("density", 1.0)
        if not np.isfinite(density) or density < 0:
            raise ValueError("leaf density must be finite and nonnegative")
        leaf_count = int(round(foliage["per_branch"] * density))
        # Foliage grows only on terminal, thin, outer segments; never bbox samples.
        if (branch.id in has_children or branch.order < foliage["min_branch_order"]
                or max(branch.radius_start, branch.radius_end) > foliage["max_branch_radius"]
                or np.linalg.norm(branch.point_at(0.5)[:2]) < foliage["min_radial_distance"]):
            continue
        axis = unit_direction(np.subtract(branch.end, branch.start))
        frame = branch.frame
        phase = rng.uniform(0, 2 * np.pi)
        for i in range(leaf_count):
            lo, hi = foliage["attachment_range"]
            t = lo + (hi - lo) * (i + 0.5 + rng.uniform(-foliage["attachment_jitter"], foliage["attachment_jitter"])) / leaf_count
            position = branch.point_at(t)
            azimuth = phase + i * np.deg2rad(foliage["azimuth_step_deg"])
            radial = frame[:, 0] * np.cos(azimuth) + frame[:, 1] * np.sin(azimuth)
            outward = unit_direction([position[0], position[1], 0])
            heading = (radial + outward * foliage["outward_bias"] + axis * foliage["axial_bias"])
            heading[2] = 0
            if np.linalg.norm(heading) < 1e-8:
                heading = np.array([np.cos(azimuth), np.sin(azimuth), 0])
            heading = unit_direction(heading)
            pitch = np.deg2rad(rng.uniform(*foliage["pitch_deg"]))
            x_axis = heading * np.cos(pitch) + np.array([0, 0, np.sin(pitch)])
            y_axis = unit_direction(np.cross([0, 0, 1], x_axis))
            rotmat = np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])
            roll = np.deg2rad(rng.uniform(-foliage["roll_deg"], foliage["roll_deg"]))
            rotmat = rotmat @ np.array([[1., 0., 0.], [0., np.cos(roll), -np.sin(roll)], [0., np.sin(roll), np.cos(roll)]])
            scale = rng.uniform(1 - foliage["size_jitter"], 1 + foliage["size_jitter"])
            leaf = LeafPlacement(
                branch.id, tuple(position), tuple(tuple(row) for row in rotmat),
                foliage["length"] * scale, foliage["width"] * scale,
                int(rng.integers(shade_count)), float(t))
            if _intersects_foliage_gap(leaf, shape, gaps):
                excluded_leaf_count += 1
                continue
            leaves.append(leaf)
    if statistics is not None:
        statistics['excluded_leaf_count'] = excluded_leaf_count
    return leaves
