"""Small NumPy-only blade/batch helpers; no renderer or physics dependencies."""
import numpy as np

from .spec import LeafShape


def leaf_mesh(length: float, width: float, shape: LeafShape):
    """Pointed lanceolate blade: (vertices, triangle indices), in blade axes.

    +X follows the midrib, +Y crosses the blade, +Z is the upper face. A pair
    of pointed ends avoids degenerate triangles. Two sloping strips meet at
    the folded midrib; curl arches the length, droop lowers the distal end.
    Reverse-wound faces have separate, slightly offset vertices so WRS's
    vertex weld does not cancel the upper/lower surface normals.
    """
    shape.validate()
    if not np.isfinite([length, width]).all() or length <= 0 or width <= 0:
        raise ValueError("leaf length and width must be finite and positive")
    vertices, rows = [], []
    for i, t in enumerate(np.linspace(0, 1, shape.stations)):
        half_width = width * 0.5 * np.sin(np.pi * t) ** shape.profile_power
        z = length * (shape.curl * np.sin(np.pi * t) - shape.droop * t * t)
        ys = [0.0] if i in (0, shape.stations - 1) else [-half_width, 0.0, half_width]
        rows.append(list(range(len(vertices), len(vertices) + len(ys))))
        vertices.extend((length * t, y, z - abs(y) * np.tan(shape.fold)) for y in ys)
    faces = []
    for a, b in zip(rows[:-1], rows[1:]):
        if len(a) == 1:
            faces.extend([(a[0], b[0], b[1]), (a[0], b[1], b[2])])
        elif len(b) == 1:
            faces.extend([(a[0], b[0], a[1]), (a[1], b[0], a[2])])
        else:
            for j in range(2):
                faces.extend([(a[j], b[j], b[j + 1]), (a[j], b[j + 1], a[j + 1])])
    upper = np.asarray(vertices, dtype=float)
    lower = upper.copy()
    upper[:, 2] += shape.thickness * 0.5
    lower[:, 2] -= shape.thickness * 0.5
    fs = np.asarray(faces, dtype=np.uint32)
    return np.concatenate([upper, lower]), np.concatenate([fs, fs[:, ::-1] + len(upper)])


def leaf_batches(spec):
    """Return {color_class: (tree-local vertices, faces)} in shade order."""
    groups = {}
    for leaf in spec.leaves:
        vs, fs = leaf_mesh(leaf.length, leaf.width, spec.leaf_shape)
        vs = vs @ np.asarray(leaf.rotmat).T + leaf.position
        vertices, faces, count = groups.setdefault(leaf.color_class, ([], [], [0]))
        vertices.append(vs)
        faces.append(fs + count[0])
        count[0] += len(vs)
    return {shade: (np.concatenate(groups[shade][0]).astype(np.float32),
                    np.concatenate(groups[shade][1]).astype(np.uint32))
            for shade in sorted(groups)}


def leaf_contact_boxes(length, width, shape, *, sections, padding, minimum_thickness):
    """Fit thin boxes to contiguous blade strips, in blade-local coordinates.

    Returns (centre, rotation, half_extents) per strip. Boundaries follow the
    existing visual mesh stations, so no triangles are skipped at strip seams.
    X follows the strip's midrib chord, Y the blade width, Z its local normal.
    Fold/curl geometry is enclosed; padding and minimum_thickness are metres,
    not a claim of biological leaf thickness. No engine imports or randomness.
    """
    if type(sections) is not int or not 1 <= sections <= shape.stations - 1:
        raise ValueError('leaf contact sections must be 1..(leaf stations - 1)')
    if not np.isfinite([padding, minimum_thickness]).all() or padding < 0 or minimum_thickness <= 0:
        raise ValueError('invalid leaf contact padding/thickness')
    vertices, _ = leaf_mesh(length, width, shape)
    stations = np.unique(vertices[:, 0])
    result = []
    for intervals in np.array_split(np.arange(len(stations) - 1), sections):
        start, end = stations[intervals[0]], stations[intervals[-1] + 1]
        points = vertices[(vertices[:, 0] >= start) & (vertices[:, 0] <= end)]
        midrib_start = points[(points[:, 0] == start) & (points[:, 1] == 0)].mean(axis=0)
        midrib_end = points[(points[:, 0] == end) & (points[:, 1] == 0)].mean(axis=0)
        x = midrib_end - midrib_start
        x /= np.linalg.norm(x)
        y = np.array([0., 1., 0.])
        rotation = np.column_stack((x, y, np.cross(x, y)))
        local = points @ rotation
        lo, hi = local.min(axis=0), local.max(axis=0)
        half = (hi - lo) / 2 + padding
        half[2] = max(half[2], minimum_thickness / 2)
        result.append((rotation @ ((lo + hi) / 2), rotation, half))
    return result
