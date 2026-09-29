"""SI plant-local stem graph. No WRS, renderer, or physics imports."""
from dataclasses import dataclass, field
import numpy as np


def unit_direction(value):
    """Normalize growth vectors; a zero radial bias stays zero (V1 convention)."""
    v = np.asarray(value, dtype=np.float32)
    length = np.linalg.norm(v)
    return v / length if length > np.finfo(np.float32).eps else np.zeros_like(v)


def stem_frame(direction):
    """Right-handed 3x3 frame, columns X/Y/Z; Z follows growth.

    Same up-projection convention as WRS rotmat_from_normal. Kept as a small
    NumPy geometry operation here because importing wrs eagerly imports engines.
    At a vertical stem, +X world defines local Y (the WRS fallback).
    """
    z = np.asarray(direction, dtype=float)
    length = np.linalg.norm(z)
    if z.shape != (3,) or not np.isfinite(z).all() or length <= 0:
        raise ValueError('stem direction must be a nonzero finite 3-vector')
    z = z / length
    y = np.array([0., 0., 1.]) - z[2] * z
    if np.linalg.norm(y) < 1e-8:
        y = np.array([1., 0., 0.]) - z[0] * z
    y /= np.linalg.norm(y)
    return np.column_stack((np.cross(y, z), y, z)).astype(np.float32)


@dataclass
class StemSegment:
    id: str
    parent_id: str | None
    start: tuple
    end: tuple
    radius_start: float
    radius_end: float
    order: int
    collidable: bool = True
    attachment_t: float = 1.0
    rotmat: tuple | None = None  # optional authored roll; otherwise canonical frame
    semantic_role: str | None = None

    def point_at(self, t):
        return np.asarray(self.start, dtype=float) * (1 - t) + np.asarray(self.end) * t

    @property
    def length(self):
        return float(np.linalg.norm(np.subtract(self.end, self.start)))

    @property
    def frame(self):
        return stem_frame(np.subtract(self.end, self.start)) if self.rotmat is None else np.asarray(self.rotmat, dtype=float)


@dataclass
class PlantSkeleton:
    segments: list[StemSegment] = field(default_factory=list)

    @property
    def by_id(self):
        return {s.id: s for s in self.segments}

    @property
    def roots(self):
        return [s.id for s in self.segments if s.parent_id is None]

    @property
    def children(self):
        result = {s.id: [] for s in self.segments}
        for s in self.segments:
            if s.parent_id is not None:
                result[s.parent_id].append(s.id)
        return result

    @property
    def terminals(self):
        return [key for key, children in self.children.items() if not children]

    def descendants(self, segment_id, include_self=False):
        children = self.children
        if segment_id not in children:
            raise KeyError(segment_id)
        result, pending = [], [segment_id]
        while pending:
            key = pending.pop()
            result.append(key)
            pending.extend(reversed(children[key]))
        return result if include_self else result[1:]

    def depth(self, segment_id):
        by_id = self.by_id
        current, depth = by_id[segment_id], 0
        while current.parent_id is not None:
            current, depth = by_id[current.parent_id], depth + 1
        return depth

    def validate(self):
        from .spec import _finite_number
        by_id = {}
        for s in self.segments:
            if not isinstance(s.id, str) or not s.id or s.id in by_id:
                raise ValueError('segment IDs must be nonempty and unique')
            by_id[s.id] = s
            if s.parent_id is not None and not isinstance(s.parent_id, str):
                raise ValueError(f'{s.id}: invalid parent_id')
            for point in (s.start, s.end):
                if np.shape(point) != (3,) or not np.isfinite(point).all():
                    raise ValueError(f'{s.id}: invalid endpoint')
            for key, value in [('length', s.length), ('radius_start', s.radius_start), ('radius_end', s.radius_end)]:
                _finite_number(value, key)
                if value <= 0:
                    raise ValueError(f'{s.id}: {key} must be positive')
            if type(s.order) is not int or s.order < 0 or type(s.collidable) is not bool:
                raise ValueError(f'{s.id}: invalid order/collidable')
            _finite_number(s.attachment_t, 'attachment_t')
            if not 0 <= s.attachment_t <= 1:
                raise ValueError(f'{s.id}: invalid attachment_t')
            r = s.frame
            if (r.shape != (3, 3) or not np.isfinite(r).all()
                    or not np.allclose(r.T @ r, np.eye(3), atol=2e-6, rtol=0)
                    or not np.isclose(np.linalg.det(r), 1, atol=2e-6, rtol=0)
                    or not np.allclose(r[:, 2] * s.length, np.subtract(s.end, s.start), atol=1e-7, rtol=1e-6)):
                raise ValueError(f'{s.id}: frame must be a proper rotation with +Z along the segment')
        if not self.roots:
            raise ValueError('skeleton needs at least one root')
        for s in self.segments:
            if s.parent_id is not None:
                if s.parent_id not in by_id:
                    raise ValueError(f'{s.id}: unknown parent')
                parent = by_id[s.parent_id]
                if s.order < parent.order or not np.allclose(s.start, parent.point_at(s.attachment_t), atol=1e-7, rtol=0):
                    raise ValueError(f'{s.id}: invalid parent attachment/order')
        complete = set()
        for s in self.segments:
            path = set()
            while s.id not in complete:
                if s.id in path:
                    raise ValueError(f'cycle at {s.id}')
                path.add(s.id)
                if s.parent_id is None:
                    break
                s = by_id[s.parent_id]
            complete.update(path)
        return self

    def bounds(self):
        if not self.segments:
            return np.zeros(3), np.zeros(3)
        points = []
        for s in self.segments:
            ends = np.asarray([s.start, s.end])
            r = max(s.radius_start, s.radius_end)
            points.extend([ends - r, ends + r])
        points = np.concatenate(points)
        return points.min(axis=0), points.max(axis=0)

    def height(self):
        lo, hi = self.bounds()
        return float(hi[2] - lo[2])

    def total_length(self):
        return sum(s.length for s in self.segments)

    def summary(self):
        return dict(segment_count=len(self.segments), roots=self.roots, terminals=self.terminals,
                    height_m=self.height(), total_length_m=self.total_length(),
                    bbox_m=[v.tolist() for v in self.bounds()])
