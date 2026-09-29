"""Renderer/physics-independent plant rest geometry, in metres and plant-local axes.

+Z is up; the root starts at (0, 0, 0). A branch's attachment_t locates its
start on its parent centreline. Leaf position is the blade base, fruit position
is the sphere centre. Their attachment_t also refers to the parent centreline.
These rest attachments and the segment graph are retained for a future builder.
"""
from dataclasses import asdict, dataclass, field
from copy import deepcopy
import json
from numbers import Real
from pathlib import Path

import numpy as np

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")


from .skeleton import StemSegment, PlantSkeleton


@dataclass
class LeafShape:
    """Blade profile: fold in radians, curl/droop as fractions of blade length.

    thickness separates front/back vertices for renderers that weld vertices.
    Seven stations produce 40 triangles INCLUDING the reverse-facing surface.
    """
    stations: int = 7
    fold: float = 0.20
    curl: float = 0.045
    droop: float = 0.09
    profile_power: float = 1.1
    thickness: float = 0.00008

    def validate(self):
        if type(self.stations) is not int or self.stations < 4:
            raise ValueError("leaf stations must be an integer >= 4")
        for key in ("fold", "curl", "droop", "profile_power", "thickness"):
            _finite_number(getattr(self, key), f"leaf {key}")
        if not 0 <= self.fold < np.pi / 2:
            raise ValueError("leaf fold must be in [0, pi/2)")
        if self.droop < 0 or self.profile_power <= 0 or self.thickness <= 0:
            raise ValueError("droop must be nonnegative; profile_power and thickness must be positive")


@dataclass
class LeafPlacement:
    parent_segment: str
    position: Vec3
    rotmat: Mat3  # blade-local -> plant-local; +X base-to-tip, +Y width, +Z upper face
    length: float
    width: float
    color_class: int
    attachment_t: float = 1.0
    visual_metadata: dict = field(default_factory=dict)

    @property
    def parent_branch(self):
        return self.parent_segment

    @parent_branch.setter
    def parent_branch(self, value):
        self.parent_segment = value


@dataclass
class FruitPlacement:
    id: str
    parent_segment: str
    position: Vec3
    radius: float
    stem_length: float
    targetable: bool = True
    attachment_t: float = 1.0
    stem_direction: Vec3 = (0.0, 0.0, 1.0)  # centre toward attachment; unit vector
    stem_radius: float = 0.002
    visual_metadata: dict = field(default_factory=dict)

    @property
    def parent_branch(self):
        return self.parent_segment

    @parent_branch.setter
    def parent_branch(self, value):
        self.parent_segment = value


@dataclass
class PlantSpec:
    name: str
    skeleton: PlantSkeleton = field(default_factory=PlantSkeleton)
    leaves: list[LeafPlacement] = field(default_factory=list)
    fruits: list[FruitPlacement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    leaf_shape: LeafShape = field(default_factory=LeafShape)
    species: str = 'generic'
    preset_name: str = ''

    @property
    def branches(self):
        """V1-compatible segment list; canonical ownership is skeleton.segments."""
        return self.skeleton.segments

    def validate(self):
        """Raise ValueError on invalid rest geometry/topology; otherwise return self.

        List order need not be topological. Continuations may retain their
        parent's order, but a child cannot have a lower order than its parent.
        """
        def vector(value, label):
            a = np.asarray(value, dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError(f"{label} must be a finite 3-vector")
            return a

        def positive(value, label, allow_zero=False):
            _finite_number(value, label)
            if (value < 0 if allow_zero else value <= 0):
                raise ValueError(f"invalid {label}: {value}")

        def attachment(value, label):
            _finite_number(value, label + ".attachment_t")
            if not 0 <= value <= 1:
                raise ValueError(f"{label}: attachment_t must be in [0, 1]")

        def flag(value, label):
            if type(value) is not bool:
                raise ValueError(f"{label} must be bool")

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tree name must be nonempty")
        self.leaf_shape.validate()
        self.skeleton.validate()
        by_id = self.skeleton.by_id
        for i, leaf in enumerate(self.leaves):
            label = f"leaf[{i}]"
            if not isinstance(leaf.parent_segment, str) or leaf.parent_segment not in by_id:
                raise ValueError(f"{label}: unknown parent")
            attachment(leaf.attachment_t, label)
            pos = vector(leaf.position, label)
            if not np.allclose(pos, by_id[leaf.parent_segment].point_at(leaf.attachment_t), atol=1e-7, rtol=0):
                raise ValueError(f"{label}: blade base does not match branch attachment")
            rot = np.asarray(leaf.rotmat, dtype=float)
            if (rot.shape != (3, 3) or not np.isfinite(rot).all()
                    or not np.allclose(rot.T @ rot, np.eye(3), atol=2e-6)
                    or not np.isclose(np.linalg.det(rot), 1, atol=2e-6)):
                raise ValueError(f"{label}: rotmat must be a proper rotation")
            positive(leaf.length, label + ".length")
            positive(leaf.width, label + ".width")
            if type(leaf.color_class) is not int or leaf.color_class < 0:
                raise ValueError(f"{label}: color_class must be a nonnegative integer")
        ids = set(by_id)
        for fruit in self.fruits:
            if not isinstance(fruit.id, str) or not fruit.id or fruit.id in ids:
                raise ValueError(f"invalid/duplicate fruit id: {fruit.id}")
            ids.add(fruit.id)
            if not isinstance(fruit.parent_segment, str) or fruit.parent_segment not in by_id:
                raise ValueError(f"{fruit.id}: unknown parent")
            attachment(fruit.attachment_t, fruit.id)
            pos = vector(fruit.position, fruit.id)
            direction = vector(fruit.stem_direction, fruit.id + ".stem_direction")
            if not np.isclose(np.linalg.norm(direction), 1, atol=1e-7, rtol=0):
                raise ValueError(f"{fruit.id}: stem_direction must be unit length")
            positive(fruit.radius, fruit.id + ".radius")
            positive(fruit.stem_length, fruit.id + ".stem_length", allow_zero=True)
            positive(fruit.stem_radius, fruit.id + ".stem_radius")
            flag(fruit.targetable, fruit.id + ".targetable")
            tip = pos + direction * (fruit.radius + fruit.stem_length)
            if not np.allclose(tip, by_id[fruit.parent_segment].point_at(fruit.attachment_t), atol=1e-7, rtol=0):
                raise ValueError(f"{fruit.id}: stem tip does not match branch attachment")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        try:
            for metadata in [self.metadata] + [p.visual_metadata for p in self.leaves + self.fruits]:
                if not isinstance(metadata, dict):
                    raise ValueError('metadata must be a JSON object')
                json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain only finite JSON values") from exc
        return self

    def bounds(self):
        """Conservative plant-local AABB, including radii, leaf blades and stems.

        Branch bounds also enclose the simplified capsule's rounded end caps.
        Leaves use the actual procedural vertices, not just their attachment.
        """
        from .geometry import leaf_mesh

        points = []
        for branch in self.branches:
            radius = max(branch.radius_start, branch.radius_end)
            ends = np.asarray([branch.start, branch.end])
            points.extend((ends - radius, ends + radius))
        for leaf in self.leaves:
            vs, _ = leaf_mesh(leaf.length, leaf.width, self.leaf_shape)
            points.append(vs @ np.asarray(leaf.rotmat).T + leaf.position)
        for fruit in self.fruits:
            points.extend([np.asarray([fruit.position]) - fruit.radius,
                           np.asarray([fruit.position]) + fruit.radius])
            direction = np.asarray(fruit.stem_direction)
            ends = np.asarray(fruit.position) + np.outer(
                [fruit.radius, fruit.radius + fruit.stem_length], direction)
            points.extend([ends - fruit.stem_radius, ends + fruit.stem_radius])
        if not points:
            return np.zeros(3), np.zeros(3)
        vs = np.concatenate(points)
        return vs.min(axis=0), vs.max(axis=0)

    def height(self):
        """AABB Z span in metres (includes conservative capsule caps)."""
        lower, upper = self.bounds()
        return float(upper[2] - lower[2])

    def canopy_bounds(self):
        """Leaf-only AABB in plant-local metres; independent of trunk end caps."""
        from .geometry import leaf_batches
        batches = leaf_batches(self)
        if not batches:
            return np.zeros(3), np.zeros(3)
        vertices = np.concatenate([vs for vs, _ in batches.values()])
        return vertices.min(axis=0), vertices.max(axis=0)

    def scaled(self, factor):
        """Copy and uniformly scale all metric rest geometry about the root.

        Orientations, topology, attachment_t, counts and collision flags stay
        unchanged. Metadata is copied; recipe/provenance entries remain nominal.
        """
        _finite_number(factor, "scale factor")
        if factor <= 0:
            raise ValueError("scale factor must be finite and positive")
        result = deepcopy(self)
        def position(value):
            return tuple(float(v) * factor for v in value)
        for branch in result.branches:
            branch.start, branch.end = position(branch.start), position(branch.end)
            branch.radius_start *= factor
            branch.radius_end *= factor
        for leaf in result.leaves:
            leaf.position = position(leaf.position)
            leaf.length *= factor
            leaf.width *= factor
        for fruit in result.fruits:
            fruit.position = position(fruit.position)
            fruit.radius *= factor
            fruit.stem_length *= factor
            fruit.stem_radius *= factor
        result.leaf_shape.thickness *= factor
        result.metadata["scale_multiplier"] = self.metadata.get("scale_multiplier", 1.0) * factor
        return result.validate()

    def summary(self):
        lower, upper = self.bounds()
        canopy_lower, canopy_upper = self.canopy_bounds()
        return {"name": self.name, "branch_count": len(self.branches),
                "collidable_branch_count": sum(b.collidable for b in self.branches),
                "leaf_count": len(self.leaves), "fruit_count": len(self.fruits),
                "bbox_m": [lower.tolist(), upper.tolist()],
                "height_m": float(upper[2] - lower[2]),
                "bbox_size_m": (upper - lower).tolist(),
                "canopy_bbox_m": [canopy_lower.tolist(), canopy_upper.tolist()],
                "canopy_size_m": (canopy_upper - canopy_lower).tolist(),
                "root_to_top_m": float(upper[2]),
                "leaf_triangle_count": len(self.leaves) * (8 * self.leaf_shape.stations - 16)}

    def to_dict(self):
        """JSON-compatible values, also accepting NumPy-authored placements."""
        def plain(value):
            if isinstance(value, np.ndarray):
                return plain(value.tolist())
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            return value

        return plain(asdict(self))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        skeleton = data.pop('skeleton')
        data['skeleton'] = PlantSkeleton([StemSegment(**s) for s in skeleton['segments']])
        data['leaves'] = [LeafPlacement(**leaf) for leaf in data.get('leaves', [])]
        data['fruits'] = [FruitPlacement(**fruit) for fruit in data.get('fruits', [])]
        data["leaf_shape"] = LeafShape(**data.get("leaf_shape", {}))
        return cls(**data).validate()

    def to_json(self, path=None):
        self.validate()
        text = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path):
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
