"""Serializable qualitative cluster dynamics, independent of WRS and MuJoCo."""
from dataclasses import dataclass, field, asdict
import json
import numpy as np


@dataclass
class BranchDynamicsProfile:
    stiffness: float = 1.5       # N m / rad
    damping: float = .04        # N m s / rad
    springref: float = 0.0      # radians in rest joint coordinates
    frictionloss: float = 0.0
    armature: float = .0001
    limits: tuple = (-.65, .65)
    wood_density: float = 500.0 # kg/m^3, qualitative
    fruit_density: float = 850.0
    leaf_mass: float = .00015   # kg per visual blade, lumped into its cluster
    minimum_mass: float = .001
    spacer_mass: float = .0001
    spacer_radius: float = .003


@dataclass
class BranchClusterSpec:
    id: str
    root_segment: str
    member_segments: list[str]
    parent_cluster: str | None = None
    dof: int = 1
    profile: str = 'default'


@dataclass
class FoliageProxySettings:
    sections_per_leaf: int = 3
    padding: float = .0005      # metres, applied after plant scale
    minimum_thickness: float = .0015  # full contact thickness along local normal


@dataclass
class PlantDynamicsSpec:
    clusters: list[BranchClusterSpec] = field(default_factory=list)
    profiles: dict[str, BranchDynamicsProfile] = field(default_factory=lambda: {'default': BranchDynamicsProfile()})
    foliage_proxy: FoliageProxySettings = field(default_factory=FoliageProxySettings)

    def segment_clusters(self, plant):
        result = {s.id: None for s in plant.skeleton.segments}
        for cluster in self.clusters:
            for key in cluster.member_segments:
                result[key] = cluster.id
        return result

    def validate(self, plant):
        plant.validate()
        by_id = plant.skeleton.by_id
        owners, clusters = {}, {}
        for name, p in self.profiles.items():
            if not isinstance(name, str) or not name:
                raise ValueError('invalid dynamics profile name')
            for key, value in asdict(p).items():
                if key == 'limits':
                    if np.shape(value) != (2,) or not np.isfinite(value).all() or value[0] >= value[1]:
                        raise ValueError('invalid joint limits')
                elif not np.isfinite(value) or (key != 'springref' and value < 0):
                    raise ValueError(f'invalid dynamics {key}')
            if min(p.wood_density, p.fruit_density, p.minimum_mass, p.spacer_mass, p.spacer_radius) <= 0:
                raise ValueError('mass/density/spacer size must be positive')
            if not p.limits[0] <= p.springref <= p.limits[1]:
                raise ValueError('springref must lie inside joint limits')
        for c in self.clusters:
            if not isinstance(c.id, str) or not c.id or c.id in clusters:
                raise ValueError('cluster IDs must be unique and nonempty')
            clusters[c.id] = c
            if type(c.dof) is not int or c.dof not in (1, 2) or c.profile not in self.profiles:
                raise ValueError(f'{c.id}: unsupported DOF/profile')
            if c.root_segment not in c.member_segments or not c.member_segments:
                raise ValueError(f'{c.id}: root must be a member')
            for key in c.member_segments:
                if key not in by_id or key in owners:
                    raise ValueError(f'{c.id}: missing or multiply owned segment {key}')
                owners[key] = c.id
                if key != c.root_segment and by_id[key].parent_id not in c.member_segments:
                    raise ValueError(f'{c.id}: cluster members must form a connected subtree')
            if by_id[c.root_segment].parent_id is None:
                raise ValueError('plant roots must remain static')
        for c in self.clusters:
            parent = owners.get(by_id[c.root_segment].parent_id)
            if parent != c.parent_cluster or (parent is not None and parent not in clusters):
                raise ValueError(f'{c.id}: parent_cluster does not match plant topology')
        # No static child may be left floating in space when its parent moves.
        for s in plant.skeleton.segments:
            if owners.get(s.parent_id) is not None and s.id not in owners:
                raise ValueError(f'{s.id}: descendant of moving segment cannot remain static')
        p = self.foliage_proxy
        if type(p.sections_per_leaf) is not int or not 1 <= p.sections_per_leaf < plant.leaf_shape.stations:
            raise ValueError('foliage sections_per_leaf must be 1..(leaf stations - 1)')
        if not np.isfinite([p.padding, p.minimum_thickness]).all() or p.padding < 0 or p.minimum_thickness <= 0:
            raise ValueError('invalid foliage proxy size')
        return self

    def to_json(self):
        return json.dumps(asdict(self), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, data):
        return cls([BranchClusterSpec(**c) for c in data.get('clusters', [])],
                   {k: BranchDynamicsProfile(**p) for k, p in data.get('profiles', {}).items()},
                   FoliageProxySettings(**data.get('foliage_proxy', {})))

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_config(cls, plant, config):
        """Expand authored cluster roots into exclusive descendant ownership.

        Nested roots are allowed: the nearest root owns the segment, and the
        child's parent_cluster is inferred from the actual segment graph.
        """
        settings = config['dynamics']
        roots = settings['cluster_roots']
        by_id = plant.skeleton.by_id
        selected = {entry['root_segment']: entry['id'] for entry in roots}
        if len(selected) != len(roots) or any(key not in by_id for key in selected):
            raise ValueError('cluster roots must be unique existing segments')
        ownership = {}
        for s in plant.skeleton.segments:
            current = s
            while current.id not in selected and current.parent_id is not None:
                current = by_id[current.parent_id]
            ownership[s.id] = selected.get(current.id)
        data = {key: value for key, value in settings.items() if key != 'cluster_roots'}
        data['clusters'] = [dict(**entry, member_segments=[s.id for s in plant.skeleton.segments if ownership[s.id] == entry['id']],
                                 parent_cluster=ownership.get(by_id[entry['root_segment']].parent_id)) for entry in roots]
        return cls.from_dict(data).validate(plant)
