"""Small V1 constructor/JSON adapter. No engines; no duplicated geometry."""
import numpy as np
from .spec import PlantSpec, LeafPlacement as PlantLeaf, FruitPlacement as PlantFruit, LeafShape
from .skeleton import StemSegment, PlantSkeleton

BranchSegment = StemSegment


class LeafPlacement(PlantLeaf):
    def __init__(self, *args, parent_branch=None, **kwargs):
        if parent_branch is not None:
            kwargs['parent_segment'] = parent_branch
        super().__init__(*args, **kwargs)


class FruitPlacement(PlantFruit):
    def __init__(self, *args, parent_branch=None, **kwargs):
        if parent_branch is not None:
            kwargs['parent_segment'] = parent_branch
        super().__init__(*args, **kwargs)


class TreeSpec(PlantSpec):
    def __init__(self, name, branches=None, leaves=None, fruits=None, metadata=None, leaf_shape=None,
                 *, skeleton=None, species='citrus', preset_name=None):
        super().__init__(name, skeleton if skeleton is not None else PlantSkeleton([] if branches is None else branches),
                         [] if leaves is None else leaves, [] if fruits is None else fruits,
                         {} if metadata is None else metadata, LeafShape() if leaf_shape is None else leaf_shape,
                         species=species, preset_name=name if preset_name is None else preset_name)

    def validate(self):
        super().validate()
        roots = [s for s in self.branches if s.parent_id is None]
        if len(roots) != 1 or roots[0].order != 0 or not np.allclose(roots[0].start, 0, atol=1e-7, rtol=0):
            raise ValueError('tree needs one order-0 root starting at (0,0,0)')
        return self

    def to_dict(self):
        data = super().to_dict()
        data['branches'] = data.pop('skeleton')['segments']
        data.pop('species')
        data.pop('preset_name')
        for key in ('leaves', 'fruits'):
            for placement in data[key]:
                placement['parent_branch'] = placement.pop('parent_segment')
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(data['name'], [StemSegment(**b) for b in data.get('branches', [])],
                   [LeafPlacement(**p) for p in data.get('leaves', [])],
                   [FruitPlacement(**p) for p in data.get('fruits', [])],
                   data.get('metadata', {}), LeafShape(**data.get('leaf_shape', {}))).validate()

    def as_plant(self):
        """Independent canonical PlantSpec copy, suitable for either builder."""
        return PlantSpec.from_dict(PlantSpec.to_dict(self))
