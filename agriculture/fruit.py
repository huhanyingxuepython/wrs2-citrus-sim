"""Fruit attachment records only; builders decide rendering and rigid mounts."""
import numpy as np
from .spec import FruitPlacement


def place_fruits(skeleton, placements, defaults):
    by_id, fruits = skeleton.by_id, []
    for i, placement in enumerate(placements):
        params = defaults | placement
        parent = params.get('parent_segment', params.get('parent_branch'))
        if parent not in by_id:
            raise ValueError(f'fruit references unknown segment {parent}')
        radius, stem_length = params['radius'], params['stem_length']
        direction = np.asarray(params['stem_direction'], dtype=float)
        position = by_id[parent].point_at(params['attachment_t']) - direction * (radius + stem_length)
        fruits.append(FruitPlacement(params.get('id', f'orange_{i:03d}'), parent, tuple(position),
                                     radius, stem_length, params['targetable'], params['attachment_t'],
                                     tuple(direction), params['stem_radius'], params.get('visual_metadata', {})))
    return fruits
