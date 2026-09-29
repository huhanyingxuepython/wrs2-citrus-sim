"""Small generator protocol and two concrete paths; no rendering/physics."""
from copy import deepcopy
from typing import Protocol
import numpy as np
from .config import load_config
from .spec import PlantSpec
from .skeleton import StemSegment, PlantSkeleton
from .reference import generate_tree
from .morphology import apply_pipe_model


class PlantGenerator(Protocol):
    def generate(self, config: dict, seed: int) -> PlantSpec: ...


class ReferenceFittedGenerator:
    def generate(self, config, seed=None):
        plant = generate_tree(config, seed=seed).as_plant()
        plant.species = config.get('species', 'generic')
        plant.preset_name = config.get('preset_name', config['name'])
        return plant.validate()


class ProceduralTreeGenerator:
    """Seeded central trunk, lateral shoots, attached foliage and fruit.

    This deliberately small path proves that neither PlantSpec nor builders
    require lab IDs, reference images, or a particular species generator.
    """
    def generate(self, config, seed=None):
        config = deepcopy(config)
        seed = config['seed'] if seed is None else seed
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        rng = np.random.default_rng(seed)
        p = config['procedural']
        segments = [StemSegment('trunk', None, (0, 0, 0), (0, 0, p['trunk_height']),
                                p['trunk_radius'], p['trunk_radius'] * p['taper'], 0)]
        phase = rng.uniform(0, 2 * np.pi)
        for i in range(p['primary_count']):
            t = rng.uniform(*p['attachment_range'])
            az = phase + i * np.deg2rad(p['azimuth_step_deg'])
            pitch = np.deg2rad(rng.uniform(*p['pitch_deg']))
            direction = np.array([np.cos(az) * np.cos(pitch), np.sin(az) * np.cos(pitch), np.sin(pitch)])
            start = segments[0].point_at(t)
            end = start + direction * rng.uniform(*p['primary_length'])
            primary = StemSegment(f'primary_{i:03d}', 'trunk', tuple(start), tuple(end),
                                  p['primary_radius'], p['primary_radius'] * p['taper'], 1, attachment_t=t)
            segments.append(primary)
            for j in range(p['shoots_per_primary']):
                a = az + rng.uniform(*np.deg2rad(p['shoot_spread_deg']))
                pitch = np.deg2rad(rng.uniform(*p['shoot_pitch_deg']))
                direction = np.array([np.cos(a) * np.cos(pitch), np.sin(a) * np.cos(pitch), np.sin(pitch)])
                end = np.asarray(primary.end) + direction * rng.uniform(*p['shoot_length'])
                segments.append(StemSegment(f'shoot_{i:03d}_{j:02d}', primary.id, primary.end, tuple(end),
                                            p['shoot_radius'], p['shoot_radius'] * p['taper'], 2))
        skeleton = PlantSkeleton(segments).validate()
        if p.get('pipe_model'):
            skeleton = apply_pipe_model(skeleton, **p['pipe_model'])
        terminals = skeleton.terminals
        if not 0 <= p['fruit_count'] <= len(terminals):
            raise ValueError('fruit_count exceeds available terminal shoots')
        choices = rng.choice(len(terminals), size=p['fruit_count'], replace=False)
        config['macro_branches'] = skeleton.segments
        config['twig_generation']['parent_ids'] = None
        config['branch_profiles'] = {}
        config['leaf_placements'] = []
        config['fruit_placements'] = [dict(id=f'fruit_{i:03d}', parent_segment=terminals[index],
                                          attachment_t=float(rng.uniform(*p['fruit_attachment_range']))) for i, index in enumerate(choices)]
        return ReferenceFittedGenerator().generate(config, seed)


def generate(config=None, seed=None):
    config = load_config() if config is None else config
    kind = config.get('generator', 'reference')
    generators = {'reference': ReferenceFittedGenerator, 'procedural': ProceduralTreeGenerator}
    if kind not in generators:
        raise ValueError(f'unknown plant generator {kind!r}')
    return generators[kind]().generate(config, seed)
