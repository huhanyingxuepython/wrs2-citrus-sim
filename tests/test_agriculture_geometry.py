"""Engine-independent contracts: run with stdlib unittest and NumPy."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

from agriculture import PlantSkeleton, StemSegment, PlantSpec
from agriculture.config import load_config, GENERIC_CONFIG
from agriculture.generator import generate
from agriculture.morphology import apply_pipe_model
from agriculture.foliage import place_leaves
from agriculture.geometry import leaf_mesh, leaf_contact_boxes
from agriculture.spec import LeafShape
from agriculture.dynamics import PlantDynamicsSpec


def skeleton():
    return PlantSkeleton([
        StemSegment('root', None, (0, 0, 0), (0, 0, 1), .03, .02, 0),
        StemSegment('left', 'root', (0, 0, 1), (-.3, 0, 1.2), .01, .005, 1),
        StemSegment('right', 'root', (0, 0, 1), (.3, 0, 1.2), .01, .005, 1),
        StemSegment('tip', 'right', (.3, 0, 1.2), (.4, .1, 1.3), .003, .001, 2)])


class PlantGeometryTests(unittest.TestCase):
    def test_topology_queries_frames_and_bounds(self):
        s = skeleton().validate()
        self.assertEqual(s.roots, ['root'])
        self.assertEqual(s.terminals, ['left', 'tip'])
        self.assertEqual(s.descendants('right'), ['tip'])
        self.assertEqual(s.descendants('right', include_self=True), ['right', 'tip'])
        self.assertEqual(s.depth('tip'), 2)
        self.assertAlmostEqual(s.total_length(), 1 + 2 * np.hypot(.3, .2) + np.sqrt(.03))
        lo, hi = s.bounds()
        self.assertAlmostEqual(s.height(), hi[2] - lo[2])
        for stem in s.segments:
            np.testing.assert_allclose(stem.frame[:, 2] * stem.length, np.subtract(stem.end, stem.start), atol=1e-7)
            np.testing.assert_allclose(stem.frame.T @ stem.frame, np.eye(3), atol=1e-6)
        s.segments.reverse()
        s.validate()
        self.assertEqual(s.depth('tip'), 2)

    def test_invalid_skeleton_and_authored_frame(self):
        for key, value in [('end', (0, 0, 0)), ('radius_start', -1), ('parent_id', 'missing'),
                           ('rotmat', np.eye(3)), ('rotmat', np.zeros((3, 3)))]:
            s = skeleton()
            target = s.segments[1] if key == 'rotmat' else s.segments[0]
            setattr(target, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                s.validate()

    def test_multiple_root_shoots_are_supported(self):
        s = skeleton()
        s.segments.append(StemSegment('basal_shoot', None, (.03, 0, 0), (.1, 0, .5), .004, .002, 0))
        plant = PlantSpec('shrub', s).validate()
        self.assertEqual(len(plant.skeleton.roots), 2)
        self.assertEqual(PlantSpec.from_json(plant.to_json()).skeleton.roots, plant.skeleton.roots)

    def test_optional_pipe_model_preserves_geometry_and_topology(self):
        s = skeleton()
        radii = [p.radius_start for p in s.segments]
        out = apply_pipe_model(s, .002, beta=2)
        self.assertEqual(radii, [p.radius_start for p in s.segments])
        for stem in out.segments:
            self.assertEqual(stem.start, s.by_id[stem.id].start)
            if out.children[stem.id]:
                self.assertAlmostEqual(stem.radius_start ** 2,
                                      sum(out.by_id[key].radius_start ** 2 for key in out.children[stem.id]))

    def test_generators_serialization_and_seed_variation(self):
        for config in (load_config(), load_config(GENERIC_CONFIG)):
            untouched = deepcopy(config)
            a, b = generate(config, 12), generate(config, 12)
            self.assertEqual(a.to_json(), b.to_json())
            self.assertEqual(a.to_json(), PlantSpec.from_json(a.to_json()).to_json())
            self.assertEqual(config, untouched)
            self.assertEqual([f.id for f in a.fruits], [f.id for f in generate(config, 13).fruits])
            self.assertTrue(all(s.radius_end > 0 for s in a.branches))
        a, b = generate(load_config(GENERIC_CONFIG), 12), generate(load_config(GENERIC_CONFIG), 13)
        self.assertNotEqual([s.end for s in a.branches], [s.end for s in b.branches])
        self.assertNotEqual([f.position for f in a.fruits], [f.position for f in b.fruits])

    def test_foliage_standalone_density_winding_and_parents(self):
        config = load_config(GENERIC_CONFIG)
        plant = generate(config)
        parameters = config['leaf_generation']
        a = place_leaves(plant.skeleton, parameters, plant.leaf_shape, seed=99)
        b = place_leaves(plant.skeleton, parameters, plant.leaf_shape, seed=99)
        self.assertEqual(a, b)
        self.assertEqual(place_leaves(plant.skeleton, parameters | {'density': 0}, plant.leaf_shape), [])
        for leaf in a:
            self.assertIn(leaf.parent_segment, plant.skeleton.terminals)
        vs, fs = leaf_mesh(a[0].length, a[0].width, plant.leaf_shape)
        tri = vs[fs]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        np.testing.assert_allclose(n[:len(n)//2], -n[len(n)//2:], atol=1e-12)

    def test_dynamics_partition_and_roundtrip(self):
        c = load_config()
        plant = generate(c)
        d = PlantDynamicsSpec.from_config(plant, c)
        self.assertEqual(d.to_json(), PlantDynamicsSpec.from_json(d.to_json()).validate(plant).to_json())
        self.assertEqual(len(d.clusters), 16)
        self.assertEqual(sum(c.dof for c in d.clusters), 19)
        for key in plant.skeleton.roots:
            self.assertIsNone(d.segment_clusters(plant)[key])
        for mutation in ('missing', 'overlap', 'static_descendant', 'parent', 'dof'):
            bad = deepcopy(d)
            if mutation == 'missing': bad.clusters[0].member_segments.append('missing')
            elif mutation == 'overlap': bad.clusters[1].member_segments.append(bad.clusters[0].root_segment)
            elif mutation == 'static_descendant': bad.clusters[1].member_segments.pop()
            elif mutation == 'parent': bad.clusters[1].parent_cluster = bad.clusters[0].id
            else: bad.clusters[0].dof = 3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): bad.validate(plant)

    def test_lab_approach_foliage_has_contact_ownership_with_bounded_cost(self):
        # Cover the central front approach shoots, not the whole front hemisphere.
        config = load_config()
        plant = generate(config)
        owners = PlantDynamicsSpec.from_config(plant, config).segment_clusters(plant)
        required = [f'front_middle_shoot_{i:02d}' for i in range(5)]
        required += [f'front_lower_shoot_{i:02d}' for i in range(4)]
        required += ['front_upper_shoot_02', 'front_upper_shoot_03']
        for root in required:
            for key in plant.skeleton.descendants(root, include_self=True):
                self.assertIsNotNone(owners[key], key)
        contact_count = sum(owners[leaf.parent_segment] is not None for leaf in plant.leaves)
        self.assertLessEqual(contact_count, 400)
        self.assertIsNone(owners['apex_shoot_00'])
        # Preserve the cheaper non-interaction foliage and fixed hard skeleton.
        self.assertTrue(any(owners[leaf.parent_segment] is None for leaf in plant.leaves))
        self.assertTrue(all(owners[s.id] is None for s in plant.branches if s.order <= 1))

    def test_leaf_contact_strips_enclose_blade_without_thick_global_box(self):
        # Include different station counts: strip boundaries must not leave gaps
        # when the visual tessellation is not divisible by the strip count.
        for stations in (7, 8, 11):
            shape = LeafShape(stations=stations, fold=.22, curl=.06, droop=.12)
            vs, fs = leaf_mesh(.10, .03, shape)
            boxes = leaf_contact_boxes(.10, .03, shape, sections=3,
                                       padding=.0005, minimum_thickness=.0015)
            other = leaf_contact_boxes(.10, .03, shape, sections=3,
                                       padding=.0005, minimum_thickness=.0015)
            # Triangle vertices + barycentres are independent surface probes.
            samples = np.concatenate((vs, vs[fs].mean(axis=1)))
            covered = np.zeros(len(samples), dtype=bool)
            self.assertEqual(len(boxes), 3)
            for (center, rotation, half), repeated in zip(boxes, other):
                np.testing.assert_array_equal(np.concatenate((center, rotation.ravel(), half)),
                                              np.concatenate((repeated[0], repeated[1].ravel(), repeated[2])))
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1)
                self.assertLess(2 * half[2], .007)
                covered |= np.all(np.abs((samples - center) @ rotation) <= half + 1e-12, axis=1)
            self.assertTrue(covered.all())

    def test_leaf_contact_settings_validation_and_minimum_thickness(self):
        shape = LeafShape(fold=0, curl=0, droop=0)
        args = dict(sections=3, padding=0, minimum_thickness=.0015)
        for _, _, half in leaf_contact_boxes(.1, .03, shape, **args):
            self.assertAlmostEqual(half[2] * 2, .0015)
        config = load_config()
        plant = generate(config)
        dynamics = PlantDynamicsSpec.from_config(plant, config)
        for key, bad in [('sections', 0), ('sections', True), ('sections', 7),
                         ('padding', -.001), ('padding', np.nan), ('minimum_thickness', 0)]:
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                leaf_contact_boxes(.1, .03, shape, **(args | {key: bad}))
            changed = deepcopy(dynamics)
            setattr(changed.foliage_proxy, 'sections_per_leaf' if key == 'sections' else key, bad)
            with self.assertRaises(ValueError):
                changed.validate(plant)

    def test_pure_generation_and_export_block_engine_imports(self):
        code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('wrs','mujoco','newton','warp'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from agriculture.config import load_config, GENERIC_CONFIG
from agriculture.generator import generate
from agriculture.spec import PlantSpec
from agriculture.dynamics import PlantDynamicsSpec
from agriculture.geometry import leaf_contact_boxes
for c in (load_config(), load_config(GENERIC_CONFIG)):
    p = PlantSpec.from_json(generate(c).to_json()).scaled(.8)
    p.summary()
    if 'dynamics' in c:
        PlantDynamicsSpec.from_config(p, c).to_json()
    leaf_contact_boxes(.1, .03, p.leaf_shape, sections=3, padding=.0005, minimum_thickness=.0015)
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
