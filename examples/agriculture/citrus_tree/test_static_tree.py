"""Contract checks: python -m unittest examples.agriculture.citrus_tree.test_static_tree."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from wrs import wss, wum
from wrs.scene.collision_shape import CapsuleCollisionShape, SphereCollisionShape
from wrs.viewer.protocol import describe, iter_scene_models, scene_message, unpack

from .builder import build_tree
from .generator import GENERIC_CONFIG, _intersects_foliage_gap, generate_tree, load_config
from .spec import BranchSegment, TreeSpec


class StaticTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(GENERIC_CONFIG)
        cls.spec = generate_tree(cls.config)

    def test_determinism_roundtrip_and_input_not_mutated(self):
        original = deepcopy(self.config)
        first = self.spec.to_json()
        self.assertEqual(first, generate_tree(self.config).to_json())
        self.assertNotEqual(first, generate_tree(self.config, seed=17).to_json())
        self.assertEqual(self.config, original)
        self.assertEqual(first, TreeSpec.from_json(first).to_json())
        arrays = deepcopy(self.spec)
        arrays.leaves[0].rotmat = np.asarray(arrays.leaves[0].rotmat)
        arrays.branches[0].start = np.asarray(arrays.branches[0].start)
        self.assertEqual(self.spec.to_dict(), TreeSpec.from_json(arrays.to_json()).to_dict())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "expanded.json"
            self.spec.to_json(path)
            self.assertEqual(first, TreeSpec.load(path).to_json())

    def test_rng_isolation_and_cross_process_determinism(self):
        numpy_state, python_state = np.random.get_state(), random.getstate()
        generate_tree(self.config)
        for before, after in zip(numpy_state, np.random.get_state()):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(python_state, random.getstate())
        expected = hashlib.sha256(self.spec.to_json().encode()).hexdigest()
        code = """
import hashlib
import random
import numpy as np
from examples.agriculture.citrus_tree.generator import GENERIC_CONFIG, load_config, generate_tree
random.seed(765)
np.random.seed(456)
print(hashlib.sha256(generate_tree(load_config(GENERIC_CONFIG)).to_json().encode()).hexdigest())
"""
        for hash_seed in ("1", "99"):
            result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                                    cwd=Path(__file__).resolve().parents[3],
                                    env=dict(os.environ, PYTHONHASHSEED=hash_seed))
            self.assertEqual(result.stdout.strip(), expected)

    def test_hand_fitted_skeleton_and_parent_attachments(self):
        skeleton = [BranchSegment(**b) for b in self.config["macro_branches"]]
        before = deepcopy(skeleton)
        result = generate_tree(self.config, skeleton=skeleton)
        self.assertEqual(skeleton, before)
        self.assertEqual(self.spec.to_json(), result.to_json())
        # Arbitrary ordering of a serialized graph is valid.
        result.branches.reverse()
        result.validate()
        by_id = {b.id: b for b in result.branches}
        parents = {b.parent_id for b in result.branches}
        foliage = self.config["leaf_generation"]
        for leaf in result.leaves:
            parent = by_id[leaf.parent_branch]
            self.assertNotIn(parent.id, parents)
            self.assertGreaterEqual(parent.order, foliage["min_branch_order"])
            self.assertLessEqual(max(parent.radius_start, parent.radius_end), foliage["max_branch_radius"])
            self.assertGreaterEqual(np.linalg.norm(parent.point_at(.5)[:2]), foliage["min_radial_distance"])
            np.testing.assert_allclose(leaf.position, parent.point_at(leaf.attachment_t), atol=1e-7)

    def test_empty_foliage_and_fruits_build_normally(self):
        for leaves, fruits in ((False, False), (False, True), (True, False)):
            with self.subTest(leaves=leaves, fruits=fruits):
                config = deepcopy(self.config)
                if not leaves:
                    config['leaf_generation']['per_branch'] = 0
                if not fruits:
                    config['fruit_placements'] = []
                spec = generate_tree(config)
                tree = build_tree(spec, config)
                self.assertEqual(bool(tree.leaf_objects), leaves)
                self.assertEqual(bool(tree.fruit_objects), fruits)
                self.assertEqual(tree.summary()['scene_object_count'], len(tree.objects))
                if not leaves:
                    self.assertEqual(len(build_tree(spec.scaled(.01), config).leaf_objects), 0)

    def test_adapter_rejects_unsupported_render_geometry(self):
        tiny = self.spec.scaled(.01)
        tiny.validate()  # Renderer limits do not restrict pure rest data.
        with self.assertRaisesRegex(ValueError, 'thickness'):
            build_tree(tiny, self.config)
        for key, value in (('branch_taper_steps', 0), ('branch_sides', 2),
                           ('fruit_subdivisions', -1), ('fruit_subdivisions', 1.5), ('stem_sides', 2)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                config = deepcopy(self.config)
                config['visual'][key] = value
                build_tree(self.spec, config)

    def test_native_collision_shapes_leaf_normals_and_bounds(self):
        tree = build_tree(self.spec, self.config)
        self.assertEqual(len(tree.leaf_objects), 3)
        self.assertTrue(150 <= len(self.spec.leaves) <= 300)
        self.assertTrue(4 <= len(tree.fruit_objects) <= 8)
        for obj in tree.leaf_objects:
            self.assertFalse(obj.collisions)
            geom = obj.visuals[0].geom
            # Test AFTER WRS's vertex welding; double-sided normals must survive.
            np.testing.assert_allclose(np.linalg.norm(geom.vns, axis=1), 1, atol=1e-5)
            self.assertTrue(np.isfinite(geom.fns).all())
        for obj, branch in zip(tree.branch_objects, self.spec.branches):
            self.assertEqual(len(obj.collisions), int(branch.collidable))
            self.assertFalse(obj.is_floating)
            if not branch.collidable:
                continue
            shape = obj.collisions[0]
            self.assertIsInstance(shape, CapsuleCollisionShape)
            length = float(np.linalg.norm(np.subtract(branch.end, branch.start)))
            np.testing.assert_allclose(obj.tf[:3, :3] @ [0, 0, length] + obj.pos, branch.end, atol=3e-7)
            for visual in obj.visuals:
                vs = visual.geom.vs @ visual.rotmat.T + visual.pos
                closest = np.column_stack([np.zeros((len(vs), 2)), np.clip(vs[:, 2], 0, length)])
                self.assertTrue(np.all(np.linalg.norm(vs - closest, axis=1) <= shape.radius + 1e-7))
        lower, upper = self.spec.bounds()
        for obj in tree.objects:
            for visual in obj.visuals:
                tf = obj.tf @ visual.loc_tf
                vs = visual.geom.vs @ tf[:3, :3].T + tf[:3, 3]
                self.assertTrue(np.all(vs >= lower - 3e-7))
                self.assertTrue(np.all(vs <= upper + 3e-7))

    def test_fruit_centres_stems_and_tree_placement(self):
        tree = build_tree(self.spec, self.config)
        rotation = wum.rotmat_from_axangle([0, 0, 1], .73)
        translation = np.array([2.0, -.5, .1])
        tree.set_pos_rotmat(translation, rotation)
        for i, (obj, fruit) in enumerate(zip(tree.fruit_objects, self.spec.fruits)):
            self.assertEqual(obj.name, f"orange_{i:03d}")
            self.assertIs(obj, tree.fruit_by_id[fruit.id])
            self.assertEqual(len(obj.collisions), 1)
            self.assertIsInstance(obj.collisions[0], SphereCollisionShape)
            self.assertEqual(obj.collisions[0].radius, fruit.radius)
            np.testing.assert_allclose(obj.pos, rotation @ fruit.position + translation, atol=3e-7)
            stem = obj.visuals[1]
            tip_local = stem.pos + stem.rotmat @ [0, 0, fruit.stem_length / 2]
            tip_world = obj.pos + obj.rotmat @ tip_local
            branch = next(b for b in self.spec.branches if b.id == fruit.parent_branch)
            np.testing.assert_allclose(tip_world, rotation @ branch.point_at(fruit.attachment_t) + translation, atol=3e-7)
        previous = tree.fruit_objects[1].pos
        tree.fruit_objects[0].pos += [0.1, 0, 0]
        np.testing.assert_array_equal(previous, tree.fruit_objects[1].pos)
        # Reapplying a rigid pose is not cumulative.
        tree.set_pos_rotmat(translation, rotation)
        np.testing.assert_allclose(tree.fruit_objects[0].pos,
                                   rotation @ self.spec.fruits[0].position + translation, atol=3e-7)

    def test_scene_publication_and_collision_toggle(self):
        tree = build_tree(self.spec, self.config)
        scene = wss.Scene()
        tree.add_to_scene(scene)
        visible = list(iter_scene_models(scene))
        tree.show_collision(True)
        debug = list(iter_scene_models(scene))
        self.assertEqual(len(debug) - len(visible), sum(b.collidable for b in self.spec.branches) + len(self.spec.fruits))
        models, geometries = describe([(key, model) for key, model, _ in debug], set())
        header, _ = unpack(scene_message("scene_init", models, geometries))
        self.assertEqual(len(header["models"]), len(debug))
        tree.remove_from_scene(scene)
        self.assertEqual(len(list(scene)), 0)


class ReferenceFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.spec = generate_tree(cls.config)

    def test_reference_snapshot_and_seed_stable_fruit_anchors(self):
        self.assertEqual(len(self.spec.fruits), 3)
        lower, upper, left = self.spec.fruits
        self.assertGreater(upper.position[2], left.position[2])
        self.assertGreater(left.position[2], lower.position[2])
        self.assertLess(left.position[0], min(lower.position[0], upper.position[0]))
        another = generate_tree(self.config, seed=42)
        self.assertEqual([f.position for f in another.fruits], [f.position for f in self.spec.fruits])
        self.assertEqual(self.spec.to_json(), generate_tree(self.config).to_json())
        self.assertEqual(self.spec.to_json(), TreeSpec.from_json(self.spec.to_json()).to_json())
        self.assertGreater(len(self.spec.leaves), 900)
        self.assertEqual(len(build_tree(self.spec, self.config).leaf_objects), 3)
        self.assertIsNone(self.spec.metadata['reference']['scale']['verified_scale_reference'])

    def test_profile_density_gaps_and_attachments(self):
        counts = self.spec.metadata['leaf_counts_by_profile']
        self.assertGreater(counts['middle_dense'], counts['right_open'] * 3)
        self.assertGreater(counts['upper_dense'], counts['lower_sparse'] * 3)
        by_id = {b.id: b for b in self.spec.branches}
        authored_count = self.spec.metadata['authored_leaf_count']
        procedural = self.spec.leaves[:-authored_count] if authored_count else self.spec.leaves
        for leaf in procedural:
            self.assertFalse(_intersects_foliage_gap(leaf, self.spec.leaf_shape, self.config['foliage_exclusions']))
            np.testing.assert_allclose(leaf.position, by_id[leaf.parent_branch].point_at(leaf.attachment_t), atol=1e-7)
        # Suppressing a named profile affects its descendants, not the fruit anchors.
        config = deepcopy(self.config)
        config['growth_profiles']['interior']['leaf']['per_branch'] = 0
        reduced = generate_tree(config)
        self.assertNotIn('interior', reduced.metadata['leaf_counts_by_profile'])
        self.assertLess(len(reduced.leaves), len(self.spec.leaves))
        self.assertEqual([f.position for f in reduced.fruits], [f.position for f in self.spec.fruits])

    def test_uniform_scale_preserves_topology_and_camera_framing(self):
        from .demo_static_tree import camera_pose
        config = deepcopy(self.config)
        config['scale_multiplier'] = 1.25
        scaled = generate_tree(config)
        for a, b in zip(self.spec.branches, scaled.branches):
            self.assertEqual((a.id, a.parent_id, a.attachment_t, a.collidable),
                             (b.id, b.parent_id, b.attachment_t, b.collidable))
            np.testing.assert_allclose(np.array(a.end) * 1.25, b.end)
            self.assertAlmostEqual(a.radius_start * 1.25, b.radius_start)
        for a, b in zip(self.spec.leaves, scaled.leaves):
            np.testing.assert_array_equal(a.rotmat, b.rotmat)
            self.assertAlmostEqual(a.length * 1.25, b.length)
        for a, b in zip(self.spec.fruits, scaled.fruits):
            np.testing.assert_allclose(np.array(a.position) * 1.25, b.position)
            self.assertAlmostEqual(a.radius * 1.25, b.radius)
        np.testing.assert_allclose(np.array(self.spec.bounds()) * 1.25, scaled.bounds(), atol=1e-7)
        for view in ('front', 'front-left', 'front-right'):
            np.testing.assert_allclose(np.array(camera_pose(self.config, view)) * 1.25,
                                       camera_pose(config, view))

    def test_debug_modes_do_not_mutate_geometry(self):
        from .demo_static_tree import MODES, scene_for_mode
        tree = build_tree(self.spec, self.config)
        original = tree.spec.to_json()
        counts = {'full': len(tree.objects), 'branches': len(tree.branch_objects),
                  'leaves': 3, 'fruits': 3,
                  'collision': sum(bool(obj.collisions) for obj in tree.objects)}
        for mode in MODES:
            scene = scene_for_mode(tree, self.config, mode)
            self.assertEqual(len(scene.sobjs), counts[mode] + 1)  # ground
            self.assertEqual(tree.spec.to_json(), original)
            self.assertTrue(all(obj.visuals for obj in tree.objects))
        lower, upper = self.spec.bounds()
        for obj in tree.objects:
            for visual in obj.visuals:
                tf = obj.tf @ visual.loc_tf
                vs = visual.geom.vs @ tf[:3, :3].T + tf[:3, 3]
                self.assertTrue(np.all(vs >= lower - 3e-7))
                self.assertTrue(np.all(vs <= upper + 3e-7))

    def test_config_reload_keeps_last_good_scene(self):
        from .demo_static_tree import ConfigReloader
        applied = []
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            path.write_text(json.dumps(self.config), encoding='utf-8')
            reloader = ConfigReloader(path, lambda config, tree: applied.append(tree))
            self.assertFalse(reloader.poll())
            signature = reloader.signature
            for invalid in ('{incomplete edit', '[]', 'null', '{"schema_version": 1, "units": "metre"}'):
                path.write_text(invalid, encoding='utf-8')
                signature += 1_000_000
                os.utime(path, ns=(signature, signature))
                self.assertFalse(reloader.poll())
                self.assertEqual(applied, [])
                self.assertIsNotNone(reloader.last_error)
            config = deepcopy(self.config)
            config['scale_multiplier'] = 1.1
            path.write_text(json.dumps(config), encoding='utf-8')
            os.utime(path, ns=(signature + 1_000_000, signature + 1_000_000))
            self.assertTrue(reloader.poll())
            self.assertIsNone(reloader.last_error)
            self.assertEqual(len(applied), 1)
            self.assertAlmostEqual(applied[0].spec.height(), self.spec.height() * 1.1)


if __name__ == "__main__":
    unittest.main()
