"""Pure rest-data/geometry tests: only stdlib and NumPy are required."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from .geometry import leaf_batches, leaf_mesh
from .spec import BranchSegment, FruitPlacement, LeafPlacement, LeafShape, TreeSpec


def small_spec():
    return TreeSpec("small", branches=[
        BranchSegment("root", None, (0, 0, 0), (0, 0, 1), .02, .01, 0),
        BranchSegment("twig", "root", (0, 0, .7), (.3, 0, .9), .004, .002, 1,
                      attachment_t=.7)], leaves=[
        LeafPlacement("twig", (.15, 0, .8), ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                      .12, .03, 0, attachment_t=.5)], fruits=[
        FruitPlacement("orange_000", "twig", (.3, 0, .85), .04, .01)])


class TreeSpecTests(unittest.TestCase):
    def test_roundtrip_scaling_and_local_bounds(self):
        spec = small_spec().validate()
        original = spec.to_json()
        self.assertEqual(original, TreeSpec.from_json(original).to_json())
        for factor in (.01, 2):
            with self.subTest(factor=factor):
                scaled = spec.scaled(factor)
                np.testing.assert_allclose(scaled.bounds(), np.asarray(spec.bounds()) * factor)
                self.assertAlmostEqual(scaled.height(), spec.height() * factor)
                for a, b in zip(spec.branches, scaled.branches):
                    self.assertEqual((a.id, a.parent_id, a.order, a.attachment_t),
                                     (b.id, b.parent_id, b.order, b.attachment_t))
                    self.assertEqual(b.radius_start, a.radius_start * factor)
                np.testing.assert_array_equal(spec.leaves[0].rotmat, scaled.leaves[0].rotmat)
        self.assertEqual(original, spec.to_json())
        spec.branches.reverse()  # Serialized segment lists need not be topological.
        spec.validate()
        for factor in (0, -1, float("nan"), float("inf"), True, [1]):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                spec.scaled(factor)

    def test_invalid_branches_are_rejected(self):
        mutations = [(0, "end", (0, 0, 0)), (0, "start", (0, 0, .1)),
                     (0, "end", (0, 0, float("inf"))),
                     (1, "parent_id", "missing"), (1, "parent_id", []),
                     (1, "parent_id", None), (1, "id", "root"),
                     (1, "start", (0, 0, .6)), (1, "attachment_t", 1.1),
                     (1, "order", -1), (1, "order", 1.5), (1, "collidable", 1)]
        for key in ("radius_start", "radius_end"):
            mutations.extend((0, key, value) for value in
                             (0, -.01, float("nan"), float("inf"), True, [.02], "0.02", None))
        for index, key, value in mutations:
            with self.subTest(index=index, key=key, value=value), self.assertRaises(ValueError):
                spec = small_spec()
                setattr(spec.branches[index], key, value)
                spec.validate()
        cycle = TreeSpec("cycle", [
            BranchSegment("root", None, (0, 0, 0), (0, 0, 1), .02, .01, 0),
            BranchSegment("a", "b", (0, 0, 0), (1, 0, 0), .01, .01, 1, attachment_t=0),
            BranchSegment("b", "a", (0, 0, 0), (0, 1, 0), .01, .01, 1, attachment_t=0)])
        with self.assertRaisesRegex(ValueError, "cycle"):
            cycle.validate()

    def test_invalid_leaf_and_fruit_attachments_are_rejected(self):
        for group, key, value in (
                ("leaves", "parent_branch", "missing"), ("leaves", "position", (0, 0, 0)),
                ("leaves", "rotmat", np.diag([1, 1, -1])), ("leaves", "length", 0),
                ("fruits", "id", "twig"), ("fruits", "parent_branch", "missing"),
                ("fruits", "stem_direction", (0, 0, 2)), ("fruits", "position", (0, 0, 0))):
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                spec = small_spec()
                setattr(getattr(spec, group)[0], key, value)
                spec.validate()
        for metadata in ({"handle": object()}, {"value": float("nan")}):
            with self.assertRaisesRegex(ValueError, "metadata"):
                replace(small_spec(), metadata=metadata).validate()

    def test_empty_leaves_and_fruits(self):
        for leaves, fruits in ((False, False), (False, True), (True, False)):
            with self.subTest(leaves=leaves, fruits=fruits):
                spec = small_spec()
                if not leaves:
                    spec.leaves.clear()
                if not fruits:
                    spec.fruits.clear()
                spec.validate()
                stats = TreeSpec.from_json(spec.to_json()).summary()
                self.assertEqual(stats["leaf_count"], int(leaves))
                self.assertEqual(stats["fruit_count"], int(fruits))
                self.assertGreater(spec.height(), 0)
                if not leaves:
                    self.assertEqual(leaf_batches(spec), {})
                    np.testing.assert_array_equal(spec.canopy_bounds(), np.zeros((2, 3)))

    def test_leaf_winding_batch_transforms_and_indices(self):
        for stations in (5, 7, 8):
            shape = replace(LeafShape(), stations=stations)
            vs, fs = leaf_mesh(.12, .05, shape)
            self.assertEqual(len(fs), 8 * stations - 16)
            triangles = vs[fs]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            half = len(fs) // 2
            self.assertTrue(np.all(np.linalg.norm(normals, axis=1) > 1e-8))
            self.assertTrue(np.all(normals[:half, 2] > 0))
            np.testing.assert_allclose(normals[:half], -normals[half:], atol=1e-12)
            self.assertEqual(np.ptp(vs[vs[:, 0] == 0, 1]), 0)
            self.assertEqual(np.ptp(vs[vs[:, 0] == .12, 1]), 0)
            self.assertGreater(np.ptp(vs[:, 1]), .04)
        spec = small_spec()
        spec.leaves += [replace(spec.leaves[0], rotmat=((0, -1, 0), (1, 0, 0), (0, 0, 1))),
                        replace(spec.leaves[0], color_class=2)]
        batches = leaf_batches(spec)
        self.assertEqual(list(batches), [0, 2])
        vs, fs = leaf_mesh(.12, .03, spec.leaf_shape)
        merged_vs, merged_fs = batches[0]
        np.testing.assert_allclose(merged_vs[:len(vs)], vs + spec.leaves[0].position, atol=1e-7)
        np.testing.assert_allclose(merged_vs[len(vs):],
                                   vs @ np.asarray(spec.leaves[1].rotmat).T + spec.leaves[1].position, atol=1e-7)
        np.testing.assert_array_equal(merged_fs[len(fs):], fs + len(vs))

    def test_data_operations_never_import_wrs(self):
        # Block WRS imports, rather than merely checking which modules were loaded.
        code = """
import importlib.abc
import sys
class NoWRS(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'wrs' or fullname.startswith('wrs.'):
            raise AssertionError('pure layer imported ' + fullname)
sys.meta_path.insert(0, NoWRS())
from examples.agriculture.citrus_tree.test_spec import small_spec
from examples.agriculture.citrus_tree.spec import TreeSpec
from examples.agriculture.citrus_tree.geometry import leaf_batches
spec = TreeSpec.from_json(small_spec().to_json()).scaled(.01)
spec.validate()
assert spec.summary()['leaf_count'] == 1
assert spec.height() > 0
assert leaf_batches(spec)
assert not any(n == 'wrs' or n.startswith('wrs.') for n in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
