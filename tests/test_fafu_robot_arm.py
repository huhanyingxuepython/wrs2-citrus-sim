"""Regression checks against the legacy Panthera Python frame convention."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from wrs import wss, wssop, wuc
from wrs.robots.base.kine.numik import NumIKSolver
from wrs.robots.manipulators.fafu import FAFURobotArm, fafu_with_gripper
from wrs.robots.end_effectors.fafu_gripper import FAFUGripper


def legacy_flange(qs, base_tf):
    """Independent FK oracle using scipy and the original X-axis wrist.

    The old WRS wrist frame equals the URDF wrist frame times Ry(pi/2).
    Keep the Python model's 63.4 mm base offset (URDF has 58.4 mm).
    """
    origins = ((0, 0, .0634), (.018199, 0, .053), (-.26, 0, 0),
               (.23, 0, .06), (.07, 0, .036319), (.02345, 0, -.039))
    axes = np.array(((0, 0, 1), (0, 1, 0), (0, -1, 0),
                     (0, -1, 0), (0, 0, -1), (1, 0, 0)))
    result = base_tf.copy()
    for pos, axis, q in zip(origins, axes, qs):
        step = np.eye(4)
        step[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
        step[:3, 3] = pos
        result = result @ step
    result[:3, :3] = result[:3, :3] @ Rotation.from_euler('y', np.pi / 2).as_matrix()
    return result


class FAFURobotTests(unittest.TestCase):
    def test_legacy_fk_and_tcp_at_zero_random_and_moved_base(self):
        arm, gripper = fafu_with_gripper()
        self.assertEqual(arm.ndof, 6)
        self.assertFalse(arm.is_floating)
        np.testing.assert_allclose(arm.tcp('flange').pos, [.081649, 0, .173719], atol=1e-7)
        np.testing.assert_allclose(gripper.tcp('grasp_center').pos, [.251649, 0, .173719], atol=1e-7)
        np.testing.assert_array_equal(arm.home_qs, np.zeros(6))
        lo, hi = arm.chain_joint_limits('main')
        np.testing.assert_allclose(lo, [-2.4, 0, 0, -1.6, -1.7, -2.5])
        np.testing.assert_allclose(hi, [2.4, 3.2, 4, 1.6, 1.7, 2.5])
        arm.set_pos_rotmat(pos=[.3, -.4, .2],
                           rotmat=Rotation.from_euler('xyz', [.3, -.5, .7]).as_matrix())
        for qs in np.random.default_rng(14).uniform(lo, hi, size=(20, 6)):
            arm.fk(qs=qs)
            expected = legacy_flange(qs, arm.tf)
            np.testing.assert_allclose(arm.tcp('flange').tf, expected, atol=4e-7)
            np.testing.assert_allclose(gripper.tf, expected, atol=4e-7)
            np.testing.assert_allclose(gripper.tcp('grasp_center').pos,
                                       expected[:3, 3] + expected[:3, :3] @ [0, 0, .17],
                                       atol=4e-7)

    def test_gripper_matches_legacy_serial_fingers_and_rejects_bad_widths(self):
        gripper = FAFUGripper(pos=[.2, .3, -.1],
                              rotmat=Rotation.from_euler('xyz', [.2, .6, -.4]).as_matrix())
        self.assertEqual(gripper.ndof, 1)
        for width in (0, .02, .05, .085):
            gripper.set_opening(width)
            relative = np.linalg.inv(gripper.tf) @ gripper.gl_lnk_tfarr
            # Legacy serial commands [w/2, w] yield these two world offsets.
            np.testing.assert_allclose(relative[1, :3, 3], [0, width / 2, .005], atol=1e-7)
            np.testing.assert_allclose(relative[2, :3, 3], [0, -width / 2, .005], atol=1e-7)
            np.testing.assert_allclose(gripper.tcp('grasp_center').loc_tf[:3, 3], [0, 0, .17])
        for width in (-.01, .086, np.nan, np.inf):
            with self.assertRaises(ValueError):
                gripper.set_opening(width)

    def test_mount_clone_and_scene_membership(self):
        arm, gripper = fafu_with_gripper(jaw_width=.05)
        scene = wss.Scene()
        arm.add_to_scene(scene)
        clone = arm.clone()
        self.assertIs(arm.end_effector, gripper)
        self.assertIsNot(clone.end_effector, gripper)
        self.assertIs(clone.end_effector.mounted_by, clone)
        self.assertIs(clone.structure, arm.structure)
        clone.fk([.2, 1.1, 1.5, -.4, .3, .2])
        clone.end_effector.close()
        np.testing.assert_array_equal(arm.qs, np.zeros(6))
        np.testing.assert_allclose(gripper.qs, [.025, .025])
        np.testing.assert_allclose(clone.end_effector.tf, clone.tcp('flange').tf)
        self.assertIs(gripper._scene, scene)
        self.assertIsNone(clone._scene)
        arm.remove_from_scene(scene)
        self.assertIsNone(gripper._scene)

    def test_fixed_gripper_preserves_geometry_tcp_and_default_mimic(self):
        normal = FAFUGripper()
        original_structure = normal.structure
        for width in (0, .05, .085):
            normal.set_opening(width)
            fixed = FAFUGripper(fixed_opening=width)
            self.assertEqual(fixed.ndof, 0)
            self.assertIsNot(fixed.structure, original_structure)
            np.testing.assert_allclose(fixed.gl_lnk_tfarr, normal.gl_lnk_tfarr, atol=1e-7)
            np.testing.assert_array_equal(fixed.tcp('grasp_center').tf, normal.tcp('grasp_center').tf)
            for link, expected in zip(fixed.runtime_lnks, normal.runtime_lnks):
                self.assertEqual(len(link.collisions), len(expected.collisions))
                np.testing.assert_array_equal(link.visuals[0].geom.vs, expected.visuals[0].geom.vs)
            cloned = fixed.clone()
            cloned.set_opening(width)
            np.testing.assert_array_equal(cloned.gl_lnk_tfarr, fixed.gl_lnk_tfarr)
            with self.assertRaises(ValueError):
                fixed.set_opening(.02 if width != .02 else .03)
        self.assertEqual(FAFUGripper().ndof, 1)
        self.assertIsNotNone(original_structure.jnts[1].mmc)
        for width in (-.01, .086, np.nan, np.inf):
            with self.assertRaises(ValueError):
                FAFUGripper(fixed_opening=width)

    def test_fk_ik_roundtrip_for_flange_and_mounted_tcp(self):
        # Exercise the public solver hook without building the default large
        # SELIK database; use perturbed seeds, not the known exact solution.
        arm, gripper = fafu_with_gripper(
            pos=[.3, -.2, .1],
            rotmat=Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix(),
            solver=NumIKSolver)
        for qs in ([.2, 1.1, 1.5, -.4, .3, .2], [-.4, 2, 2.4, .5, -.6, .7]):
            arm.fk(qs)
            for tcp in (arm.tcp('flange'), gripper.tcp('grasp_center')):
                target = tcp.tf.copy()
                solutions = arm.ik(target[:3, 3], target[:3, :3], tcp=tcp,
                                   qs_active_init=np.asarray(qs) + .025)
                self.assertTrue(solutions)
                arm.fk(solutions[0])
                np.testing.assert_allclose(tcp.pos, target[:3, 3], atol=2e-4)
                np.testing.assert_allclose(tcp.rotmat, target[:3, :3], atol=2e-3)
        self.assertEqual(arm.ik([10, 10, 10], np.eye(3)), [])

    def test_bare_arm_and_assembled_collision_worlds(self):
        bare = FAFURobotArm()
        collider = bare.build_collider()
        self.assertFalse(collider.is_collided(bare.qs))
        arm, gripper = fafu_with_gripper(jaw_width=.05)
        arm.fk([.2, 1.1, 1.5, -.4, .3, .2])
        collider = arm.build_collider()
        self.assertFalse(collider.is_collided(arm.qs))
        # Enclose the wrist in an external obstacle; self-collision exclusions
        # must not exempt the obstacle, even at the initial configuration.
        obstacle = wssop.box(xyz_lengths=(.3, .3, .3), pos=gripper.pos,
                            collision_type=wuc.CollisionType.AABB,
                            is_floating=False)
        blocked = arm.build_collider(fixtures=[obstacle])
        self.assertTrue(blocked.is_collided(arm.qs))
        # Topological exclusions must leave non-adjacent arm pairs available.
        self.assertNotIn((0, 4), arm.structure.compiled.collision_ignores_idx)


if __name__ == '__main__':
    unittest.main()
