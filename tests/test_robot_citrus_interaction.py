"""Headless control/contact regressions for the interactive arm example."""
import unittest
import numpy as np
from wrs import wuc
from wrs.robots.manipulators.fafu import FAFURobotArm
from wrs.robots.end_effectors.fafu_gripper import FAFUGripper
from examples.agriculture.robot_citrus_interaction import RobotPlantInteraction, run_smoke


class RobotCitrusInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = RobotPlantInteraction()

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def setUp(self):
        self.demo.paused = False
        self.demo.reset()

    def test_cartesian_jogs_keep_orientation_and_respect_rates(self):
        d = self.demo
        solver = d.robot.get_solver(d.robot.chain('main'))
        for axis in range(3):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    d.reset()
                    qpos, command = d.env.data.qpos.copy(), d.command.copy()
                    start = d.command_position.copy()
                    goal = start.copy()
                    goal[axis] += sign * .01
                    d.jog(axis, sign * .01)
                    np.testing.assert_allclose(d.target_position, goal)
                    np.testing.assert_array_equal(d.env.data.qpos, qpos)
                    np.testing.assert_array_equal(d.command, command)
                    # Check native FK at waypoints AND between them against the
                    # Cartesian line, independent of the stored command positions.
                    for i in range(1, len(d._path_angles)):
                        for fraction in (.5, 1):
                            q = (1 - fraction) * d._path_angles[i - 1] + fraction * d._path_angles[i]
                            expected = (1 - fraction) * d._path_positions[i - 1] + fraction * d._path_positions[i]
                            tcp = solver.fk(q, d.robot.tf) @ d.tcp.loc_tf
                            np.testing.assert_allclose(tcp[:3, 3], expected, atol=2e-4)
                            np.testing.assert_allclose(tcp[:3, :3], d.tool_rotation, atol=2e-3)
                            self.assertTrue(np.all((q >= d.limits[0]) & (q <= d.limits[1])))
                    d.step(.04)
                    self.assertLessEqual(np.max(np.abs(d.command - command)),
                                         np.deg2rad(d.settings['joint_speed_deg_s']) * .04 + 1e-12)
                    self.assertLessEqual(np.linalg.norm(d.command_position - start),
                                         d.settings['cartesian_speed_m_s'] * .04 + 1e-12)
                    self.assertGreater(np.linalg.norm(d.env.data.qpos - qpos), 1e-4)

    def test_invalid_or_unreachable_path_does_not_replace_command(self):
        d = self.demo
        d.jog(1, .02)
        target, angles, times = d.target_position.copy(), d._path_angles.copy(), d._path_times.copy()
        qpos = d.env.data.qpos.copy()
        for point in ([0], [np.nan] * 3, [10, 10, 10], [0, -.59, .39]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                d.set_cartesian_target(point)
            np.testing.assert_array_equal(d.target_position, target)
            np.testing.assert_array_equal(d._path_angles, angles)
            np.testing.assert_array_equal(d._path_times, times)
            np.testing.assert_array_equal(d.env.data.qpos, qpos)

    def test_pause_stop_and_complete_physics_reset(self):
        d = self.demo
        qpos, command, initial_position = d.env.data.qpos.copy(), d.command.copy(), d.target_position.copy()
        d.jog(1, .02)
        d.step(.04)
        time, paused_q = d.env.data.time, d.env.data.qpos.copy()
        d.paused = True
        d.step(.5)
        self.assertEqual(d.env.data.time, time)
        np.testing.assert_array_equal(d.env.data.qpos, paused_q)
        d.stop_motion()
        stopped = d.command.copy()
        np.testing.assert_array_equal(d.env.data.qpos, paused_q)
        d.paused = False
        d.step(.1)
        np.testing.assert_array_equal(d.command, stopped)
        d.reset()
        np.testing.assert_array_equal(d.env.data.qpos, qpos)
        np.testing.assert_array_equal(d.command, command)
        np.testing.assert_array_equal(d.target_position, initial_position)
        self.assertEqual(d.motion_duration, 0)
        self.assertEqual(d.report()['simulated_seconds'], 0)
        self.assertEqual(d.contact_samples, {})

    def test_arm_contacts_both_foliage_and_fruit_then_releases(self):
        d = self.demo
        self.assertIsInstance(d.robot, FAFURobotArm)
        self.assertEqual(d.env.model.nu, 8)
        self.assertEqual(d.env.model.nv, d.robot.ndof + d.gripper.ndof + d.plant.mech.ndof)
        self.assertIsInstance(d.gripper, FAFUGripper)
        self.assertIs(d.robot.end_effector, d.gripper)
        self.assertEqual(d.gripper.ndof, 2)
        for link in d.gripper.runtime_lnks:
            self.assertEqual(link.collision_group, wuc.CollisionGroup.ACTIVE)
        result = run_smoke(d)
        for name, role, fruit in [('foliage', 'FOLIAGE', 'orange_001'),
                                   ('fruit', 'FRUIT', d.settings['fruit_id'])]:
            with self.subTest(name=name):
                contact, released = result[name]['touch'], result[name]['released']
                self.assertGreater(contact['contact_samples'].get(role, 0), 0)
                self.assertGreater(contact['fruit_displacement_m'][fruit], .01)
                self.assertGreater(contact['plant_deflection_rad'], .03)
                self.assertEqual(contact['plant_actuators'], 0)
                self.assertEqual(released['contacts'], {})
                self.assertLess(released['fruit_displacement_m'][fruit], .002)
                self.assertLess(released['plant_deflection_rad'], .01)
        self.assertTrue(np.isfinite(d.env.data.qpos).all())
        self.assertTrue(np.isfinite(d.env.data.qvel).all())
        self.assertEqual(sum(int(w.number) for w in d.env.data.warning), 0)
        for link in d.gripper.runtime_lnks:
            body = d.env.model.body(d.env.sync.rutl2bdy[link].name).id
            np.testing.assert_allclose(link.pos, d.env.data.xpos[body], atol=3e-7)
            np.testing.assert_allclose(link.rotmat, d.env.data.xmat[body].reshape(3, 3), atol=3e-7)


if __name__ == '__main__':
    unittest.main()
