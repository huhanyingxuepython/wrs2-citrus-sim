"""Physical FAFU contact regressions; detached and grasped are separate states."""
import unittest
import numpy as np
from agriculture.config import load_config
from examples.agriculture.robot_citrus_interaction import RobotPlantInteraction
from examples.agriculture.fruit_harvest import DEFAULT_CONFIG, HarvestSequence, run_case, CASES
from examples.agriculture.free_grasp_validation import run_free_grasp


class FruitHarvestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # First validate actual fingers lifting a completely untethered fruit.
        _, _, cls.free_report = run_free_grasp()
        cls.demo = RobotPlantInteraction(load_config(DEFAULT_CONFIG))
        cls.sequence = HarvestSequence(cls.demo)
        cls.reports = {case: run_case(cls.sequence, case) for case in CASES}

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def test_free_fruit_grasp_before_stem(self):
        r = self.free_report
        self.assertTrue(r['success'])
        self.assertEqual((r['tendon_count'], r['equality_count']), (0, 0))
        self.assertLess(r['milestones']['hold']['relative_displacement_m'], .003)

    def test_pull_hold_weak_slip_and_release_to_ground(self):
        for case, report in self.reports.items():
            with self.subTest(case=case):
                self.assertTrue(report['success'])
                self.assertEqual(report['milestones']['close']['grasp_state'], 'bilateral_stable')
                self.assertTrue(all(report['robot_actuator_force_limited']))
                for key in ('orange_001', 'orange_002'):
                    self.assertTrue(report['all_attachments'][key]['attached'])
        hold = self.reports['pull_hold']['milestones']['hold']
        self.assertLess(hold['relative_displacement_m'], .01)
        self.assertEqual(len(self.reports['pull_hold']['attachment']['break_events']), 1)
        self.assertEqual(self.reports['weak_grip']['attachment']['break_events'], [])
        self.assertGreater(self.reports['release']['target_fruit_ground_contacts'], 0)

    def test_reset_and_render_fps_reproduce_rupture(self):
        d = self.demo
        model, fruit = d.env.model, d.plant.fruits['orange_000']
        reference = self.reports['pull_hold']['attachment']['break_events'][0]
        report = run_case(self.sequence, 'pull_hold', frame_dt=1 / 100)
        self.assertTrue(report['success'])
        event = report['attachment']['break_events'][0]
        # Rupture substep is identical. Existing mounted-TCP FK/IK uses float32
        # scene transforms, so accept micronewton-level solver roundoff.
        self.assertEqual(event['time'], reference['time'])
        self.assertEqual(event['overload_time_s'], reference['overload_time_s'])
        self.assertAlmostEqual(event['extension'], reference['extension'], delta=1e-6)
        self.assertAlmostEqual(event['tension'], reference['tension'], delta=1e-4)
        self.assertIs(d.env.model, model)
        self.assertIs(d.plant.fruits['orange_000'], fruit)
        np.testing.assert_allclose(fruit.pos, d.env.data.xpos[self.sequence.monitor.fruit_body], atol=1e-7)
        d.reset()
        for key, c in d.plant.fruit_connections.items():
            h = d.env.connection(c)
            self.assertTrue(h.attached)
            self.assertEqual(h.events, [])
            self.assertEqual(h.overload_time_s, 0)
            self.assertEqual(d.plant.stem_objects[key].alpha, 1)
        np.testing.assert_allclose(fruit.pos, d._initial_fruit['orange_000'], atol=1e-7)
        # Reset cancels commands even while paused, before the first substep.
        self.sequence.start('weak_grip')
        d.paused = True
        d.reset()
        self.assertFalse(self.sequence.running)
        self.assertIsNone(self.sequence.case)
        self.assertEqual(self.sequence.monitor.phase, 'idle')
        np.testing.assert_array_equal(d.env.model.actuator_forcerange[d.gripper_actuators, 1], [20., 20.])

    def test_half_timestep_remains_physical_grasp_and_detachment(self):
        d = self.demo
        dt = d.env.get_timestep()
        try:
            d.env.model.opt.timestep = dt / 2
            report = run_case(self.sequence, 'pull_hold')
            self.assertTrue(report['success'])
            a = self.reports['pull_hold']['attachment']['break_events'][0]
            b = report['attachment']['break_events'][0]
            self.assertLess(abs(a['time'] - b['time']), .08)
            self.assertLess(abs(a['extension'] - b['extension']), .002)
        finally:
            d.env.model.opt.timestep = dt
            d.reset()

    def test_wrist_camera_reads_detached_physical_fruit_without_stepping(self):
        d = self.demo
        run_case(self.sequence, 'pull_hold')
        qpos, qvel, time = d.env.data.qpos.copy(), d.env.data.qvel.copy(), d.env.data.time
        events = d.env.connection(d.plant.fruit_connections['orange_000']).events.copy()
        frame = d.capture_rgbd()
        # Clean geometry depth, independent of the existing camera noise model.
        excluded = list(d.rgbd.excluded_objects) + list(getattr(d.scene, '_force_arrows', ()))
        excluded.append(d.plant.fruits['orange_000'])
        without = d.camera.capture(d.scene, exclude=excluded)
        self.assertGreater(np.count_nonzero(abs(frame.depth_gt - without.depth_gt) > .001), 100)
        np.testing.assert_array_equal(qpos, d.env.data.qpos)
        np.testing.assert_array_equal(qvel, d.env.data.qvel)
        self.assertEqual(d.env.data.time, time)
        self.assertEqual(d.env.connection(d.plant.fruit_connections['orange_000']).events, events)


if __name__ == '__main__':
    unittest.main()
