"""Calibrated flange mounting and live sensor integration; no browser required."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from agriculture.config import CONFIG_DIR, load_config
from examples.agriculture.fafu_d405 import camera_mount_transform, load_handeye
from wrs import wum
from wrs.robots.manipulators.fafu import fafu_with_gripper
from examples.agriculture.robot_citrus_interaction import DEFAULT_CONFIG, RobotPlantInteraction


class FafuCameraMountTests(unittest.TestCase):
    def test_calibrated_flange_transform_without_tcp_or_urdf_offset(self):
        settings = load_config(DEFAULT_CONFIG)['robot_demo']['d405']
        self.assertEqual(settings['depth_noise']['preset'], 'agriculture_foliage')
        self.assertTrue(settings['depth_noise']['enabled'])
        source = json.loads((CONFIG_DIR / settings['handeye_file']).read_text())
        matrix = camera_mount_transform(settings)
        np.testing.assert_array_equal(matrix, source['handeye_mat'])
        np.testing.assert_allclose(matrix[:3, 3], [-.04691661693, .00735122042, .05014299275], atol=1e-11)
        np.testing.assert_allclose(matrix[:3, 2], [.01078914544, -.00894360278, .99990179833], atol=1e-11)
        arm, gripper = fafu_with_gripper(pos=[.2, -.3, .4],
            rotmat=wum.rotmat_from_euler(.3, -.5, .7))
        for q in ([0, 0, 0, 0, 0, 0], [.2, 1.1, 1.5, -.4, .3, .2]):
            arm.fk(q)
            tcp = gripper.tcp('grasp_center')
            # User's TCP -> flange formula; independent of mounting helper.
            flange = wum.tf_from_pos_rotmat(tcp.pos - tcp.rotmat @ [0, 0, .17], tcp.rotmat)
            np.testing.assert_allclose(flange, arm.tcp('flange').tf, atol=1e-7)
            np.testing.assert_allclose(flange @ matrix, arm.tcp('flange').tf @ matrix, atol=1e-7)
        with self.assertRaises(ValueError):
            camera_mount_transform(dict(settings, wrs_parent_link='tool_link'))

    def test_invalid_optical_transform_is_rejected(self):
        settings = load_config(DEFAULT_CONFIG)['robot_demo']['d405']
        source = json.loads((CONFIG_DIR / settings['handeye_file']).read_text())
        invalid = []
        for bad in (np.eye(3), np.full((4, 4), np.nan), np.diag([1, 1, -1, 1]),
                    np.diag([2, 1, 1, 1]), np.diag([1, 1, 1, 2])):
            invalid.append(dict(source, handeye_mat=bad.tolist()))
        invalid.extend([dict(source, frame='T_tcp_cam'), dict(source, convention='cam_from_flange'),
                        dict(source, affine_mat=np.eye(4).tolist())])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'handeye.json'
            for bad in invalid:
                path.write_text(json.dumps(bad), encoding='utf-8')
                with self.subTest(data=bad), self.assertRaises(ValueError):
                    load_handeye(path)


class FafuLiveCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = RobotPlantInteraction()

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def setUp(self):
        self.demo.rgbd.set_noise_enabled(False)
        self.demo.rgbd.set_show_cloud(False)
        self.demo.reset()

    def test_native_mount_follows_actual_arm_and_frame_is_snapshot(self):
        d = self.demo
        self.assertIn(d.camera, tuple(d.scene))
        self.assertFalse(d.camera.collisions)
        self.assertEqual(d.env.model.nu, 8)
        self.assertEqual(d.env.model.nv, d.robot.ndof + d.gripper.ndof + d.plant.mech.ndof)
        state = d.env.data.qpos.copy()
        first = d.capture_rgbd()
        np.testing.assert_array_equal(d.env.data.qpos, state)
        saved_tf, saved_points = first.T_world_camera.copy(), first.points_world.copy()
        d.jog(1, .02)
        d.step(d.motion_duration + .3)
        second = d.capture_rgbd()
        expected = d.rgbd.parent_link.tf @ d.rgbd.mount_tf
        np.testing.assert_allclose(second.T_world_camera, expected, atol=1e-7)
        self.assertGreater(np.linalg.norm(second.T_world_camera[:3, 3] - saved_tf[:3, 3]), .01)
        np.testing.assert_array_equal(first.T_world_camera, saved_tf)
        np.testing.assert_array_equal(first.points_world, saved_points)
        points, colors = second.get_point_cloud(world_frame=False, stride=4)
        world, registered_colors = second.get_point_cloud(world_frame=True, stride=4)
        np.testing.assert_allclose(world, points @ expected[:3, :3].T + expected[:3, 3], atol=1e-7)
        np.testing.assert_array_equal(colors, registered_colors)
        self.assertEqual(second.rgb.shape, (240, 320, 3))
        self.assertEqual(second.depth_m.shape, (240, 320))
        self.assertGreater(len(world), 100)
        self.assertTrue(np.all(second.depth_m[second.valid_mask] >= d.camera.min_depth - 1e-6))
        self.assertTrue(np.all(second.depth_m[second.valid_mask] <= d.camera.max_depth + 1e-6))

    def test_cloud_replacement_empty_depth_and_reset(self):
        d, rig = self.demo, self.demo.rgbd
        count = len(tuple(d.scene))
        reference = d.capture_rgbd()
        rig.set_show_cloud(True)
        self.assertIsNotNone(rig.cloud)
        self.assertFalse(rig.cloud.collisions)
        for _ in range(3):
            previous = rig.cloud
            frame = d.capture_rgbd()
            self.assertEqual(len(tuple(d.scene)), count + 1)
            self.assertNotIn(previous, tuple(d.scene))
            # Display points must not become measured geometry on the next frame.
            np.testing.assert_array_equal(frame.depth_raw, reference.depth_raw)
        rig.last_frame = d.camera.capture([])
        rig.set_show_cloud(True)
        self.assertIsNone(rig.cloud)
        self.assertEqual(len(tuple(d.scene)), count)
        stats = rig.statistics()
        self.assertEqual(stats['valid_pixels'], 0)
        self.assertEqual(stats['sampled_points'], 0)
        self.assertIsNone(stats['depth_range_m'])
        self.assertIsNone(stats['point_bbox_world_m'])
        d.capture_rgbd()
        self.assertIsNotNone(rig.cloud)
        d.reset()
        self.assertIsNone(rig.last_frame)
        self.assertIsNone(rig.cloud)
        self.assertEqual(rig.statistics(), {'frame_count': 0})
        self.assertTrue(rig.show_cloud)
        self.assertEqual(len(tuple(d.scene)), count)

    def test_depth_noise_controls_update_measurement_and_cloud(self):
        d, rig = self.demo, self.demo.rgbd
        reference = d.capture_rgbd()
        qpos = d.env.data.qpos.copy()
        rig.set_show_cloud(True)
        rig.set_noise_sigma(.10)
        rig.set_noise_enabled(True)
        self.assertIsNone(rig.last_frame)
        self.assertIsNone(rig.cloud)
        noisy = d.capture_rgbd()
        self.assertIsNotNone(rig.cloud)
        self.assertTrue(rig.statistics()['depth_noise_enabled'])
        self.assertEqual(rig.statistics()['disparity_std_px'], .10)
        np.testing.assert_array_equal(d.env.data.qpos, qpos)
        np.testing.assert_array_equal(noisy.depth_gt, reference.depth_gt)
        np.testing.assert_array_equal(noisy.rgb, reference.rgb)
        self.assertFalse(np.array_equal(noisy.depth_raw, reference.depth_raw))
        rig.set_noise_enabled(False)
        np.testing.assert_array_equal(d.capture_rgbd().depth_raw, reference.depth_raw)
        self.assertFalse(rig.camera.noise_config.enabled)
        self.assertEqual(rig.last_frame.timings_ms['noise_model_cpu'], 0)
        with self.assertRaises(ValueError):
            rig.set_noise_sigma(-1)

    def test_noise_state_reset_and_zero_sigma_preserves_holes(self):
        d, rig = self.demo, self.demo.rgbd
        d.capture_rgbd()
        renderer = d.camera._renderer
        rig.set_noise_sigma(0)
        rig.set_noise_enabled(True)
        first = d.capture_rgbd()
        support = (first.depth_gt >= d.camera.min_depth) & (first.depth_gt <= d.camera.max_depth)
        self.assertTrue(np.any(support & ~first.valid_mask))
        second = d.capture_rgbd()
        self.assertFalse(np.array_equal(first.depth_raw, second.depth_raw))
        self.assertEqual(d.camera.noise_model.state.frame_index, 2)
        d.reset()
        self.assertEqual(d.camera.noise_model.state.frame_index, 0)
        np.testing.assert_array_equal(d.capture_rgbd().depth_raw, first.depth_raw)
        self.assertIs(d.camera._renderer, renderer)

    def test_batched_contact_debug_is_excluded_from_rgbd(self):
        d = self.demo
        reference = d.capture_rgbd()
        state = d.env.data.qpos.copy()
        try:
            d.plant.show_foliage_proxies()
            frame = d.capture_rgbd()
            np.testing.assert_array_equal(frame.rgb, reference.rgb)
            np.testing.assert_array_equal(frame.depth_raw, reference.depth_raw)
            np.testing.assert_array_equal(d.env.data.qpos, state)
        finally:
            d.plant.show_foliage_proxies(False)


if __name__ == '__main__':
    unittest.main()
