"""Analytic and independent raycast checks for the offscreen optical camera."""

import unittest

import numpy as np

from wrs import wss, wsso, wsrm, wssop, wum
from wrs.scene.geometry_ops import ray_shoot_flat
from wrs.sensor import CameraModel, StereoDepthNoise, RGBLighting, VirtualDepthCamera, VirtualD405


def cpu_depth(objects, model, camera_tf, near, far):
    """Slow test oracle: intersect existing WRS visual triangles per pixel."""
    triangles = []
    for owner in objects:
        for visual in owner.visuals:
            if visual.geom.fs is None or visual.alpha <= 0:
                continue
            tf = np.linalg.inv(camera_tf) @ owner.tf @ visual.loc_tf
            vs = visual.geom.vs @ tf[:3, :3].T + tf[:3, 3]
            triangles.append((vs, visual.geom.fs, visual.geom.fns @ tf[:3, :3].T))
    output = np.zeros((model.height, model.width), np.float32)
    rays = model.deproject(np.ones_like(output))
    for v in range(model.height):
        for u in range(model.width):
            best = np.inf
            for vertices, faces, normals in triangles:
                hit = ray_shoot_flat(np.zeros(3), rays[v, u], vertices, faces, normals)
                if hit is not None:
                    z = hit[0][:, 2]
                    z = z[(z >= near) & (z <= far)]
                    if len(z):
                        best = min(best, z.min())
            if np.isfinite(best):
                output[v, u] = best
    return output


class VirtualDepthCameraTest(unittest.TestCase):
    def camera(self, model=None, **kwargs):
        camera = VirtualDepthCamera(model or CameraModel(33, 25, 31, 27, 13.2, 10.7),
                                    **kwargs)
        self.addCleanup(camera.close)
        return camera

    def test_front_plane_metric_z_raw_and_reprojection(self):
        camera = self.camera()
        plane = wssop.box(pos=(0, 0, .55), xyz_lengths=(4, 4, .1), rgb=(1, .5, 0))
        frame = camera.capture([plane])
        np.testing.assert_allclose(frame.depth_gt, .5, atol=2e-6)
        np.testing.assert_allclose(frame.depth_m, .5, atol=1e-7)
        np.testing.assert_array_equal(frame.depth_raw, 500)
        self.assertEqual(frame.depth_m.dtype, np.float32)
        self.assertEqual(frame.depth_raw.dtype, np.uint16)
        self.assertEqual(frame.valid_mask.dtype, np.bool_)
        np.testing.assert_allclose(frame.points_cam[:, 2], .5)
        k = camera.camera_model
        pixels = frame.points_cam[:, :2] / frame.points_cam[:, 2, None]
        pixels = pixels * [k.fx, k.fy] + [k.cx, k.cy]
        v, u = np.indices(frame.depth_m.shape)
        np.testing.assert_allclose(pixels, np.column_stack([u.ravel(), v.ravel()]), atol=3e-6)
        # Optical +Y points down, and off-axis Z is not Euclidean range.
        self.assertGreater(frame.points_cam[-1, 1], frame.points_cam[0, 1])
        self.assertGreater(np.linalg.norm(frame.points_cam[-1]), .5)

    def test_ray_reference_off_axis_pose_and_local_visual_transform(self):
        model = CameraModel(32, 24, 28, 35, 11.3, 9.8)
        camera = self.camera(model, far=2, max_depth=2)
        camera.set_pos_rotmat((.1, -.2, .1), wum.rotmat_from_axangle((0, 1, 0), .2))
        background = wssop.box(pos=(0, 0, .9), xyz_lengths=(2, 2, .1))
        foreground = wsso.SceneObject()
        foreground.add_visual(wsrm.RenderModel(
            geom=wssop.box(xyz_lengths=(.25, .22, .12)).visuals[0].geom,
            pos=(.09, -.08, .52), rotmat=wum.rotmat_from_axangle((1, 0, 0), .27)))
        foreground.pos = (.04, -.08, .04)
        objects = [background, foreground]
        frame = camera.capture(objects)
        expected = cpu_depth(objects, model, camera.tf, camera.near, camera.far)
        np.testing.assert_allclose(frame.depth_gt, expected, atol=2e-5)

    def test_occlusion_range_and_dynamic_scene(self):
        camera = self.camera(min_depth=.1, max_depth=.8, far=1)
        scene = wss.Scene()
        back = wssop.box(pos=(0, 0, .55), xyz_lengths=(4, 4, .1))
        front = wssop.box(pos=(0, 0, .25), xyz_lengths=(.2, .2, .1))
        scene.add(back)
        scene.add(front)
        self.assertAlmostEqual(camera.capture(scene).depth_m[11, 13], .2, places=5)
        front.pos = (0, 0, .08)
        # Too-close surfaces must still hide the back plane.
        self.assertEqual(camera.capture(scene).depth_m[11, 13], 0)
        self.assertAlmostEqual(camera.capture(scene, exclude=[front]).depth_m[11, 13], .5, places=5)
        scene.remove(front)
        np.testing.assert_allclose(camera.capture(scene).depth_m, .5)
        back.pos = (0, 0, 1.2)
        frame = camera.capture(scene)
        self.assertFalse(frame.valid_mask.any())
        self.assertEqual(frame.points_world.shape, (0, 3))

    def test_mount_extrinsic_and_frame_snapshot(self):
        offset = wum.tf_from_pos_rotmat((.02, .03, .04),
                                       wum.rotmat_from_axangle((0, 1, 0), .15))
        camera = self.camera(T_mount_camera=offset)
        mount = wum.tf_from_pos_rotmat((.1, -.2, .1),
                                      wum.rotmat_from_axangle((0, 0, 1), .25))
        back = wssop.box(pos=(0, 0, .6), xyz_lengths=(4, 4, .1))
        frame = camera.capture([back], T_world_mount=mount)
        expected_tf = mount @ offset
        np.testing.assert_allclose(frame.T_world_camera, expected_tf, atol=1e-7)
        expected_points = frame.points_cam @ expected_tf[:3, :3].T + expected_tf[:3, 3]
        camera.capture([back], T_world_mount=np.eye(4))
        np.testing.assert_allclose(frame.points_world, expected_points, atol=1e-7)

    def test_invalid_depth_encoding_and_noise_repeatability(self):
        model = CameraModel(4, 2, 100, 100, 1.5, .5)
        camera = self.camera(model, min_depth=.07, max_depth=.5, depth_scale=1e-4)
        source = np.array([[0, np.nan, np.inf, -.1], [.069, .2, .6, .30004]])
        depth, raw = camera.process_depth(source)
        np.testing.assert_array_equal(raw, [[0, 0, 0, 0], [0, 2000, 0, 3000]])
        np.testing.assert_allclose(depth, raw * camera.depth_scale, atol=1e-8)
        noise = StereoDepthNoise(disparity_std_px=.3, disparity_step_px=1/32, dropout_rate=.2)
        a = VirtualD405(model, noise=noise, seed=4)
        b = VirtualD405(model, noise=noise, seed=4)
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        np.testing.assert_array_equal(a.process_depth(np.full((2, 4), .3))[1],
                                      b.process_depth(np.full((2, 4), .3))[1])
        with self.assertRaises(ValueError):
            self.camera(max_depth=10, depth_scale=1e-4)

    def test_gpu_inclusive_range_endpoints_with_decimal_depth_scale(self):
        camera = self.camera(min_depth=.1, max_depth=.3, depth_scale=1e-4, far=1)
        for z in (.1, .2, .3):
            with self.subTest(z=z):
                plane = wssop.box(pos=(0, 0, z+.005), xyz_lengths=(4, 4, .01))
                frame = camera.capture([plane])
                self.assertTrue(frame.valid_mask.all())
                np.testing.assert_array_equal(frame.depth_raw, round(z/camera.depth_scale))

    def test_direct_intrinsics_clone_and_resource_recreation(self):
        with VirtualDepthCamera(width=31, height=23, fx=31, fy=35, cx=11, cy=8) as camera:
            camera.capture([])
            camera.close()
            self.assertFalse(camera.capture([]).valid_mask.any())
            clone = camera.clone()
            self.addCleanup(clone.close)
            self.assertIsNone(clone._renderer)
            self.assertIsNone(clone.mounted_by)
            self.assertEqual(clone.camera_model, camera.camera_model)
        self.assertIsInstance(VirtualD405(), VirtualDepthCamera)

    def test_brown_lut_gpu_remap_and_distorted_reprojection(self):
        model = CameraModel(41, 29, 32, 37, 17.8, 12.2, 'brown_conrady',
                            (-.12, .015, .002, -.003, .001))
        camera = self.camera(model)
        back = wssop.box(pos=(0, 0, .65), xyz_lengths=(4, 4, .1))
        front = wssop.box(pos=(.04, -.03, .31), xyz_lengths=(.15, .13, .02))
        frame = camera.capture([back, front])
        self.assertTrue(frame.valid_mask.all())
        self.assertIs(model.ray_lut, model.ray_lut)
        self.assertFalse(model.ray_lut.flags.writeable)
        self.assertGreater(model.render_model.width, model.width)
        # Nearest remap must not blend the .3 m foreground into .6 m background.
        distance = np.minimum(abs(frame.depth_m-.3), abs(frame.depth_m-.6))
        self.assertLess(distance.max(), 1e-6)
        v, u = np.indices(frame.depth_m.shape)
        np.testing.assert_allclose(model.project(frame.points_cam),
                                   np.column_stack((u.ravel(), v.ravel())), atol=1e-5)
        np.testing.assert_allclose(frame.points_cam_image,
                                   model.deproject(frame.depth_m), atol=1e-7)

    def test_distortion_validation(self):
        with self.assertRaises(ValueError):
            CameraModel(10, 10, 8, 8, 4, 4, 'fisheye')
        with self.assertRaises(ValueError):
            CameraModel(10, 10, 8, 8, 4, 4, 'none', (.1, 0, 0, 0, 0))

    def test_fast_reprojection_occlusion_matches_cpu_and_preserves_truth(self):
        model = CameraModel(80, 60, 80, 80, 39.5, 29.5)
        camera = VirtualD405(model)
        self.addCleanup(camera.close)
        back = wssop.box(pos=(0, 0, .41), xyz_lengths=(4, 4, .02))
        front = wssop.box(pos=(0, 0, .21), xyz_lengths=(.06, .06, .02))
        frame = camera.capture([back, front])
        _, expected_raw = camera.process_depth(frame.depth_gt)
        np.testing.assert_array_equal(frame.depth_raw, expected_raw)
        self.assertTrue((frame.depth_gt > 0).all())
        self.assertFalse(frame.valid_mask[:, 0].any())  # right imager's FOV boundary
        self.assertTrue((~frame.valid_mask[:, 10:50]).any())  # foreground occlusion
        self.assertTrue(frame.valid_mask[30, 40])

    def test_fast_gpu_seeded_noise_quantization_and_dropout(self):
        model = CameraModel(64, 48, 70, 70, 31.5, 23.5)
        noise = StereoDepthNoise(disparity_std_px=.2, disparity_step_px=1/32,
                                 dropout_rate=.1, stereo_occlusion=False)
        a = VirtualD405(model, noise=noise, seed=123)
        b = VirtualD405(model, noise=noise, seed=123)
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        plane = wssop.box(pos=(0, 0, .31), xyz_lengths=(4, 4, .02))
        first = a.capture([plane])
        np.testing.assert_array_equal(first.depth_raw, b.capture([plane]).depth_raw)
        second = a.capture([plane])
        np.testing.assert_array_equal(second.depth_raw, b.capture([plane]).depth_raw)
        self.assertFalse(np.array_equal(first.depth_raw, second.depth_raw))
        self.assertGreater(np.std(first.depth_m[first.valid_mask]), .001)
        np.testing.assert_allclose(first.depth_gt, .3, atol=1e-6)
        np.testing.assert_array_equal(first.depth_raw, np.rint(first.depth_m/a.depth_scale).astype(np.uint16))
        dropout = VirtualD405(model, noise=StereoDepthNoise(dropout_rate=1))
        self.addCleanup(dropout.close)
        self.assertEqual(dropout.capture([plane]).points_cam.shape, (0, 3))

    def test_fast_texture_confidence_and_brown_output(self):
        model = CameraModel(41, 29, 32, 37, 17.8, 12.2, 'brown_conrady',
                            (-.12, .015, .002, -.003, .001))
        plane = wssop.box(pos=(0, 0, .31), xyz_lengths=(4, 4, .02), rgb=(.5, .5, .5))
        for noise in (StereoDepthNoise(confidence_threshold=.01, stereo_occlusion=False),
                      StereoDepthNoise(texture_dropout_strength=1, stereo_occlusion=False)):
            with VirtualD405(model, noise=noise) as camera:
                frame = camera.capture([plane])
                self.assertFalse(frame.valid_mask.any())
                self.assertTrue((frame.depth_gt > 0).all())
                np.testing.assert_allclose(frame.confidence, 0, atol=1e-6)

    def test_fast_depth_error_grows_with_distance(self):
        model = CameraModel(64, 48, 80, 80, 31.5, 23.5)
        noise = StereoDepthNoise(disparity_std_px=.03, stereo_occlusion=False)
        deviations = []
        for distance in (.2, .4):
            with VirtualD405(model, noise=noise, seed=123, depth_scale=1e-5) as camera:
                plane = wssop.box(pos=(0, 0, distance+.01), xyz_lengths=(4, 4, .02))
                frame = camera.capture([plane])
                self.assertTrue(frame.valid_mask.all())
                deviations.append(np.std(frame.depth_m-distance))
        # Constant disparity noise produces approximately quadratic depth error.
        self.assertGreater(deviations[1]/deviations[0], 3.5)
        self.assertLess(deviations[1]/deviations[0], 4.5)

    def test_lighting_and_distant_rgb_do_not_change_depth(self):
        camera = VirtualD405(width=80, height=60, far=2, rgb_background=(242/255, 242/255, 240/255))
        self.addCleanup(camera.close)
        sphere = wssop.icosphere(pos=(0, 0, .3), radius=.06, subdivisions=2, rgb=(.2, .6, .9))
        wall = wssop.box(pos=(0, 0, 1.01), xyz_lengths=(4, 4, .02), rgb=(.8, .7, .6))
        unlit = camera.capture([sphere, wall])
        camera.rgb_lighting = RGBLighting()
        lit = camera.capture([sphere, wall])
        np.testing.assert_array_equal(lit.depth_gt, unlit.depth_gt)
        np.testing.assert_array_equal(lit.depth_raw, unlit.depth_raw)
        near = lit.depth_gt < .5
        self.assertEqual(len(np.unique(unlit.rgb[near], axis=0)), 1)
        self.assertGreater(len(np.unique(lit.rgb[near], axis=0)), 12)
        self.assertTrue(np.all(lit.depth_m[~near] == 0))
        self.assertTrue(np.all(lit.rgb[~near] > 0))
        empty = camera.capture([])
        np.testing.assert_array_equal(empty.rgb, np.broadcast_to([242, 242, 240], empty.rgb.shape))
        self.assertFalse(empty.valid_mask.any())
        self.assertEqual(empty.points_world.shape, (0, 3))

    def test_lighting_linear_color_and_reverse_faces(self):
        camera = self.camera(rgb_lighting=RGBLighting(key_direction=(0, 0, -1),
                                                     key_intensity=.4, fill_intensity=0, ambient=.1))
        rgb = np.array([.5, .25, .75])
        # Independent linear-intensity oracle: key .4 + ambient .1 = .5.
        linear = ((rgb + .055) / 1.055) ** 2.4 * .5
        expected = np.rint((1.055 * linear ** (1/2.4) - .055) * 255)
        vertices = np.array([[-1, -1, .3], [1, -1, .3], [1, 1, .3], [-1, 1, .3]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        for winding in (faces, faces[:, ::-1]):
            plane = wssop.mesh(vertices, winding, rgb=rgb)
            frame = camera.capture([plane])
            np.testing.assert_allclose(frame.rgb, np.broadcast_to(expected, frame.rgb.shape), atol=1)
            np.testing.assert_allclose(frame.depth_m, .3, atol=1e-7)

    def test_runtime_noise_toggle_and_clone_preserve_rgb_and_resources(self):
        camera = VirtualD405(width=64, height=48, seed=41, rgb_lighting=RGBLighting())
        self.addCleanup(camera.close)
        plane = wssop.box(pos=(0, 0, .36), xyz_lengths=(4, 4, .02), rgb=(.7, .4, .2))
        baseline = camera.capture([plane])
        renderer = camera._renderer
        camera.noise = StereoDepthNoise(disparity_std_px=.08, disparity_step_px=1/32)
        clone = camera.clone()
        self.addCleanup(clone.close)
        noisy = camera.capture([plane])
        np.testing.assert_array_equal(noisy.depth_raw, clone.capture([plane]).depth_raw)
        np.testing.assert_array_equal(noisy.depth_gt, baseline.depth_gt)
        np.testing.assert_array_equal(noisy.rgb, baseline.rgb)
        self.assertFalse(np.array_equal(noisy.depth_raw, baseline.depth_raw))
        self.assertEqual(clone.rgb_lighting, camera.rgb_lighting)
        self.assertEqual(clone.rgb_background, camera.rgb_background)
        camera.noise = None
        np.testing.assert_array_equal(camera.capture([plane]).depth_raw, baseline.depth_raw)
        self.assertIs(camera._renderer, renderer)
        with self.assertRaises(TypeError):
            camera.noise = {'disparity_std_px': .1}
        self.assertEqual(camera.noise, StereoDepthNoise())
        ideal = self.camera()
        with self.assertRaises(ValueError):
            ideal.noise = StereoDepthNoise(dropout_rate=.1)

    def test_rgb_settings_validation(self):
        for kwargs in (dict(key_direction=(0, 0, 0)), dict(fill_direction=(1, 2)),
                       dict(ambient=-1), dict(key_power=0), dict(fill_intensity=np.nan)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RGBLighting(**kwargs)
        for background in ((1, 0), (1, 0, np.nan), (1.01, 0, 0)):
            with self.subTest(background=background), self.assertRaises(ValueError):
                self.camera(rgb_background=background)
        with self.assertRaises(TypeError):
            self.camera(rgb_lighting=True)

    def test_robot_mount_updates_optical_pose_without_scene_changes(self):
        from wrs.robots.manipulators.xarm.lite6.lite6 import Lite6
        robot, scene = Lite6(), wss.Scene()
        robot.add_to_scene(scene)
        camera = self.camera()
        offset = wum.tf_from_pos_rotmat((.02, 0, .04),
                                       wum.rotmat_from_axangle((1, 0, 0), np.pi))
        link = robot.runtime_lnks[-1]
        robot.mount(camera, link, loc_tf=offset, update=True)
        self.addCleanup(robot.unmount, camera)
        self.assertIn(camera, scene.sobjs)
        np.testing.assert_allclose(camera.tf, link.tf @ offset, atol=1e-7)
        before = camera.tf.copy()
        robot.fk(qs=[.1, .2, -.1, .2, .1, -.1])
        self.assertFalse(np.allclose(before, camera.tf))
        np.testing.assert_allclose(camera.capture(scene, exclude=robot.runtime_lnks).T_world_camera,
                                   link.tf @ offset, atol=1e-7)
        with self.assertRaises(ValueError):
            camera.capture(scene, T_world_mount=np.eye(4))


if __name__ == '__main__':
    unittest.main()
