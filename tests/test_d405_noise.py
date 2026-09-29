"""Raw stereo failure morphology, temporal state, calibration and camera integration."""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi

from wrs import wssop
from wrs.sensor import (CameraModel, D405NoiseConfig, D405NoiseModel, VirtualD405,
                        CorruptionReason, StereoDepthNoise)
from wrs.sensor.d405_noise_analysis import analyze_depth_sequence, fit_noise_config, load_depth_recording


class D405NoiseTest(unittest.TestCase):
    def setUp(self):
        self.model = CameraModel(120, 80, 120, 120, 59.5, 39.5)
        self.plane = np.full((80, 120), .3, np.float32)

    def config(self, **kwargs):
        return D405NoiseConfig.preset('ideal', disparity_sigma_px=0, **kwargs)

    def test_disparity_precision_quadratic_bias_and_calibration(self):
        c = D405NoiseConfig.preset('ideal', disparity_sigma_px=.05, spatial_correlation_px=0,
                                 max_depth_m=.6, depth_unit_m=.00001)
        sigma = []
        for distance in (.2, .4):
            result = D405NoiseModel(c, seed=3).apply(np.full(self.plane.shape, distance), self.model, .018)
            sigma.append(np.std(result.depth-distance))
        self.assertAlmostEqual(sigma[1]/sigma[0], 4, delta=.15)
        for fx, baseline in ((120, .018), (240, .03)):
            model = replace(self.model, fx=fx)
            bias = replace(c, disparity_sigma_px=0, disparity_bias_px=.25)
            measured = D405NoiseModel(bias, seed=2).apply(self.plane, model, baseline).depth
            expected = fx*baseline/(fx*baseline/.3+.25)
            np.testing.assert_allclose(measured, expected, atol=1e-5)

    def test_correlated_holes_and_markov_persistence(self):
        engine = D405NoiseModel(self.config(base_invalid_prob=.18), seed=4)
        frames = [engine.apply(self.plane, self.model, .018).depth == 0 for _ in range(16)]
        invalid = np.stack(frames)
        neighbor = np.mean(invalid[:, :, 1:][invalid[:, :, :-1]])
        self.assertGreater(neighbor, .45)
        retention = np.mean(invalid[1:][invalid[:-1]])
        self.assertGreater(retention, .65)
        self.assertGreater(np.mean(invalid[1:] != invalid[:-1]), .005)
        labeled, _ = ndi.label(invalid[0])
        areas = np.bincount(labeled.ravel())[1:]
        self.assertGreater(np.median(areas), 2)

    def test_horizontal_donors_and_all_mismatch_modes(self):
        rows = np.arange(80, dtype=np.float32)[:, None]*.0001
        z = np.broadcast_to(.3+rows, self.plane.shape).copy()
        z[:, 60:] += .2
        for name, reason in (('background_takeover_weight', CorruptionReason.BACKGROUND_BLEED),
                             ('foreground_bleed_weight', CorruptionReason.FOREGROUND_BLEED),
                             ('wrong_disparity_weight', CorruptionReason.WRONG_DISPARITY),
                             ('reject_to_invalid_weight', CorruptionReason.HOLE)):
            weights = dict(background_takeover_weight=0, foreground_bleed_weight=0,
                           wrong_disparity_weight=0, reject_to_invalid_weight=0)
            weights[name] = 1
            c = self.config(edge_outlier_gain=1, horizontal_edge_weight=1,
                            invalid_persistence=0, outlier_persistence=0, **weights)
            result = D405NoiseModel(c, seed=11).apply(z, self.model, .018, debug=True)
            selected = result.debug['corruption_reason'] == reason
            self.assertTrue(selected.any(), name)
            self.assertFalse(result.debug['outlier_mask'][:, :45].any())
            if reason == CorruptionReason.HOLE:
                self.assertTrue((result.depth[selected] == 0).all())
            else:
                row = np.nonzero(selected)[0]
                donor = result.depth[selected]
                error = np.minimum(abs(donor-z[row, 0]), abs(donor-z[row, -1]))
                self.assertLess(error.max(), 1e-5)
                if reason == CorruptionReason.BACKGROUND_BLEED:
                    self.assertTrue((donor > z[selected]).all())
                elif reason == CorruptionReason.FOREGROUND_BLEED:
                    self.assertTrue((donor < z[selected]).all())

    def test_thin_structures_fail_more_than_interiors(self):
        z = np.full_like(self.plane, .5)
        z[:, :35] = .3
        z[:, 70:73] = .3
        engine = D405NoiseModel(self.config(thin_structure_invalid_gain=.65), seed=4)
        result = engine.apply(z, self.model, .018, debug=True)
        self.assertTrue(result.debug['thin_structure_mask'][:, 70:73].all())
        self.assertFalse(result.debug['thin_structure_mask'][:, 8:20].any())
        self.assertGreater(np.mean(result.depth[:, 70:73] == 0), .3)
        self.assertEqual(np.mean(result.depth[:, 8:20] == 0), 0)

    def test_rgb_optional_texture_normals_semantics_and_motion(self):
        c = self.config(low_texture_invalid_gain=.4, texture_gain=1,
                        grazing_invalid_gain=.4, motion_invalid_gain=.4,
                        semantic_multipliers={'FOLIAGE': 2})
        flat = np.full((*self.plane.shape, 3), 128, np.uint8)
        textured = np.repeat(((np.indices(self.plane.shape).sum(0) % 2)*255).astype(np.uint8)[..., None], 3, -1)
        no_rgb = D405NoiseModel(c, seed=1).apply(self.plane, self.model, .018)
        self.assertTrue((no_rgb.depth > 0).all())
        low = D405NoiseModel(c, seed=1).apply(self.plane, self.model, .018, rgb=flat)
        high = D405NoiseModel(c, seed=1).apply(self.plane, self.model, .018, rgb=textured)
        self.assertGreater(np.mean(low.depth == 0), np.mean(high.depth == 0)+.2)
        ids = np.full(self.plane.shape, 'FOLIAGE')
        semantic = D405NoiseModel(c, seed=1).apply(self.plane, self.model, .018, rgb=flat, semantic_ids=ids)
        self.assertGreater(np.mean(semantic.depth == 0), np.mean(low.depth == 0)+.2)
        normal = np.zeros((*self.plane.shape, 3), np.float32)
        normal[..., 0] = 1
        grazing = D405NoiseModel(c, seed=1).apply(self.plane, self.model, .018, normals=normal)
        self.assertGreater(np.mean(grazing.depth == 0), .05)
        z = self.plane.copy()
        z[:, 60:] = .5
        stationary = D405NoiseModel(c, seed=1).apply(z, self.model, .018, motion_px=0)
        moving = D405NoiseModel(c, seed=1).apply(z, self.model, .018, motion_px=4)
        self.assertGreater(np.mean(moving.depth == 0), np.mean(stationary.depth == 0))

    def test_state_roundtrip_reset_clone_and_no_global_rng(self):
        c = D405NoiseConfig()
        a, b = D405NoiseModel(c, seed=10), D405NoiseModel(c, seed=10)
        before = np.random.get_state()
        for _ in range(3):
            np.testing.assert_array_equal(a.apply(self.plane, self.model, .018).depth_raw,
                                           b.apply(self.plane, self.model, .018).depth_raw)
        snapshot = a.get_state()
        expected = a.apply(self.plane, self.model, .018).depth_raw
        a.set_state(snapshot)
        np.testing.assert_array_equal(a.apply(self.plane, self.model, .018).depth_raw, expected)
        a.reset()
        fresh = D405NoiseModel(c, seed=10).apply(self.plane, self.model, .018).depth_raw
        np.testing.assert_array_equal(a.apply(self.plane, self.model, .018).depth_raw, fresh)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_ar1_and_scene_change_drops_stale_state(self):
        c = D405NoiseConfig.preset('ideal', disparity_sigma_px=.08, temporal_rho=.85)
        engine = D405NoiseModel(c, seed=5)
        first = engine.apply(self.plane, self.model, .018, debug=True)
        second = engine.apply(self.plane, self.model, .018, debug=True)
        corr = np.corrcoef(first.debug['normal_noise_map'].ravel(), second.debug['normal_noise_map'].ravel())[0, 1]
        self.assertAlmostEqual(corr, .85, delta=.035)
        changed = engine.apply(self.plane+.2, self.model, .018, debug=True)
        corr = np.corrcoef(second.debug['normal_noise_map'].ravel(), changed.debug['normal_noise_map'].ravel())[0, 1]
        self.assertLess(abs(corr), .1)
        empty = engine.apply(np.zeros_like(self.plane), self.model, .018, debug=True)
        self.assertFalse(empty.depth.any())
        self.assertFalse(empty.debug['outlier_mask'].any())

    def test_motion_estimation_has_no_float32_stationary_drift(self):
        engine = D405NoiseModel(self.config(), seed=4)
        tf = np.eye(4, dtype=np.float32)
        angle = .117
        tf[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        engine.apply(self.plane, self.model, .018, camera_tf=tf)
        still = engine.apply(self.plane, self.model, .018, camera_tf=tf, debug=True)
        self.assertEqual(still.debug['motion_px'], 0)
        tf[0, 3] += .003
        moved = engine.apply(self.plane, self.model, .018, camera_tf=tf, debug=True)
        self.assertAlmostEqual(moved.debug['motion_px'], self.model.fx*.003/.3, places=5)

    def test_range_soft_degradation_and_final_units(self):
        z = np.tile(np.array([0, .06, .07, .3, .6, 1.2, 2.1, np.nan], np.float32), (80, 15))
        config = self.config(enabled=False)
        result = D405NoiseModel(config, seed=1).apply(z, self.model, .018, debug=True)
        self.assertTrue((result.depth[:, 4::8] > .5).all())
        np.testing.assert_allclose(result.depth, result.depth_raw*config.depth_unit_m, atol=1e-7)
        for col in (0, 1, 6, 7):
            self.assertFalse(result.depth[:, col::8].any())
        engine = D405NoiseModel(self.config(far_range_invalid_gain=.8), seed=2)
        near = engine.apply(self.plane, self.model, .018).depth
        engine.reset()
        far = engine.apply(self.plane+1, self.model, .018).depth
        self.assertGreater(np.mean(far == 0), np.mean(near == 0)+.5)

    def test_camera_integration_profile_clone_and_world_cloud(self):
        obj = wssop.box(pos=(0, 0, .35), xyz_lengths=(1, 1, .1))
        camera = VirtualD405(self.model, noise_config='default', seed=7, pos=(.01, 0, 0))
        self.addCleanup(camera.close)
        first = camera.capture([obj], debug=True)
        self.assertIsNotNone(first.noise_debug)
        np.testing.assert_allclose(first.points_cam_image, self.model.deproject(first.depth_m), atol=1e-7)
        valid = first.valid_mask
        np.testing.assert_allclose(first.points_world_image[valid], first.points_cam_image[valid]+(.01, 0, 0), atol=1e-7)
        self.assertFalse(first.points_world_image[~valid].any())
        clone = camera.clone()
        self.addCleanup(clone.close)
        next_frame = camera.capture([obj])
        cloned = clone.capture([obj])
        np.testing.assert_array_equal(next_frame.depth_raw, cloned.depth_raw)
        self.assertIsNone(next_frame.noise_debug)
        self.assertIsNot(camera.noise_model.state, clone.noise_model.state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'profile.json'
            camera.noise_config.save_profile(path, calibration={'camera_model': asdict(self.model), 'baseline_m': .02})
            loaded = VirtualD405(noise_profile=path)
            self.addCleanup(loaded.close)
            self.assertEqual(loaded.camera_model, self.model)
            self.assertEqual(loaded.baseline_m, .02)
            self.assertEqual(loaded.noise_config, camera.noise_config)
        with self.assertRaises(ValueError):
            camera.noise = StereoDepthNoise()

    def test_profiles_validation_and_debug_is_optional(self):
        for name in ('ideal', 'default', 'agriculture_foliage', 'harsh'):
            config = D405NoiseConfig.preset(name)
            self.assertEqual(D405NoiseConfig.from_dict(config.to_dict()), config)
        for settings in ({'hole_correlation_px': 0}, {'invalid_persistence': 2},
                         {'depth_unit_m': 1e-7}, {'texture_window_px': 4},
                         {'depth_edge_threshold_m': 0}, {'disparity_sigma_px': -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                D405NoiseConfig(**settings)
        result = D405NoiseModel(seed=1).apply(self.plane, self.model, .018)
        self.assertIsNone(result.debug)

    def test_disabled_capture_reuses_ideal_gpu_outputs(self):
        obj = wssop.box(pos=(0, 0, .35), xyz_lengths=(1, 1, .1))
        config = replace(D405NoiseConfig(), enabled=False)
        with VirtualD405(self.model, noise_config=config, seed=2) as camera:
            with VirtualD405(self.model, mode='ideal', max_depth=config.max_depth_m,
                             pos=camera.pos, rotmat=camera.rotmat) as ideal:
                actual, expected = camera.capture([obj]), ideal.capture([obj])
            np.testing.assert_array_equal(actual.depth_raw, expected.depth_raw)
            np.testing.assert_array_equal(actual.points_world_image, expected.points_world_image)
            self.assertEqual(actual.timings_ms['noise_model_cpu'], 0)
            self.assertEqual(actual.timings_ms['noise_pointcloud_cpu'], 0)
            self.assertEqual(camera.noise_model.state.frame_index, 0)
            diagnostics = camera.capture([obj], debug=True)
            self.assertIsNotNone(diagnostics.noise_debug)
            self.assertFalse(diagnostics.noise_debug['outlier_mask'].any())

    def test_statistics_known_board_fit_and_integer_recording_units(self):
        c = D405NoiseConfig.preset('ideal', disparity_sigma_px=.08, disparity_bias_px=.1,
                                 spatial_correlation_px=0, temporal_rho=.65)
        model = replace(self.model, fx=600, fy=600)
        engine = D405NoiseModel(c, seed=1)
        sequence = np.stack([engine.apply(self.plane, model, .018).depth for _ in range(40)])
        fitted, stats = fit_noise_config(sequence, fx=model.fx, baseline_m=.018, known_depth_m=.3)
        self.assertAlmostEqual(fitted.disparity_bias_px, .1, delta=.012)
        self.assertAlmostEqual(fitted.disparity_sigma_px, .08, delta=.012)
        self.assertAlmostEqual(fitted.temporal_rho, .65, delta=.04)
        self.assertEqual(stats['fill_rate'], 1)
        holes = np.full((3, 4, 4), .3, np.float32)
        holes[:, 0:2, 0:2] = 0
        metrics = analyze_depth_sequence(holes, known_depth_m=.3)
        self.assertEqual(metrics['fill_rate'], .75)
        self.assertEqual(metrics['invalid_persistence'], 1)
        self.assertEqual(metrics['hole_components']['quantiles_px']['p50'], 4)
        all_missing = analyze_depth_sequence(np.zeros_like(holes))
        self.assertIsNone(all_missing['valid_pixel_depth_MAE_m'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'raw.npy'
            np.save(path, np.full((2, 4, 4), 3000, np.uint16))
            with self.assertRaises(ValueError):
                load_depth_recording(path)
            depth, rgb = load_depth_recording(path, depth_unit_m=.0001)
            np.testing.assert_allclose(depth, .3, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
