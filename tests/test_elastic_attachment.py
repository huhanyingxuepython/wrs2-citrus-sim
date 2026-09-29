"""Native two-site spring/rupture tests; no renderer, no grip assistance."""
import unittest
import xml.etree.ElementTree as ET
from copy import deepcopy
import mujoco
import numpy as np
from wrs import wss, wum, wuc, wssop
from wrs.physics.mj_env import MJEnv
from wrs.physics.mj_runtime import MJRuntime
from examples.agriculture.attachment_validation import fixed_fixture, branch_fixture, body_id, run_fixed, run_branch


class ElasticAttachmentTests(unittest.TestCase):
    def test_native_tendon_schema_gravity_and_linear_load(self):
        e, fruit, c = fixed_fixture()
        root = ET.fromstring(e.xml_string)
        tendon = root.find('tendon/spatial')
        self.assertEqual(len(tendon.findall('site')), 2)
        self.assertIsNone(root.find('equality'))
        self.assertIsNone(root.find('actuator'))
        self.assertEqual(tendon.get('limited'), 'false')
        self.assertEqual(tendon.get('armature'), '0')
        self.assertEqual(tendon.get('frictionloss'), '0')
        self.assertEqual(e.model.nv, 6)
        self.assertTrue(np.all(e.model.tendon_stiffnesspoly == 0))
        self.assertTrue(np.all(e.model.tendon_dampingpoly == 0))
        e.runtime.step(2000)
        self.assertAlmostEqual(e.connection(c).extension, .2 * 9.81 / 200, places=7)
        self.assertAlmostEqual(e.connection(c).tension, .2 * 9.81, places=6)
        e.data.xfrc_applied[body_id(e, fruit), 2] = -1.
        e.runtime.step(2000)
        self.assertAlmostEqual(e.connection(c).extension, (.2 * 9.81 + 1.) / 200, places=7)
        self.assertTrue(e.connection(c).attached)

    def test_break_at_substep_preserves_state_and_free_fall(self):
        e, fruit, c = fixed_fixture()
        h = e.connection(c)
        body = body_id(e, fruit)
        e.runtime.step(1000)
        e.data.xfrc_applied[body, 2] = -7.
        # An identical native step is the continuity oracle at the actual event.
        oracle = MJRuntime(e.xml_string)
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        while h.attached:
            snapshot = e.snapshot()
            mujoco.mj_setState(oracle.model, oracle.data, snapshot['state'], kind)
            oracle.forward()
            oracle.step()
            e.runtime.step()
        np.testing.assert_array_equal(e.data.qpos, oracle.data.qpos)
        np.testing.assert_array_equal(e.data.qvel, oracle.data.qvel)
        self.assertEqual(len(h.events), 1)
        self.assertEqual((e.model.tendon_stiffness[h.tendon_id], e.model.tendon_damping[h.tendon_id]), (0, 0))
        self.assertAlmostEqual(h.events[0]['overload_time_s'], .04)
        e.data.xfrc_applied[:] = 0
        v = e.data.qvel[2]
        e.runtime.step(50)
        self.assertAlmostEqual(e.data.qvel[2] - v, -9.81 * .1, places=9)
        self.assertEqual(len(h.events), 1)
        self.assertEqual(h.tension, 0.)

    def test_snapshot_reset_debounce_and_forward_does_not_advance(self):
        e, fruit, c = fixed_fixture()
        h = e.connection(c)
        e.runtime.step(1000)
        e.data.xfrc_applied[body_id(e, fruit), 2] = -7.
        while h.overload_time_s < .01:
            e.runtime.step()
        snapshot = e.snapshot()
        elapsed, time = h.overload_time_s, e.data.time
        for _ in range(30):
            e.runtime.forward()
            e.is_collided()
            e.runtime.exit_cd()
            e.sync_scene()
        self.assertEqual(h.overload_time_s, elapsed)
        self.assertEqual(e.data.time, time)
        e.runtime.step(100)
        expected = e.data.qpos.copy(), deepcopy(h.events)
        e.restore(snapshot)
        self.assertTrue(h.attached)
        self.assertEqual(h.events, [])
        self.assertEqual(e.model.tendon_stiffness[h.tendon_id], 200.)
        e.runtime.step(100)
        np.testing.assert_array_equal(e.data.qpos, expected[0])
        self.assertEqual(h.events, expected[1])
        # A below-threshold substep clears consecutive overload time.
        e.restore(snapshot)
        e.data.qpos[2] = .86
        e.data.qvel[:] = 0
        e.data.xfrc_applied[:] = 0
        e.runtime.step()
        self.assertEqual(h.overload_time_s, 0.)
        e.reset()
        self.assertTrue(h.attached)
        self.assertEqual(h.events, [])
        self.assertEqual(h.overload_time_s, 0.)
        self.assertEqual(e.data.time, 0.)
        self.assertFalse(np.any(e.data.xfrc_applied))

    def test_models_are_isolated_and_env_step_covers_policy(self):
        a, f, c = fixed_fixture()
        # Same scene, same connection spec, independent model and runtime state.
        b = MJEnv(a.scene)
        self.assertIsNot(a.model, b.model)
        a.data.xfrc_applied[body_id(a, f), 2] = -10
        a.step(.2)
        self.assertFalse(a.connection(c).attached)
        self.assertTrue(b.connection(c).attached)
        self.assertEqual(b.model.tendon_stiffness[0], 200.)
        with self.assertRaises(ValueError):
            b.restore(a.snapshot())

    def test_half_timestep_and_render_grouping(self):
        a, _, ra = run_fixed(.002)
        b, _, rb = run_fixed(.001)
        ea, eb = ra['final']['break_events'][0], rb['final']['break_events'][0]
        self.assertLess(abs(ea['time'] - eb['time']), .004)
        self.assertLess(abs(ea['extension'] - eb['extension']), .002)
        # Publish poses at 20 / 100 Hz; every physical substep gets identical load.
        records = []
        for publish_every in (25, 5):
            e, f, c = fixed_fixture()
            for i in range(1500):
                e.data.xfrc_applied[body_id(e, f), 2] = -7. if i >= 1000 else 0.
                e.runtime.step()
                if i % publish_every == 0:
                    e.sync_scene()
            records.append((e.data.qpos.copy(), e.connection(c).events[0]))
        np.testing.assert_array_equal(records[0][0], records[1][0])
        for key in ('time', 'extension', 'tension', 'overload_time_s'):
            self.assertEqual(records[0][1][key], records[1][1][key])

    def test_branch_reaction_rebound_and_selective_break(self):
        e, rows, reports = run_branch()
        self.assertFalse(reports['orange_000']['attached'])
        self.assertTrue(reports['orange_001']['attached'])
        attached_rows = [r for r in rows if r['attached']]
        self.assertGreater(max(r['branch_displacement_rad'] for r in attached_rows), .02)
        # Unloaded branch rebounds towards q=0; loaded second branch stays bent.
        self.assertLess(np.linalg.norm(e.data.qpos[:2]), .01)
        self.assertGreater(abs(rows[-1]['fruit_y_m'] - rows[0]['fruit_y_m']), .5)

    def test_local_anchors_placement_same_names_and_visual_reset(self):
        p = (.5, -.2, .3)
        r = wum.rotmat_from_euler(.2, -.4, .7)
        e, plant = branch_fixture(pos=p, rotmat=r)
        original = plant.spec.to_json()
        for f in plant.spec.fruits:
            c = plant.fruit_connections[f.id]
            h = e.connection(c)
            a, b = e.data.site_xpos[list(h.site_ids)]
            expected = plant.spec.skeleton.by_id[f.parent_segment].point_at(f.attachment_t)
            np.testing.assert_allclose(a, r @ expected + p, atol=2e-7)
            np.testing.assert_allclose(b, r @ (np.asarray(f.position) + np.asarray(f.stem_direction) * f.radius) + p, atol=2e-7)
            self.assertAlmostEqual(h.length, f.stem_length, places=6)
            self.assertIsNone(plant.fruits[f.id].mounted_by)
            self.assertEqual(len(plant.fruits[f.id].visuals), 1)
        e.data.xfrc_applied[body_id(e, plant.fruits['orange_000']), :3] = (0, -50, 0)
        e.step(.3)
        self.assertEqual(plant.stem_objects['orange_000'].alpha, 0.)
        e.reset()
        self.assertEqual(plant.stem_objects['orange_000'].alpha, 1.)
        self.assertEqual(plant.spec.to_json(), original)
        with self.assertRaises(RuntimeError):
            plant.set_pos_rotmat((2, 0, 0))
        # Two plants deliberately reuse every botanical/fruit name.
        _, other = branch_fixture(pos=(-1, 0, 0))
        scene = wss.Scene()
        plant.add_to_scene(scene)
        other.add_to_scene(scene)
        both = MJEnv(scene)
        self.assertEqual(both.model.ntendon, 4)
        self.assertEqual(len({h.tendon_name for h in both.runtime.connections.values()}), 4)
        plant.remove_from_scene(scene)
        self.assertEqual(len(scene.connections), 2)
        self.assertNotIn(plant.fruits['orange_000'], scene.sobjs)

    def test_legacy_rigid_fruit_mode(self):
        e, plant = branch_fixture(mode='rigid')
        self.assertEqual(e.model.ntendon, 0)
        self.assertEqual(e.model.nv, 4)
        self.assertFalse(plant.fruit_connections)
        self.assertTrue(all(f.mounted_by is plant.mech for f in plant.fruit_objects))
        e.step(.05)
        self.assertTrue(np.isfinite(e.data.qpos).all())

    def test_sites_survive_empty_body_merge(self):
        from wrs.robots.base.mech_structure import MechStruct, Link, Joint
        from wrs.robots.base.mech_base import MechBase
        from wrs.physics.connections import BodyAnchor, LinearSpringConnectionSpec
        from wrs.scene.collision_shape import SphereCollisionShape
        root, child = Link(), Link()
        child.add_collision(SphereCollisionShape(.01))
        structure = MechStruct()
        rotation = wum.rotmat_from_euler(.2, .3, -.4)
        structure.add_jnt(Joint(wuc.JntType.FIXED, root, child, (1, 0, 0),
                               pos=(.1, .2, .3), rotmat=rotation))
        mech = MechBase(structure=structure, pos=(.4, -.2, .5))
        scene = wss.Scene()
        mech.add_to_scene(scene)
        connections, expected = [], []
        for link in mech.runtime_lnks:
            point = np.array([.03, -.01, .05])
            expected.append(link.rotmat @ point + link.pos)
            c = LinearSpringConnectionSpec('same_name', BodyAnchor(link, point), BodyAnchor(None, (0, 0, 2)), 1., 0., 0.)
            scene.add_connection(c)
            connections.append(c)
        env = MJEnv(scene)
        self.assertIs(env.sync.rutl2bdy[mech.runtime_lnks[0]], env.sync.rutl2bdy[mech.runtime_lnks[1]])
        for c, point in zip(connections, expected):
            np.testing.assert_allclose(env.data.site_xpos[env.connection(c).site_ids[0]], point, atol=2e-7)

    def test_tendon_velocity_uses_both_moving_anchors(self):
        e, plant = branch_fixture()
        e.data.qvel[:] = np.linspace(-.4, .7, e.model.nv)
        e.runtime.forward()
        h = e.connection(plant.fruit_connections['orange_000'])
        jacp, jacr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))
        velocities = []
        for site in h.site_ids:
            mujoco.mj_jacSite(e.model, e.data, jacp, jacr, site)
            velocities.append(jacp @ e.data.qvel)
        a, b = e.data.site_xpos[list(h.site_ids)]
        expected = (b - a) @ (velocities[1] - velocities[0]) / np.linalg.norm(b - a)
        self.assertAlmostEqual(h.velocity, expected, places=12)
        self.assertGreater(np.linalg.norm(velocities[0]), .01)

    def test_pre_scene_placement_visual_endpoints_and_empty_fruits(self):
        from agriculture.dynamic import DynamicPlantBuilder
        from agriculture.config import load_config
        from agriculture.attachment import FruitAttachmentSettings
        old, plant = branch_fixture()
        plant.remove_from_scene(old.scene)
        plant.set_pos_rotmat((.3, -.5, .2), wum.rotmat_from_euler(.4, -.1, .7))
        scene = wss.Scene()
        plant.add_to_scene(scene)
        env = MJEnv(scene)
        env.step(.02)
        for key, c in plant.fruit_connections.items():
            stem = plant.stem_objects[key]
            a, b = env.data.site_xpos[list(env.connection(c).site_ids)]
            np.testing.assert_allclose(stem.tf[:3, 3], a, atol=2e-7)
            np.testing.assert_allclose(stem.tf[:3, 3] + stem.tf[:3, 2], b, atol=2e-7)
            self.assertFalse(stem.collisions)
        spec = deepcopy(plant.spec)
        spec.fruits.clear()
        config = load_config()
        config['fruit_attachment'] = dict(mode='breakable_axial')
        empty = DynamicPlantBuilder(config).build(spec, plant.dynamics)
        scene = wss.Scene()
        empty.add_to_scene(scene)
        env = MJEnv(scene)
        env.step(.1)
        self.assertEqual(env.model.ntendon, 0)
        self.assertEqual(empty.fruits, {})
        for invalid in ({'mode': 'weld'}, {'density_kg_m3': -1}, {'stiffness_N_m': -1},
                        {'damping_N_s_m': float('nan')}, {'break_force_N': 0}, {'overload_hold_s': -1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FruitAttachmentSettings(**invalid)


if __name__ == '__main__':
    unittest.main()
