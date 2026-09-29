"""Small WRS core regression suite: generated structures and passive MJCF."""
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from wrs import wuc, wss, wssop
from wrs.robots.base.mech_structure import MechStruct, Link, Joint
from wrs.robots.base.mech_base import MechBase
from wrs.physics.mj_wrs_cvter import MJWRSConverter
from wrs.physics.mj_env import MJEnv
from wrs.scene.collision_shape import SphereCollisionShape


def mechanism(**joint_options):
    root, tip = Link(), Link()
    tip.add_collision(SphereCollisionShape(.03, pos=(0, 0, .2)))
    tip.set_inertia(mass=.1)
    s = MechStruct()
    s.add_jnt(Joint(wuc.JntType.REVOLUTE, root, tip, (1, 0, 0), **joint_options))
    return MechBase(structure=s, is_floating=False)


class PassiveJointTests(unittest.TestCase):
    def test_defaults_and_passive_parameters_reach_mujoco(self):
        for passive in (False, True):
            options = dict(actuated=False, stiffness=3.5, damping=.07, springref=.12,
                           frictionloss=.002, armature=.003) if passive else {}
            mech = mechanism(**options)
            scene = wss.Scene(); mech.add_to_scene(scene)
            env = MJEnv(scene, require_ctrl=True)
            self.assertEqual(env.model.nu, int(not passive))
            self.assertEqual(env.model.nv, 1)
            node = ET.fromstring(env.xml_string).find('.//joint')
            for key, expected in (options or dict(stiffness=0, damping=1, springref=0, frictionloss=.01, armature=.02)).items():
                if key != 'actuated': self.assertAlmostEqual(float(node.attrib[key]), expected)
            self.assertAlmostEqual(env.model.jnt_stiffness[0], options.get('stiffness', 0))
            self.assertAlmostEqual(env.model.qpos_spring[0], options.get('springref', 0))

    def test_instance_structures_clone_and_mount_are_independent(self):
        a, b = mechanism(actuated=False), mechanism()
        self.assertIsNot(a.structure, b.structure)
        marker = wssop.sphere(radius=.01)
        a.mount(marker, a.runtime_lnks[-1], update=True)
        clone = a.clone()
        self.assertIs(clone.structure, a.structure)
        self.assertIsNot(clone.runtime_lnks[-1], a.runtime_lnks[-1])
        clone.fk([.4])
        np.testing.assert_array_equal(a.qs, [0])
        self.assertIs(next(iter(clone._mountings)).mounted_by, clone)
        self.assertEqual(b.structure.jnts[0].stiffness, 0)

    def test_existing_robot_apis_and_actuation(self):
        from wrs.robots.vehicle.xytheta import XYThetaRobot
        from wrs.robots.manipulators.kawasaki.rs007l.rs007l import RS007L
        for cls, ndof in ((XYThetaRobot, 3), (RS007L, 6)):
            robot, another = cls(), cls()
            robot.is_floating = False
            self.assertIs(robot.structure, another.structure)
            self.assertEqual(robot.ndof, ndof)
            q = np.linspace(.01, .05, len(robot.qs))
            robot.fk(q)
            clone = robot.clone()
            np.testing.assert_allclose(clone.gl_lnk_tfarr, robot.gl_lnk_tfarr)
            scene = wss.Scene(); robot.add_to_scene(scene)
            env = MJEnv(scene, require_ctrl=True)
            self.assertEqual(env.model.nu, ndof)
            env.step(.01)
            self.assertTrue(np.isfinite(env.data.qpos).all())


if __name__ == '__main__': unittest.main()
