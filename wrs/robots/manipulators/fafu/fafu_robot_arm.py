"""FAFU six-axis arm, ported from the legacy PantheraHT Python definition.

Source: blanketxx/wrs-sealp-assemble, commit
7260d56e7eba7176eabc23063210d341583671ae (MIT; see LICENSE).
The Python model, rather than its differing URDF, defines the frame convention.
"""
import os
import numpy as np
import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.robots.base.mech_structure as wrbms
import wrs.robots.base.mech_base as wrbmb
from wrs.manipulation.arm import SingleArmManipulation


def prepare_mechstruct():
    structure = wrbms.MechStruct()
    links = []
    for name in ('base_link', 'link1', 'link2', 'link3', 'link4', 'link5'):
        gray = 0.55 if name == 'base_link' else 0.35
        link = wrbms.Link.from_file(
            os.path.join(structure.default_mesh_dir, name + '.stl'),
            # link2 exceeds MuJoCo's 200,000-face STL limit; keep its full
            # visual mesh and use WRS's inline convex hull for collision.
            collision_type=(wuc.CollisionType.CVXHULL if name == 'link2'
                            else wuc.CollisionType.MESH),
            rgb=(gray, gray, gray))
        link.name = name
        links.append(link)

    # The legacy link6 is a frame only; the mounted gripper supplies its mesh.
    flange = wrbms.Link()
    flange.name = 'flange'
    # A meshless moving body needs nonzero inertia for standalone MJCollider.
    # Numerical placeholder only: this port does not calibrate dynamics.
    flange.set_inertia(mass=1e-6, com=np.zeros(3), inrtmat=np.eye(3) * 1e-9)
    links.append(flange)
    for link in links:
        structure.add_lnk(link)

    # Joint offsets (m), axes, and limits (rad) from the legacy Python model.
    specs = (
        ((0, 0, 0.0634), (0, 0, 1), (-2.4, 2.4)),
        ((0.018199, 0, 0.053), (0, 1, 0), (0.0, 3.2)),
        ((-0.26, 0, 0), (0, -1, 0), (0.0, 4.0)),
        ((0.23, 0, 0.06), (0, -1, 0), (-1.6, 1.6)),
        ((0.07, 0, 0.036319), (0, 0, -1), (-1.7, 1.7)),
        ((0.02345, 0, -0.039), (0, 0, 1), (-2.5, 2.5)),
    )
    for i, (pos, axis, limits) in enumerate(specs):
        structure.add_jnt(wrbms.Joint(
            jnt_type=wuc.JntType.REVOLUTE,
            parent_lnk=links[i], child_lnk=links[i + 1],
            pos=pos, axis=axis,
            rotmat=wum.rotmat_from_euler(0, np.pi / 2, 0) if i == 5 else None,
            lmt_lo=limits[0], lmt_up=limits[1]))
    # Connected links are excluded by compile(); other self-collision pairs
    # remain enabled. MJCollider can detect structural overlaps on compilation.
    structure.compile()
    return structure


class FAFURobotArm(wrbmb.MechBase, SingleArmManipulation):
    """Six-axis arm with a 'main' chain and a 'flange' TCP.

    The default solver is WRS's lazy SELIK (builds a seed database on first IK).
    ``solver`` optionally accepts a chain -> solver factory, as in add_chain().
    Use fafu_with_gripper() to assemble the legacy arm and two-finger gripper.
    """

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None, *, solver=None):
        super().__init__(rotmat=rotmat, pos=pos, is_floating=False,
                         home_qs=np.zeros(6, dtype=np.float32))
        c = self.structure.compiled
        self.add_chain('main', c.root_lnk, c.tip_lnks[0], solver=solver)
        self.add_tcp('flange', self.runtime_lnks[-1])


def fafu_with_gripper(rotmat=None, pos=None, jaw_width=0.085, *, solver=None):
    """Return (FAFURobotArm, FAFUGripper), mounted and ready for FK/planning.

    Full-pose IK defaults to the arm flange; for grasp targets pass
    ``tcp=gripper.tcp('grasp_center')`` explicitly.
    """
    from wrs.robots.end_effectors.fafu_gripper import FAFUGripper

    arm = FAFURobotArm(rotmat=rotmat, pos=pos, solver=solver)
    gripper = FAFUGripper()
    gripper.set_opening(jaw_width)
    arm.mount(gripper, arm.tcp('flange').parent_lnk,
              arm.tcp('flange').loc_tf, update=True)
    return arm, gripper
