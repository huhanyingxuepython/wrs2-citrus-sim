"""Legacy Panthera two-finger gripper expressed as a branching MechStruct.

Source: blanketxx/wrs-sealp-assemble, commit
7260d56e7eba7176eabc23063210d341583671ae (MIT; see LICENSE).
"""
import os
import numpy as np
import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.robots.base.mech_structure as wrbms
import wrs.robots.base.mech_base as wrbmb
from wrs.robots.end_effectors.ee_mixins import GripperMixin


def prepare_mechstruct(fixed_opening=None):
    structure = wrbms.MechStruct()
    # The gripper root stays at the arm flange. The old coupling's 5 mm
    # offset belongs to the palm geometry and both finger joint origins.
    coupling_pos = (0, 0, 0.005)
    palm = wrbms.Link.from_file(
        os.path.join(structure.default_mesh_dir, 'palm.stl'),
        loc_pos=coupling_pos, collision_type=wuc.CollisionType.MESH,
        rgb=(0.35, 0.35, 0.35))
    left = wrbms.Link.from_file(
        os.path.join(structure.default_mesh_dir, 'left_finger.stl'),
        collision_type=wuc.CollisionType.MESH, rgb=(0.3, 0.3, 0.3))
    right = wrbms.Link.from_file(
        os.path.join(structure.default_mesh_dir, 'right_finger.stl'),
        collision_type=wuc.CollisionType.MESH, rgb=(0.3, 0.3, 0.3))
    for link, name in ((palm, 'palm'), (left, 'left_finger'), (right, 'right_finger')):
        link.name = name
        structure.add_lnk(link)
    left_joint = wrbms.Joint(
        jnt_type=wuc.JntType.PRISMATIC, parent_lnk=palm, child_lnk=left,
        pos=coupling_pos, axis=wuc.StandardAxis.Y, lmt_lo=0, lmt_up=0.0425)
    right_joint = wrbms.Joint(
        jnt_type=wuc.JntType.PRISMATIC, parent_lnk=palm, child_lnk=right,
        pos=coupling_pos, axis=-wuc.StandardAxis.Y, lmt_lo=0, lmt_up=0.0425,
        mmc=(left_joint, 1.0, 0.0))
    if fixed_opening is not None:
        # Bake the existing prismatic poses into fixed joints. This keeps the
        # native meshes/TCP and needs no unsupported physical mimic constraint.
        for joint in (left_joint, right_joint):
            joint.pos = joint.pos + joint.rotmat @ (joint.ax * fixed_opening / 2)
            joint.jtype = wuc.JntType.FIXED
            joint.mmc = None
            joint.actuated = False
            joint.lmt_lo = joint.lmt_up = 0.0
    structure.add_jnt(left_joint)
    structure.add_jnt(right_joint)
    structure.ignore_collision(left, right)
    structure.compile()
    return structure


class FAFUGripper(wrbmb.MechBase, GripperMixin):
    """Symmetric 0--85 mm gripper; grasp center is 170 mm from its root."""

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None, *, fixed_opening=None):
        """Optionally hold an opening rigidly for contact simulation (metres).

        The default retains the original movable, kinematically linked fingers.
        A fixed instance has its own structure and zero active finger DOF.
        """
        if fixed_opening is not None:
            fixed_opening = float(fixed_opening)
            if not np.isfinite(fixed_opening) or not 0 <= fixed_opening <= .085:
                raise ValueError('fixed_opening must be within [0, 0.085] metres')
        self._fixed_opening = fixed_opening
        structure = None if fixed_opening is None else prepare_mechstruct(fixed_opening)
        super().__init__(rotmat=rotmat, pos=pos, structure=structure)
        self.add_tcp('grasp_center', self.runtime_root_lnk,
                     wum.tf_from_pos_rotmat(pos=(0, 0, 0.17)))
        self.jaw_range = np.array([0.0, 0.085])
        if fixed_opening is not None:
            self.jaw_range[:] = fixed_opening
        self.open_dir = wuc.StandardAxis.Y
        self.contact_pattern = np.zeros((1, 3), dtype=np.float32)
        self.open()

    def set_opening(self, width):
        width = float(width)
        if not np.isfinite(width) or not self.jaw_range[0] <= width <= self.jaw_range[1]:
            raise ValueError(f'opening {width} out of range {self.jaw_range}')
        if self._fixed_opening is None:
            self.fk(qs=[width * 0.5, width * 0.5])

    def clone(self):
        new = super().clone()
        new._fixed_opening = self._fixed_opening
        new.jaw_range = self.jaw_range.copy()
        new.open_dir = self.open_dir.copy()
        new.contact_pattern = self.contact_pattern.copy()
        return new
