"""Plant-to-physics adapter for a synthetic, pull-oriented axial attachment.

PlantSpec retains only rest geometry/topology. These WRS connection specs live
on the built instance; no engine IDs or XML enter the botanical data model.
"""
from dataclasses import dataclass
import numpy as np
from wrs import wssop, wum
from wrs.physics.connections import BodyAnchor, LinearSpringConnectionSpec, TensileBreakPolicy


@dataclass(frozen=True)
class FruitAttachmentSettings:
    mode: str = 'rigid'
    density_kg_m3: float = 850.
    stiffness_N_m: float = 200.
    damping_N_s_m: float = 2.
    break_force_N: float = 8.
    overload_hold_s: float = .04

    def __post_init__(self):
        if self.mode not in ('rigid', 'breakable_axial'):
            raise ValueError('fruit attachment mode must be rigid or breakable_axial')
        if not np.isfinite(self.density_kg_m3) or self.density_kg_m3 <= 0:
            raise ValueError('fruit density must be finite and positive')
        # Reuse the generic validator, even for an empty fruit list.
        LinearSpringConnectionSpec('validation', BodyAnchor(None), BodyAnchor(None), 1.,
            self.stiffness_N_m, self.damping_N_s_m,
            TensileBreakPolicy(self.break_force_N, self.overload_hold_s))


def make_attachment(fruit, parent, cluster_link, cluster_frame, body, settings, visual):
    a = parent.point_at(fruit.attachment_t)
    b = np.asarray(fruit.stem_direction) * fruit.radius
    rest_length = float(np.linalg.norm(a - (np.asarray(fruit.position) + b)))
    if rest_length <= 1e-7:
        raise ValueError(f'{fruit.id}: breakable_axial requires a nonzero stem length')
    a_local = cluster_frame[:3, :3].T @ (a - cluster_frame[:3, 3])
    connection = LinearSpringConnectionSpec(f'{fruit.id}_stem', BodyAnchor(cluster_link, a_local),
        BodyAnchor(body, b), rest_length, settings.stiffness_N_m, settings.damping_N_s_m,
        TensileBreakPolicy(settings.break_force_N, settings.overload_hold_s))
    # Reuse one unit cylinder. Its SceneObject transform scales only the visual
    # Z axis, so geometry/scene membership need not be rebuilt every frame.
    stem = wssop.cylinder(spos=(0, 0, 0), epos=(0, 0, 1), radius=fruit.stem_radius,
                         segments=visual['stem_sides'], rgb=visual['stem_color'],
                         name=f'{fruit.id}_elastic_stem')
    update_stem(stem, cluster_link.tf[:3, :3] @ a_local + cluster_link.pos,
                body.tf[:3, :3] @ b + body.pos, True)
    return connection, stem


def update_stem(stem, a, b, attached):
    stem.alpha = 1. if attached else 0.
    if attached:
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length > 1e-10:
            tf = wum.tf_from_pos_rotmat(a, wum.rotmat_from_normal(delta))
            tf[:3, 2] *= length
            stem.tf = tf  # render-only affine transform; never a physics body
