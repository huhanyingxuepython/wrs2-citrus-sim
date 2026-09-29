"""Bunker Mini cart + 3030 frame, loaded from the Onshape export.

The raw Assembly 1 STL is ~1.4M triangles. The viewer uses the decimated
``cart_viz.stl``. Motion is a planar x/y/yaw chain; the mesh does not need
separate wheel parts. Collision is a box around the chassis and cabinet only.
"""
from pathlib import Path
import json
import struct
import numpy as np
from wrs import wsrm, wuc, wum
import wrs.geom.geometry as wgg
import wrs.robots.base.mech_base as wrbmb
import wrs.robots.base.mech_structure as wrbms
import wrs.scene.render_model_primitive as wsrmp

ASSET_DIR = Path(__file__).resolve().parent / 'assets' / 'bunker_cart'
SRC_STL = ASSET_DIR / 'cart.stl'
VIZ_STL = ASSET_DIR / 'cart_viz.stl'
META_PATH = ASSET_DIR / 'cart_meta.json'
DEFAULT_RGB = (0.55, 0.58, 0.60)


def _load_meta():
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    body_max = list(meta['body_max'])
    body_max[2] = max(meta['mount_local'][2], meta['zmin'] + meta['mount_height_m'])
    meta['body_max'] = body_max
    return meta


def _load_stl_numpy(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('ab', '<u2')])
    tris = np.frombuffer(raw, dtype=dt, count=count, offset=84)
    verts = np.ascontiguousarray(tris['v'].reshape(-1, 3), dtype=np.float32)
    faces = np.arange(count * 3, dtype=np.int32).reshape(count, 3)
    return verts, faces


def cart_mesh_path():
    if VIZ_STL.is_file():
        return VIZ_STL
    if SRC_STL.is_file():
        return SRC_STL
    return None


def mount_height_m(meta=None):
    meta = _load_meta() if meta is None else meta
    return float(meta['mount_height_m'])


def _dummy_link(name):
    link = wrbms.Link()
    link.name = name
    link.set_inertia(mass=1e-6, com=np.zeros(3), inrtmat=np.eye(3) * 1e-9)
    return link


def prepare_cart_struct():
    path = cart_mesh_path()
    if path is None or not META_PATH.is_file():
        raise FileNotFoundError('bunker cart mesh missing')
    meta = _load_meta()
    rgb = DEFAULT_RGB
    structure = wrbms.MechStruct()
    root = _dummy_link('cart_world')
    dummy_x = _dummy_link('cart_x')
    dummy_y = _dummy_link('cart_y')
    body = wrbms.Link(collision_type=wuc.CollisionType.AABB)
    body.name = 'cart_body'
    verts, faces = _load_stl_numpy(path)
    body.add_visual(
        wsrm.RenderModel(geom=wgg.gen_geom_from_raw(verts, faces), rgb=rgb),
        auto_make_collision=False)
    body_min = np.asarray(meta['body_min'], dtype=float)
    body_max = np.asarray(meta['body_max'], dtype=float)
    body_max[2] = meta['mount_local'][2]
    box = wsrmp.gen_box_rmodel(xyz_lengths=body_max - body_min, rgb=rgb, alpha=0.0)
    body.add_visual(
        wsrm.RenderModel(geom=box.geom, pos=0.5 * (body_min + body_max), rgb=rgb, alpha=0.0),
        auto_make_collision=True)
    body.set_inertia(mass=80.0, com=0.5 * (body_min + body_max), inrtmat=np.eye(3) * 8.0)
    for link in (root, dummy_x, dummy_y, body):
        structure.add_lnk(link)
    structure.add_jnt(wrbms.Joint(
        jnt_type=wuc.JntType.PRISMATIC, parent_lnk=root, child_lnk=dummy_x,
        axis=wuc.StandardAxis.X, lmt_lo=-8.0, lmt_up=8.0, actuated=False))
    structure.add_jnt(wrbms.Joint(
        jnt_type=wuc.JntType.PRISMATIC, parent_lnk=dummy_x, child_lnk=dummy_y,
        axis=wuc.StandardAxis.Y, lmt_lo=-8.0, lmt_up=8.0, actuated=False))
    structure.add_jnt(wrbms.Joint(
        jnt_type=wuc.JntType.REVOLUTE, parent_lnk=dummy_y, child_lnk=body,
        axis=wuc.StandardAxis.Z, lmt_lo=-12.6, lmt_up=12.6, actuated=False))
    structure.compile()
    return structure


class BunkerCart(wrbmb.MechBase):
    """Planar bunker: qs = [x, y, yaw], mesh origin at the Onshape frame."""

    @classmethod
    def _build_structure(cls):
        return prepare_cart_struct()

    def __init__(self, home_xy_yaw, z, arm_loc_tf):
        home = np.asarray(home_xy_yaw, dtype=np.float32)
        super().__init__(pos=(0.0, 0.0, z), home_qs=home, is_floating=False)
        self.arm_loc_tf = np.asarray(arm_loc_tf, dtype=np.float32)
        self.fk(home)

    @property
    def body_lnk(self):
        return next(link for link in self.runtime_lnks if link.name == 'cart_body')

    @property
    def yaw(self):
        return float(self.qs[2])

    def drive(self, lin_m_s, ang_rad_s, dt):
        """Unicycle step in the cart heading (Onshape +X / world after yaw)."""
        x, y, yaw = (float(v) for v in self.qs)
        x = x + float(lin_m_s) * np.cos(yaw) * float(dt)
        y = y + float(lin_m_s) * np.sin(yaw) * float(dt)
        yaw = yaw + float(ang_rad_s) * float(dt)
        self.fk(np.array((x, y, yaw), dtype=np.float32))


def _cart_placement(settings):
    spec = settings.get('cart') or {}
    meta = _load_meta()
    mount = np.asarray(meta['mount_local'], dtype=float)
    base = np.asarray(settings['base_position'], dtype=float)
    base[2] = float(meta['mount_height_m'])
    yaw_deg = float(spec.get('yaw_deg', settings.get('base_yaw_deg', 90)))
    rotmat = wum.rotmat_from_axangle((0, 0, 1), np.deg2rad(yaw_deg))
    pitch = float(spec.get('hole_pitch_m', 0.025))
    offset_local = np.array((
        float(spec.get('mount_forward_holes', 0)) * pitch,
        float(spec.get('mount_lateral_holes', 0)) * pitch,
        0.0,
    ), dtype=float)
    mount = mount + offset_local
    base[:2] = base[:2] + (rotmat @ offset_local)[:2]
    settings['base_position'] = base.tolist()
    mount_world = rotmat @ mount
    pos = np.array((
        base[0] - mount_world[0],
        base[1] - mount_world[1],
        -float(meta['zmin']),
    ), dtype=float)
    cart_tf = wum.tf_from_pos_rotmat(pos, rotmat)
    arm_tf = wum.tf_from_pos_rotmat(
        base, wum.rotmat_from_axangle((0, 0, 1), np.deg2rad(settings['base_yaw_deg'])))
    arm_loc_tf = np.linalg.inv(cart_tf) @ arm_tf
    return pos, float(np.deg2rad(yaw_deg)), float(-meta['zmin']), arm_loc_tf


def add_bunker_cart(scene, settings):
    """Add the planar cart. Writes ``settings['base_position']`` to the plate."""
    spec = settings.get('cart') or {}
    if spec.get('enabled', True) is False:
        return None
    if cart_mesh_path() is None or not META_PATH.is_file():
        return None
    pos, yaw, z, arm_loc_tf = _cart_placement(settings)
    cart = BunkerCart(home_xy_yaw=(pos[0], pos[1], yaw), z=z, arm_loc_tf=arm_loc_tf)
    cart.add_to_scene(scene)
    return cart
