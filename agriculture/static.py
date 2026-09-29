"""Static PlantSpec adapter and shared native WRS primitive factories."""
from dataclasses import dataclass, field

import numpy as np
from wrs import wum, wsso, wssop
import wrs.scene.collision_shape as wsc
import wrs.scene.render_model_primitive as wsrmp

from .geometry import leaf_batches
from .config import load_config
from .spec import PlantSpec

# WRS gen_geom_from_raw welds on a 1 um grid. Keep opposite leaf surfaces
# safely separated under arbitrary rotations. This is an adapter limitation,
# not a restriction on renderer-independent rest geometry or scaling.
WRS_MIN_LEAF_THICKNESS = 1e-5


@dataclass
class StaticPlantInstance:
    spec: PlantSpec
    branch_objects: list
    leaf_objects: list
    fruit_objects: list
    _local_poses: list = field(repr=False)

    @property
    def objects(self):
        return self.branch_objects + self.leaf_objects + self.fruit_objects

    @property
    def fruits(self):
        return self.fruit_by_id

    @property
    def semantic_groups(self):
        return {'TRUNK': [o for o, s in zip(self.branch_objects, self.spec.branches) if s.order == 0],
                'HARD_BRANCH': [o for o, s in zip(self.branch_objects, self.spec.branches) if s.order > 0],
                'COMPLIANT_BRANCH': [], 'FOLIAGE': self.leaf_objects, 'FRUIT': self.fruit_objects}

    @property
    def branch_by_id(self):
        return {obj.name: obj for obj in self.branch_objects}

    @property
    def fruit_by_id(self):
        return {obj.name: obj for obj in self.fruit_objects}

    def add_to_scene(self, scene):
        for obj in self.objects:
            obj.add_to_scene(scene)
        return self

    def remove_from_scene(self, scene):
        for obj in self.objects:
            obj.remove_from_scene(scene)

    def set_pos_rotmat(self, pos=None, rotmat=None):
        """Place the rest tree rigidly in world axes; spec always stays tree-local.

        This reapplies all rest poses, including any individually moved fruit.
        It is not a hierarchy update or a dynamics/articulation implementation.
        """
        tf = wum.tf_from_pos_rotmat(pos, rotmat)
        for obj, local in zip(self.objects, self._local_poses):
            obj.tf = tf @ local
        return self

    def show_collision(self, enabled=True):
        for obj in self.objects:
            obj.toggle_render_collision = bool(enabled)

    def summary(self):
        return self.spec.summary() | {
            "leaf_batch_count": len(self.leaf_objects),
            "scene_object_count": len(self.objects),
            "visual_triangle_count": sum(len(m.geom.fs) for obj in self.objects for m in obj.visuals)}


def build_tree(spec, config=None, *, pos=None, rotmat=None):
    """One object per branch/fruit, one leaf mesh per shade; no joints/physics.

    branch_objects/fruit_objects have the same order as their spec lists.
    Fruit object origins are at fruit centres, so obj.pos is a planning target.
    Collision is authored explicitly, independent of the visual tessellation.
    """
    spec.validate()
    config = load_config() if config is None else config
    visual = validate_render_config(spec, config)
    branches = [make_branch(branch, visual, spec.metadata.get('scale_multiplier', 1.0)) for branch in spec.branches]
    leaves = []
    for shade, (vs, fs) in leaf_batches(spec).items():
        obj = wssop.mesh(vs, fs, rgb=visual["leaf_colors"][shade], name=f"leaves_{shade:02d}")
        leaves.append(obj)
    fruits = [make_fruit(fruit, visual) for fruit in spec.fruits]
    objects = branches + leaves + fruits
    tree = StaticPlantInstance(spec, branches, leaves, fruits, [obj.tf for obj in objects])
    return tree.set_pos_rotmat(pos, rotmat)


def validate_render_config(spec, config):
    if config["collision"]["leaves_collision"]:
        raise ValueError("static leaf collision is unsupported; use DynamicPlantBuilder foliage proxies")
    visual = config["visual"]
    for key, minimum in (("branch_taper_steps", 1), ("branch_sides", 3),
                         ("fruit_subdivisions", 0), ("stem_sides", 3)):
        if type(visual[key]) is not int or visual[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if spec.leaves and spec.leaf_shape.thickness < WRS_MIN_LEAF_THICKNESS:
        raise ValueError("WRS leaf rendering requires thickness >= 10 um after scaling to prevent vertex welding")
    for color in [visual[k] for k in ("trunk_color", "branch_color", "twig_color", "fruit_color", "stem_color")] + visual["leaf_colors"]:
        if np.shape(color) != (3,) or not np.isfinite(color).all() or np.any(np.asarray(color) < 0) or np.any(np.asarray(color) > 1):
            raise ValueError("visual colors must be RGB triples in [0, 1]")
    if any(leaf.color_class >= len(visual['leaf_colors']) for leaf in spec.leaves):
        raise ValueError('no visual leaf color for color_class')
    return visual


def make_branch(branch, visual, scale=1.0):
    steps, sides = visual["branch_taper_steps"], visual["branch_sides"]
    length = float(np.linalg.norm(np.subtract(branch.end, branch.start)))
    rotation = branch.frame
    obj = wsso.SceneObject(name=branch.id)
    color = visual["trunk_color"] if branch.order == 0 else (
        visual["twig_color"] if max(branch.radius_start, branch.radius_end) <= visual["twig_color_radius"] * scale
        else visual["branch_color"])
    for i in range(steps):
        t = (i + 0.5) / steps
        radius = branch.radius_start * (1 - t) + branch.radius_end * t
        obj.add_visual(wsrmp.gen_cylinder_rmodel(
            length=length / steps, radius=radius, n_segs=sides,
            pos=(0, 0, length * i / steps), rgb=color), auto_make_collision=False)
    if branch.collidable:
        # Capsule endpoints lie exactly on segment endpoints; the max
        # radius encloses every taper cylinder, with conservative end caps.
        obj.add_collision(wsc.CapsuleCollisionShape(
            radius=max(branch.radius_start, branch.radius_end), half_length=length / 2,
            pos=(0, 0, length / 2)))
    obj.set_pos_rotmat(branch.start, rotation)
    return obj


def make_fruit(fruit, visual, *, include_stem=True):
    obj = wssop.icosphere(pos=fruit.position, radius=fruit.radius,
                         subdivisions=visual["fruit_subdivisions"],
                         rgb=visual["fruit_color"], name=fruit.id)
    obj.add_collision(wsc.SphereCollisionShape(radius=fruit.radius))
    if include_stem and fruit.stem_length > 0:
        direction = np.asarray(fruit.stem_direction)
        obj.add_visual(wsrmp.gen_cylinder_rmodel(
            length=fruit.stem_length, radius=fruit.stem_radius,
            n_segs=visual["stem_sides"],
            pos=direction * fruit.radius,
            rotmat=wum.rotmat_from_normal(direction),
            rgb=visual["stem_color"]), auto_make_collision=False)
    return obj


class StaticPlantBuilder:
    def __init__(self, config=None):
        self.config = load_config() if config is None else config

    def build(self, spec, *, pos=None, rotmat=None):
        return build_tree(spec, self.config, pos=pos, rotmat=rotmat)


