"""Run from the repository root with python -m examples.agriculture.citrus_tree.demo_static_tree."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

# Match the existing examples' direct-script usage as well as module execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from wrs import wuc, wss, wsso, wssop, wvw
from examples.agriculture.citrus_tree.builder import build_tree
from examples.agriculture.citrus_tree.generator import DEFAULT_CONFIG, generate_tree, load_config

SHOW_COLLISION = False
MODES = ("full", "branches", "leaves", "fruits", "collision")


def camera_pose(config, view="front"):
    """Fixed nominal camera poses scale with the tree; no camera calibration claim."""
    settings = config["demo"]
    poses = settings.get("reference_cameras", {"front": {
        "pos": settings["camera_pos"], "lookat": settings["camera_lookat"]}})
    if view not in poses:
        raise ValueError(f"unknown camera {view!r}; available: {list(poses)}")
    factor = config.get("scale_multiplier", 1.0)
    return (np.asarray(poses[view]["pos"]) * factor,
            np.asarray(poses[view]["lookat"]) * factor)


def scene_for_mode(tree, config, mode="full", show_collision=False):
    """Create a scene atomically; debug filtering never changes TreeSpec/visuals."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    scene = wss.Scene()
    settings = config["demo"]
    size = np.asarray(settings["ground_size"]) * config.get("scale_multiplier", 1.0)
    ground = wssop.box(pos=(0, 0, -size[2] / 2), xyz_lengths=size,
                        rgb=settings["ground_color"], collision_type=wuc.CollisionType.AABB,
                        name="ground")
    ground.add_to_scene(scene)
    tree.show_collision(show_collision)
    if mode == "collision":
        for owner in tree.objects:
            if not owner.collisions:
                continue
            proxy = wsso.SceneObject(name=f"collision_{owner.name}")
            proxy.tf = owner.tf
            for shape in owner.collisions:
                model = shape.to_render_model()
                model.alpha = 1.0
                proxy.add_visual(model, auto_make_collision=False)
            proxy.add_to_scene(scene)
    else:
        groups = {"full": tree.objects, "branches": tree.branch_objects,
                  "leaves": tree.leaf_objects, "fruits": tree.fruit_objects}
        for obj in groups[mode]:
            obj.add_to_scene(scene)
    return scene


def fitting_report(tree):
    """Machine-readable summary, including explicit tree-local planning targets."""
    return tree.summary() | {
        "scale_multiplier": tree.spec.metadata.get("scale_multiplier", 1.0),
        "reference_status": tree.spec.metadata.get("reference_status"),
        "leaf_counts_by_profile": tree.spec.metadata.get("leaf_counts_by_profile", {}),
        "fruits": [{"id": fruit.id, "parent_branch": fruit.parent_branch,
                    "tree_local_position_m": list(fruit.position),
                    "world_position_m": obj.pos.tolist(), "radius_m": fruit.radius,
                    "targetable": fruit.targetable}
                   for fruit, obj in zip(tree.spec.fruits, tree.fruit_objects)]}


def create_demo(config, *, seed=None, show_collision=SHOW_COLLISION, port=8000,
                view="front", mode="full"):
    camera, lookat = camera_pose(config, view)
    base = wvw.World(cam_pos=camera, cam_lookat_pos=lookat, port=port)
    base.set_caption(f"Static Citrus Tree V1 | {view}")
    tree = build_tree(generate_tree(config, seed=seed), config)
    base.set_scene(scene_for_mode(tree, config, mode, show_collision))
    return base, tree


class ConfigReloader:
    """Reload only after successful validation/build, keeping the last good scene."""
    def __init__(self, path, apply, *, seed=None, scale=None):
        self.path = Path(path)
        self.apply = apply
        self.seed = seed
        self.scale = scale
        self.signature = self.path.stat().st_mtime_ns
        self.last_error = None

    def poll(self, dt=0):
        try:
            signature = self.path.stat().st_mtime_ns
            if signature == self.signature:
                return False
            self.signature = signature
            config = load_config(self.path)
            if self.scale is not None:
                config["scale_multiplier"] = self.scale
            tree = build_tree(generate_tree(config, seed=self.seed), config)
            self.apply(config, tree)
            self.last_error = None
            return True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            message = f"Config reload rejected; keeping previous scene: {exc}"
            if message != self.last_error:
                print(message, flush=True)
            self.last_error = message
            return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scale", type=float, help="Override the preset's uniform scale_multiplier.")
    parser.add_argument("--view", choices=("front", "front-left", "front-right"), default="front")
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--watch", action="store_true", help="Rebuild on config saves; keep the current viewer camera.")
    parser.add_argument("--no-ui", action="store_true", help="Hide the mode selector for comparison screenshots.")
    parser.add_argument("--show-collision", action=argparse.BooleanOptionalAction, default=SHOW_COLLISION)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--duration", type=float, help="Close the World after N seconds (smoke run).")
    parser.add_argument("--export-spec", type=Path, help="Write expanded pure TreeSpec JSON.")
    parser.add_argument("--report", type=Path, help="Write fitting statistics and all fruit positions as JSON.")
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if args.scale is not None and (not np.isfinite(args.scale) or args.scale <= 0):
        parser.error("--scale must be finite and positive")
    config = deepcopy(load_config(args.config))
    if args.scale is not None:
        config["scale_multiplier"] = args.scale
    base, tree = create_demo(config, seed=args.seed, view=args.view, mode=args.mode,
                             show_collision=args.show_collision, port=args.port)
    state = {"tree": tree, "config": config, "mode": args.mode}

    def report(current_tree):
        text = json.dumps(fitting_report(current_tree), indent=2, allow_nan=False)
        print(text, flush=True)
        if args.export_spec:
            current_tree.spec.to_json(args.export_spec)
        if args.report:
            args.report.write_text(text + "\n", encoding="utf-8")

    def change_mode(mode):
        base.set_scene(scene_for_mode(state["tree"], state["config"], mode, args.show_collision))
        state["mode"] = mode

    def apply_config(new_config, new_tree):
        scene = scene_for_mode(new_tree, new_config, state["mode"], args.show_collision)
        # Do not mutate the live scene while the publisher walks it.
        report(new_tree)
        base.set_scene(scene)
        state.update(config=new_config, tree=new_tree)
        if not args.no_ui:
            base.ui.set_value("counts", f"{len(new_tree.spec.leaves)} leaves / {len(new_tree.spec.fruits)} fruits")

    if not args.no_ui:
        base.ui.configure(title="Citrus reference fitting", description=f"Camera preset: {args.view}", width=260)
        base.ui.add_select("mode", options=MODES, value=args.mode, label="Tree layers", on_change=change_mode)
        base.ui.add_label("counts", value=f"{len(tree.spec.leaves)} leaves / {len(tree.spec.fruits)} fruits")
    report(tree)
    if args.watch:
        reloader = ConfigReloader(args.config, apply_config, seed=args.seed, scale=args.scale)
        base.schedule_interval(reloader.poll, .5)
        print("Watching config. Camera changes require restart with --view; geometry reload preserves the current view.", flush=True)
    if args.duration is not None:
        base.schedule_once(lambda dt: base.close(), args.duration)
    base.run()


if __name__ == "__main__":
    main()
