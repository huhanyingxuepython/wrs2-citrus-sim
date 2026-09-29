"""Lab Citrus V2: static reference, cluster inspection, or contact/rebound."""
from pathlib import Path
import argparse
import colorsys
import json
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from wrs import wuc, wss, wssop, wvw
from agriculture.config import LAB_CONFIG, load_config
from agriculture.generator import generate
from agriculture.static import StaticPlantBuilder
from agriculture.dynamic import DynamicPlantBuilder
from agriculture.dynamics import PlantDynamicsSpec
from examples.agriculture.push_experiment import PushExperiment

SHOW_FULL_PLANT = True
SHOW_SKELETON = False
SHOW_STATIC_BRANCHES = False
SHOW_DYNAMIC_CLUSTERS = False
SHOW_LEAVES = True
SHOW_FRUITS = True
SHOW_BRANCH_COLLISION = False
SHOW_FOLIAGE_PROXY = False
SHOW_JOINT_AXES = False
FLAGS = ('FULL_PLANT', 'SKELETON', 'STATIC_BRANCHES', 'DYNAMIC_CLUSTERS', 'LEAVES',
         'FRUITS', 'BRANCH_COLLISION', 'FOLIAGE_PROXY', 'JOINT_AXES')


def debug_view(plant, flags, config):
    for key, link in plant.cluster_links.items():
        show = flags['FULL_PLANT'] or flags['STATIC_BRANCHES' if key is None else 'DYNAMIC_CLUSTERS']
        link.alpha = float(show)
        link.toggle_render_collision = flags['BRANCH_COLLISION']
    if flags['DYNAMIC_CLUSTERS']:
        for i, cluster in enumerate(plant.dynamics.clusters):
            plant.cluster_links[cluster.id].rgb = colorsys.hsv_to_rgb(i / len(plant.dynamics.clusters), .7, .85)
    for leaf in plant.leaf_objects:
        leaf.alpha = float(flags['LEAVES'])
    for obj in plant.fruits.values():
        obj.alpha = float(flags['FRUITS'])
    plant.show_foliage_proxies(flags['FOLIAGE_PROXY'])
    if flags['SKELETON']:
        for key, link in plant.cluster_links.items():
            stems = [s for s in plant.spec.branches if plant.segment_clusters[s.id] == key]
            if not stems:
                continue
            frame = plant.cluster_frames[key]
            points = (np.array([[s.start, s.end] for s in stems]) - frame[:3, 3]) @ frame[:3, :3]
            obj = wssop.linsegs(points, radius=config['demo']['skeleton_radius'], srgbs=np.array([.6, .1, .8]))
            plant.mech.mount(obj, link, update=True)
    if flags['JOINT_AXES']:
        for cluster in plant.dynamics.clusters:
            obj = wssop.frame(length_scale=config['demo']['joint_axis_length_scale'])
            plant.mech.mount(obj, plant.cluster_links[cluster.id], update=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=LAB_CONFIG)
    parser.add_argument('--case', choices=('static', 'clusters', 'push'), default='push')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float, default=None, help='Viewer duration for static/clusters cases.')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--trace', type=Path, help='Write contact experiment trajectory JSON.')
    for flag in FLAGS:
        parser.add_argument('--show-' + flag.lower().replace('_', '-'), action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be finite and positive')
    config = load_config(args.config)
    spec = generate(config)
    scene = wss.Scene()
    size = np.asarray(config['demo']['ground_size']) * config.get('scale_multiplier', 1)
    wssop.box(pos=(0, 0, -size[2] / 2), xyz_lengths=size, rgb=config['demo']['ground_color'],
              collision_type=wuc.CollisionType.AABB, name='ground').add_to_scene(scene)
    flags = {name: globals()['SHOW_' + name] for name in FLAGS}
    if args.case == 'clusters':
        flags.update(FULL_PLANT=False, LEAVES=False, SKELETON=True, DYNAMIC_CLUSTERS=True,
                     JOINT_AXES=True, FOLIAGE_PROXY=True)
    flags.update({name: getattr(args, 'show_' + name.lower()) for name in FLAGS if getattr(args, 'show_' + name.lower()) is not None})
    experiment = None
    if args.case == 'static':
        plant = StaticPlantBuilder(config).build(spec)
        for obj in plant.leaf_objects:
            obj.alpha = float(flags['LEAVES'])
        for obj in plant.fruits.values():
            obj.alpha = float(flags['FRUITS'])
        for obj in plant.branch_objects:
            obj.toggle_render_collision = flags['BRANCH_COLLISION']
        plant.add_to_scene(scene)
    else:
        plant = DynamicPlantBuilder(config).build(spec, PlantDynamicsSpec.from_config(spec, config))
        debug_view(plant, flags, config)
        plant.add_to_scene(scene)
        if args.case == 'push':
            experiment = PushExperiment(plant, scene, config['push_demo'])
    print(json.dumps(plant.summary(), indent=2), flush=True)

    def finish():
        report = experiment.report() if experiment else plant.summary()
        print(json.dumps(report, indent=2), flush=True)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if args.trace and experiment:
            args.trace.write_text(json.dumps(experiment.rows), encoding='utf-8')

    if args.headless:
        if experiment:
            experiment.step(experiment.duration)
        finish()
        return
    scale = config.get('scale_multiplier', 1)
    base = wvw.World(cam_pos=np.asarray(config['demo']['camera_pos']) * scale,
                     cam_lookat_pos=np.asarray(config['demo']['camera_lookat']) * scale, port=args.port)
    base.set_scene(scene)
    base.set_caption(f'Lab Citrus V2 | {args.case}')
    if experiment:
        def advance(dt):
            experiment.step(1 / 60)
            if experiment.time >= experiment.duration:
                base.close()
        base.schedule_interval(advance, 1 / 60)
    elif args.duration is not None:
        base.schedule_once(lambda dt: base.close(), args.duration)
    base.run()
    finish()


if __name__ == '__main__':
    main()
