"""Small deterministic fixtures for axial attachment validation (no gripper).

External forces here are TEST loads, not a second spring implementation.
All masses, stiffnesses and rupture thresholds are synthetic SI demo values.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from wrs import wss, wssop, wuc
from wrs.physics.connections import BodyAnchor, LinearSpringConnectionSpec, TensileBreakPolicy
from wrs.physics.inertial import inertia_sphere
from wrs.physics.mj_env import MJEnv
from agriculture.config import load_config
from agriculture.spec import PlantSpec, PlantSkeleton, StemSegment, FruitPlacement
from agriculture.dynamics import PlantDynamicsSpec, BranchClusterSpec, BranchDynamicsProfile
from agriculture.dynamic import DynamicPlantBuilder


def fixed_fixture(dt=.002):
    scene = wss.Scene()
    fruit = wssop.icosphere(pos=(0, 0, .86), radius=.04, subdivisions=2,
        rgb=(1., .45, .02), is_floating=True, collision_type=wuc.CollisionType.SPHERE,
        name='orange_000')
    fruit.set_inertia(inertia_sphere(.2, .04), np.zeros(3), .2)
    fruit.add_to_scene(scene)
    connection = LinearSpringConnectionSpec('orange_000_stem',
        BodyAnchor(None, (0, 0, 1)), BodyAnchor(fruit, (0, 0, .04)),
        .1, 200., 2., TensileBreakPolicy(5., .04))
    scene.add_connection(connection)
    env = MJEnv(scene)
    env.model.opt.timestep = dt
    return env, fruit, connection


def branch_fixture(*, pos=None, rotmat=None, mode='breakable_axial'):
    spec = PlantSpec('two_fruits', PlantSkeleton([
        StemSegment('trunk', None, (0, 0, 0), (0, 0, .8), .012, .01, 0),
        StemSegment('right', 'trunk', (0, 0, .8), (.30, 0, .8), .006, .004, 1),
        StemSegment('left', 'trunk', (0, 0, .8), (-.30, 0, .8), .006, .004, 1)]),
        fruits=[FruitPlacement(f'orange_{i:03d}', parent, (x, 0, .71), .04, .05)
                for i, (parent, x) in enumerate([('right', .3), ('left', -.3)])])
    dynamics = PlantDynamicsSpec([
        BranchClusterSpec('right', 'right', ['right'], dof=2),
        BranchClusterSpec('left', 'left', ['left'], dof=2)],
        {'default': BranchDynamicsProfile(stiffness=4., damping=.12)})
    config = load_config()
    config['fruit_attachment'] = dict(mode=mode, stiffness_N_m=200., damping_N_s_m=2.,
                                     break_force_N=8., overload_hold_s=.04)
    plant = DynamicPlantBuilder(config).build(spec, dynamics, pos=pos, rotmat=rotmat)
    scene = wss.Scene()
    plant.add_to_scene(scene)
    env = MJEnv(scene, require_ctrl=True)
    return env, plant


def body_id(env, fruit):
    return env.model.body(env.sync.sobj2bdy[fruit].name).id


def trace_row(env, connection, **extra):
    h = env.connection(connection)
    return dict(time_s=float(env.data.time), extension_m=h.extension, tension_N=h.tension,
                attached=h.attached, overload_time_s=h.overload_time_s, **extra)


def run_fixed(dt=.002):
    env, fruit, connection = fixed_fixture(dt)
    body = body_id(env, fruit)
    rows, support = [], None
    for i in range(round(4 / dt)):
        t = i * dt
        # Gravity support, then a known 1 N load (below threshold), then overload.
        load = 0. if t < 1.5 else 1. if t < 2.5 else 7. if t < 3.0 else 0.
        if not env.connection(connection).attached:
            load = 0.  # released test load; subsequent motion is gravity alone
        env.data.xfrc_applied[body, 2] = -load
        env.runtime.step()
        rows.append(trace_row(env, connection, test_load_N=load,
                              fruit_z_m=float(env.data.xpos[body, 2])))
        if i == round(1.5 / dt) - 1:
            support = env.connection(connection).report()
    return env, rows, dict(gravity_support=support, final=env.connection(connection).report())


def run_branch(dt=.002):
    env, plant = branch_fixture()
    env.model.opt.timestep = dt
    env.runtime.step(round(2 / dt))
    initial_q = env.data.qpos[:4].copy()
    fruit = plant.fruits['orange_000']
    body = body_id(env, fruit)
    rows = []
    for i in range(round(3 / dt)):
        loading = i * dt < 1 and env.connection(plant.fruit_connections['orange_000']).attached
        env.data.xfrc_applied[body, :3] = (0, -12. if loading else 0., 0)
        env.runtime.step()
        rows.append(trace_row(env, plant.fruit_connections['orange_000'],
            branch_displacement_rad=float(np.linalg.norm(env.data.qpos[:4] - initial_q)),
            fruit_y_m=float(env.data.xpos[body, 1])))
    env.sync_scene()
    return env, rows, {key: env.connection(c).report() for key, c in plant.fruit_connections.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {'mujoco_version': mujoco.__version__, 'source': 'synthetic demo parameters'}
    for name, run in [('fixed', run_fixed), ('branch', run_branch)]:
        env, rows, report = run()
        reports[name] = report
        env.save(args.output / f'{name}.xml')
        with (args.output / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / 'attachment_report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
