"""Before testing stems: a real FAFU gripper lifts an untethered fruit off a stand.

One native vertical test-stage servo carries the normal two-finger gripper.
There are no tendons, equalities, fruit mounts, gravity compensation or assists.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from wrs import wss, wssop, wuc, wum
from wrs.robots.base.mech_structure import MechStruct, Link, Joint
from wrs.robots.base.mech_base import MechBase
from wrs.physics.inertial import inertia_sphere
from wrs.physics.mj_env import MJEnv
from agriculture.config import load_config
from examples.agriculture.robot_citrus_interaction import _ServoGripper
from examples.agriculture.fruit_harvest import DEFAULT_CONFIG
from examples.agriculture.harvest_diagnostics import HarvestDiagnostics, configure_grasp_contacts


def run_free_grasp(config=None):
    config = load_config(DEFAULT_CONFIG) if config is None else config
    settings = config['harvest_demo']
    p = settings['free_grasp_fixture']
    scene = wss.Scene()
    stand = wssop.box(pos=(0, 0, p['stand_top_m'] - p['stand_thickness_m'] / 2),
        xyz_lengths=(p['stand_width_m'], p['stand_width_m'], p['stand_thickness_m']),
        collision_type=wuc.CollisionType.AABB, name='test_stand')
    stand.add_to_scene(scene)
    radius = p['fruit_radius_m']
    fruit = wssop.icosphere(pos=(0, 0, p['stand_top_m'] + radius), radius=radius,
        is_floating=True, collision_type=wuc.CollisionType.SPHERE, name='orange_000')
    mass = config['fruit_attachment']['density_kg_m3'] * 4 / 3 * np.pi * radius ** 3
    fruit.set_inertia(inertia_sphere(mass, radius), np.zeros(3), mass)
    fruit.add_to_scene(scene)
    root, carrier = Link(), Link()
    carrier.set_inertia(inertia_sphere(p['stage_mass_kg'], p['stage_radius_m']), np.zeros(3), p['stage_mass_kg'])
    structure = MechStruct()
    structure.add_jnt(Joint(wuc.JntType.PRISMATIC, root, carrier, (0, 0, 1),
                            lmt_lo=0., lmt_up=p['lift_m'] * 2))
    # The stage axes stay world-aligned; the mounting owns the gripper orientation.
    stage = MechBase(structure=structure, is_floating=False)
    grip = _ServoGripper()
    grip.set_opening(grip.jaw_range[1])
    rotation = wum.rotmat_from_normal((0, 1, 0)) @ wum.rotmat_from_axangle((0, 0, 1), -np.pi / 2)
    origin = fruit.pos - rotation @ grip.tcp('grasp_center').loc_tf[:3, 3]
    origin += np.array([0, p['grasp_depth_m'], 0])
    stage.mount(grip, stage.runtime_lnks[-1], wum.tf_from_pos_rotmat(origin, rotation), update=True)
    stage.add_to_scene(scene)
    env = MJEnv(scene, require_ctrl=True)
    model = env.model
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    acts = {}
    for mech in (stage, grip):
        acts[mech] = np.array([np.flatnonzero(model.actuator_trnid[:, 0] ==
            model.joint(env.sync.mecj2jnt[mech, i].name).id).item() for i in range(mech.ndof)])
    for ids, kp, kv, limit in [(acts[stage], p['stage_kp_N_m'], p['stage_kv_N_s_m'], p['stage_force_N']),
            (acts[grip], settings['finger_kp_N_m'], settings['finger_kv_N_s_m'], settings['finger_force_N'])]:
        model.actuator_gainprm[ids, 0] = kp
        model.actuator_biasprm[ids, 1:3] = [-kp, -kv]
        model.actuator_forcelimited[ids] = True
        model.actuator_forcerange[ids] = [-limit, limit]
    configure_grasp_contacts(env, grip, [fruit], settings)
    demo = SimpleNamespace(env=env, gripper=grip, tcp=grip.tcp('grasp_center'), gripper_actuators=acts[grip],
        plant=SimpleNamespace(fruits={'orange_000': fruit}, fruit_connections={}, mech=SimpleNamespace(ndof=0)),
        harvest_settings=settings, gripper_target=0., gripper_command=float(grip.jaw_range[1]),
        _initial_plant_q=np.zeros(0))
    monitor = HarvestDiagnostics(demo, 'orange_000')
    dt = env.get_timestep()
    milestones = {}
    for phase, duration in [('close', settings['grasp_wait_s']), ('lift', p['lift_duration_s']),
                            ('hold', settings['hold_after_pull_s']), ('release', settings['release_wait_s'])]:
        monitor.phase = phase
        demo.gripper_target = float(grip.jaw_range[1]) if phase == 'release' else 0.
        if phase == 'lift':
            monitor.mark_reference()
        for i in range(round(duration / dt)):
            width_step = settings['opening_speed_m_s'] * dt
            demo.gripper_command += np.clip(demo.gripper_target - demo.gripper_command, -width_step, width_step)
            env.ctrl[acts[grip]] = demo.gripper_command / 2
            env.ctrl[acts[stage]] = p['lift_m'] * min(1., (i + 1) * dt / duration) if phase == 'lift' else (
                0. if phase == 'close' else p['lift_m'])
            env.runtime.step()
            env.runtime.forward()  # fixture has no connection observer; read current poses/contacts
            monitor.sample()
        milestones[phase] = monitor.last
    env.sync_scene()
    success = (milestones['hold']['grasp_state'] == 'bilateral_stable' and
               milestones['hold']['fruit_z_m'] > p['stand_top_m'] + radius + p['lift_m'] * .8 and
               milestones['release']['grasp_state'] == 'unsecured' and
               milestones['release']['fruit_z_m'] < p['stand_top_m'] + radius + .01)
    return env, monitor.rows, dict(success=success, milestones=milestones,
        tendon_count=model.ntendon, equality_count=model.neq, fruit_mass_kg=mass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env, rows, report = run_free_grasp()
    env.save(args.output / 'free_grasp.xml')
    (args.output / 'free_grasp_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output / 'free_grasp.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
