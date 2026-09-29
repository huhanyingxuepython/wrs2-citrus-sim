"""FAFU physical contact grasp, slow pull-off, weak-grip slip and release.

Only servo targets change during a sequence. Rupture belongs to MJRuntime, not
this controller. All contact/attachment parameters are synthetic demo settings.
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
from wrs import wvw
from agriculture.config import CONFIG_DIR, load_config
from examples.agriculture.robot_citrus_interaction import RobotPlantInteraction, add_controls
from examples.agriculture.harvest_diagnostics import HarvestDiagnostics

DEFAULT_CONFIG = CONFIG_DIR / 'presets/lab_citrus_harvest.json'
CASES = ('pull_hold', 'weak_grip', 'release')


class HarvestSequence:
    def __init__(self, demo):
        if demo.settings['fruit_id'] not in demo.plant.fruit_connections:
            raise ValueError('HarvestSequence requires breakable_axial on the target fruit')
        self.demo = demo
        self.settings = demo.harvest_settings
        self.monitor = HarvestDiagnostics(demo, demo.settings['fruit_id'])
        self.monitor.recording = False
        self.running = False
        self.case = None
        self.milestones = {}
        self._deadline = 0.
        self._approach_speed = demo.settings['cartesian_speed_m_s']
        demo.substep_observers.append(self.after_substep)
        demo.reset_observers.append(self.after_reset)

    def after_reset(self, demo):
        self.running = False
        self.case, self.milestones = None, {}
        self.monitor.recording = False
        self.monitor.phase = 'idle'
        self.monitor.reference, self.monitor.last = None, None
        self.monitor.rows.clear()
        demo.settings['cartesian_speed_m_s'] = self._approach_speed

    def start(self, case):
        if case not in CASES:
            raise ValueError(f'case must be one of {CASES}')
        self.running = False
        self.demo.reset()
        self.demo.paused = False
        self.case, self.milestones = case, {}
        self.monitor.rows.clear()
        self.monitor.reference = None
        self.monitor.recording = True
        p = self.settings
        force = p['weak_finger_force_N'] if case == 'weak_grip' else p['finger_force_N']
        self.demo.env.model.actuator_forcerange[self.demo.gripper_actuators] = [-force, force]
        self.demo.settings['cartesian_speed_m_s'] = self._approach_speed
        self.demo.set_gripper_opening(self.demo.gripper.jaw_range[1])
        self.running = True
        self._move_to_fruit('pregrasp', p['pregrasp_offset_world_m'], p['approach_wait_s'])

    def cancel(self):
        self.running = False
        self.monitor.recording = False
        self.monitor.phase = 'idle'
        self.demo.stop_motion()
        force = self.settings['finger_force_N']
        self.demo.env.model.actuator_forcerange[self.demo.gripper_actuators] = [-force, force]
        self.demo.settings['cartesian_speed_m_s'] = self._approach_speed

    def _phase(self, name, duration):
        self.monitor.phase = name
        self._deadline = float(self.demo.env.data.time) + duration

    def _move_to_fruit(self, phase, offset, wait):
        # Native pose, independent of when the renderer last published a frame.
        center = self.demo.env.data.xpos[self.monitor.fruit_body].copy()
        self.demo.set_cartesian_target(center + offset)
        self._phase(phase, self.demo.motion_duration + wait)

    def after_substep(self, demo):
        row = self.monitor.sample()
        if not self.running or demo.env.data.time + 1e-10 < self._deadline:
            return
        phase, p = self.monitor.phase, self.settings
        self.milestones[phase] = row
        if phase == 'pregrasp':
            self._move_to_fruit('enclose', p['grasp_offset_world_m'], p['enclose_wait_s'])
        elif phase == 'enclose':
            demo.set_gripper_opening(0.)
            self._phase('close', p['grasp_wait_s'])
        elif phase == 'close':
            self.monitor.mark_reference()
            demo.settings['cartesian_speed_m_s'] = p['pull_speed_m_s']
            direction = np.asarray(p['pull_direction_world'], dtype=float)
            direction /= np.linalg.norm(direction)
            demo.set_cartesian_target(demo.command_position + direction * p['pull_distance_m'])
            self._phase('pull', demo.motion_duration)
        elif phase == 'pull':
            self._phase('hold', p['hold_after_pull_s'])
        elif phase == 'hold' and self.case == 'release':
            demo.set_gripper_opening(demo.gripper.jaw_range[1])
            self._phase('release', p['release_wait_s'])
        else:
            self.running = False
            self.monitor.recording = False
            self.monitor.phase = 'done'

    def report(self):
        d, monitor = self.demo, self.monitor
        connection = d.plant.fruit_connections[monitor.fruit_id]
        attached = d.env.connection(connection).report()
        hold = self.milestones.get('hold', {})
        detached_held = hold.get('attachment_state') == 'detached' and hold.get('grasp_state') == 'bilateral_stable'
        slipped_first = (hold.get('attachment_state') == 'attached' and
                         hold.get('grasp_state') == 'unsecured' and
                         hold.get('relative_displacement_m', 0) > self.settings['slip_displacement_m'])
        ground_bodies = {d.env.model.body(d.env.sync.sobj2bdy[o].name).id
                         for o in d.scene.sobjs if o.name == 'ground'}
        ground_contacts = sum(monitor.fruit_body in pair and bool(ground_bodies & pair)
            for pair in ({int(d.env.model.geom_bodyid[c.geom1]), int(d.env.model.geom_bodyid[c.geom2])}
                         for c in d.env.data.contact))
        success = slipped_first if self.case == 'weak_grip' else detached_held
        if self.case == 'release':
            success = success and ground_contacts > 0 and monitor.last['grasp_state'] == 'unsecured'
        return dict(case=self.case, success=bool(success), physics_dt_s=d.env.get_timestep(),
            parameter_source=self.settings['parameter_source'],
            attachment=attached, milestones=self.milestones, final=monitor.last,
            target_fruit_ground_contacts=int(ground_contacts),
            peak_tension_N=max((r['tension_N'] for r in monitor.rows), default=0.),
            robot_actuator_force_limits=d.env.model.actuator_forcerange[d.actuators].tolist(),
            robot_actuator_force_limited=d.env.model.actuator_forcelimited[d.actuators].tolist(),
            robot_actuator_effort=d.env.data.actuator_force[d.actuators].tolist(),
            all_attachments={key: d.env.connection(c).report() for key, c in d.plant.fruit_connections.items()})

    def save_trace(self, path):
        with Path(path).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.monitor.rows[0]))
            writer.writeheader()
            writer.writerows(self.monitor.rows)


def run_case(sequence, case, frame_dt=1 / 30):
    sequence.start(case)
    while sequence.running:
        sequence.demo.step(frame_dt)
    return sequence.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--case', choices=(*CASES, 'all'), default='all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--capture-rgbd', action='store_true')
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    demo = RobotPlantInteraction(load_config(args.config))
    sequence = HarvestSequence(demo)
    reports = dict(mujoco_version=mujoco.__version__, cases={})
    try:
        if args.headless:
            for case in CASES if args.case == 'all' else (args.case,):
                reports['cases'][case] = run_case(sequence, case)
                if args.capture_rgbd:
                    demo.capture_rgbd()
                    reports['cases'][case]['camera'] = demo.rgbd.statistics()
                if args.output:
                    sequence.save_trace(args.output / f'{case}.csv')
        else:
            p = demo.settings
            base = wvw.World(cam_pos=p['camera_pos'], cam_lookat_pos=p['camera_lookat'], port=args.port)
            base.set_scene(demo.scene)
            base.set_caption('FAFU / physical grasp and axial fruit detachment')
            panel, _ = add_controls(base, demo)
            panel.configure(title='FAFU / Axial harvest',
                description='Scroll to Harvest for synthetic pull-off tests. Cancel a sequence before manual jogging.')
            for case, label in [('pull_hold', 'Pull off and hold'), ('weak_grip', 'Weak grip: slip first'),
                                ('release', 'Pull off, then release')]:
                panel.add_button(case, label=label, group='Harvest', on_click=lambda case=case: sequence.start(case))
            panel.add_button('cancel', label='Cancel sequence', group='Harvest', on_click=sequence.cancel)
            for key, label in [('phase', 'Sequence'), ('attachment', 'Stem'), ('grasp', 'Grasp'),
                               ('tension', 'Anchor extension / tension'), ('loads', 'Left / right normal load')]:
                panel.add_label(key, label=label, group='Harvest')

            def refresh(dt):
                r = sequence.monitor.sample(record=False)
                panel.set_value('phase', sequence.monitor.phase)
                panel.set_value('attachment', r['attachment_state'])
                panel.set_value('grasp', r['grasp_state'])
                panel.set_value('tension', f'{r["extension_m"] * 1000:.1f} mm / {r["tension_N"]:.2f} N')
                panel.set_value('loads', f'{r["left_finger_normal_N"]:.2f} / {r["right_finger_normal_N"]:.2f} N')

            base.schedule_interval(lambda dt: demo.step(min(dt, p['max_frame_dt'])), 1 / p['control_hz'])
            base.schedule_interval(refresh, 1 / p['status_hz'])
            if args.case != 'all':
                sequence.start(args.case)
            if args.duration:
                base.schedule_once(lambda dt: base.close(), args.duration)
            base.run()
            if sequence.case:
                reports['cases'][sequence.case] = sequence.report()
                if args.output and sequence.monitor.rows:
                    sequence.save_trace(args.output / f'{sequence.case}.csv')
    finally:
        demo.close()
    if args.output:
        (args.output / 'harvest_report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
