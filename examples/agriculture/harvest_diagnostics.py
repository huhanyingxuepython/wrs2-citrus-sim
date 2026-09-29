"""Target-specific measurements; bilateral contact is not proof of stable grasp."""
import mujoco
import numpy as np


def configure_grasp_contacts(env, gripper, fruits, settings):
    """Finite soft-finger contact profile, local to this example/environment.

    Elliptic cones/impratio reduce soft-contact creep, without changing Coulomb
    friction or adding holding forces. condim=4 includes bounded spin friction
    at a fingertip contact; it is not a stem bending/torsion constraint.
    """
    model = env.model
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = settings['contact_impratio']
    bodies = [model.body(env.sync.rutl2bdy[link].name).id for link in gripper.runtime_lnks]
    bodies += [model.body(env.sync.sobj2bdy[fruit].name).id for fruit in fruits]
    ids = np.flatnonzero(np.isin(model.geom_bodyid, bodies))
    model.geom_condim[ids] = settings['contact_dimension']
    model.geom_friction[ids, 0] = settings['sliding_friction']
    model.geom_friction[ids, 1] = settings['torsional_friction_m']
    model.geom_solref[ids, 0] = settings['contact_time_constant_s']


class HarvestDiagnostics:
    def __init__(self, demo, fruit_id):
        self.demo, self.fruit_id = demo, fruit_id
        env = demo.env
        self.fruit = demo.plant.fruits[fruit_id]
        self.fruit_body = env.model.body(env.sync.sobj2bdy[self.fruit].name).id
        self.tcp_body = env.model.body(env.sync.rutl2bdy[demo.tcp.parent_lnk].name).id
        self.fingers = {link.name: env.model.body(env.sync.rutl2bdy[link].name).id
                        for link in demo.gripper.runtime_lnks if link.name in ('left_finger', 'right_finger')}
        self.finger_qadr = [env.model.jnt_qposadr[env.model.actuator_trnid[a, 0]]
                           for a in demo.gripper_actuators]
        self.finger_actuators = {key: next(a for a in demo.gripper_actuators
            if env.model.jnt_bodyid[env.model.actuator_trnid[a, 0]] == body)
            for key, body in self.fingers.items()}
        self.plant_qadr = np.array([env.model.jnt_qposadr[env.model.joint(
            env.sync.mecj2jnt[demo.plant.mech, i].name).id] for i in range(demo.plant.mech.ndof)], dtype=int)
        self.reference = None
        self.rows = []
        self.recording = True
        self.last = None
        self.phase = 'idle'
        self._jacp = np.zeros((3, env.model.nv))
        self._jacr = np.zeros((3, env.model.nv))

    def _velocity(self, point, body):
        e = self.demo.env
        mujoco.mj_jac(e.model, e.data, self._jacp, self._jacr, point, body)
        return self._jacp @ e.data.qvel, self._jacr @ e.data.qvel

    def sample(self, demo=None, *, record=True):
        d = self.demo
        model, data = d.env.model, d.env.data
        b = self.tcp_body
        parent_r = data.xmat[b].reshape(3, 3)
        tcp_p = data.xpos[b] + parent_r @ d.tcp.loc_tf[:3, 3]
        tcp_r = parent_r @ d.tcp.loc_tf[:3, :3]
        fruit_p = data.xpos[self.fruit_body]
        delta = fruit_p - tcp_p
        relative = tcp_r.T @ delta
        vt, wt = self._velocity(tcp_p, b)
        vf, _ = self._velocity(fruit_p, self.fruit_body)
        relative_v = tcp_r.T @ (vf - vt - np.cross(wt, delta))
        loads = {key: 0. for key in self.fingers}
        contacts = {key: 0 for key in self.fingers}
        force = np.zeros(6)
        for i, contact in enumerate(data.contact):
            bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
            if self.fruit_body not in bodies:
                continue
            for key, body in self.fingers.items():
                if body in bodies:
                    mujoco.mj_contactForce(model, data, i, force)
                    loads[key] += max(0., float(force[0]))
                    contacts[key] += 1
        displacement = 0. if self.reference is None else float(np.linalg.norm(relative - self.reference))
        speed = float(np.linalg.norm(relative_v))
        bilateral = len(loads) == 2 and all(value > d.harvest_settings['minimum_contact_load_N'] for value in loads.values())
        slipping = (displacement > d.harvest_settings['slip_displacement_m'] or
                    speed > d.harvest_settings['held_relative_speed_m_s'])
        grasp_state = ('bilateral_sliding' if slipping else 'bilateral_stable') if bilateral else 'unsecured'
        c = d.plant.fruit_connections.get(self.fruit_id)
        h = d.env.connection(c) if c is not None else None
        row = dict(time_s=float(data.time), phase=self.phase, grasp_state=grasp_state,
            attachment_state=('attached' if h.attached else 'detached') if h else 'absent',
            extension_m=h.extension if h else 0., tension_N=h.tension if h else 0.,
            overload_time_s=h.overload_time_s if h else 0.,
            opening_actual_m=float(np.sum(data.qpos[self.finger_qadr])),
            opening_target_m=d.gripper_target, opening_command_m=d.gripper_command,
            relative_displacement_m=displacement, relative_speed_m_s=speed,
            fruit_x_m=float(fruit_p[0]), fruit_y_m=float(fruit_p[1]), fruit_z_m=float(fruit_p[2]),
            branch_displacement_rad=float(np.linalg.norm(data.qpos[self.plant_qadr] - d._initial_plant_q)))
        for axis, p, v in zip('xyz', relative, relative_v):
            row[f'fruit_tcp_{axis}_m'] = float(p)
            row[f'fruit_tcp_v{axis}_m_s'] = float(v)
        for key, a in self.finger_actuators.items():
            adr = model.jnt_qposadr[model.actuator_trnid[a, 0]]
            row[f'{key}_opening_m'] = float(data.qpos[adr])
            row[f'{key}_target_m'] = d.gripper_target / 2
            row[f'{key}_contacts'] = contacts[key]
            row[f'{key}_normal_N'] = loads[key]
            row[f'{key}_effort_N'] = float(data.actuator_force[a])
            row[f'{key}_force_limit_N'] = float(model.actuator_forcerange[a, 1])
        self.last = row
        if record and self.recording:
            self.rows.append(row)
        return row

    def mark_reference(self):
        row = self.sample()
        self.reference = np.array([row[f'fruit_tcp_{axis}_m'] for axis in 'xyz'])
