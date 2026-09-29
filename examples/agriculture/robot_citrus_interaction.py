"""Cartesian jogging of a FAFU arm in contact with Citrus V2 foliage/fruit.

TCP pose paths and gripper opening become rate-limited position-servo targets.
The arm and fingers have actuators; the plant retains its passive V2 joints.
The wrist camera switches between raw-like agriculture noise and clean depth.
"""
from collections import deque
from pathlib import Path
import argparse
import json
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from wrs import wss, wssop, wuc, wum, wvw
from wrs.robots.manipulators.fafu import FAFURobotArm
from wrs.robots.end_effectors.fafu_gripper import FAFUGripper
from wrs.robots.base.kine.numik import NumIKSolver
from wrs.physics.mj_env import MJEnv
from wrs.viewer.web_ui import Anchor
from agriculture.config import CONFIG_DIR, load_config
from agriculture.bunker_cart import add_bunker_cart
from agriculture.cart_irm import CartIRM, DEFAULT_CACHE, IRMResult, build_irm, unicycle_segments
from agriculture.generator import generate
from agriculture.dynamics import PlantDynamicsSpec
from agriculture.dynamic import DynamicPlantBuilder
from examples.agriculture.fafu_d405 import MountedD405

DEFAULT_CONFIG = CONFIG_DIR / 'presets/lab_citrus_robot.json'


class _ServoGripper(FAFUGripper):
    """Simulate both fingers; equal targets allow asymmetric contact response."""

    _structure = None

    @classmethod
    def _build_structure(cls):
        structure = super()._build_structure()
        # Kinematic mimic would hide one finger's independently simulated pose.
        for joint in structure.jnts:
            joint.mmc = None
        structure.compile()
        return structure


class RobotPlantInteraction:
    """One example scene/controller, also usable in a headless contact smoke test."""

    def __init__(self, config=None):
        config = load_config(DEFAULT_CONFIG) if config is None else config
        self.settings = p = config['robot_demo']
        self.harvest_settings = config.get('harvest_demo', {})
        self.substep_observers = []
        self.reset_observers = []
        self.scene = wss.Scene()
        size = np.asarray(p['ground_size'])
        ground = wssop.box(pos=(*p['ground_center_xy'], -size[2] / 2), xyz_lengths=size,
                          rgb=p['ground_color'], collision_type=wuc.CollisionType.AABB,
                          name='ground')
        ground.add_to_scene(self.scene)
        spec = generate(config)
        plant_dynamics = PlantDynamicsSpec.from_config(spec, config)
        if p.get('foliage_dual_axis', False):
            for cluster in plant_dynamics.clusters:
                if cluster.profile == 'foliage':
                    cluster.dof = 2
            plant_dynamics.validate(spec)
        self.plant = DynamicPlantBuilder(config).build(spec, plant_dynamics)
        self.plant.set_pos_rotmat(pos=(0.0, 0.0, float(p.get('plant_lift_m', 0.0))))
        plant_summary = self.plant.summary()
        self.foliage_coverage = {key: plant_summary[key] for key in
            ('foliage_contact_leaf_count', 'visual_only_leaf_count', 'foliage_proxy_count', 'foliage_proxy_object_count')}
        self.plant.add_to_scene(self.scene)
        # Pick the demonstration poses from the gravity-loaded tree, not q=0.
        warmup = MJEnv(self.scene, require_ctrl=True)
        warmup.step(p['settle_seconds'])
        # NumIK uses explicit neighbouring seeds, with no SELIK database writes.
        self.cart = add_bunker_cart(self.scene, p)
        self.robot = FAFURobotArm(pos=p['base_position'], solver=NumIKSolver,
                                  rotmat=wum.rotmat_from_axangle((0, 0, 1), np.deg2rad(p['base_yaw_deg'])))
        compiled = self.robot.structure.compiled
        self.limits = np.array([compiled.jlmt_low_by_idx, compiled.jlmt_high_by_idx], dtype=float)
        if self.cart is None and p['base_position'][2] > 0:
            pedestal = wssop.box(pos=(*p['base_position'][:2], p['base_position'][2] / 2),
                xyz_lengths=(p['pedestal_width'], p['pedestal_width'], p['base_position'][2]),
                rgb=p['pedestal_color'], collision_type=wuc.CollisionType.AABB, name='arm_pedestal')
            pedestal.add_to_scene(self.scene)
        self.gripper = _ServoGripper()
        self.gripper.set_opening(p['gripper']['opening_m'])
        self.robot.mount(self.gripper, self.robot.tcp('flange').parent_lnk, update=True)
        if self.cart is not None:
            self.cart.mount(self.robot, self.cart.body_lnk, self.cart.arm_loc_tf, update=True)
        self.tcp = self.gripper.tcp('grasp_center')
        self.rgbd = MountedD405(self.robot, self.scene, p['d405'], exclude=self.plant.foliage_proxy_visuals)
        self.camera = self.rgbd.camera
        self.presets = self._make_presets()
        self._ready_q = self._solve_ready_q()
        self._home_q = self._solve_home_q()
        self.robot.fk(self._ready_q)
        self.robot.add_to_scene(self.scene)
        # Palm and the wrist housing meet at their mechanical mounting surface.
        excludes = [(self.gripper.runtime_root_lnk, self.robot.runtime_lnks[-2])]
        self.env = MJEnv(self.scene, require_ctrl=True, extra_excludes=excludes)
        model = self.env.model
        # Explicit mapping: scene insertion order is not an actuator ordering API.
        joint_ids = [model.joint(self.env.sync.mecj2jnt[self.robot, i].name).id
                     for i in range(self.robot.ndof)]
        self.actuators = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == jid).item()
                                   for jid in joint_ids])
        finger_ids = [model.joint(self.env.sync.mecj2jnt[self.gripper, i].name).id
                      for i in range(self.gripper.ndof)]
        self.gripper_actuators = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == jid).item()
                                          for jid in finger_ids])
        if model.nu != len(self.actuators) + len(self.gripper_actuators):
            raise RuntimeError('Only the arm and fingers may have actuators')
        # Example-specific native position-servo tuning; no changes to robot defaults.
        model.actuator_gainprm[self.actuators, 0] = p['servo_kp']
        model.actuator_biasprm[self.actuators, 1] = -np.asarray(p['servo_kp'])
        model.actuator_biasprm[self.actuators, 2] = -np.asarray(p['servo_kv'])
        finger_kp = self.harvest_settings.get('finger_kp_N_m', 800.)
        finger_kv = self.harvest_settings.get('finger_kv_N_s_m', 10.)
        finger_force = self.harvest_settings.get('finger_force_N', 20.)
        model.actuator_gainprm[self.gripper_actuators, 0] = finger_kp
        model.actuator_biasprm[self.gripper_actuators, 1:3] = [-finger_kp, -finger_kv]
        model.actuator_forcelimited[self.gripper_actuators] = True
        model.actuator_forcerange[self.gripper_actuators] = [-finger_force, finger_force]
        if self.harvest_settings:
            from examples.agriculture.harvest_diagnostics import configure_grasp_contacts
            configure_grasp_contacts(self.env, self.gripper, self.plant.fruit_objects, self.harvest_settings)
            limits = np.asarray(self.harvest_settings['arm_force_limits_Nm'])
            model.actuator_forcelimited[self.actuators] = True
            model.actuator_forcerange[self.actuators] = np.column_stack((-limits, limits))
        foliage_contact = p.get('foliage_contact') or {}
        foliage_objects = [
            obj for objects in self.plant.foliage_proxies.values() for obj in objects
        ]
        foliage_body_ids = [
            model.body(node.name).id
            for obj in foliage_objects
            if (node := self.env.sync.sobj2bdy.get(obj)) is not None
        ]
        self.foliage_geom_ids = np.flatnonzero(
            np.isin(model.geom_bodyid, foliage_body_ids))
        if len(self.foliage_geom_ids) and foliage_contact:
            ids = self.foliage_geom_ids
            model.geom_condim[ids] = int(foliage_contact.get('contact_dimension', 3))
            model.geom_friction[ids, 0] = float(
                foliage_contact.get('sliding_friction', 0.25))
            model.geom_friction[ids, 1] = float(
                foliage_contact.get('torsional_friction_m', 0.001))
            model.geom_solref[ids, 0] = float(
                foliage_contact.get('contact_time_constant_s', 0.04))
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.command = self._ready_q.copy()
        self.command_position = self.presets['ready'].copy()
        self.command_rotation = self.tool_rotation.copy()
        self.gripper_command = p['gripper']['opening_m']
        self.stop_motion()
        self.env.ctrl[self.actuators] = self.command
        self.env.ctrl[self.gripper_actuators] = self.gripper_command / 2
        self._cart_hold_qs = None if self.cart is None else self.cart.qs.copy()
        self._cart_lin = 0.0
        self._cart_ang = 0.0
        self._cart_cmd_remaining = 0.0
        self._cart_segments = deque()
        self._cart_irm = None
        self._cart_irm_goal = None
        self._cart_grasp_seed = None
        self._pending_gripper_close = False
        self._fruit_pull_deadline = None
        self._fruit_pull_active = False
        self._irm_visuals = []
        self._irm_visual_enabled = True
        self._last_irm_visual = None
        self._cart_dof_adr = []
        if self.cart is not None:
            model = self.env.model
            for mecba, qadr in ((m, adr) for (m, _j), adr in self.env.sync._qpos_map.items() if m is self.cart):
                jid = int(np.flatnonzero(model.jnt_qposadr == qadr)[0])
                self._cart_dof_adr.append(int(model.jnt_dofadr[jid]))
        self.paused = False
        self.show_forces = False
        self._remainder = 0.0
        self.robot_bodies = {model.body(self.env.sync.rutl2bdy[link].name).id
                             for link in (*self.robot.runtime_lnks, *self.gripper.runtime_lnks)}
        self.cart_bodies = set() if self.cart is None else {
            model.body(self.env.sync.rutl2bdy[link].name).id
            for link in self.cart.runtime_lnks if link in self.env.sync.rutl2bdy
        }
        self.plant_bodies = {}
        for role, objects in self.plant.semantic_groups.items():
            for obj in objects:
                node = self.env.sync.rutl2bdy.get(obj) or self.env.sync.sobj2bdy.get(obj)
                self.plant_bodies[model.body(node.name).id] = role
        self.contacts = {}
        self.contact_samples = {}
        self.step(p['settle_seconds'])
        self._initial_state = self.env.snapshot()
        self._initial_time = self.env.data.time
        self._initial_fruit = {key: obj.pos.copy() for key, obj in self.plant.fruits.items()}
        self._initial_plant_q = self.plant.mech.qs.copy()
        self._initial_base_position = list(self.settings['base_position'])
        self._initial_ready = self.presets['ready'].copy()
        self._initial_cart_qs = None if self.cart is None else self.cart.qs.copy()
        self.reset()

    def _make_presets(self):
        p = self.settings
        direction = np.asarray(p['approach_direction'], dtype=float)
        direction /= np.linalg.norm(direction)
        self.tool_rotation = (wum.rotmat_from_normal(direction) @
                              wum.rotmat_from_axangle((0, 0, 1), np.deg2rad(p['tool_roll_deg'])))
        fruit = self.plant.fruits[p['fruit_id']]
        proxy = self.plant.foliage_proxies[p['foliage_cluster']][p['foliage_proxy_index']]
        shape = proxy.collisions[p['foliage_shape_index']]
        shape_tf = proxy.tf @ shape.loc_tf
        extent = np.sum(np.abs(direction @ shape_tf[:3, :3]) * shape.half_extents)
        # Aim a real fingertip, not the empty grasp centre, at the surface.
        # Native finger geometry is measured in gripper-local coordinates here.
        finger = next(link for link in self.gripper.runtime_lnks
                      if link.name == p['gripper']['contact_link'])
        vertices = []
        for visual in finger.visuals:
            tf = np.linalg.inv(self.gripper.tf) @ finger.tf @ visual.loc_tf
            vertices.append(visual.geom.vs @ tf[:3, :3].T + tf[:3, 3])
        vertices = np.concatenate(vertices)
        tip_z = vertices[:, 2].max()
        tip = vertices[vertices[:, 2] >= tip_z - p['gripper']['contact_tip_band_m']].mean(axis=0)
        tip[2] = tip_z
        self.contact_point_tcp = tip - self.tcp.loc_tf[:3, 3]
        tip_offset = self.tool_rotation @ self.contact_point_tcp
        positions = dict(ready=fruit.pos + p['ready_offset'],
                         foliage=shape_tf[:3, 3] - direction * (extent - p['foliage_push_depth']) - tip_offset,
                         fruit=fruit.pos.copy())
        return {key: np.asarray(value, dtype=float) for key, value in positions.items()}

    def _base_ready_pose(self):
        """Park above the cart, toward the tree. Independent of fruit droop."""
        p = self.settings
        base = np.asarray(p['base_position'], dtype=float)
        planar = np.asarray(p['approach_direction'], dtype=float)
        planar[2] = 0.
        if np.linalg.norm(planar) < 1e-9:
            planar = np.array((0., 1., 0.))
        planar /= np.linalg.norm(planar)
        return base + planar * p.get('ready_forward_m', 0.26) + np.array(
            (0., 0., p.get('ready_up_m', 0.22)))

    def _ik_seed_list(self, seed):
        p = self.settings
        seeds = [np.asarray(seed, dtype=float)]
        for raw in p.get('ik_fallback_seeds_deg', (
                [-53, 39, 24, -5, -52, -15],
                [-45, 44, 25, 3, -44, -11],
                [-14, 160, 70, 79, -14, -3],
                [0, 80, 90, 0, -20, 0],
        )):
            seeds.append(np.deg2rad(raw))
        rng = np.random.default_rng(0)
        lo, hi = self.limits
        seeds.extend(rng.uniform(lo, hi) for _ in range(int(p.get('ik_search_seeds', 40))))
        return seeds

    def _try_ik(self, position, seed, rotation=None, max_iter=None):
        rotation = self.tool_rotation if rotation is None else rotation
        solutions = self.robot.ik(
            position, rotation, tcp=self.tcp, qs_active_init=seed,
            max_iter=self.settings['ik_max_iter'] if max_iter is None else max_iter)
        if not solutions:
            return None
        q = np.asarray(solutions[0], dtype=float)
        if (not np.isfinite(q).all() or np.any(q < self.limits[0]) or np.any(q > self.limits[1])):
            return None
        return q

    def _solve_tcp(self, position, seed, rotation=None, fallback_seeds=()):
        rotation = self.tool_rotation if rotation is None else rotation
        last_error = 'No nearby IK solution for this TCP pose'
        for qs_init in (seed, *fallback_seeds):
            q = self._try_ik(position, qs_init, rotation)
            if q is not None:
                return q
        raise ValueError(last_error)

    def _solve_ready_q(self):
        """Solve a start pose. Fruit+offset can become unreachable after the cart
        mount or gravity settle; fall back to a park above the cabinet."""
        seed = np.deg2rad(self.settings['ik_seed_deg'])
        seeds = self._ik_seed_list(seed)
        for pose_name, pose in (
                ('fruit_offset', self.presets['ready']),
                ('cart_park', self._base_ready_pose())):
            for qs_init in seeds:
                q = self._try_ik(pose, qs_init, max_iter=100)
                if q is not None:
                    self.presets['ready'] = np.asarray(pose, dtype=float)
                    return q
        raise ValueError('No nearby IK solution for this TCP pose')

    def _solve_home_q(self):
        """A cart-relative park above the cabinet, independent of fruit pose."""
        home = self._base_ready_pose()
        for seed in self._ik_seed_list(self._ready_q):
            q = self._try_ik(home, seed, max_iter=100)
            if q is not None:
                return q
        raise ValueError('No IK solution for the arm home pose')

    def set_cartesian_target(self, position, rotation=None):
        """Check the entire straight TCP path before changing any command/state.

        Short Cartesian waypoints keep seeded IK on the same branch. Linear
        interpolation between their joint solutions approximates the straight
        path while enforcing both Cartesian and joint command speed limits.
        This is not an obstacle-avoiding planner: contact is the purpose here.
        """
        p = self.settings
        goal = np.asarray(position, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError('Expected three finite world coordinates in metres')
        if np.any(goal < p['workspace_min']) or np.any(goal > p['workspace_max']):
            raise ValueError('Target outside the configured jogging workspace')
        rotation = np.asarray(self.target_rotation if rotation is None else rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
            raise ValueError('Expected a 3 by 3 rotation matrix')
        distance = np.linalg.norm(goal - self.command_position)
        angle = np.linalg.norm(wum.delta_rotvec_between_rotmats(self.command_rotation, rotation))
        if distance < 1e-9 and angle < 1e-9:
            self.stop_motion()
            return
        n = max(1, int(np.ceil(distance / p['cartesian_waypoint_m'])),
                int(np.ceil(angle / np.deg2rad(2))))
        points = np.linspace(self.command_position, goal, n + 1)
        rotations = wum.rotmat_slerp(self.command_rotation, rotation, n + 1)
        angles, times = [self.command.copy()], [0.0]
        for i, point in enumerate(points[1:], 1):
            q = self._solve_tcp(point, angles[-1], rotations[i])
            max_delta = float(np.max(np.abs(q - angles[-1])))
            if max_delta > np.deg2rad(p['ik_max_step_deg']):
                raise ValueError('IK jump near a singularity; try a smaller move or another direction')
            duration = max(np.linalg.norm(point - points[i - 1]) / p['cartesian_speed_m_s'],
                           angle / n / np.deg2rad(20),
                           max_delta / np.deg2rad(p['joint_speed_deg_s']))
            angles.append(q)
            times.append(times[-1] + duration)
        self._path_times = np.asarray(times)
        self._path_angles = np.asarray(angles)
        self._path_positions = points
        self._path_rotations = rotations
        self._path_rotvecs = np.array([wum.delta_rotvec_between_rotmats(a, b)
                                      for a, b in zip(rotations[:-1], rotations[1:])])
        self._motion_elapsed = 0.0
        self.target = angles[-1].copy()
        self.target_position = goal.copy()
        self.target_rotation = rotation.copy()
        self.motion_status = 'Moving'

    @property
    def motion_duration(self):
        return float(self._path_times[-1])

    def jog(self, axis, distance):
        if axis not in (0, 1, 2) or not np.isfinite(distance):
            raise ValueError('Jog requires axis 0/1/2 and a finite distance')
        # Keep held input at most one step ahead instead of building a long path.
        position = self.command_position.copy()
        position[axis] += distance
        self.set_cartesian_target(position, self.command_rotation)

    def jog_rotation(self, axis, angle):
        """Rotate about a world axis at the current TCP position (radians)."""
        if axis not in (0, 1, 2) or not np.isfinite(angle):
            raise ValueError('Rotation requires axis 0/1/2 and a finite angle')
        rotation = wum.rotmat_from_axangle(np.eye(3)[axis], angle) @ self.command_rotation
        self.set_cartesian_target(self.command_position, rotation)

    def set_gripper_opening(self, width):
        """Set the total opening in metres; native servos move both fingers."""
        if not np.isfinite(width) or not self.gripper.jaw_range[0] <= width <= self.gripper.jaw_range[1]:
            raise ValueError('Gripper opening must be between 0 and 0.085 metres')
        self.gripper_target = float(width)

    def stop_motion(self):
        """Hold the current servo command without teleporting the physical arm."""
        self.target = self.command.copy()
        self.target_position = self.command_position.copy()
        self.target_rotation = self.command_rotation.copy()
        self.gripper_target = self.gripper_command
        self._path_times = np.array([0.0])
        self._path_angles = self.command[None, :].copy()
        self._path_positions = self.command_position[None, :].copy()
        self._path_rotations = self.command_rotation[None, :, :].copy()
        self._path_rotvecs = np.empty((0, 3))
        self._motion_elapsed = 0.0
        if hasattr(self, '_cart_segments'):
            self._cart_segments.clear()
            self._cart_lin = 0.0
            self._cart_ang = 0.0
            self._cart_cmd_remaining = 0.0
        if hasattr(self, '_pending_gripper_close'):
            self._pending_gripper_close = False
            self._fruit_pull_deadline = None
            self._fruit_pull_active = False
        self.motion_status = 'Holding target'

    def drive_cart(self, lin_m_s, ang_rad_s, dt=0.1):
        """Hold a chassis velocity; ``step`` integrates it at the physics dt."""
        if self.cart is None:
            raise ValueError('No cart in this scene')
        if not np.isfinite(lin_m_s) or not np.isfinite(ang_rad_s) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Cart drive requires finite lin, ang and a positive dt')
        self.stop_motion()
        self._cart_lin = float(lin_m_s)
        self._cart_ang = float(ang_rad_s)
        self._cart_cmd_remaining = float(dt)
        self.motion_status = 'Driving cart'

    def drive_cart_trajectory(self, segments, *, status='Driving cart trajectory'):
        """Replace the current cart command with queued ``(v, omega, dt)`` legs."""
        if self.cart is None:
            raise ValueError('No cart in this scene')
        checked = []
        for lin_m_s, ang_rad_s, duration_s in segments:
            values = np.asarray((lin_m_s, ang_rad_s, duration_s), dtype=float)
            if not np.isfinite(values).all() or duration_s <= 0:
                raise ValueError('Cart trajectory segments require finite velocities and positive duration')
            checked.append((float(lin_m_s), float(ang_rad_s), float(duration_s)))
        if not checked:
            raise ValueError('Cart is already at the requested pose')
        self.stop_motion()
        self._cart_segments.extend(checked)
        self._start_next_cart_segment()
        self.motion_status = status

    def _start_next_cart_segment(self):
        if not self._cart_segments:
            self._cart_lin = 0.0
            self._cart_ang = 0.0
            self._cart_cmd_remaining = 0.0
            return False
        self._cart_lin, self._cart_ang, self._cart_cmd_remaining = self._cart_segments.popleft()
        return True

    def _follow_cart_mount(self):
        self.settings['base_position'] = np.asarray(self.robot.pos, dtype=float).tolist()
        self.presets['ready'] = self._base_ready_pose()
        self.command_position = np.asarray(self.tcp.pos, dtype=float)
        self.command_rotation = np.asarray(self.tcp.rotmat, dtype=float)
        self.target_position = self.command_position.copy()
        self.target_rotation = self.command_rotation.copy()

    def _lock_cart(self):
        """Keep the chassis kinematic: commanded qpos, zero slide/yaw rate."""
        self.env.sync.push_one_mecba_qpos(self.cart, self._cart_hold_qs)
        qvel = self.env.data.qvel
        for adr in self._cart_dof_adr:
            qvel[adr] = 0.0

    def _advance_cart(self, dt):
        remaining_dt = float(dt)
        moved = False
        while remaining_dt > 1e-12:
            if self._cart_cmd_remaining <= 1e-12 and not self._start_next_cart_segment():
                break
            used = min(remaining_dt, self._cart_cmd_remaining)
            self.cart.drive(self._cart_lin, self._cart_ang, used)
            self._cart_hold_qs = self.cart.qs.copy()
            self._cart_cmd_remaining -= used
            remaining_dt -= used
            moved = True
        if moved:
            self._follow_cart_mount()
        if self._cart_cmd_remaining <= 1e-12 and not self._cart_segments:
            cart_was_active = (
                abs(self._cart_lin) > 1e-12
                or abs(self._cart_ang) > 1e-12
                or self.motion_status.startswith(('Driving cart', 'IRM driving'))
            )
            self._cart_lin = 0.0
            self._cart_ang = 0.0
            self._cart_cmd_remaining = 0.0
            if cart_was_active:
                self.motion_status = 'Holding target'

    def _validate_irm_candidate(self, candidate, target_position, target_rotation):
        """Return an IK solution when a candidate cart pose clears the plant."""
        snapshot = self.env.snapshot()
        try:
            self.cart.fk(candidate.cart_qs)
            arm_qs = self._try_ik(
                target_position, candidate.seed_qs, rotation=target_rotation,
                max_iter=max(100, self.settings['ik_max_iter']))
            if arm_qs is None:
                for seed in (self.command, *self._ik_seed_list(candidate.seed_qs)[:4]):
                    arm_qs = self._try_ik(
                        target_position, seed, rotation=target_rotation,
                        max_iter=max(100, self.settings['ik_max_iter']))
                    if arm_qs is not None:
                        break
            if arm_qs is None:
                return None

            self.robot.fk(arm_qs)
            self.env.sync.push_one_mecba_qpos(self.cart, candidate.cart_qs)
            self.env.sync.push_one_mecba_qpos(self.robot, arm_qs)
            mujoco.mj_forward(self.env.model, self.env.data)
            for contact in self.env.data.contact:
                body_a, body_b = (
                    int(self.env.model.geom_bodyid[contact.geom1]),
                    int(self.env.model.geom_bodyid[contact.geom2]),
                )
                if ((body_a in self.cart_bodies and body_b in self.plant_bodies)
                        or (body_b in self.cart_bodies and body_a in self.plant_bodies)):
                    return None
            return arm_qs.copy()
        finally:
            self.env.restore(snapshot)

    def _fruit_grasp_targets(self):
        """Current gravity/physics fruit centre and an approach-line pregrasp."""
        fruit = self.plant.fruits[self.settings['fruit_id']]
        centre = np.asarray(fruit.pos, dtype=float).copy()
        direction = np.asarray(self.settings['approach_direction'], dtype=float)
        direction /= np.linalg.norm(direction)
        pregrasp = centre - direction * float(self.settings.get('fruit_pregrasp_m', 0.10))
        grasp = centre + np.asarray(
            self.harvest_settings.get('grasp_offset_world_m', (0, 0, 0)),
            dtype=float)
        return grasp, pregrasp

    def _solve_line_qpath(self, start_position, goal_position, start_qs, rotation):
        """Continuous seeded IK along one Cartesian line, including both ends."""
        start = np.asarray(start_position, dtype=float)
        goal = np.asarray(goal_position, dtype=float)
        count = max(
            1,
            int(np.ceil(np.linalg.norm(goal - start) / self.settings['cartesian_waypoint_m'])),
        )
        points = np.linspace(start, goal, count + 1)
        qpath = [np.asarray(start_qs, dtype=float).copy()]
        for point in points[1:]:
            q = self._try_ik(point, qpath[-1], rotation=rotation)
            if q is None:
                return None
            if np.max(np.abs(q - qpath[-1])) > np.deg2rad(self.settings['ik_max_step_deg']):
                return None
            qpath.append(q)
        return np.asarray(qpath)

    def _plan_grasp_at_candidate(self, candidate, centre, pregrasp):
        """Check cart clearance and pregrasp-to-centre continuity at one pose."""
        snapshot = self.env.snapshot()
        try:
            self.cart.fk(candidate.cart_qs)
            pregrasp_q = None
            seeds = (candidate.seed_qs, self.command, *self._ik_seed_list(candidate.seed_qs))
            for seed in seeds:
                pregrasp_q = self._try_ik(
                    pregrasp, seed, rotation=self.tool_rotation,
                    max_iter=max(100, self.settings['ik_max_iter']))
                if pregrasp_q is not None:
                    break
            if pregrasp_q is None:
                return None
            approach_qpath = self._solve_line_qpath(
                pregrasp, centre, pregrasp_q, self.tool_rotation)
            if approach_qpath is None:
                return None

            self.robot.fk(approach_qpath[-1])
            self.env.sync.push_one_mecba_qpos(self.cart, candidate.cart_qs)
            self.env.sync.push_one_mecba_qpos(self.robot, approach_qpath[-1])
            mujoco.mj_forward(self.env.model, self.env.data)
            for contact in self.env.data.contact:
                body_a, body_b = (
                    int(self.env.model.geom_bodyid[contact.geom1]),
                    int(self.env.model.geom_bodyid[contact.geom2]),
                )
                if ((body_a in self.cart_bodies and body_b in self.plant_bodies)
                        or (body_b in self.cart_bodies and body_a in self.plant_bodies)):
                    return None
            return pregrasp_q.copy(), approach_qpath[-1].copy()
        finally:
            self.env.restore(snapshot)

    def _clear_irm_visualization(self):
        for visual in self._irm_visuals:
            visual.remove_from_scene(self.scene)
        self._irm_visuals.clear()

    def _show_irm_visualization(self, candidates, selected_index, centre, start_qs):
        """Draw candidate parking poses and the selected unicycle translation."""
        self._clear_irm_visualization()
        self._last_irm_visual = (
            tuple(candidates), int(selected_index), np.asarray(centre).copy(),
            np.asarray(start_qs).copy())
        if not self._irm_visual_enabled:
            return

        z = 0.035
        for index, candidate in enumerate(candidates[:30]):
            q = candidate.cart_qs
            if index < selected_index:
                color = (0.90, 0.18, 0.12)  # rejected complete-grasp route
            elif index == selected_index:
                color = (0.10, 0.85, 0.20)  # selected
            else:
                color = (0.12, 0.45, 0.95)  # untested/other IRM candidate
            radius = 0.024 if index == selected_index else 0.010
            marker = wssop.sphere(pos=(q[0], q[1], z), radius=radius, rgb=color, alpha=0.9)
            marker.add_to_scene(self.scene)
            self._irm_visuals.append(marker)
            if index < 15 or index == selected_index:
                heading = np.array((np.cos(q[2]), np.sin(q[2]), 0.0))
                arrow = wssop.arrow(
                    spos=(q[0], q[1], z + 0.006),
                    epos=np.array((q[0], q[1], z + 0.006)) + heading * 0.10,
                    shaft_radius=0.0025, head_radius=0.007, head_length=0.016,
                    rgb=color, alpha=0.9)
                arrow.add_to_scene(self.scene)
                self._irm_visuals.append(arrow)

        selected = candidates[selected_index].cart_qs
        path = wssop.dashed_arrow(
            spos=(start_qs[0], start_qs[1], z + 0.025),
            epos=(selected[0], selected[1], z + 0.025),
            shaft_radius=0.003, head_radius=0.010, head_length=0.025,
            rgb=(1.0, 0.72, 0.05), alpha=0.95)
        path.add_to_scene(self.scene)
        self._irm_visuals.append(path)

        fruit_ground = wssop.sphere(
            pos=(centre[0], centre[1], z), radius=0.018,
            rgb=(1.0, 0.45, 0.02), alpha=0.95)
        fruit_ground.add_to_scene(self.scene)
        self._irm_visuals.append(fruit_ground)
        relation = wssop.dashed_cylinder(
            spos=(selected[0], selected[1], z + 0.01),
            epos=(centre[0], centre[1], z + 0.01),
            radius=0.002, rgb=(0.75, 0.20, 0.90), alpha=0.8)
        relation.add_to_scene(self.scene)
        self._irm_visuals.append(relation)

    def set_irm_visualization(self, enabled):
        self._irm_visual_enabled = bool(enabled)
        if not enabled:
            self._clear_irm_visualization()
        elif self._last_irm_visual is not None:
            self._show_irm_visualization(*self._last_irm_visual)

    def drive_cart_to_fruit(self, linear_speed=None, angular_speed=None):
        """Choose a cart pose supporting the complete pregrasp-to-centre route."""
        if self.cart is None:
            raise ValueError('No cart in this scene')
        if self._cart_irm is None:
            if not DEFAULT_CACHE.is_file():
                build_irm(self.robot, self.tcp, DEFAULT_CACHE)
            self._cart_irm = CartIRM(DEFAULT_CACHE)

        centre, pregrasp = self._fruit_grasp_targets()
        candidates = self._cart_irm.candidates(
            centre,
            self.tool_rotation,
            self._cart_hold_qs,
            self.cart.arm_loc_tf,
            max_candidates=120,
            arm_base_height_m=float(self.robot.pos[2]),
        )
        selected = None
        for checked, candidate in enumerate(candidates, start=1):
            grasp_plan = self._plan_grasp_at_candidate(candidate, centre, pregrasp)
            if grasp_plan is not None:
                selected = candidate, grasp_plan, checked
                break
        if selected is None:
            raise ValueError(
                'IRM found no cart pose supporting the complete fruit approach '
                f'after checking {len(candidates)} candidates')
        candidate, (pregrasp_q, centre_q), checked = selected
        result = IRMResult(
            cart_qs=candidate.cart_qs.copy(),
            arm_qs=centre_q,
            score=candidate.score,
            checked_candidates=checked,
        )
        start_qs = self._cart_hold_qs.copy()
        self._show_irm_visualization(candidates, checked - 1, centre, start_qs)
        cart_spec = self.settings.get('cart') or {}
        linear_speed = float(cart_spec.get('lin_m_s', 0.25) if linear_speed is None else linear_speed)
        angular_speed = float(
            np.deg2rad(cart_spec.get('ang_deg_s', 30))
            if angular_speed is None else angular_speed
        )
        segments = unicycle_segments(
            self._cart_hold_qs,
            result.cart_qs,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
        )
        self._cart_irm_goal = result.cart_qs.copy()
        self._cart_grasp_seed = pregrasp_q
        if segments:
            self.drive_cart_trajectory(
                segments,
                status=f'IRM driving · checked {result.checked_candidates} candidate(s)',
            )
        else:
            self.stop_motion()
            self.motion_status = f'IRM parked · checked {result.checked_candidates} candidate(s)'
        return result

    def move_arm_home(self):
        """Cancel the current sequence and return to the cart-relative park."""
        self.stop_motion()
        joint_count = max(
            1,
            int(np.ceil(
                np.max(np.abs(self._home_q - self.command)) / np.deg2rad(2.0))),
        )
        qpath = np.linspace(self.command, self._home_q, joint_count + 1)
        old_qs = self.robot.qs.copy()
        poses = []
        for q in qpath:
            self.robot.fk(q)
            poses.append(self.tcp.tf.copy())
        self.robot.fk(old_qs)
        positions = np.asarray([pose[:3, 3] for pose in poses])
        rotations = np.asarray([pose[:3, :3] for pose in poses])
        durations = np.max(np.abs(np.diff(qpath, axis=0)), axis=1) / np.deg2rad(
            self.settings['joint_speed_deg_s'])
        durations = np.maximum(durations, 1e-4)
        self._path_angles = qpath
        self._path_times = np.r_[0.0, np.cumsum(durations)]
        self._path_positions = positions
        self._path_rotations = rotations
        self._path_rotvecs = np.asarray([
            wum.delta_rotvec_between_rotmats(a, b)
            for a, b in zip(rotations[:-1], rotations[1:])
        ])
        self._motion_elapsed = 0.0
        self.target = self._home_q.copy()
        self.target_position = positions[-1].copy()
        self.target_rotation = rotations[-1].copy()
        self.motion_status = 'Returning arm home'

    def approach_fruit(self):
        """Open, move in joint space to pregrasp, approach current fruit, close."""
        centre, pregrasp = self._fruit_grasp_targets()
        if np.any(centre < self.settings['workspace_min']) or np.any(
                centre > self.settings['workspace_max']):
            raise ValueError('Current fruit centre is outside the configured workspace')

        pregrasp_q = None
        seeds = (
            *((self._cart_grasp_seed,) if self._cart_grasp_seed is not None else ()),
            self.command,
            *self._ik_seed_list(self.command),
        )
        for seed in seeds:
            pregrasp_q = self._try_ik(
                pregrasp, seed, rotation=self.tool_rotation,
                max_iter=max(100, self.settings['ik_max_iter']))
            if pregrasp_q is not None:
                break
        if pregrasp_q is None:
            raise ValueError('No IK solution for the current fruit pregrasp')
        approach_qpath = self._solve_line_qpath(
            pregrasp, centre, pregrasp_q, self.tool_rotation)
        if approach_qpath is None:
            raise ValueError('No continuous IK path from pregrasp to the current fruit')

        joint_count = max(
            1,
            int(np.ceil(
                np.max(np.abs(pregrasp_q - self.command)) / np.deg2rad(2.0))),
        )
        joint_qpath = np.linspace(self.command, pregrasp_q, joint_count + 1)
        qpath = np.concatenate((joint_qpath, approach_qpath[1:]), axis=0)

        old_qs = self.robot.qs.copy()
        poses = []
        for q in qpath:
            self.robot.fk(q)
            poses.append(self.tcp.tf.copy())
        self.robot.fk(old_qs)
        positions = np.asarray([pose[:3, 3] for pose in poses])
        rotations = np.asarray([pose[:3, :3] for pose in poses])
        joint_dt = np.max(np.abs(np.diff(qpath, axis=0)), axis=1) / np.deg2rad(
            self.settings['joint_speed_deg_s'])
        cartesian_dt = np.linalg.norm(np.diff(positions, axis=0), axis=1) / self.settings[
            'cartesian_speed_m_s']
        durations = np.maximum(joint_dt, cartesian_dt)
        durations = np.maximum(durations, 1e-4)

        self._path_angles = qpath
        self._path_times = np.r_[0.0, np.cumsum(durations)]
        self._path_positions = positions
        self._path_rotations = rotations
        self._path_rotvecs = np.asarray([
            wum.delta_rotvec_between_rotmats(a, b)
            for a, b in zip(rotations[:-1], rotations[1:])
        ])
        self._motion_elapsed = 0.0
        self.target = qpath[-1].copy()
        self.target_position = centre.copy()
        self.target_rotation = self.tool_rotation.copy()
        self.presets['fruit'] = centre.copy()
        self.gripper_target = self.gripper.jaw_range[1]
        self._pending_gripper_close = True
        self.motion_status = 'Approaching current fruit'

    def _advance_command(self, dt):
        if self._motion_elapsed >= self.motion_duration:
            if (self._fruit_pull_deadline is not None
                    and self.env.data.time + 1e-10 >= self._fruit_pull_deadline):
                self._fruit_pull_deadline = None
                direction = np.asarray(
                    self.harvest_settings.get('pull_direction_world', (0, -1, 0)),
                    dtype=float)
                direction /= np.linalg.norm(direction)
                distance = float(self.harvest_settings.get('pull_distance_m', 0.14))
                old_speed = self.settings['cartesian_speed_m_s']
                self.settings['cartesian_speed_m_s'] = float(
                    self.harvest_settings.get('pull_speed_m_s', 0.025))
                try:
                    self.set_cartesian_target(
                        self.command_position + direction * distance,
                        self.command_rotation)
                except ValueError as error:
                    self._fruit_pull_active = False
                    self.motion_status = 'Pull IK failed · use Arm home'
                    print(f'Automatic fruit pull cancelled: {error}')
                    return
                finally:
                    self.settings['cartesian_speed_m_s'] = old_speed
                self._fruit_pull_active = True
                self.motion_status = 'Pulling fruit'
            elif self._fruit_pull_active:
                connection = self.plant.fruit_connections.get(self.settings['fruit_id'])
                handle = self.env.connection(connection) if connection is not None else None
                if handle is not None and not handle.attached:
                    self.motion_status = 'Fruit detached'
                    self._fruit_pull_active = False
            return
        self._motion_elapsed = min(self.motion_duration, self._motion_elapsed + dt)
        i = min(int(np.searchsorted(self._path_times, self._motion_elapsed, side='right')),
                len(self._path_times) - 1)
        u = ((self._motion_elapsed - self._path_times[i - 1]) /
             (self._path_times[i] - self._path_times[i - 1]))
        self.command = (1 - u) * self._path_angles[i - 1] + u * self._path_angles[i]
        self.command_position = (1 - u) * self._path_positions[i - 1] + u * self._path_positions[i]
        self.command_rotation = wum.rotmat_from_rotvec(u * self._path_rotvecs[i - 1]) @ self._path_rotations[i - 1]
        if self._motion_elapsed >= self.motion_duration:
            if self._pending_gripper_close:
                self.gripper_target = self.gripper.jaw_range[0]
                self._pending_gripper_close = False
                self._fruit_pull_deadline = (
                    float(self.env.data.time)
                    + float(self.harvest_settings.get('grasp_wait_s', 3.0))
                )
                self.motion_status = 'Closing gripper'
            elif self._fruit_pull_active:
                connection = self.plant.fruit_connections.get(self.settings['fruit_id'])
                handle = self.env.connection(connection) if connection is not None else None
                self.motion_status = (
                    'Fruit detached' if handle is not None and not handle.attached
                    else 'Pull complete')
                self._fruit_pull_active = False
            else:
                self.motion_status = 'Holding target'

    def select_pose(self, name):
        if name == 'fruit':
            self.approach_fruit()
        else:
            self.set_cartesian_target(self.presets[name], self.tool_rotation)

    def _read_contacts(self):
        counts = {}
        for contact in self.env.data.contact:
            a, b = (int(self.env.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            plant_body = b if a in self.robot_bodies else a if b in self.robot_bodies else None
            role = self.plant_bodies.get(plant_body)
            if role:
                counts[role] = counts.get(role, 0) + 1
        self.contacts = counts
        for role, count in counts.items():
            self.contact_samples[role] = self.contact_samples.get(role, 0) + count

    def _sync_scene(self):
        self.env.sync_scene()
        if self.show_forces:
            self.env.contact_viz.update_from_data(self.env.model, self.env.data)
        else:
            self.env.contact_viz.clear()

    def step(self, dt):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('dt must be finite and nonnegative')
        if self.paused:
            return
        self._remainder += dt
        h = self.env.get_timestep()
        n = int((self._remainder + 1e-12) / h)
        self._remainder -= n * h
        for _ in range(n):
            if self.cart is not None:
                self._advance_cart(h)
                self._lock_cart()
            self._advance_command(h)
            self.env.ctrl[self.actuators] = self.command
            opening_step = self.harvest_settings.get('opening_speed_m_s', .04) * h
            self.gripper_command += np.clip(self.gripper_target - self.gripper_command, -opening_step, opening_step)
            self.env.ctrl[self.gripper_actuators] = self.gripper_command / 2
            self.env.runtime.step()
            if self.cart is not None:
                self._lock_cart()
            self._read_contacts()
            for observer in self.substep_observers:
                observer(self)
        # Physics runs at its native timestep; publish rigid mounts once per frame.
        self._sync_scene()
        if self.cart is not None:
            self.cart.fk(self._cart_hold_qs)

    def reset(self):
        self.env.restore(self._initial_state)
        force = self.harvest_settings.get('finger_force_N', 20.)
        self.env.model.actuator_forcerange[self.gripper_actuators] = [-force, force]
        self.settings['base_position'] = list(self._initial_base_position)
        self.presets['ready'] = self._initial_ready.copy()
        if self.cart is not None:
            self._cart_hold_qs = self._initial_cart_qs.copy()
            self._cart_lin = 0.0
            self._cart_ang = 0.0
            self._cart_cmd_remaining = 0.0
            self.cart.fk(self._cart_hold_qs)
            self._lock_cart()
            self.env.runtime.forward()
        self.command = self._ready_q.copy()
        self.command_position = self.presets['ready'].copy()
        self.command_rotation = self.tool_rotation.copy()
        self.gripper_command = self.settings['gripper']['opening_m']
        self.stop_motion()
        self._cart_irm_goal = None
        self._cart_grasp_seed = None
        self._clear_irm_visualization()
        self._last_irm_visual = None
        self._remainder = 0.0
        self.contacts, self.contact_samples = {}, {}
        self._sync_scene()
        if self.cart is not None:
            self.cart.fk(self._cart_hold_qs)
        self.rgbd.invalidate()
        for observer in self.reset_observers:
            observer(self)

    def capture_rgbd(self, dt=0):
        return self.rgbd.capture(simulation_time=float(self.env.data.time - self._initial_time))

    def close(self):
        self.rgbd.close()

    def report(self):
        return dict(simulated_seconds=float(self.env.data.time - self._initial_time),
                    robot_actuators=len(self.actuators), plant_actuators=0,
                    contacts=self.contacts, contact_samples=self.contact_samples.copy(),
                    foliage_coverage=self.foliage_coverage.copy(),
                    joint_target_deg=np.rad2deg(self.target).tolist(),
                    joint_actual_deg=np.rad2deg(self.robot.qs).tolist(),
                    tcp_target_m=self.target_position.tolist(), tcp_actual_m=self.tcp.pos.tolist(),
                    gripper_opening_m=float(np.sum(self.gripper.qs)), gripper_target_m=self.gripper_target,
                    gripper_actuators=len(self.gripper_actuators), gripper_mode='position_servo',
                    attachments={key: self.env.connection(c).report()
                                 for key, c in self.plant.fruit_connections.items()},
                    T_flange_cam=self.rgbd.mount_tf.tolist(),
                    motion_status=self.motion_status,
                    plant_deflection_rad=float(np.linalg.norm(self.plant.mech.qs - self._initial_plant_q)),
                    fruit_displacement_m={k: float(np.linalg.norm(o.pos - self._initial_fruit[k]))
                                          for k, o in self.plant.fruits.items()},
                    cart_pose_m=None if self.cart is None else [float(v) for v in self.cart.qs],
                    cart_irm_goal=None if self._cart_irm_goal is None else self._cart_irm_goal.tolist(),
                    camera=self.rgbd.statistics())


def add_controls(base, demo):
    arm = base.ui.add_panel('arm', title='FAFU Cartesian jog', anchor=Anchor.TOP_LEFT,
                            width=300, columns=6, font_size=12, movable=True,
                            description='Hold a key or button to jog. WASD/QE move the arm; arrows drive the cart. World axes: +Y toward the tree, +X right, +Z up.')
    status = base.ui.add_panel('interaction', title='VirtualD405 / Citrus contact', anchor=Anchor.TOP_RIGHT,
                               width=340, font_size=12, movable=True,
                               description='Live wrist RGB-D. Switch Clean / Noise to compare depth. Black depth = invalid/out of range.')
    demo.rgbd.add_controls(status)
    jog_step = [demo.settings['jog_step_m']]
    rotation_step = [2.0]

    def request(action):
        try:
            action()
        except ValueError as error:
            # A rejected path never replaces the previous valid target/path.
            arm.set_value('request', f'Not applied: {error}')
            raise  # Let the button stop repeating on a rejected move.
        else:
            arm.set_value('request', 'Applied')
        refresh(0)

    def set_step(value):
        jog_step[0] = value / 1000

    arm.add_slider('step', label='Move per step', unit='mm', min_value=1, max_value=30, group='Move',
                   step=1, value=jog_step[0] * 1000, on_change=set_step)
    for key, label, axis, sign, shortcut in [
            ('up', 'Up +Z', 2, 1, 'q'), ('forward', 'Forward +Y', 1, 1, 'w'), ('down', 'Down −Z', 2, -1, 'e'),
            ('left', 'Left −X', 0, -1, 'a'), ('backward', 'Back −Y', 1, -1, 's'), ('right', 'Right +X', 0, 1, 'd')]:
        arm.add_button(key, label=label, group='Move', variant='keycap', column_span=2,
                       repeat=True, repeat_hz=10, shortcut=shortcut,
                       on_click=lambda axis=axis, sign=sign: request(lambda: demo.jog(axis, sign * jog_step[0])))

    def set_rotation_step(value):
        rotation_step[0] = value

    arm.add_slider('rotation_step', label='Rotate per step', unit='°', min_value=1, max_value=10,
                   step=1, value=rotation_step[0], group='Rotate',
                   on_change=set_rotation_step)
    for axis, sign, shortcut in [(0, 1, 'i'), (1, 1, 'j'), (2, 1, 'u'),
                                 (0, -1, 'k'), (1, -1, 'l'), (2, -1, 'o')]:
        arm.add_button(f'rotate_{shortcut}', label=f'{"XYZ"[axis]} {"+" if sign > 0 else "−"}',
                       group='Rotate', variant='keycap', column_span=2, repeat=True, repeat_hz=10,
                       shortcut=shortcut, on_click=lambda axis=axis, sign=sign: request(
                           lambda: demo.jog_rotation(axis, np.deg2rad(sign * rotation_step[0]))))

    arm.add_label('gripper', label='Opening · actual / target', group='Gripper')
    arm.add_button('open', label='Open', group='Gripper', variant='keycap', column_span=3, shortcut='r',
                   on_click=lambda: request(lambda: demo.set_gripper_opening(demo.gripper.jaw_range[1])))
    arm.add_button('close', label='Close', group='Gripper', variant='keycap', column_span=3, shortcut='f',
                   on_click=lambda: request(lambda: demo.set_gripper_opening(demo.gripper.jaw_range[0])))

    cart_spec = demo.settings.get('cart') or {}
    lin_speed = [float(cart_spec.get('lin_m_s', 0.25))]
    ang_speed = [np.deg2rad(float(cart_spec.get('ang_deg_s', 30)))]
    cart_dt = 0.1
    if demo.cart is not None:
        arm.add_slider('cart_lin', label='Cart lin speed', unit='m/s', min_value=0.05, max_value=0.80,
                       step=0.05, value=lin_speed[0], group='Cart',
                       on_change=lambda value: lin_speed.__setitem__(0, value))
        arm.add_slider('cart_ang', label='Cart yaw speed', unit='°/s', min_value=5, max_value=90,
                       step=5, value=np.rad2deg(ang_speed[0]), group='Cart',
                       on_change=lambda value: ang_speed.__setitem__(0, np.deg2rad(value)))
        for key, label, lin, ang, shortcut in [
                ('cart_forward', 'Forward +v', 1, 0, 'ArrowUp'),
                ('cart_back', 'Back −v', -1, 0, 'ArrowDown'),
                ('cart_left', 'Left +ω', 0, 1, 'ArrowLeft'),
                ('cart_right', 'Right −ω', 0, -1, 'ArrowRight')]:
            arm.add_button(key, label=label, group='Cart', variant='keycap', column_span=3,
                           repeat=True, repeat_hz=10, shortcut=shortcut,
                           on_click=lambda lin=lin, ang=ang: request(
                               lambda: demo.drive_cart(lin * lin_speed[0], ang * ang_speed[0], cart_dt)))
        arm.add_button(
            'cart_irm', label=f'IRM to {demo.settings["fruit_id"]}', group='Cart',
            column_span=6, on_click=lambda: request(
                lambda: demo.drive_cart_to_fruit(lin_speed[0], ang_speed[0])))
        arm.add_checkbox(
            'cart_irm_visual', label='Show IRM candidates', value=True, group='Cart',
            on_change=demo.set_irm_visualization)
        arm.add_label(
            'cart_irm_legend',
            label='IRM map',
            value='green selected · red rejected · blue other · yellow cart path',
            group='Cart')
        arm.add_label('cart_pose', label='Cart x / y / yaw', group='Cart')
        arm.add_label('cart_irm_goal', label='IRM goal x / y / yaw', group='Cart')

    def reset():
        demo.reset()
        arm.set_value('request', 'Reset')
        refresh(0)

    def proxies(checked):
        demo.plant.show_foliage_proxies(checked)

    def collision(checked):
        demo.robot.toggle_render_collision = checked
        for obj in (*demo.plant.branch_objects, *demo.plant.fruit_objects):
            obj.toggle_render_collision = checked

    def forces(checked):
        demo.show_forces = checked
        demo._sync_scene()

    arm.add_button('stop', label='Stop', group='Actions', column_span=3, shortcut='Escape',
                   on_click=lambda: request(demo.stop_motion))
    arm.add_button('arm_home', label='Arm home', group='Actions', column_span=3,
                   on_click=lambda: request(demo.move_arm_home))
    arm.add_button('ready', label='Retract / ready', group='Actions', column_span=3,
                   on_click=lambda: request(lambda: demo.select_pose('ready')))
    arm.add_button('foliage', label='Touch leaves', group='Actions', column_span=3,
                   on_click=lambda: request(lambda: demo.select_pose('foliage')))
    arm.add_button('fruit', label=f'Grasp {demo.settings["fruit_id"]}', group='Actions', column_span=3,
                   on_click=lambda: request(lambda: demo.select_pose('fruit')))
    arm.add_button('reset', label='Reset robot and tree', group='Actions', on_click=reset)
    arm.add_label('request', label='Last request', value='Ready')
    status.add_checkbox('paused', label='Pause physics', on_change=lambda value: setattr(demo, 'paused', value))
    status.add_checkbox('proxies', label='Show foliage contact proxies', on_change=proxies)
    coverage = demo.foliage_coverage
    status.add_label('foliage_coverage', label='Foliage contact coverage',
        value=f'{coverage["foliage_contact_leaf_count"]:,} contact leaves / '
              f'{coverage["visual_only_leaf_count"]:,} visual-only\n'
              f'{coverage["foliage_proxy_object_count"]} compound objects · '
              f'{demo.plant.dynamics.foliage_proxy.sections_per_leaf} strips per leaf')
    status.add_checkbox('collisions', label='Show arm / branch / fruit collisions', on_change=collision)
    status.add_checkbox('forces', label='Show contact force arrows', on_change=forces)
    for key, label in [('time', 'Simulation'), ('motion', 'Motion'),
                       ('target', 'Target TCP · X / Y / Z metres'), ('actual', 'Actual TCP · X / Y / Z metres'),
                       ('contacts', 'Arm ↔ plant contact'), ('bend', 'Plant deflection'),
                       ('fruit_position', f'{demo.settings["fruit_id"]} · world metres')]:
        status.add_label(key, label=label)

    def refresh(dt):
        report = demo.report()
        arm.set_value('gripper', f'{report["gripper_opening_m"] * 1000:.0f} / '
                                f'{report["gripper_target_m"] * 1000:.0f} mm')
        if demo.cart is not None and report.get('cart_pose_m') is not None:
            x, y, yaw = report['cart_pose_m']
            arm.set_value('cart_pose', f'{x:.3f} / {y:.3f} m · {np.rad2deg(yaw):.1f}°')
            goal = report.get('cart_irm_goal')
            arm.set_value(
                'cart_irm_goal',
                '—' if goal is None
                else f'{goal[0]:.3f} / {goal[1]:.3f} m · {np.rad2deg(goal[2]):.1f}°')
        status.set_value('time', f'{report["simulated_seconds"]:.1f} s' + (' · paused' if demo.paused else ''))
        status.set_value('motion', report['motion_status'])
        status.set_value('target', ' / '.join(f'{x:.3f}' for x in report['tcp_target_m']))
        status.set_value('actual', ' / '.join(f'{x:.3f}' for x in report['tcp_actual_m']))
        status.set_value('contacts', ', '.join(f'{k}: {v}' for k, v in demo.contacts.items()) or 'No contact')
        status.set_value('bend', f'{report["plant_deflection_rad"]:.3f} rad')
        status.set_value('fruit_position', ' / '.join(f'{x:.3f}' for x in demo.plant.fruits[demo.settings['fruit_id']].pos))

    refresh(0)
    base.schedule_interval(refresh, 1 / demo.settings['status_hz'])
    demo.capture_rgbd()
    base.schedule_interval(demo.capture_rgbd, 1 / demo.camera.fps)
    return arm, status


def run_smoke(demo, *, capture_rgbd=False):
    results = {}
    for name in ('foliage', 'fruit'):
        demo.reset()
        if name == 'fruit' and demo.cart is not None:
            demo.drive_cart_to_fruit()
            cart_travel = demo._cart_cmd_remaining + sum(
                segment[2] for segment in demo._cart_segments)
            demo.step(cart_travel + demo.env.get_timestep())
        demo.select_pose(name)
        travel = demo.motion_duration
        demo.step(travel + demo.settings['smoke_hold_seconds'])
        if capture_rgbd:
            demo.capture_rgbd()
        touch = demo.report()
        demo.select_pose('ready')
        demo.step(demo.motion_duration + demo.settings['smoke_release_seconds'])
        if capture_rgbd:
            demo.capture_rgbd()
        results[name] = dict(touch=touch, released=demo.report())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--duration', type=float, help='Close viewer after this many wall-clock seconds')
    parser.add_argument('--headless', action='store_true', help='Run both contact presets and release without a viewer')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be finite and positive')
    demo = RobotPlantInteraction(load_config(args.config))
    try:
        if args.headless:
            report = run_smoke(demo, capture_rgbd=True)
        else:
            p = demo.settings
            base = wvw.World(cam_pos=p['camera_pos'], cam_lookat_pos=p['camera_lookat'], port=args.port)
            base.set_scene(demo.scene)
            base.set_caption('FAFU / Citrus V2 / VirtualD405')
            add_controls(base, demo)
            base.schedule_interval(lambda dt: demo.step(min(dt, p['max_frame_dt'])), 1 / p['control_hz'])
            if args.duration is not None:
                base.schedule_once(lambda dt: base.close(), args.duration)
            base.run()
            report = demo.report()
    finally:
        demo.close()
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
