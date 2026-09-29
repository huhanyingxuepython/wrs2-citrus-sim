"""A native WRS slider pushes a foliage proxy; only the slider is actuated."""
import numpy as np
from wrs import wuc, wssop
from wrs.robots.base.mech_structure import MechStruct, Link, Joint
from wrs.robots.base.mech_base import MechBase
from wrs.physics.mj_env import MJEnv


class PushExperiment:
    def __init__(self, plant, scene, settings):
        self.plant, self.settings = plant, settings
        # Settle under gravity before positioning the probe at the actual canopy.
        warmup = MJEnv(scene, require_ctrl=True)
        warmup.runtime.step(int(round(settings['settle_before'] / warmup.get_timestep())))
        warmup.sync.pull_all_mecba_qpos()
        target = plant.foliage_proxies[settings['cluster']][settings['proxy_index']]
        direction = plant.mech.rotmat @ np.asarray(settings['direction_local'], dtype=float)
        direction /= np.linalg.norm(direction)
        shape = target.collisions[settings['shape_index']]
        shape_tf = target.tf @ shape.loc_tf
        extent = np.sum(np.abs(direction @ shape_tf[:3, :3]) * shape.half_extents)
        origin = shape_tf[:3, 3] - direction * (extent + settings['sphere_radius'] + settings['clearance'])
        root, probe = Link(), Link()
        root.name, probe.name = 'probe_base', 'probe'
        sphere = wssop.icosphere(radius=settings['sphere_radius'], rgb=(.2, .6, 1),
                                collision_type=wuc.CollisionType.SPHERE, mass=settings['pusher_mass'])
        for model in sphere.visuals:
            probe.add_visual(model, auto_make_collision=False)
        for shape in sphere.collisions:
            probe.add_collision(shape)
        probe.set_inertia(mass=settings['pusher_mass'])
        structure = MechStruct()
        structure.add_jnt(Joint(wuc.JntType.PRISMATIC, root, probe, direction,
                               lmt_lo=0, lmt_up=settings['travel'] * 1.2,
                               damping=settings['pusher_damping'], armature=0))
        self.probe = MechBase(structure=structure, pos=origin, is_floating=False)
        self.probe.add_to_scene(scene)
        self.env = MJEnv(scene, require_ctrl=True)
        if self.env.model.nu != 1:
            raise RuntimeError('Expected only the test probe actuator; plant joints must be passive')
        self.duration = sum(settings[key] for key in ('push_duration', 'hold_duration', 'retract_duration', 'settle_after'))
        self.time, self.rows, self.proxy_contacts = 0.0, [], 0
        self.baseline = plant.mech.qs.copy()
        self.proxy_body_ids = {self.env.model.body(self.env.sync.sobj2bdy[obj].name).id
                              for group in plant.foliage_proxies.values() for obj in group}
        self.target_indices = [i for i, j in enumerate(plant.mech.structure.jnts)
                               if j.clnk.name in (settings['cluster'], settings['cluster'] + '_bend_x')]
        self.leaf = next(obj for obj, owner in plant.leaf_clusters.items() if owner == settings['cluster'])
        self.fruit = next((obj for key, obj in plant.fruits.items()
                           if plant.segment_clusters[next(f.parent_segment for f in plant.spec.fruits if f.id == key)] == settings['cluster']), None)
        self.leaf_origin = self.leaf_point()
        self.fruit_origin = None if self.fruit is None else self.fruit.pos

    def leaf_point(self):
        return self.leaf.rotmat @ self.leaf.visuals[0].geom.vs[-1] + self.leaf.pos

    def step(self, dt):
        h = self.env.get_timestep()
        for _ in range(max(1, int(round(dt / h)))):
            p = self.settings
            t = self.time
            if t < p['push_duration']:
                amount = t / p['push_duration']
            elif t < p['push_duration'] + p['hold_duration']:
                amount = 1.0
            else:
                amount = max(0.0, 1 - (t - p['push_duration'] - p['hold_duration']) / p['retract_duration'])
            self.env.ctrl[0] = p['travel'] * amount
            self.env.step(h)
            for contact in self.env.data.contact:
                if (int(self.env.model.geom_bodyid[contact.geom1]) in self.proxy_body_ids or
                        int(self.env.model.geom_bodyid[contact.geom2]) in self.proxy_body_ids):
                    self.proxy_contacts += 1
            self.time += h
            self.rows.append(dict(t=self.time, q=self.plant.mech.qs.astype(float).tolist(),
                                  leaf=self.leaf_point().tolist(),
                                  fruit=None if self.fruit is None else self.fruit.pos.tolist()))

    def report(self):
        q = np.array([row['q'] for row in self.rows])
        displacement = np.linalg.norm(q[:, self.target_indices] - self.baseline[self.target_indices], axis=1)
        release = self.settings['push_duration'] + self.settings['hold_duration'] + self.settings['retract_duration']
        late = np.array([row['t'] > self.duration - .5 for row in self.rows])
        free = np.array([row['t'] > release + .1 for row in self.rows])
        equilibrium = q[late].mean(axis=0)
        dominant = self.target_indices[int(np.argmax(np.ptp(q[:, self.target_indices], axis=0)))]
        residual = q[free, dominant] - equilibrium[dominant]
        significant = residual[np.abs(residual) > .001]
        crossings = int(np.sum(significant[:-1] * significant[1:] < 0))
        return self.plant.summary() | dict(
            simulated_seconds=self.time, plant_actuators=0, probe_actuators=self.env.model.nu,
            proxy_contact_samples=self.proxy_contacts, peak_joint_displacement_rad=float(displacement.max()),
            final_joint_residual_rad=float(np.linalg.norm(q[-1, self.target_indices] - self.baseline[self.target_indices])),
            late_joint_peak_to_peak_rad=float(np.ptp(q[late, :][:, self.target_indices], axis=0).max()),
            release_oscillation_crossings=crossings,
            peak_leaf_displacement_m=float(np.linalg.norm(np.array([r['leaf'] for r in self.rows]) - self.leaf_origin, axis=1).max()),
            peak_fruit_displacement_m=None if self.fruit is None else float(np.linalg.norm(np.array([r['fruit'] for r in self.rows]) - self.fruit_origin, axis=1).max()))
