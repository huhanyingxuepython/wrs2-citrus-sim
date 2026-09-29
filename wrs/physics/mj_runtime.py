import mujoco
import numpy as np
from copy import deepcopy
from wrs.physics.connections import RuntimeConnectionHandle


class MJRuntime:
    def __init__(self, xml_string, connections=()):
        # Model parameters are mutable at rupture: never share this model with
        # another runtime/environment, even when XML/specifications are shared.
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self._cd_mode = False
        self._dyn_backup = None
        self.connections = {node.spec: RuntimeConnectionHandle(node.spec, node.name,
            self.model.tendon(node.name).id,
            tuple(self.model.site(site.name).id for site in node.sites)) for node in connections}
        self._state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
        self._snapshot_owner = object()
        self.forward()
        self.save_reset_state()

    def step(self, substeps=1):
        self.exit_cd()
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
            if self.connections:
                # mj_step advances qpos/qvel; refresh derived tendon values at
                # that completed substep. No extra integration or Python force.
                self.forward()
                broken = False
                for handle in self.connections.values():
                    broken |= handle.advance(self.model, self.data, self.model.opt.timestep)
                if broken:
                    self.forward()  # clear disabled passive forces, retain state

    def forward(self):
        mujoco.mj_forward(self.model, self.data)
        for handle in self.connections.values():
            handle.observe(self.model, self.data)

    def snapshot(self):
        """Integration state PLUS mutable spring parameters and rupture history.

        Snapshots belong to this runtime only; no renderer state or object poses
        are copied. Call MJEnv.restore to also publish restored scene visuals.
        """
        state = np.empty(mujoco.mj_stateSize(self.model, self._state_kind))
        mujoco.mj_getState(self.model, self.data, state, self._state_kind)
        return dict(owner=self._snapshot_owner, state=state,
            parameters={name: getattr(self.model, name).copy() for name in
                        ('tendon_stiffness', 'tendon_damping', 'tendon_lengthspring')},
            connections=[(h.attached, h.overload_time_s, deepcopy(h.events))
                         for h in self.connections.values()])

    def restore(self, snapshot):
        if snapshot['owner'] is not self._snapshot_owner:
            raise ValueError('snapshot belongs to a different runtime')
        self._cd_mode, self._dyn_backup = False, None
        for name, value in snapshot['parameters'].items():
            getattr(self.model, name)[:] = value
        mujoco.mj_setState(self.model, self.data, snapshot['state'], self._state_kind)
        for h, (attached, elapsed, events) in zip(self.connections.values(), snapshot['connections']):
            h.attached, h.overload_time_s, h.events = attached, elapsed, deepcopy(events)
        self.forward()

    def save_reset_state(self):
        """Explicitly choose the current state as the reset baseline."""
        self._initial_snapshot = self.snapshot()

    def reset(self):
        self.restore(self._initial_snapshot)

    def enter_cd(self):
        if self._cd_mode:
            return
        self._backup()
        self._cd_mode = True

    def exit_cd(self):
        if not self._cd_mode:
            return
        self._restore()
        self._cd_mode = False

    def is_collided(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        # import wrs.physics.mj_contact as mjc
        # mjc.debug_contacts(self)
        return self.data.ncon > 0

    def _backup(self):
        self._dyn_backup = self.snapshot()

    def _restore(self):
        self.restore(self._dyn_backup)
