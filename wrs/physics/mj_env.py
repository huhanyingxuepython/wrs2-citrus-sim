from wrs.physics.mj_wrs_cvter import MJWRSConverter
from wrs.physics.mj_runtime import MJRuntime
from wrs.physics.mj_wrs_sync import MJSynchronizer
from wrs.physics.mj_contact import MJContactForceViz


class MJEnv:
    def __init__(self, scene, margin=0.0, require_ctrl=False,
                 extra_excludes=None):
        self._cvter = MJWRSConverter(margin)
        self.scene = scene
        self._world, sobj2bdy, rutl2bdy, mecj2jnt = (
            self._cvter.convert(scene, extra_excludes=extra_excludes))
        if not require_ctrl:
            self._world.actuators = None
        self.xml_string = self._world.compile_mjcf()
        # print(self.xml_string)
        self.runtime = MJRuntime(self.xml_string, self._world.connections)
        self.sync = MJSynchronizer(
            self.runtime, scene,
            sobj2bdy, rutl2bdy, mecj2jnt)
        # contact viz
        self.contact_viz = MJContactForceViz(scene)
        self.sync.push_all_mecba_qpos()
        self.sync.push_all_sobj_qpos()
        self.runtime.forward()
        self.runtime.save_reset_state()
        self.sync_scene()

    def step(self, dt):
        self.runtime.exit_cd()
        h = self.runtime.model.opt.timestep
        n = int(round(dt / h))
        # self.sync.push_qpos()
        self.runtime.step(n)
        self.sync_scene()
        self.contact_viz.update_from_data(self.model, self.data)

    def sync_scene(self):
        """Publish current physics poses, including connection visual callbacks."""
        self.sync.pull_all_sobj_pose()
        self.sync.pull_all_mecba_qpos()
        self.sync.pull_all_mecba_freebase_pose()
        for callback in tuple(self.scene.physics_sync_callbacks):
            callback(self)

    def connection(self, spec):
        return self.runtime.connections[spec]

    def snapshot(self):
        return self.runtime.snapshot()

    def restore(self, snapshot):
        self.runtime.restore(snapshot)
        self.sync_scene()
        self.contact_viz.clear()

    def is_collided(self):
        self.runtime.enter_cd()
        collided = self.runtime.is_collided()
        return collided

    def reset(self):
        self.runtime.reset()
        self.sync_scene()
        self.contact_viz.clear()

    def get_timestep(self):
        return float(self.runtime.model.opt.timestep)

    def save(self, filepath, encoding="utf-8"):
        if self.xml_string is None:
            raise RuntimeError("XML not built yet")
        with open(filepath, "w", encoding=encoding) as f:
            f.write(self.xml_string)

    @property
    def data(self):
        return self.runtime.data

    @property
    def model(self):
        return self.runtime.model

    @property
    def ctrl(self):
        return self.runtime.data.ctrl
