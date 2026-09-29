import numpy as np
import wrs.utils.math as wum
import wrs.utils.constant as wuc
import wrs.scene.collision_shape as sco
import wrs.robots.base.mech_base as wrbmb
import wrs.physics.inertial as wpi
import wrs.physics.mj_nodes as wpmno
import wrs.physics.mj_naming as wpmna


class MJWRSConverter:

    def __init__(self, margin=0.0):
        self._opt = wpmno.OptionNode()
        self._opt.gravity = (0, 0, -9.81)
        self._opt.timestep = 0.002
        self._default = wpmno.DefaultNode()
        # self._default.geom["friction"] = (1.0, 0.1, 0.1)
        self._default.geom["margin"] = margin
        self._default.geom["solref"] = (0.02, 1.0)
        self._default.geom["solimp"] = (0.9, 0.95, 0.002)
        self._mesh_assets = {}  # key = file_path, value = MeshAsset
        self._actuators = []  # list of ActuatorNode
        # mappings
        self._sobj2bdy = {}
        self._sobj2site = {}  # collision-free mounted sobjs -> marker sites
        self._rutl2bdy = {}  # use runtime lnk as key
        self._mecj2jnt = {}  # use (mecba, jidx) as key
        self._bdy_alias = {}
        # mounted children
        self._mounted_children = set()

    def convert(self, scene, extra_excludes=None):
        self._mesh_assets.clear()
        self._actuators.clear()
        self._sobj2bdy.clear()
        self._sobj2site.clear()
        self._rutl2bdy.clear()
        self._mecj2jnt.clear()
        self._bdy_alias.clear()
        self._mounted_children.clear()
        for mecba in scene.mecbas:
            self._collect_mounted_children(mecba)
        world = wpmno.WorldNode()
        world.option = self._opt
        world.default = self._default
        root = wpmno.BodyNode("world_root")
        world.root_body = root
        for mecba in scene.mecbas:
            if mecba in self._mounted_children:
                continue
            robot = self._cvt_robot(mecba)
            robot.parent = root
            root.children.append(robot)
        for sobj in scene.sobjs:
            if sobj in self._mounted_children:
                continue
            if not getattr(sobj, "collisions", None):
                # NOT turned into a site, unlike a collision-free MOUNTED child.
                # A free-standing collision-free object here is render-layer
                # scenery -- MJContactForceViz alone parks 64 force arrows in the
                # scene, and re-parks them on every MJEnv rebuild -- so siting
                # them would pour hundreds of meaningless markers into the model.
                # A mounted marker is different: it is named, bounded, and
                # attached to a link whose pose is worth reading back.
                continue
            body = self._cvt_sobj(sobj, ref_tf=sobj.tf)
            self._sobj2bdy[sobj] = body
            body.parent = root
            root.children.append(body)
        world.assets = list(self._mesh_assets.values())
        world.actuators = self._actuators
        # Attach sites in the original body frames BEFORE folding empty bodies.
        # Site names survive aliasing; the folding code preserves their frames.
        from wrs.physics.connections import LinearSpringConnectionSpec
        for spec in scene.connections:
            if not isinstance(spec, LinearSpringConnectionSpec):
                raise TypeError('unsupported physics connection')
            sites = []
            for anchor in (spec.anchor_a, spec.anchor_b):
                body = root if anchor.body is None else self._lookup_body_node(anchor.body)
                site = wpmno.SiteNode(wpmna.alloc_name('spring_site'))
                site.pos = anchor.local_pos
                site.rgba = (0, 0, 0, 0)
                body.sites.append(site)
                sites.append(site)
            world.connections.append(wpmno.SpatialSpringNode(wpmna.alloc_name('spring'), spec, sites))
        for robot in root.children:
            self._merge_empty_geoms(robot, is_root=True)
        self._finalize_alias_maps()
        # process collision ignores
        for mecba in scene.mecbas:
            struct = mecba.structure
            for alidx, blidx in struct.compiled.collision_ignores_idx:
                rta = mecba.runtime_lnks[alidx]
                rtb = mecba.runtime_lnks[blidx]
                body_a = self._rutl2bdy[rta]
                body_b = self._rutl2bdy[rtb]
                world.contact_excludes.append((body_a, body_b))
        # extra scene-level excludes, given as (sobj_a, sobj_b) pairs where each
        # is a converted runtime Link OR a mounted/free SceneObject (a Link IS a
        # SceneObject). Resolved through both body maps, so an exclude can name a
        # held object vs the gripper's finger links, not just link-link pairs.
        # They are object refs, so they survive a rebuild and cross mecba
        # boundaries (mounted EE vs arm, held object vs gripper).
        for sobj_a, sobj_b in (extra_excludes or []):
            world.contact_excludes.append(
                (self._lookup_body_node(sobj_a),
                 self._lookup_body_node(sobj_b)))
        return world, self._sobj2bdy, self._rutl2bdy, self._mecj2jnt

    @property
    def gravity(self):
        return self._opt.gravity

    @gravity.setter
    def gravity(self, value):
        self._opt.gravity[:] = value

    @property
    def timestep(self):
        return self._opt.timestep

    @timestep.setter
    def timestep(self, value):
        self._opt.timestep = value

    def _cvt_robot(self, mecba):
        compiled = mecba._compiled

        def _scan_child(mecba, jidx, lidx):
            # hosting joint frame
            jnode = wpmno.JointNode(wpmna.alloc_name("jnt"))
            self._mecj2jnt[(mecba, jidx)] = jnode
            jnode.actuated = bool(compiled.actuated_by_idx[jidx])
            jtype = compiled.jtypes_by_idx[jidx]
            if jtype == wuc.JntType.REVOLUTE:
                jnode.jtype_str = "hinge"
            elif jtype == wuc.JntType.PRISMATIC:
                jnode.jtype_str = "slide"
            else:
                jnode.jtype_str = "fixed"
            if jnode.jtype_str != "fixed" and compiled.actuated_by_idx[jidx]:
                prefix = "ra" if jnode.jtype_str == "hinge" else "sa"
                act = wpmno.ActuatorNode(wpmna.alloc_name(prefix))
                act.joint = jnode
                self._actuators.append(act)
            for key, values in compiled.passive_params.items():
                setattr(jnode, key, float(values[jidx]))
            jnode.ax = tuple(compiled.jax_by_idx[jidx])
            jnode.range = (float(compiled.jlmt_low_by_idx[jidx]),
                           float(compiled.jlmt_high_by_idx[jidx]))
            # lnk
            lnk = mecba.runtime_lnks[lidx]
            if lnk.is_floating:
                raise ValueError(
                    "Free link cannot be child link of a joint")
            jtf0 = compiled.jtf0_by_idx[jidx]
            body = self._cvt_sobj(lnk, ref_tf=jtf0)
            self._rutl2bdy[lnk] = body
            # attach joint to parent body
            body.hosting_jnts.append(jnode)
            # recurse into grandchildren
            for clidx in compiled.clnk_ids_of_lidx[lidx]:
                pjidx = compiled.pjidx_of_lidx[clidx]
                if pjidx >= 0:
                    child = _scan_child(mecba, pjidx, clidx)
                    child.parent = body
                    body.children.append(child)
            return body

        ridx = compiled.root_lnk_idx
        root_lnk = mecba.runtime_lnks[ridx]
        root = self._cvt_sobj(root_lnk, ref_tf=mecba.tf)
        self._rutl2bdy[root_lnk] = root
        for clidx in compiled.clnk_ids_of_lidx[ridx]:
            pjidx = compiled.pjidx_of_lidx[clidx]
            if pjidx >= 0:
                child = _scan_child(mecba, pjidx, clidx)
                child.parent = root
                root.children.append(child)
        self._attach_mountings(mecba)
        return root

    def _cvt_sobj(self, sobj, ref_tf=None):
        if ref_tf is None:
            ref_tf = np.eye(4, dtype=np.float32)
        b = wpmno.BodyNode(wpmna.alloc_name("sobj"))
        if sobj.is_floating:
            jnode = wpmno.JointNode("free_root")
            jnode.jtype_str = "free"
            b.hosting_jnts.append(jnode)
        b.pos, b.quat = wum.pos_quat_from_tf(ref_tf)
        # Explicit inertia also supports collision-free articulated spacer links.
        if sobj.mass is not None and sobj.com is not None and sobj.inrtmat is not None:
            b.inertial = wpmno.InertialNode(sobj.mass, sobj.com, sobj.inrtmat)
        if sobj.collisions:
            if b.inertial is None and sobj.mass is not None:
                com, inrtmat = wpi.inertia_from_collisions(
                    sobj.collisions, sobj.mass)
                b.inertial = wpmno.InertialNode(
                    mass=sobj.mass, com=com,
                    inertia=inrtmat)
            for c in sobj.collisions:
                g = self._cvt_geom(
                    c, wpmna.alloc_name("geom"))
                self._apply_collision_filter(sobj, g)
                b.geoms.append(g)
        return b

    def _cvt_sobj_as_site(self, sobj, loc_tf):
        """A collision-free SceneObject -> a marker site at ``loc_tf`` on its host
        link. Sites carry no mesh, so the object's visuals are NOT transferred:
        what survives is the POSE (readable as ``data.site_xpos`` / ``site_xmat``)
        and the color."""
        s = wpmno.SiteNode(wpmna.alloc_name("site"))
        s.pos, s.quat = wum.pos_quat_from_tf(loc_tf)
        s.rgba = sobj.rgba
        self._sobj2site[sobj] = s
        return s

    def _cvt_geom(self, c, name=None):
        g = wpmno.GeomNode(name)
        g.pos = c.pos
        g.quat = c.quat
        if isinstance(c, sco.SphereCollisionShape):
            g.gtype = "sphere"
            g.size = (c.radius,)
        elif isinstance(c, sco.CapsuleCollisionShape):
            g.gtype = "capsule"
            g.size = (c.radius, c.half_length)
        elif isinstance(c, sco.CylinderCollisionShape):
            g.gtype = "cylinder"
            g.size = (c.radius, c.half_length)
        elif isinstance(c, (sco.AABBCollisionShape, sco.OBBCollisionShape)):
            g.gtype = "box"
            g.size = tuple(c.half_extents)
        elif isinstance(c, sco.PlaneCollisionShape):
            g.gtype = "plane"
            g.size = (1.0, 1.0, 0.1)
        elif isinstance(c, sco.CvxHullCollisionShape):
            # A convex-hull proxy: emit the hull's vertices INLINE (MuJoCo rebuilds
            # the hull from them) instead of the source STL file, which may be an
            # ASCII / oversized mesh MuJoCo cannot load. Keyed per shape (each hull
            # is distinct); the verts are in the same local frame as the file mesh
            # would be, so the geom pos/quat place it identically.
            g.gtype = "mesh"
            key = id(c)
            if key not in self._mesh_assets:
                name = f"mesh_{len(self._mesh_assets)}"
                verts = np.asarray(c.geom.vs, dtype=float).reshape(-1)
                self._mesh_assets[key] = wpmno.MeshAsset(name=name, vertices=verts)
            g.mesh_ref = self._mesh_assets[key]
        elif isinstance(c, sco.MeshCollisionShape):
            g.gtype = "mesh"
            if c.file_path not in self._mesh_assets:
                name = f"mesh_{len(self._mesh_assets)}"
                self._mesh_assets[c.file_path] = wpmno.MeshAsset(
                    name=name, path=c.file_path)
            g.mesh_ref = self._mesh_assets[c.file_path]
        else:
            raise NotImplementedError(f"Unsupported collision: {type(c)}")
        return g

    def _collect_mounted_children(self, mecba):
        for m in mecba._mountings.values():
            child = m.child
            self._mounted_children.add(child)
            if getattr(child, '_mountings', None):
                self._collect_mounted_children(child)

    def _attach_mountings(self, mecba):
        for m in mecba._mountings.values():
            child = m.child
            loc_tf = m.loc_tf
            # A mounted SceneObject with NO collision geometry (a tcp coordinate
            # frame from toggle_tcp, a marker) is not a physical body: as a
            # zero-geom child body it would contribute nothing yet still be
            # foldable by _merge_empty_geoms, which rewrites the HOST link's pose
            # and aliases the marker onto that link. It becomes a site instead --
            # massless, contact-free, outside the body tree.
            if (not isinstance(child, wrbmb.MechBase) and
                    not child.collisions):
                plnk_bdy = self._rutl2bdy[m.plnk]
                plnk_bdy.sites.append(self._cvt_sobj_as_site(child, loc_tf))
                continue
            # child subtree
            if isinstance(child, wrbmb.MechBase):  # MechBase
                child_root = self._cvt_robot(child)
            else:  # SceneObject
                child_root = self._cvt_sobj(child)
                self._sobj2bdy[child] = child_root
            child_root.pos, child_root.quat = wum.pos_quat_from_tf(loc_tf)
            # find parent link body
            plnk_bdy = self._rutl2bdy[m.plnk]
            child_root.parent = plnk_bdy
            plnk_bdy.children.append(child_root)
            if isinstance(child, type(mecba)):
                self._attach_mountings(child)

    def _apply_collision_filter(self, sobj, geom):
        geom.contype = int(sobj.collision_group)
        geom.conaffinity = int(sobj.collision_affinity)

    def _merge_empty_geoms(self, body, is_root=False):
        for child in list(body.children):
            self._merge_empty_geoms(child, is_root=False)
        if is_root:
            return
        parent = body.parent
        if parent is None:
            return
        # A moving or explicitly inertial frame is not an empty structural stub.
        # Folding it changes joint pivots/axes and can discard spacer inertia.
        if parent.inertial is not None or any(not j.actuated for j in parent.hosting_jnts + body.hosting_jnts):
            return
        merge = 0
        if len(parent.geoms) == 0 and len(body.geoms) > 0:
            merge = 1
        if len(parent.geoms) == 0 and len(body.geoms) == 0:
            merge = 2
        if merge == 0:
            return
        # Folding moves the PARENT onto the child's frame, so everything that was
        # expressed relative to the parent moves with it. That is only sound when
        # the child is the parent's ONLY child: a sibling's pos/quat is relative
        # to the OLD parent frame, and folding would silently shift it in the
        # world without touching its numbers.
        if len(parent.children) > 1:
            return
        # BodyNode.pos/quat are written by pos_quat_from_tf, i.e. quat is XYZW.
        # rotmat_from_quat (scipy) reads XYZW as well; tf_from_quat_pos does NOT
        # -- it reads WXYZ, which turns an identity quat into a 180-deg z flip.
        ptfmat = wum.tf_from_pos_rotmat(parent.pos,
                                        wum.rotmat_from_quat(parent.quat))
        ctfmat = wum.tf_from_pos_rotmat(body.pos,
                                        wum.rotmat_from_quat(body.quat))
        newtfmat = ptfmat @ ctfmat
        # Parent moves onto the child frame; its existing sites must not move in
        # world space. Child sites are already expressed in the destination frame.
        for site in parent.sites:
            tf = np.linalg.inv(ctfmat) @ wum.tf_from_pos_rotmat(site.pos, wum.rotmat_from_quat(site.quat))
            site.pos, site.quat = wum.pos_quat_from_tf(tf)
        parent.pos, parent.quat = wum.pos_quat_from_tf(newtfmat)
        if merge == 1:
            parent.geoms.extend(body.geoms)
            parent.inertial = body.inertial
        parent.sites.extend(body.sites)
        parent.hosting_jnts.extend(body.hosting_jnts)
        for gc in body.children:
            gc.parent = parent
            parent.children.append(gc)
        parent.children.remove(body)
        self._bdy_alias[body] = parent

    def _finalize_alias_maps(self):
        for k, b in list(self._sobj2bdy.items()):
            self._sobj2bdy[k] = self._resolve_body(b)
        for k, b in list(self._rutl2bdy.items()):
            self._rutl2bdy[k] = self._resolve_body(b)

    def _resolve_body(self, b):
        while b in self._bdy_alias:
            b = self._bdy_alias[b]
        return b

    def _lookup_body_node(self, sobj):
        """BodyNode for an existing runtime Link/SceneObject, keyed by identity.

        Before folding, anchors use the original frame. After finalizing aliases,
        collision excludes use the surviving body. Missing bodies fail loudly.
        """
        b = self._rutl2bdy.get(sobj)
        if b is None:
            b = self._sobj2bdy.get(sobj)
        assert sobj not in self._sobj2site, (
            f"body reference {sobj!r} was converted to a marker site; "
            f"reference its actual parent body instead")
        assert b is not None, (
            f"body reference {sobj!r} has no body in the "
            f"model (not a converted link / scene object)")
        return b
