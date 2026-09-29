# Citrus V2 implementation notes

本文记录最初的 cluster V2 实现，以下 rigid fruit / 未实现 breaking 的描述属于该基线。
当前可选的 `breakable_axial` 适配、WRS 通用连接和 reset/substep 语义见 [HARVEST.md](HARVEST.md)；
默认 rigid 模式保持兼容，既有 passive joint 能力没有重新实现。

## Audit before implementation

已阅读 V1 全部模块、两套配置和文档，以及 SceneObject / primitive / RenderModel /
collision shapes、MechStruct / MechBase、MJ nodes / converter / compiler / env / runtime /
inertial / synchronizer 和 API_INDEX。

检查结论：

- V1 的纯数据、双面叶 mesh、native cylinder/capsule/sphere、确定性生成和 reference fitting 可复用。
- SceneObject 是扁平世界位姿；RenderModel / collision shape 的 loc_tf 相对所属 SceneObject。
- MechBase 原先按类缓存结构，并复制 runtime links；procedural plant 需要显式实例结构。
- mount 是原生跟随机制，负责对象世界位姿和 scene 成员；可以直接用于 batch 和固定果实。
- converter 原先给所有 revolute/prismatic joint 建 actuator，JointNode 已有 damping/frictionloss/armature，
  但 Joint 无法传递这些参数，也没有 stiffness/springref。
- 原生 STATIC collision role 已允许与 ACTIVE 交互、屏蔽角色内部接触，足够支持当前 V2。
- collision-free visual mount 转为 marker site，不产生 body/DOF；有碰撞的固定 mount 没有 free joint。

## Final layout

```text
agriculture/
  __init__.py
  skeleton.py             StemSegment / PlantSkeleton / local growth frames
  spec.py                 PlantSpec / placements / validation / JSON
  compat.py               V1 constructor, name and JSON adapters
  config.py               species + preset composition
  generator.py            protocol + reference/procedural entry points
  reference.py            V1 fitted macro + micro twig generation
  morphology.py           optional pipe model, copy-on-transform
  foliage.py              skeleton -> leaf placements
  fruit.py                skeleton -> fruit placements
  geometry.py             V1 folded/curled double-sided blade and batching
  static.py               static builder + shared WRS primitive factories
  dynamics.py             pure cluster/profile/proxy settings and validation
  dynamic.py              cluster-local geometry, MechBase and mounts
  configs/species/citrus.json
  configs/presets/lab_citrus_v2.json
  configs/presets/generic_citrus_v1.json
  configs/presets/generic_plant.json
  README.md / ARCHITECTURE.md / .gitignore

examples/agriculture/
  citrus_tree/            retained V1 entry, tests, reference notes; compatibility modules/configs
  dynamic_citrus.py       static / clusters / push cases and nine debug flags
  generic_plant_demo.py
  push_experiment.py      test probe and measured contact/rebound trajectory

tests/
  test_agriculture_geometry.py
  test_agriculture_dynamics.py
  test_passive_joints.py
```

The independent top-level package avoids importing the eager `wrs.__init__`
when generating geometry. Only `static.py` / `dynamic.py` import WRS. Packaging
includes agriculture and its JSON data; no dependency was added. Pure tests
actively block imports of wrs, mujoco, newton and warp while generating and exporting.

## OrchardBench reading and provenance

Reviewed upstream commit `6313313db8b1a7d23fb2cc3afd67cac46f29399a` (Apache-2.0).
No nontrivial upstream source code was copied into this repository. The existing
WRS V1 leaf implementation was retained. The new implementation follows the ideas
below, with WRS conventions and APIs:

| Source | Idea adopted | Deliberately different here |
| --- | --- | --- |
| [skeleton.py](https://github.com/humphreymunn/orchardbench/blob/6313313db8b1a7d23fb2cc3afd67cac46f29399a/treesim/skeleton.py) | Pure skeleton IR, topology and geometry queries | Generic PlantSkeleton, string IDs, derived children/depth, 3×3 rotations |
| [lsystem.py](https://github.com/humphreymunn/orchardbench/blob/6313313db8b1a7d23fb2cc3afd67cac46f29399a/treesim/lsystem.py) | Growth frame +Z, explicit seed, optional pipe taper | Small procedural grower; no copied turtle grammar or quaternion math |
| [foliage.py](https://github.com/humphreymunn/orchardbench/blob/6313313db8b1a7d23fb2cc3afd67cac46f29399a/treesim/foliage.py) | Attached placements, folded blades and few geometry batches | Own V1 blade +X length convention; cluster-local WRS meshes; no Newton mesh/instancing API |
| [fruit.py](https://github.com/humphreymunn/orchardbench/blob/6313313db8b1a7d23fb2cc3afd67cac46f29399a/treesim/fruit.py) | Placement record separate from runtime representation | Generic FruitPlacement, rigid mount; no tether kernel or rupture state |
| [config.py](https://github.com/humphreymunn/orchardbench/blob/6313313db8b1a7d23fb2cc3afd67cac46f29399a/treesim/config.py) | Explicit morphology/physics settings | Species defaults, generator selection and scene preset separated |
| [builder.py](https://github.com/humphreymunn/orchardbench/blob/6313313db8b1a7d23fb2cc3afd67cac46f29399a/treesim/builder.py) | Geometry-to-engine adapter, stable mappings | Static/dynamic builders; many segments per physical cluster, native WRS mounts |
| [domainrand.py](https://github.com/humphreymunn/orchardbench/blob/6313313db8b1a7d23fb2cc3afd67cac46f29399a/treesim/domainrand.py) | Pure parameter/geometry transformations with explicit RNG | Only seeded generation and optional morphology utility; no batched DR framework |
| [physics.py](https://github.com/humphreymunn/orchardbench/blob/6313313db8b1a7d23fb2cc3afd67cac46f29399a/treesim/physics.py), [PHYSICS.md](https://github.com/humphreymunn/orchardbench/blob/6313313db8b1a7d23fb2cc3afd67cac46f29399a/PHYSICS.md) | Separate restoring dynamics settings, task-relevant fidelity | Qualitative native MuJoCo stiffness/damping; no beam fitting, Newton D6, spring force kernels or breaking |

The WRS adapter reuses native primitives, collision classes, MechBase/mount and
inertial utilities. The pure geometry layer has only two small NumPy direction/frame
operations to avoid importing WRS engines; the growth-frame convention agrees
with WRS `rotmat_from_normal`. It introduces no competing transform or physics library.

## Static data flow

`species + preset -> generator -> PlantSpec -> StaticPlantBuilder -> SceneObjects -> Scene`

Per-segment native cylinder visuals approximate taper; capsule colliders use maximum
radius and authored collision flags. Global plant-local leaf batches are grouped by
shade. Each fruit keeps its own SceneObject origin at its centre and sphere collision.
Applying a world transform does not mutate the spec.

## Dynamic data flow

`PlantSpec + PlantDynamicsSpec -> DynamicPlantBuilder -> MechStruct/MechBase/mount`

`Scene -> MJWRSConverter -> BodyNode/JointNode -> MJCFCompiler -> MJEnv -> WRS pose sync`

The lab preset selects three fruit-bearing supports and thirteen nearby outer
shoots in the central front approach region. The top, rear, and other peripheral
foliage remain visual-only. Selection is authored in cluster_roots and does not
change with the viewer camera. Each root owns its connected descendant subtree.
Nested roots are supported; unassigned descendants of moving members are rejected.
Trunk and primary branches stay fixed. The partition has 16 clusters, 65 moving
geometric segments, 19 passive DOF, and 20 links including the fixed root.

The cluster rest frame originates at its root segment start. Its rotation is the
segment's stored 3×3 frame. Parent-relative joint transforms are
`inverse(T_plant_parent_cluster) @ T_plant_cluster`. One DOF uses local X; two DOF
use serial local X/Y hinges with a light, explicitly inertial spacer. There is no twist DOF.

All branch vertices/models, capsule offsets, leaf vertices and proxy boxes are
transformed into their cluster frame once. Leaf batches are keyed by owner cluster
and shade, never combined across moving links. A fixed mount supplies world pose.
The fruit sphere/stem stay in a named mounted SceneObject; `instance.fruits[id].pos`
is its current world centre. Geometry/data never store runtime objects or IDs.

Each leaf owned by a dynamic cluster receives three thin strip boxes, fitted to
contiguous stations of the existing blade mesh. Each strip follows its midrib chord
and encloses fold/curl/droop; its normal follows the blade, not the branch axes.
Boxes never span different leaves, so the space between separate blades stays open.
Padding is 0.5 mm per side; minimum normal thickness is 1.5 mm (full extent).
Actual default thickness is about 3.3–5.2 mm including fold/curl and padding. This
is a conservative contact approximation; rectangular strip corners still overhang
the tapered silhouette. The thickness is numerical, not measured leaf thickness.

All boxes of one nonempty cluster share ONE mounted collision-only SceneObject.
The lab preset has 399 contacting leaves: 1,197 box shapes in 15 compound objects,
with 44 visual leaf batches (3 static). The remaining 1,129 leaves are visual-only.
This bounds the contact workload while retaining the central front approach shoots.
There are no leaf DOFs or mesh colliders. Empty foliage produces no proxy objects.
`foliage_proxies[cluster]` is a list of compound objects; inspect `collisions` for
individual strips. A strip's world pose is `object.tf @ shape.loc_tf`.

`show_foliage_proxies` batches each compound object's exact boxes into one debug
mesh, using the existing WRS box geometry. It adds only 15 drawables instead of
1,197 separately streamed collision models. Debug mounts have no collision/mass
and are removed when hidden; they follow the same cluster as their physical boxes.
Exclude `foliage_proxy_visuals` from sensor captures (the robot example does this).
The ordinary per-shape collision flag is not used for these compound objects.

Summary fields distinguish `foliage_proxy_count` (shapes), `foliage_proxy_object_count`
(objects), `foliage_contact_leaf_count`, and `visual_only_leaf_count`.
The demos still select an actual physical strip, not the batched display mesh.

`FoliageProxySettings` uses `sections_per_leaf`, `padding`, and `minimum_thickness`.
The old coarse-envelope keys `per_cluster` / `minimum_half_extent` are replaced;
custom older presets must replace those keys too. Section counts must be positive
and no larger than `LeafShape.stations - 1`. All strips move rigidly with their
existing passive cluster; this does not simulate independent leaf deformation.

Wood mass is estimated from tapered-segment volume and configured density. Native
WRS collision inertia utilities approximate each cluster's inertia, including lumped
leaf mass. Fruit mass/inertia are on the fixed fruit body. Proxy bodies have explicit
zero inertia/mass to avoid counting their box volume as solid vegetation; their
parent cluster already carries the leaf mass. No physical constants were identified
from the reference video.

Semantic groups are TRUNK, HARD_BRANCH, COMPLIANT_BRANCH, FOLIAGE and FRUIT.
Dynamic instances also map actual collision shape objects to these labels, because
one rigid link may contain different branch roles. These mappings live on the instance.

## Minimal WRS changes

| File | Change |
| --- | --- |
| `wrs/robots/base/mech_base.py` | Optional keyword-only `structure=`, separate from class cache; clone retains the structure and cloned mount ownership |
| `wrs/robots/base/mech_structure.py` | Optional Joint actuated/damping/stiffness/springref/frictionloss/armature; propagated through FlatMechStructure |
| `wrs/physics/mj_nodes.py` | JointNode stiffness, springref, actuated data |
| `wrs/physics/mj_wrs_cvter.py` | No actuator for passive joints; parameter forwarding; honor explicit inertia without colliders; retain passive/inertial spacer frames during empty-body folding |
| `wrs/physics/mj_compiler.py` | Emit native stiffness and springref attributes |

Defaults remain actuated=True, damping=1, frictionloss=.01, armature=.02,
stiffness=0, springref=0. Existing positional Joint calls and robot constructors remain valid.
Kinematic `active_jnt_ids_mask` retains its original meaning (non-fixed/non-mimic);
passive joints still have qpos/DOF and participate in FK. Actuation is a separate flag.

No plant-specific MJCF code, actuator-as-spring control loop, D6 abstraction, solver
replacement or SceneObject semantic modification was added. MJEnv/MJRuntime API did not change.

## Verification and scope

The complete existing tests directory plus new tests passes (34 cases), as does the
19-case V1 suite. Added coverage includes pure topology/frames/serialization/RNG,
pipe taper, multiple roots, cluster partition rejection, nested clusters, instance
structures, passive attributes in compiled MuJoCo, static/dynamic rest equivalence,
rigid world transforms, empty foliage/fruits, and real contact/rebound/settling.
RS007L and XYTheta public construction/FK/clone/actuation were checked. This does
not claim every interactive robot example or hardware driver was exercised.

The V1 static, V2 cluster-debug, V2 contact-push and generic procedural demos were
also launched in the native World and inspected in a rendered browser viewer;
all four exited successfully without page errors. A separate export of tracked
HEAD, overlaid only with this change's five core files, packaging and agriculture
sources, also completed the headless MuJoCo push demo. It does not depend on the
unrelated sensor/UI work currently present in the development workspace.

The probe demo first settles gravity, positions a native actuated slider/sphere at
the selected proxy, pushes, retracts and releases. The plant has zero actuators.
Representative default results: 0.153 rad peak deflection, 0.038 m leaf movement,
0.033 m fruit movement, 7 significant post-release crossings, about 0.0025 rad final
offset from the pre-push baseline and less than 0.001 rad late peak-to-peak motion.
Rebound is measured about the gravity-loaded equilibrium, not an assumed q=0 equilibrium.

V1 configuration import paths, TreeSpec/BranchSegment constructors, old JSON loading,
builder handles, original demo and test commands remain available through adapters.
The generic preset now contains procedural parameters rather than lab geometry;
seed 7 gives 92 segments, 378 leaves and 5 fruits.

Future apple work can add a species profile and placement/geometry defaults. Grapevine
or trellis work can supply another PlantGenerator, multi-root skeleton, semantic roles
and an optional pure morphology transform, then provide a corresponding cluster
DynamicsSpec. A new leaf/fruit shape would extend the geometry factory/adapter;
support ties or other constraints would need a separate dynamics extension. These
extensions, full L-systems, pruning, breaking, detachment, wind, individual leaf
dynamics, full DR, sensors and RL are intentionally outside this V2 implementation.
