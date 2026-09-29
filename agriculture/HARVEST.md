# 果实轴向弹性连接与拉脱

本实现基于 `fruit_harvest` 分支的 `5cbe75e829d8aa28701a0ad25be036e80c25320a`，
实际验证环境为 Python 3.12 / MuJoCo **3.13.0**。不引入新第三方依赖、不修改 STL、
手眼标定文件或相机噪声模型。下面参数全部是 **synthetic/demo profile**，不是柑橘实测参数。

## 运行与调参

在 WRS2 根目录运行；已有交互例子默认仍用 rigid fruit。

```powershell
# 完整树 + FAFUGripper + 腕部 D405。左侧向下滚动到 Harvest。
.\.venv\Scripts\python.exe -m examples.agriculture.fruit_harvest

# 自动展示：拉脱、保持两秒、张手落地（也可选择 pull_hold / weak_grip）
.\.venv\Scripts\python.exe -m examples.agriculture.fruit_harvest --case release

# 无界面跑三个场景，每个物理子步记录 CSV，输出报告 JSON
.\.venv\Scripts\python.exe -m examples.agriculture.fruit_harvest --headless --case all --capture-rgbd --output harvest_results

# 阶段 A/B：测试载荷，无夹爪；同时输出 fixed.xml / branch.xml
.\.venv\Scripts\python.exe -m examples.agriculture.attachment_validation --output harvest_results

# 阶段 C 的前置验证：正常 FAFU 双指举升一颗无果梗的自由果实
.\.venv\Scripts\python.exe -m examples.agriculture.free_grasp_validation --output harvest_results

# 16 个新增 headless 测试（含真实 GPU 相机采集，但不需要 viewer）
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_elastic_attachment.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_fruit_harvest.py -v
```

配置集中于 [`configs/presets/lab_citrus_harvest.json`](configs/presets/lab_citrus_harvest.json)，
继承原来的 robot/tree preset。`fruit_attachment.mode` 为 `rigid` 或 `breakable_axial`。
省略配置时保持刚性挂载；静态 builder 不受该动态配置影响。

| 参数 | 当前 demo 值 | 含义 |
| --- | ---: | --- |
| density_kg_m3 | 850 | 球体质量和惯量的合成密度 |
| stiffness_N_m | 200 | 轴向线性刚度，N/m |
| damping_N_s_m | 2 | 轴向阻尼，N·s/m |
| break_force_N | 8 | 严格大于此拉力时累积超载时间，N |
| overload_hold_s | 0.040 | 连续超载保持时间，s |
| pull_speed_m_s | 0.025 | TCP 后拉命令速度，m/s |
| finger_force_N / weak_finger_force_N | 20 / 0.5 | 每指伺服力上限，N |
| sliding_friction | 0.8 | 手指/果实滑动摩擦系数 |
| torsional_friction_m | 0.005 | 有限接触斑等效 spin friction 长度，m |
| contact_dimension / contact_impratio | 4 / 10 | 原生软指接触及椭圆摩擦锥求解设置 |
| contact_time_constant_s | 0.01 | 接触模型时间常数，s |

机械臂六轴限力矩也在该 demo 中显式设置为 `[80, 80, 60, 30, 20, 15]` N·m，
仅为合成的仿真限幅，不能当作 FAFU 硬件规格。双指各自具有独立 prismatic DOF 和限力位置伺服，
相同目标允许不对称的实际接触响应。没有调用 `hold()`、没有果实到夹爪的 mount，也没有 grip assist。

`grasp_offset_world_m` 将球体放在指尖内侧约 20 mm 的抓持深度，并给 TCP 命令留出约 13 mm 的
重力跟踪误差补偿；它只改变机械臂伺服目标，不改变果实 pose 或受力。
此预设只针对当前 `orange_000` 和工作区，不是自动抓取规划器。

MuJoCo 的软接触会有缓慢滑移；接触点数量或法向载荷不能单独保证抓持。
使用椭圆摩擦锥和有限 `impratio` 减少接触求解产生的滑移，仍保留有限摩擦和执行器限力。
参考 [MuJoCo 接触与滑移说明](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip)。
正常抓持案例仍有毫米级相对位移，不能声称无滑移。

## 数据流、连接点和位姿所有权

```text
JSON preset → PlantSpec / PlantDynamicsSpec（原有纯数据模型不变）
            → DynamicPlantBuilder
              ├─ branch clusters：原有 passive joints
              ├─ plant.fruits[id]：独立 floating SceneObject + sphere collision
              ├─ plant.fruit_connections[id]：通用连接规格，引用实际 body 对象
              └─ plant.stem_objects[id]：仅渲染的伸缩圆柱
            → Scene.connections
            → converter 两个 site → compiler spatial tendon
            → MJRuntime：mj_step、读取 tendon 状态、子步断裂策略
            → MJEnv.sync_scene：现有位姿同步 + 连接视觉更新 → viewer / D405
```

原有 `FruitPlacement` 保留 `parent_segment`、`attachment_t`、`stem_direction`、radius、stem_length。
`PlantSpec` 中没有 SceneObject、MuJoCo ID 或断裂状态。

在 plant-local rest frame 中，`A = parent_segment.point_at(attachment_t)`，
`B = fruit.position + radius * stem_direction`。连接规格中的 A 使用实际 owner cluster 的局部坐标
`R_cluster.T @ (A - p_cluster)`；B 使用 fruit body 局部坐标 `radius * stem_direction`。
`L0 = norm(A - B)`，与无载果梗长度一致。它不包含果实半径，也不依赖 TCP 后退量。
零长度果梗不允许使用 `breakable_axial`。

converter 在合并空 body **之前**绑定 site，并在空 body 合并导致父坐标系移动时变换原父 site。
编译名称由 WRS name allocator 分配，runtime 通过连接规格对象的 identity 查找 handle，
所以多个植物可以重复使用 `orange_000` 等植物名称。测试覆盖非平凡世界旋转/平移及合并前后的锚点。

`plant.set_pos_rotmat()` 对可脱落植物只允许在加入 scene 前用于初始放置。
加入 scene 后自由果实没有 mount，其 pose 由物理同步写入；植物 FK 不会把它写回 reference pose。
果实显式使用 ACTIVE collision group，继续与手指、地面和 STATIC 植物几何接触。
每颗果实一个 body、一个球形碰撞体；没有果梗 mesh collision。

旧果梗 visual 在 detachable 模式下不再属于 fruit body。替代的圆柱由两个当前 site 位置决定，
只缩放 visual transform，不重建 mesh/body；断裂后 alpha=0，reset 后恢复。D405 同样忽略 alpha=0 表面。

## 子步与 reset 语义

每颗连接使用两个 site 的原生 spatial tendon，单值 `springlength=L0`：

```text
extension = ten_length - L0
F_elastic = k * extension
F_damping = c * ten_velocity
F_tensile = max(0, F_elastic + F_damping)
```

`ten_velocity` 是两端相对距离的变化率，含枝簇运动，不能用 fruit/TCP 的单独速度替代。
MuJoCo 3.13.0 中实际使用的是 `ten_length` / `ten_velocity`，没有读取虚构的 `data.tendon_force`。
单值原生 spring 在压缩时也产生恢复力；只有正向拉力参与断裂判定。

每个 `mj_step` 后执行 `mj_forward` 更新当前子步的派生量，然后更新连续超载时间；
`F_tensile <= break_force` 时清零，达到保持时间触发一次 event。断裂只把该 tendon 的
`stiffness` / `damping` 置零，再 forward 清除旧被动力；没有额外 Python spring force。
同一次模型始终保留同一颗 free body，断裂代码不写 qpos/qvel、不加 impulse、不重新编译。

连接没有 actuator/equality/weld，`limited=false`、`frictionloss=0`、`armature=0`，
非线性系数保持零。没有对 tendon 使用 `eq_active`。
独立 `runtime.step()`、`runtime.step(N)` 和 `MJEnv.step()` 都走同一个子步策略。
forward、collision query、相机 capture 和 GUI 暂停都不推进断裂计时。

```python
connection = plant.fruit_connections['orange_000']
handle = env.connection(connection)
print(handle.report())
print(plant.fruits['orange_000'].pos)  # 最近一次 scene sync 后的世界坐标

checkpoint = env.snapshot()
env.runtime.step(100)
env.sync_scene()                     # 直接用 runtime 时显式发布位姿/连线
env.restore(checkpoint)              # 同时恢复物理、连接状态、连线显示
env.reset()                         # 恢复 env 初始化时的基线
```

snapshot 保存 `mjSTATE_INTEGRATION`、tendon stiffness/damping/rest length、attached、超载计时和 events；
不能只调用 `mj_setState`。每个 MJRuntime 从 XML 创建自己的 mjModel。
snapshot 只能还给原 runtime；环境之间不共享模型参数。它不序列化任意用户修改的全部模型配置。
`runtime.save_reset_state()` 可显式选择新的物理 reset 基线；
交互 demo 用重力稳定后的 `env.snapshot()` 保存起点，同时重置伺服命令和夹爪限力。
低层 runtime 不持有 renderer，直接调用它的 restore/reset 后需 `env.sync_scene()`；
优先使用自动同步的 `env.restore/reset`。

## 验证和诊断

- **A：固定锚点。** 0.2 kg 果实、200 N/m 弹簧，平衡伸长约 9.81 mm、拉力 1.962 N；
  加 1 N 后伸长增加 5 mm。持续超载导致断裂，测试载荷停止后由重力驱动。
  此最小 fixture 无地面，断后自由落体；落地碰撞在 C 中验证。
- **B：两个动态枝簇、两颗果实。** 拉力在断裂前牵动枝条；断后受拉枝条回弹，
  该果实自由运动，第二颗仍 attached。既验证 native reaction，也验证选择性断裂。
- **C0：无果梗举升。** 原生 FAFUGripper 安装在单轴试验台，接触抓住 stand 上的独立果实，
  举升约 75 mm、保持、张手落回 stand。零 tendon、零 equality。
- **C：完整树 + FAFUArm。** `pull_hold`、`weak_grip`、`release` 分别验证拉脱保持、
  先滑脱且果梗不断、断后张手落地。不是把 detached 自动计作采摘成功。

CSV 逐物理子步记录：实际/目标/命令开口、各指实际行程、限力、实际 actuator effort、
各指与**目标果实**的 contact 数和法向载荷、fruit 相对实际 TCP 的位移/速度、
枝簇偏转、extension/tension/overload/attachment_state/grasp_state。
相对速度使用 native Jacobian，并扣除 TCP 平动及转动项。
`bilateral_stable` 是“当前双指接触、速度/位移低于配置阈值”的观测标签，不是力闭合证明。
断后 extension 仅表示两个 anchor 的间距超过原 L0 的量，连接力已经为零。

自动测试还覆盖 qpos/qvel 与未断裂原生步进的连续性对照、snapshot 中途超载恢复、reset 重现、
两环境隔离、dt=2/1 ms、不同 scene 发布频率、body 名称重复、无叶/无果、旧 rigid/static 模式、
以及腕部 D405 在断后读取物理果实且不推进时间。
完整树改变显示刷新频率时断裂子步相同；现有 float32 mounted-TCP FK/IK 的舍入会带来微小力值差异。
仅在上述当前版本/配置验证，未验证其他 MuJoCo 版本或实机。

## 本次文件

新增：

- `wrs/physics/connections.py`：通用 body-local 连接规格、拉力阈值和 runtime handle。
- `agriculture/attachment.py`、`configs/presets/lab_citrus_harvest.json`：植物适配和合成参数。
- `examples/agriculture/{attachment_validation,free_grasp_validation,fruit_harvest,harvest_diagnostics}.py`：
  A/B/C 验证、交互演示与目标果实诊断。
- `tests/test_elastic_attachment.py`、`tests/test_fruit_harvest.py`、本文。

修改：

- `wrs/scene/scene.py`：connection 注册和 physics sync callbacks。
- `wrs/physics/{mj_nodes,mj_compiler,mj_wrs_cvter}.py`：两点 tendon/site 编译和 body 合并的 site 坐标保持。
- `wrs/physics/{mj_runtime,mj_env}.py`：子步策略、独立模型、完整连接 snapshot/reset、可视化同步入口。
- `agriculture/{dynamic,static}.py`：可选自由果实、scene 生命周期和旧 stem visual 开关。
- `examples/agriculture/robot_citrus_interaction.py`：使用完整 snapshot、子步观测回调、可配置有限接触/伺服参数。
- `agriculture/{README,ARCHITECTURE}.md`、`docs/API_INDEX.md`：入口和 API 文档；
  `agriculture/.gitignore` 显式保留新增的 HARVEST.md。

没有重做已有 passive joint，没有修改 `PlantSpec`、通用生成器、数学工具、机器人 STL、
MuJoCo synchronizer 本体或 D405 实现。保留工作区原有未提交的相机等文件。

## 明确不支持

这是 **pull-oriented axial attachment**。单根轴向连接不提供独立弯曲/扭转刚度，
不能据此准确评价扭摘。没有系统辨识、柑橘材料标定、FEM、Cosserat rod、疲劳、逐叶动力学、
自动抓取规划、真实电机/指垫/果皮辨识，也没有 hold assist、重力扣除或 reaction clipping。
