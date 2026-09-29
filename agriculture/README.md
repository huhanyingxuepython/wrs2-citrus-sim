# Agriculture / Citrus V2

轻量植物几何与 WRS 构建适配层。当前真正支持实验室 citrus reference tree、通用程序化树、
静态构建以及少量 branch-cluster 被动弯曲。没有引入 Newton、Warp、外部植物资产或新的依赖。
完整设计、OrchardBench 阅读记录、core 变更和范围见 [ARCHITECTURE.md](ARCHITECTURE.md)。
新增的独立果实、原生 tendon 弹性连接、真实夹爪拉脱与 A/B/C 验证见 [HARVEST.md](HARVEST.md)。

## 运行

在 WRS2 根目录，使用已安装项目依赖的 Python 3.12+：

```powershell
# 原 V1 入口仍然可用
.\.venv\Scripts\python.exe -m examples.agriculture.citrus_tree.demo_static_tree

# 同一份 PlantSpec：静态、枝簇检查、真实接触推动
.\.venv\Scripts\python.exe -m examples.agriculture.dynamic_citrus --case static
.\.venv\Scripts\python.exe -m examples.agriculture.dynamic_citrus --case clusters
.\.venv\Scripts\python.exe -m examples.agriculture.dynamic_citrus --case push

# FAFU 笛卡尔移动 + 实时叶簇/果实接触
.\.venv\Scripts\python.exe -m examples.agriculture.robot_citrus_interaction

# 果梗弹性拉脱：左侧滚动到 Harvest，区分抓持和连接状态
.\.venv\Scripts\python.exe -m examples.agriculture.fruit_harvest

# 确定性通用程序化路径
.\.venv\Scripts\python.exe -m examples.agriculture.generic_plant_demo --seed 7
.\.venv\Scripts\python.exe -m examples.agriculture.generic_plant_demo --seed 21

# 无 viewer 的真实 MuJoCo 接触验证及轨迹导出
.\.venv\Scripts\python.exe -m examples.agriculture.dynamic_citrus --case push --headless --report push_report.json --trace push_trace.json

# 回归测试：工作区 tests（含 agriculture / 机械臂交互）+ 原 V1 的 19 项
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s examples/agriculture/citrus_tree -t . -p "test_*.py" -v
```

新例子也支持直接运行脚本。`--port` 选择 viewer 端口。
静态/枝簇例子支持 `--duration 5` 有界退出；push 跑完接触、回弹和稳定阶段后退出，网页保留最终场景。
无浏览器自动弹出时设置 `$env:WRS_VIEWER_NO_BROWSER='1'`。

## 机械臂交互示例

`robot_citrus_interaction.py` 使用 FAFURobotArm 和仓库原生 FAFUGripper 的掌部、双指网格及碰撞模型。
本例夹爪使用两个独立限力位置伺服手指，初始开口由 `robot_demo.gripper.opening_m`（0～85 mm）决定。
机械臂安装在 0.30 m 高的底座上，树的尺度不变。左侧提供前后、左右、上下六个按钮，
`Move per click` 调节每次移动 1～30 mm，默认 10 mm。方向使用固定世界坐标：
前为 +Y（朝树）、后为 −Y、右为 +X、左为 −X、上为 +Z、下为 −Z，不随相机旋转改变。
TCP 是夹爪抓取中心，位于法兰 +Z 前方 170 mm，朝向保持固定。
右侧显示目标/实际 TCP 坐标、接触类别、植物偏转和果实世界坐标。

使用 WRS NumIKSolver，从相邻解出发检查整条直线路径；默认每 5 mm 一个 waypoint。
关节解之间插值近似直线，同时限制 TCP 命令速度与各关节命令速度。UI 不直接修改 FK/qpos，
机械臂实际位姿由 MuJoCo 位置伺服与接触共同决定，受重力和接触影响可有毫米级跟踪误差。
越界、不可达或出现过大 IK 跳变时，整次请求被拒绝并显示原因，原目标/原路径保留。
这不使用默认 SELIK，因此首次运行不生成采样数据库。

可以先点击 **Touch leaves** 或 **Touch orange_000**，观察枝簇/果实让位，再点击
**Retract / ready** 观察回弹。**Stop motion / hold** 取消后续移动并保持当前伺服命令；
**Pause physics** 暂停仿真，暂停时可调整待执行目标；
**Reset robot and tree** 恢复初始化后的完整物理状态。复位保留暂停和显示开关状态。
接触代理、机械臂/枝条/果实碰撞形状和接触力箭头均有独立显示开关。

叶片接触代理采用贴合叶片朝向的分段薄盒：每片参与接触的叶子分为 3 段，
沿实际 blade mesh 的卷曲方向拟合，保留不同叶片之间的空隙。
同一枝簇的所有薄盒合并在一个碰撞对象中，仍通过原有簇关节弯曲/回弹。
`Show foliage contact proxies` 显示实际碰撞形状；取消勾选只隐藏显示，不关闭碰撞。
可在 `lab_citrus_v2.json` 的 `dynamics.foliage_proxy` 调整 `sections_per_leaf`、
`padding`（默认每侧 0.5 mm）、`minimum_thickness`（默认全厚 1.5 mm）。
实际默认法向包络厚约 3.3–5.2 mm，包含叶片折叠、卷曲和外扩，不是实测叶厚。
薄盒边角仍是矩形近似，叶片随簇刚性运动，不提供单叶柔性变形。
旧配置的 `per_cluster` / `minimum_half_extent` 需替换为上述参数。

交互参数集中在 `configs/presets/lab_citrus_robot.json`，继承 lab citrus preset：
机械臂基座、夹爪开口、示范目标、位置伺服增益、笛卡尔/关节速度、工作区与相机均可调整。
触碰预设从 `gripper.contact_link` 指定手指的原生网格提取指尖，将实际指尖对准表面。
示范目标同样走上述笛卡尔路径。它是便于探索接触的局部控制，不是避障轨迹规划。
机械臂有六个 actuator，双指有两个 actuator，植物使用 19 个 passive DOF。叶片通过 V2 的枝簇代理接触；
原 contact preset 的橙子仍与枝条刚性连接；`fruit_harvest` 使用独立 free fruit 和可断轴向弹性连接。
夹爪连杆沿用原生 ACTIVE 碰撞角色，
手指与掌部参加真实接触；相机示意外壳只有视觉。

无界面接触与撤回验证、或限制 viewer 运行时间：

```powershell
python -m examples.agriculture.robot_citrus_interaction --headless --report robot_contact.json
python -m examples.agriculture.robot_citrus_interaction --port 8001 --duration 30
python -m unittest discover -s tests -p "test_robot_citrus_interaction.py" -v
```

FAFU 的质量/惯量/电机参数尚未标定。交互例子的双指取消 kinematic mimic，使用相同目标的
独立 native 位置伺服，允许真实接触导致的非对称开度；原始 STL 不变。
`fixed_opening` 可选模式仍保留。拉脱新增的通用 physics 能力及限幅/摩擦说明见 [HARVEST.md](HARVEST.md)。
机器人支持范围见
[FAFU 支持范围](../docs/tutorials/fafu_robot_arm.md#用于-citrus-交互仿真的支持范围)。

### 腕部 VirtualD405

同一个交互 demo 默认安装 VirtualD405。右侧面板显示同次采集的 RGB、深度图、
有效深度比例、采样点数、深度范围及世界坐标点云 bbox；勾选 **Show world point cloud (cyan)**
可在三维场景中叠加青色点云。面板向下滚动可使用原有接触、碰撞、暂停等控件。
默认 320×240、目标采集频率 10 Hz、有效光轴深度 0.07–0.50 m，黑色表示无效深度；
点云每 4 个像素采样一次。实际采集耗时显示在面板中，首次 GPU 初始化可能较慢。
RGB/depth 使用当前视觉表面，叶簇接触代理、碰撞线框、力箭头和显示点云不参与成像。

RGB 默认开启面光照（主光、补光和环境光），背景为与 viewer 相同的浅色。
`render_far_m=5` 保留远处枝叶/地面的颜色，深度仍限制在 0.07–0.50 m；
RGB 光照不改变几何真值，也不额外加入 viewer 的黑描边。
右侧新增 **Depth noise (grows with distance)** 开关，默认关闭；开启后
**Disparity noise sigma** 滑块控制视差噪声强度，深度图和点云一起更新。
勾选 **Show world point cloud (cyan)** 可观察点在表面附近散开，距离越远越明显。
面板给出 0.20 / 0.40 m 处的近似 Z 标准差，仅为当前参数的理论估计，不是实机精度指标。
它复用 `StereoDepthNoise`，误差约随 Z² 增长；不是镜头径向畸变。
具体参数位于同一 `robot_demo.d405` 下的 `rgb_lighting`、`rgb_background`、
`render_far_m` 和 `depth_noise`；其中噪声幅度与量化步长均未用真实 D405 标定。

安装使用用户提供的 `configs/wrist_d405_handeye.json`，完整保留原始矩阵与来源数据。
其中 **`T_flange_cam`** 将相机光学坐标转换到法兰坐标，平移为
`[-0.046916617, 0.007351220, 0.050142993]` m，光轴近似沿法兰 +Z。
直接使用 `T_world_cam = T_world_flange @ T_flange_cam`，不再叠加旧 URDF 转轴或 TCP 偏移。
`robot_demo.d405.handeye_file` 可替换标定文件：相对路径基于 `agriculture/configs`，也支持绝对路径。
文件中的 `color_intrinsics` 没有分辨率，本次保留而不应用，RGB-D 仍使用名义内参。
外壳仍为程序化示意，不是精确 D405 CAD，也没有建模安装支架。矩阵定义见
[安装坐标说明](../docs/tutorials/fafu_robot_arm.md#citrus-中的-virtuald405-安装)。
相机使用原生 `mount` 跟随实际法兰位姿，不额外施加质量、碰撞或关节。

```python
# RobotPlantInteraction 实例：
frame = demo.capture_rgbd()                 # 当前仿真状态的一次观测
rgb, depth_m = frame.rgb, frame.depth_m      # 同一帧；深度单位 metre
xyz_world, rgb01 = frame.get_point_cloud(world_frame=True, stride=4)
T_world_optical = frame.T_world_camera      # 随该帧保存的位姿快照
# demo.camera 是 VirtualD405；demo.rgbd.last_frame 是最近一帧。
```

`--headless` 也在接触和撤回时采集，并将点数、范围、位姿写入报告。
无需安装新依赖或在运行时下载 URDF；沿用 WRS 现有 GPU 离屏相机和 UI 图像控件。
没有模拟真实曝光、镜头畸变或标定后的 D405 误差。

```powershell
python -m unittest discover -s tests -p "test_fafu_d405_interaction.py" -v
```

## 数据与 API

```python
from agriculture.config import load_config
from agriculture.generator import generate, ReferenceFittedGenerator, ProceduralTreeGenerator
from agriculture.spec import PlantSpec
from agriculture.dynamics import PlantDynamicsSpec

config = load_config()                       # species + lab preset，普通 dict
spec = generate(config, seed=20260916)        # 不导入 WRS / MuJoCo
spec.validate()
spec.to_json('plant.json')
restored = PlantSpec.load('plant.json')

skel = spec.skeleton
print(skel.roots, skel.terminals)             # segment ID 列表
print(skel.children['fruit_support_001'])     # 派生邻接关系，无重复拓扑状态
print(skel.descendants('front_middle'))
print(skel.depth('cover_upper'), skel.total_length(), skel.summary())

from agriculture.static import StaticPlantBuilder
from agriculture.dynamic import DynamicPlantBuilder
static = StaticPlantBuilder(config).build(spec, pos=(0.5, 0, 0))
dynamics = PlantDynamicsSpec.from_config(spec, config)
dynamic = DynamicPlantBuilder(config).build(spec, dynamics, pos=(0.5, 0, 0))
# static.add_to_scene(world.scene) 或 dynamic.add_to_scene(world.scene)
orange_world_position = dynamic.fruits['orange_001'].pos
orange_rest_position = spec.fruits[1].position
```

- `StemSegment`：id、parent_id、起止点、两端半径、order、attachment_t、rotmat、semantic_role。
  children / depth 从 skeleton 推导，避免与 parent 信息失步。生成器把 3×3 rest frame 保存进 rotmat；
  手写 segment 可省略 rotmat，`frame` 属性按同一 convention 生成。
- `PlantSkeleton`：segments、roots、terminals、children、descendants、depth、bounds、height、total_length、validate、summary。
  支持多个根，用于未来灌木/基生茎；兼容 TreeSpec 仍要求单根位于原点。
- `PlantSpec`：skeleton、leaves、fruits、species、preset_name、metadata、leaf_shape；JSON、校验、bbox、缩放。
  `.branches` 是兼容 V1 的别名。
- 叶和果实使用 `parent_segment` / `attachment_t`；保留 `parent_branch` 属性兼容旧调用。
  叶片保存 rotmat / color_class；二者可带仅含 JSON 值的 visual_metadata，供后续视觉适配使用。
- `PlantGenerator` 是 Protocol：`generate(config, seed) -> PlantSpec`。
  `ReferenceFittedGenerator` 消费手工 macro skeleton；`ProceduralTreeGenerator` 随 seed 改变宏观枝条、叶和果实。
- `apply_pipe_model(skeleton, terminal_radius, beta, tip_taper)` 返回新的 skeleton，默认不用于 reference preset。

全部位置和尺寸使用 metre，角度在数据层使用 rad，配置中 `_deg` 参数使用 degree。
segment 的 local +Z 是生长方向，X/Y 是稳定垂直基；3×3 矩阵列向量表示 local axes 在 plant-local 中的方向。
叶片沿用 V1 blade +X 基部到尖端、+Y 宽度、+Z 正面。两种局部 frame 用途不同。
`T_world_plant` 只由 builder / instance 保存；spec 始终为 rest plant-local。

## 配置与反复调参

真正的参数在 `configs/species/citrus.json` 和 `configs/presets/lab_citrus_v2.json`。
原 `examples/agriculture/citrus_tree/configs/*.json` 是兼容引用，支持继续叠加 override。
配置 `extends` 按 dict 递归合并，列表整项替换；循环引用会报错。

1. 固定 seed，编辑 lab preset 的 macro_branches / growth_profiles / fruit_placements。
2. 静态比较继续使用 V1 demo 的固定相机和 watch：
   `python -m examples.agriculture.citrus_tree.demo_static_tree --config agriculture/configs/presets/lab_citrus_v2.json --watch`。
   watch 监视指定 preset 文件；修改继承的 species 文件后需要重启。
3. `dynamics.cluster_roots` 列出接触枝簇根；成员自动取其子树，嵌套根归最近的 cluster。
   lab preset 覆盖果枝以及中前部、下前部、上前部较低处的接近枝簇，优先保留末端接近区域。
   树顶、远侧和其余外围叶片保留视觉；不对整个前半树冠铺满代理。分组不随 viewer 相机切换。
   动态子树不能留下仍固定的后代，独立 dynamics.validate 会拒绝这种断连。
4. 用 `--case clusters` 检查分组与代理，再调 profile 的 stiffness / damping / limits / mass 参数。
5. 用 headless push 报告比较偏转、真实代理接触、叶果位移、释放后的振荡与稳定。

species 包含典型形态默认值，generator 决定生成方法，preset 决定这一个场景的拟合几何和动态子树。
scale_multiplier 统一缩放几何；动力学系数不自动做相似缩放，proxy padding 是缩放后的米制参数。
V2 目前没有 dynamics 热更新；修改动态配置后重启，以重新编译模型。

## 显示开关

`dynamic_citrus.py` 提供用户要求的九个 `SHOW_*` 常量。命令行有对应开关，例如：

```powershell
python -m examples.agriculture.dynamic_citrus --case push --show-foliage-proxy --show-branch-collision --show-joint-axes
python -m examples.agriculture.dynamic_citrus --case clusters --show-leaves --no-show-skeleton
```

完整集合是 full-plant、skeleton、static-branches、dynamic-clusters、leaves、fruits、
branch-collision、foliage-proxy、joint-axes。full-plant 控制正常枝条显示，叶/果由各自开关控制。
clusters case 默认显示骨架、着色动态簇、proxy 和 joint axes，保持 rest pose 便于检查。
push case 使用真实 MuJoCo 接触；蓝色测试球是唯一有 actuator 的物体。

## 当前数量与边界

lab preset：274 segments、1,528 leaves、3 fruits、16 dynamic clusters、19 passive DOF。
399 片枝簇叶片对应 1,197 个薄盒碰撞形状，合并在 15 个枝簇碰撞对象中；
其余 1,129 片为纯视觉叶片。保留中前部、下前部和果实周围的接触覆盖，树顶及远侧不做全覆盖。
每片接触叶片仍用 3 段薄盒，厚度、fold/curl 拟合精度不变；主干和一级硬枝仍然固定。
原始 8 簇方案的前侧覆盖不足；整前半树冠方案的 51 簇/3,702 个形状又增加了交互负担，
当前采用限定接近区域的 16 簇配置。需要更大接近范围时，在 `dynamics.cluster_roots` 中添加对应细枝根。

动态 builder 共生成 44 个叶片视觉 batch（其中固定组 3 个），按 cluster 和 shade 分组。
WRS articulated links 共 20 个，其中 3 个为两自由度弯曲的中间 link。
叶片没有独立 body/joint；果实固定 mount，没有额外 DOF。V1 静态 builder 仍默认叶片无碰撞。

`foliage_proxies[cluster]` 返回复合碰撞对象列表，每个对象的 `collisions` 为实际薄盒列表；
单个薄盒世界位姿为 `obj.tf @ shape.loc_tf`。显示使用 `plant.show_foliage_proxies(True/False)`，
把同簇所有真实薄盒合成一个调试网格；整个默认树只增加 15 个绘制模型，避免逐薄盒发送和绘制。
关闭显示会移除这些视觉 mount，不影响任何碰撞形状、质量或 DOF。
机器人 UI 和 `dynamic_citrus` 均使用此批量显示。自定义相机采集需
`camera.capture(scene, exclude=plant.foliage_proxy_visuals)`；机器人内置 RGB-D helper 已自动排除。

`summary()` 的 `foliage_contact_leaf_count` / `visual_only_leaf_count` 给出接触覆盖与纯视觉叶片数。
UI 在代理开关旁显示数量，headless 报告的 `foliage_coverage` 记录相同信息。
接触示例通过 `foliage_shape_index` / `push_demo.shape_index` 选择真实接触薄盒。

默认静态形态 bbox 约 `[-.4187,-.3409,-.014] -> [.3288,.2885,1.1762] m`。
所有 bbox/summary 几何数值是 rest spec 的统计；运行中的世界位姿从 instance 读取。
视频尺度仍是未经实测的估计；动力学参数只是定性调节，没有系统辨识。

植物使用 WRS 原生 STATIC **碰撞角色**，因此它与 ACTIVE 机器人/可操作物体接触，
植物内部没有接触，不需要 O(N²) exclude。关节仍然被动物理运动。
该策略也排除了不同植物之间、植物与 STATIC 场景物体的接触；需要这类交互时，
应明确扩展通用碰撞过滤，而不是把叶片换成逐叶 collider。

未实现 wind、detachment、tether、branch breaking、逐叶关节、FEM、soft body、完整 L-system、
pruning/trellis 操作、系统辨识、传感器仿真或 RL。现有 camera tests 仅作为本次 core 修改的回归验证。
