# Static Citrus Tree V1

V2 整理后，纯数据、生成器、叶 mesh 和 WRS 构建实现已迁到 repo 顶层 `agriculture/`。
本目录保留兼容入口和原 V1 测试；配置文件通过 `extends` 引用集中 preset。
新的目录、API、动力学 demo 和调参入口见 [agriculture README](../../../agriculture/README.md)。
使用 `--watch` 时建议指定真正的 preset：`--config agriculture/configs/presets/lab_citrus_v2.json`。

完全程序化的静态实验室橘子树示例，只使用当前 WRS2 / NumPy。
默认 `lab_tree_v1.json` 已根据给定视频的 12 个时刻做相对轮廓、遮挡和可见果实拟合。
**米制尺度仍是估计：80 mm 果径只是工作假设，没有可靠实测尺度或相机标定。**
原始 placeholder 保留在 `configs/generic_citrus_v1.json`。观察记录和拟合局限见 [REFERENCE_FIT.md](REFERENCE_FIT.md)。
没有下载树模型，没有新增第三方运行依赖，没有修改 WRS core，也没有创建物理环境、关节或 spring-damper。

## 运行

在 WRS2 repo 根目录执行（已安装项目依赖的 Python 3.12+）：

```powershell
.\.venv\Scripts\python.exe -m examples.agriculture.citrus_tree.demo_static_tree
.\.venv\Scripts\python.exe -m examples.agriculture.citrus_tree.demo_static_tree --show-collision
.\.venv\Scripts\python.exe -m examples.agriculture.citrus_tree.demo_static_tree --view front --watch
```

也支持直接运行 `examples/agriculture/citrus_tree/demo_static_tree.py`。
demo 创建 WRS World、地面、树和观察树冠的相机，通过现有 WRS 浏览器 viewer 显示。
`SHOW_COLLISION = False` 是脚本内开关；命令行 `--show-collision` / `--no-show-collision` 可以覆盖。
浏览器会保留查看姿态。默认 viewer 端口为 8000，`--port 8765` 可使用独立端口。

reference preset 提供 `--view front`、`front-left`、`front-right`。
使用同一窗口尺寸对照视频，建议截图窗口为 720 × 1280。它们是便于重复观察的固定相机，
不是从视频标定出的 camera extrinsics。`--no-ui` 可隐藏控制面板用于截图。

`--mode full|branches|leaves|fruits|collision` 选择树的显示层；也可用 viewer 内的下拉框切换。
`collision` 仅画枝条 capsule / 果实 sphere，`--show-collision` 则叠加在当前视觉上。
切换模式只改变 scene 成员，不修改 TreeSpec、果实目标位置或原对象的 visuals。

```powershell
# 固定 seed，导出展开后的纯数据；5 秒后关闭 World
.\.venv\Scripts\python.exe -m examples.agriculture.citrus_tree.demo_static_tree --seed 20260916 --export-spec tree_spec.json --duration 5

# 有界 smoke run 不自动打开浏览器，仍真实运行 World / 发布场景
$env:WRS_VIEWER_NO_BROWSER = '1'
.\.venv\Scripts\python.exe -m examples.agriculture.citrus_tree.demo_static_tree --duration 5
```

如需恢复自动打开浏览器，可移除该环境变量。
`--config path/to/config.json` 可加载另一套生成参数；默认路径不依赖当前工作目录。

## 文件与数据流

| 文件 | 职责 |
| --- | --- |
| `spec.py` | BranchSegment / LeafPlacement / FruitPlacement / TreeSpec / LeafShape；验证、bounds、height、summary、JSON |
| `geometry.py` | NumPy blade mesh 与按 shade 合并；不导入 WRS |
| `generator.py` | 手工宏观 segment graph + 固定 seed 的 twig / leaf 生成 |
| `builder.py` | WRS 原生视觉与简化碰撞对象、整树刚性放置、场景加入/移除 |
| `demo_static_tree.py` | 地面、相机、统计、collision 开关、JSON 导出 |
| `configs/lab_tree_v1.json` | 所有宏观尺寸、微观分布、形状、颜色、碰撞阈值、相机参数 |
| `configs/generic_citrus_v1.json` | 保留原始通用 placeholder 示例，不使用 reference-specific profiles |
| `REFERENCE_FIT.md` | 多帧观察、尺度假设、拟合范围和补测优先级 |
| `test_static_tree.py` | 几何、拓扑、确定性、序列化、WRS 构建及发布契约测试 |
| `__init__.py` | 仅导出纯数据类，导入它不会加载 WRS |
| `.gitignore` | 局部允许追踪本 README 和 config JSON（repo 默认忽略这两类文件） |

```text
config JSON / 手工 macro skeleton
    -> generate_tree(config, seed=...)
    -> TreeSpec（tree-local rest geometry + parent topology + attachments）
    -> build_tree(spec, config, pos=..., rotmat=...)
    -> StaticCitrusTree
         branch_objects: 每 segment 一个 SceneObject，多个短 cylinder visual + 可选 capsule
         leaf_objects:   3 个绿色 shade mesh batch，无碰撞
         fruit_objects:  每 orange 一个 SceneObject，icosphere + 绿色 stem visual + sphere collision
    -> tree.add_to_scene(world.scene)
```

reference preset 默认 seed `20260916`：274 段枝条（82 个宏观/手工 segment + 192 个 twig）、
17 段有碰撞、1,528 片叶、3 个可见橙子。叶片数用于拟合遮挡密度，不是从视频逐片数出的值。
树共 280 个 SceneObject，叶片只占 3 个；地面另算。
每叶 40 个三角面（包括背面），共 61,120 个叶片三角面，全部树视觉 91,336 个三角面。
通用 preset 仍为原来的 91 段 / 300 片叶 / 6 个果实。
实际 bbox 和高度由 demo 打印，不硬编码在 builder 中。

## 坐标、拓扑与 V2 接口

所有长度用 **metre**。树根 `(0,0,0)`，tree-local `+Z` 向上。
配置中带 `_deg` 的角度用度；`LeafShape.fold` 用弧度，curl/droop 是叶长比例。
所有 TreeSpec 几何保持 tree-local，不存 SceneObject、关节或 physics handle。

- `BranchSegment.parent_id` 指父 segment，`attachment_t` 是父枝中心线上的 `[0,1]` 参数；
  `start = parent.start * (1-t) + parent.end * t`。延续同一条树干可保持相同 `order`。
- `LeafPlacement.position` 是叶片基部，等于父枝 attachment 点。
  `rotmat` 把 blade-local 映射到 tree-local：`+X` 基部到尖端，`+Y` 叶宽，`+Z` 正面。
- `FruitPlacement.position` 是球心；`stem_direction` 是球心向父枝 attachment 的单位向量。
  `position + stem_direction * (radius + stem_length)` 等于父枝 attachment 点。
  果柄半径也保存在纯数据中；targetable 在 spec 中查询。
- `bounds()` 返回 tree-local `(min_xyz, max_xyz)`；覆盖实际叶 mesh、果实、果柄和保守枝条 capsule。
  capsule 端帽使树根 bbox 最低点略低于 0；`height()` 是该 AABB 的 Z 跨度。
- `canopy_bounds()` 返回实际叶片顶点的 AABB，不包含主干和果实；summary 同时报告它、
  `root_to_top_m`、整树 bbox 和宽/深/高，避免把树干 capsule 端帽误认为树根高度。
- `scaled(factor)` 返回根点不变的均匀缩放副本：所有位置、枝径、叶尺寸、叶厚、果实及果柄尺寸同比缩放，
  attachment_t、rotmat 和拓扑保持不变。纯数据仍不依赖 WRS。
- `validate()` 检查唯一 ID、单根、缺失父枝、环、附件位置、正半径、旋转矩阵和有限数值等，错误抛 `ValueError`。

这些 rest 数据足以在后续 adapter 中构造父子局部变换、关节连接点以及附着叶片/果实。
V1 叶片按色合批是一次性静态构建；未来枝条变形需要 V2 对这些 placement 重新变换或重新批处理。
本版本没有实现这部分，也没有弹簧参数或 dynamics step。

## 程序接口

```python
from examples.agriculture.citrus_tree.generator import generate_tree, load_config
from examples.agriculture.citrus_tree.builder import build_tree
from examples.agriculture.citrus_tree.spec import TreeSpec

config = load_config()
spec = generate_tree(config, seed=20260916)
spec.to_json("tree_spec.json")
restored = TreeSpec.load("tree_spec.json")
tree = build_tree(restored, config, pos=(0.5, 0, 0))

target = tree.fruit_objects[0]             # 与 spec.fruits 顺序一致
target_position_world = target.pos        # 世界坐标，WRS 原生 SceneObject.pos
same_target = tree.fruit_by_id["orange_000"]
branch = tree.branch_by_id[spec.fruits[0].parent_branch]
targetable = tree.spec.fruits[0].targetable
print(tree.summary())
```

`tree.set_pos_rotmat(pos, rotmat)` 从 TreeSpec 的 rest pose 重新放置整树，包含果实；
会覆盖单独移动过的果实位置。单独取下/移动果实时直接操作对应 SceneObject。
`tree.show_collision(True)` 使用 WRS 的 `toggle_render_collision`；
`tree.add_to_scene(scene)` / `remove_from_scene(scene)` 只管理场景成员关系。
WRS 对象是平坦世界坐标集合，以上操作不创建场景父子挂载。

**两种 JSON 不同**：config 是生成 recipe，使用 `load_config()`；展开的 TreeSpec 使用
`TreeSpec.load()` 或 `TreeSpec.from_json(text)`。后者不需要再次随机生成。
使用自定义视觉配置重建时，应同时传入相应 config；纯 TreeSpec 不保存 renderer 设置。

## 调整 reference 和 foliage

推荐反复调参流程：

1. 保持 seed 不变，启动 `--view front --watch`。先调 `macro_branches` 的总体轮廓和偏心，
   再从 `branches` 模式检查主要硬障碍。
2. 用 `growth_profiles` / `branch_profiles` 调整上、中、下、外围、内部的叶量与方向。
   保存 JSON 后自动验证、构建并替换场景；不完整或无效的编辑保留上一版。
3. 用 `foliage_exclusions` 调整右侧小开口，再检查 `full` 模式中的果实遮挡。
   `fruits` 模式便于核对位置，`leaves` 模式便于核对密度。
4. 重启到 `--view front-left` 和 `--view front-right`，检查厚度与后层分布；这两项当前置信度低。
5. 得到实测尺度后修改顶层 `scale_multiplier`，或使用 `--scale 1.1` 临时覆盖。
   名义平均果径假设为 0.080 m；若测到某个具体果实，应使用 `实测直径 / (2 * 对应配置 radius)`。
6. 用 `--report fit_report.json --export-spec fitted_spec.json` 导出统计和纯几何，记录使用的 config / seed。

`--watch` 保持用户当前相机，便于看同一视角下的改动；WRS 当前仅在初次发布时发送 camera pose。
**改变固定相机配置、切换 `--view`，或要让相机随新尺度同比缩放时，请重启 demo。**
热更新时 CLI 的 `--seed` / `--scale` 覆盖仍然有效，修改同名 config 值不会越过 CLI 覆盖。

```powershell
# 原通用示例
.\.venv\Scripts\python.exe -m examples.agriculture.citrus_tree.demo_static_tree --config examples/agriculture/citrus_tree/configs/generic_citrus_v1.json

# 实验树调参和报告
.\.venv\Scripts\python.exe -m examples.agriculture.citrus_tree.demo_static_tree --view front --watch --report fit_report.json
.\.venv\Scripts\python.exe -m examples.agriculture.citrus_tree.demo_static_tree --view front-right --mode collision
```

直接修改 config 的 `macro_branches`，或调用
`generate_tree(config, skeleton=list_of_BranchSegment)` 提供手工拟合的骨架。
换骨架后要同步更新明确引用 ID 的 `fruit_placements`，以及非 null 的 twig `parent_ids`。
使用 reference 扩展时，也要同步更新 `branch_profiles` 和 `leaf_placements` 中的枝条引用。
`parent_ids: null` 按 `min_parent_order` 选宏观末端枝；显式 ID 列表允许指定其他宿主。
可设 `twig_generation.per_parent=0`，给已经包含细枝的手工骨架直接生成叶片。

可选扩展均为通用配置机制：

- `growth_profiles.<name>.twig/leaf` 覆盖基础 twig/leaf 参数；`branch_profiles` 给宏观枝条指定 profile，
  后代继承最近的祖先设置。profile 名称和分区不在 generator 中写死。
- `foliage_exclusions` 是名义 tree-local ellipsoid 列表 `{center, radii}`。
  如果程序叶片的顶点或三角面中心落入其中，就跳过这片叶。这是视觉开口启发式，
  **不是完整三角形相交算法、机械臂扫掠体检查或自由通道证明**；不删除枝条碰撞。
- `leaf_placements` 接受少量显式 LeafPlacement，例如遮挡果实的标志性叶片；这些点仍必须附着在父枝上。
  手工叶片不受程序叶片的 gap 过滤影响，便于稳定保留观察到的遮挡。
- reference 果实使用手工 `fruit_support_*` 挂点，修改随机 seed 或叶密度不会改变果实位置。
- `scale_multiplier` 最后统一缩放 TreeSpec。recipe 本身与 `reference` 中的注释仍按名义米制坐标存储。

twig 的 attachment、长度、锥角、azimuth、向外/向上偏置均在配置中。
只有满足 order、半径、距树轴阈值的**末端细枝**生成叶片，叶基沿中心线分布；
不会在 bbox 内撒点。叶片沿枝改变 azimuth，按 pitch 朝外/朝上，roll 轻微扰动，
尺寸默认 ±17%。所有随机数来自局部 NumPy Generator，相同 config / seed 结果一致。
reference preset 改为 ±18%，各 profile 可以覆盖长度、宽度、pitch、roll 与数量。

叶 mesh 有尖椭圆轮廓、中脉折面、纵向 curl 和 droop，7 个纵向站点。
reference 叶长主要为 0.075–0.100 m、叶宽 0.024–0.030 m（均为估计），较原 placeholder 更狭长；
下层使用负 pitch 形成下垂叶簇。
背面使用反向绕序三角形，与正面有 80 μm 间隔，避免 WRS 合并重合顶点后法线抵消。
纯 TreeSpec 只要求叶厚为正；WRS builder 要求缩放后的叶厚至少 10 μm，
不足时明确报错。这个渲染限制不影响纯数据缩放，也不限制没有叶片的树。
所有叶片先按自身变换放入 tree-local，再按 `color_class` 合并。
这层极薄表面用于视觉，不是封闭体积。

粗枝碰撞是按端点和最大半径直接构造的 capsule，与 cylinder taper 视觉分离。
`collidable=False` 明确关闭某段；生成器还会用 `branch_radius_threshold` 关闭细枝碰撞。
builder 尊重展开后 spec 中的 collidable，不会重新从视觉 mesh 拟合碰撞。
果实只使用 sphere collision，绿色果柄仅为视觉。V1 要求 `leaves_collision=False`，
配置为 True 会明确报错，避免误以为已建立叶片碰撞。

## 验证

```powershell
# 完整测试，不启动浏览器或 World
.\.venv\Scripts\python.exe -m unittest discover -s examples/agriculture/citrus_tree -t . -p "test_*.py" -v

# 仅纯几何 / TreeSpec 测试，只需要 NumPy
.\.venv\Scripts\python.exe -m unittest examples.agriculture.citrus_tree.test_spec -v
```

测试包括相同 seed 的完整 JSON 一致性和 round-trip、数据层不导入 WRS、手工骨架不被修改、
无效拓扑/附件拒绝、双面三角面绕序与 WRS 顶点合并后的法线、
capsule 包含枝条视觉、bounds 包含几何、整树变换后的果实球心/果柄连接点，以及 collision 发布开关。
截图检查使用现有 WRS WebGPU viewer；浏览器自动化仅用于开发验证，不是此示例的运行依赖。
新增测试覆盖 profile 继承、视觉 gap、固定果实挂点、均匀尺度和相机同比缩放、5 种 debug mode，
以及无效 config 编辑保留上一版场景。共 19 项测试，其中 6 项为独立的纯数据/几何测试，
并覆盖 zero-length / negative-radius / missing-parent / cycle、空叶片/果实、缩放、
禁止 WRS 导入时执行完整数据操作、不同进程/hash seed 的一致性，以及不改变全局 RNG 状态。
测试使用 stdlib unittest，没有新增测试依赖或 viewer UI 自动化测试。

## 文件组织

```text
citrus_tree/
  spec.py                  # 纯 rest geometry / topology / JSON
  geometry.py              # NumPy 叶片 mesh 和按色合批
  generator.py             # 配置 + 显式 RNG -> TreeSpec，复用 WRS math
  builder.py               # TreeSpec -> 原生 WRS visuals / primitive collisions
  demo_static_tree.py      # World 入口、固定相机和配置热更新
  test_spec.py             # 纯数据 / 几何测试
  test_static_tree.py      # generator / builder 集成测试，无 UI 测试
  configs/
    generic_citrus_v1.json
    lab_tree_v1.json
  README.md
  REFERENCE_FIT.md
  __init__.py
  .gitignore
```
