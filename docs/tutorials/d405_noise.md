# VirtualD405：raw-like 空洞、错配与时序噪声

这是可配置的解析/随机传感器模型，不是 D4 固件复刻，也不是工厂 D405 标定参数。
它分别模拟“有没有深度”和“有效值是什么”，输出保留空洞及离群点，供真实相机和
仿真相机共用后续预处理。没有加入深度平滑、补洞、rolling shutter、主动投影器或神经网络。

## 使用与兼容

```python
from wrs.sensor import VirtualD405, D405NoiseConfig

camera = VirtualD405(noise_config='agriculture_foliage', seed=42)
frame = camera.capture(scene)
depth_m, depth_raw = frame.depth_m, frame.depth_raw
points, colors = frame.get_point_cloud(world_frame=True)

# Per-stream calibration takes precedence over preset output parameters.
camera = VirtualD405(camera_model=calibration, baseline_m=measured_baseline,
                     depth_scale=reported_depth_scale,
                     noise_config=D405NoiseConfig.preset('default'))

# At episode boundaries, restore the original seed and clear temporal history.
camera.reset_noise()
saved = camera.noise_model.get_state()
frame = camera.capture(scene)
camera.noise_model.set_state(saved)  # Same scene/pose gives exactly the same next frame.
```

现有 `VirtualD405()` / `noise=StereoDepthNoise(...)` 保持旧版 GPU 路径、默认 0.5 m
截止距离和运行时 `.noise` 调参行为。使用 `noise_config` 才启用新模型，避免改变已有训练、
RGB 照明示例与测试。新旧噪声不能叠加，混用会报错。

新配置默认推荐最远距离 0.5 m、硬截止 2 m；超过推荐距离逐渐退化，而非到 0.5 m 立即归零。
这不意味着真实 D405 在 2 m 有可靠性能。`min_depth`、`max_depth`、`depth_scale` 构造参数
覆盖配置中的对应值，`fx`/基线来自当前相机标定。`mode='ideal'` 保持无随机扰动的理想输出；
`noise_config='ideal'` 则保留轻微视差噪声，这两个名称的区别是有意的。

`process_depth(depth_gt, rgb=None)` 仍返回 `(depth_m, depth_raw)`；新模型下每次调用推进一个
时序样本，不能把它与同一相机的 `capture()` 当作无状态参考混用。
`clone()` 深拷贝噪声状态和 RNG；`close()` 释放渲染资源但不重置时序状态。
`enabled=False` 且未请求诊断/附加线索的 `capture()` 直接复用原 GPU 的理想 Z16/XYZ，
跳过 CPU 噪声和第二次反投影，也不推进噪声状态。运行时切换配置应调用 `reset_noise()`；
机器人示例的开关、sigma 调整和场景复位已经处理这一点。

## 管线与模块

```text
clean metric Z / RGB
  -> 硬量程有效性（保留原始遮挡几何）
  -> 小幅视差误差：fx * baseline / Z，偏置 + 空间相关 Gaussian + AR(1)
  -> 场景条件化的相关空洞
  -> 显式深度边缘、细结构与虚拟右视图遮挡
  -> 同一图像行的 background takeover / foreground bleed / wrong disparity / reject
  -> 空洞和离群点的 Markov 保留与刷新
  -> 最终量程检查、depth_unit_m 量化、Z16 编码（无效严格为 0）
  -> 由测量深度重新反投影相机/世界 XYZ
```

GPU 继续提供几何 Z 和 RGB；CPU 模型读取未量化的 `depth_gt`，跳过旧 GPU 误差。
最终点云由新的测量值计算，不会把干净 XYZ 错配给噪声深度。旧 GPU 回读中的理想 Z16/XYZ
仍会计算，但不进入新噪声模型。内部度量深度只在末端进行 Z16 量化。

- `d405_noise_config.py`：配置、四个 preset、JSON profile。
- `d405_noise.py`：`D405HoleNoiseModel`、`D405ValueNoiseModel`、`D405NoiseState` 与组合模型。
- `d405_noise_analysis.py`：真实/仿真序列统计、初始拟合、PLY 导出。
- `VirtualD405`：可选组合模型、既有帧/姿态/挂载接口适配。

`D405HoleNoiseModel.probability(cues, config)` 返回无效概率图；
`D405ValueNoiseModel.precision(...)` 与 `mismatch_probability(...) / mismatch(...)`
分别负责小误差和大误匹配。可向 `D405NoiseModel(hole_model=..., value_model=...)`
注入同接口实现，再传给 `VirtualD405(noise_model=...)`。未来替换学习模型不需要改观测接口。

## 空间与时序行为

深度跳变和有效/无效边界生成 `depth_edge_mask`。采用 Chebyshev 像素距离，边缘概率在
配置的带宽内衰减；水平跳变权重更大。一个前景像素在水平或垂直两侧的小邻域都能看到更远
深度/空白时，被判作细结构。这里只使用屏幕几何，未编码叶片物理厚度。

背景接管取同行窗口的较远有效深度，前景扩张取较近深度，错误视差从同行随机偏移取样；
越界不循环到另一侧，缺少有效不同表面时不凭空生成大 Z。遮挡来自左图向虚拟右图的 z-buffer
重投影近似，它不能发现仅右相机可见的表面。

失效随机场经 Gaussian 空间相关与 Gaussian CDF 后比较概率阈值，默认在 2×2 网格计算，
随机偏移网格相位。`failure_field_stride=1` 可恢复逐像素随机场。只对随机场使用网格加速，
不会缩放/插值深度、边缘或点云。少量单像素缺测仍可能出现在细轮廓与概率边界处；整体不是
独立 salt-and-pepper dropout。水平相关长度是垂直长度的 `epipolar_correlation_ratio` 倍。

连续视差误差使用 AR(1)。空洞和离群点各自按 persistence 概率保留上一状态，否则重采样；
它们共享相关的保留随机场，使用独立的候选误差场。persistence 是“保留旧状态”的混合系数，
不是直接的 `P(invalid_t | invalid_t-1)`。对单一稳定空洞概率 p，后者约为 `rho + (1-rho)*p`。

时序状态位于传感器像素网格，不声称做了精确 optical-flow 跟踪。深度明显改变的像素刷新
历史；大幅相机运动刷新整幅历史，避免新出现的表面继承陈旧噪声。可提供 `motion_px`，否则
根据相邻 optical pose 的平移/典型距离及旋转估计帧间像素位移。农业/harsh preset 让运动
小幅增加边缘、细结构的失效率。它与帧间位移有关，不是曝光模型，也没有 rolling shutter。

RGB 使用局部亮度标准差作为 texture confidence，并可对很暗/过曝区域增大失效概率。
无 RGB 时这些项关闭。此项不会从当前无纹理/简单漫反射 RGB 凭空恢复真实果皮/叶脉。
`normals`（optical H×W×3）、`material_difficulty`（H×W [0,1]）、`semantic_ids`（H×W）
是可选输入；当前渲染器没有回读这些通道时不伪造它们。语义 ID 转成字符串匹配配置，
例如 `{'1': 1.15, '2': 1.3}`。农业 preset 提供 `FOLIAGE` 和 `TWIG` 名称映射，
只有调用者实际提供对应语义图时生效。柑橘演示使用几何与 RGB，没有借颜色猜测语义。

当前新模型在相机输出图像网格上运行。Brown 畸变后的水平 donor 是近似，不等同于在
rectified 双目图像里匹配；深度/XYZ 的标定和畸变反投影仍正确。精确双目失效需要将模型
移动到 rectified 源域，或以后接入双目 matcher；本版没有声称实现这两项。

## 诊断数据与策略隔离

```python
frame = camera.capture(scene, debug=True)
diagnostics = frame.noise_debug
# process_depth(..., debug=True) returns D405NoiseResult with .debug.
```

默认 `noise_debug=None`，不自动把诊断输入策略。可选字典包含：
`clean_depth`、`noisy_depth`、`valid_mask`、`invalid_mask`、`normal_noise_map`（视差扰动，px）、
`outlier_mask`、`edge_mask`、`distance_to_depth_edge`、`thin_structure_mask`、
`texture_confidence`、`corruption_reason` 和标量 `motion_px`。
旧 API 中的 `depth_gt` 原本就存在，同样只用于诊断/仿真评估。

原因编码为 0 VALID、1 OUT_OF_RANGE（含没有几何）、2 HOLE、3 BACKGROUND_BLEED、
4 FOREGROUND_BLEED、5 WRONG_DISPARITY。相机外参快照和颜色配准行为保持原约定。

## 预设和参数性质

| 预设 | 相对 default 的改动 |
| --- | --- |
| `ideal` | sigma=0.025 px；所有 `*_prob` 和 `*_gain` 为 0，只保留精度噪声/编码/量程 |
| `default` | 下表默认值 |
| `agriculture_foliage` | edge invalid=.23、thin invalid=.24、edge outlier=.14、thin outlier=.13；motion invalid=.035/outlier=.02；语义 FOLIAGE=1.15、TWIG=1.3 |
| `harsh` | sigma=.12、base invalid=.01、edge invalid=.35、thin invalid=.30、low texture invalid=.08、edge outlier=.22、thin outlier=.18、far invalid=.4、motion invalid=.07/outlier=.04 |

**有物理依据的关系**：双目 `d=fx*b/Z` 及其距离平方误差增长、沿 epipolar 方向的错误表面
匹配、遮挡、低纹理/弱光/掠射角难以匹配、有效量程和 Z16 单位。fx、基线、单位及裁剪/量程
应来自当前 stream 的标定/设备设置。

**启发式数值**：下面所有 sigma、概率、gain、像素带宽、窗口、混合权重、相关长度、
Markov/AR 系数、运动/纹理/法线阈值，以及没有设备 profile 时的量程默认值。
它们不是实机精度承诺；材质难度也只是输入线索，不是镜面反射或照明仿真。

## 真实记录分析与后续拟合

保存未经补洞/平滑的静态 D405 深度序列，例如：

```python
np.savez_compressed('recording.npz', depth_raw=z16_frames,
                    depth_unit_m=reported_depth_scale, rgb=rgb_frames)
```

输入支持 metric `(T,H,W)` NPY、带 `depth_m` 或 `depth_raw` 的 NPZ、按文件名排序的 Z16 PNG
目录；RGB 可放 NPZ 的 `rgb` 键。整数深度必须提供真实单位，不猜测 D400 默认值。

```powershell
python -m tools.analyze_d405_noise recording.npz --fx 600 --fy 600 --cx 319.5 --cy 239.5 --baseline-m 0.018 --known-depth-m 0.30 --output d405_noise_profile.json
```

命令中的数值仅为语法示例，替换为设备当前 stream 的标定。平板已知深度必须是正对相机的
Z；有背景时用 `--mask board_roi.npy` 限定平板。倾斜面/已知完整真值使用 `--reference ref.npy`。
不提供真值时采用时间中位数作为伪参考，要求相机与场景静止；无法识别固定偏差、持续错误
匹配或从未有有效值的区域。报告单列 reference coverage，避免把未知区域当作干净真值。

工具输出 fill/invalid、有效像素 MAE/RMSE、temporal std、边缘失效、离群误差分位数、
8 邻接空洞连通域面积分布和无效状态保持率。它为 base/edge 概率、小视差 sigma/bias、rho
与 invalid persistence 生成**初始估计**；没有数据支持的参数继续保留启发式值并标明估计字段。
无已知参考不拟合绝对 bias，无 fx/基线不拟合视差参数。只有 metric 记录且没报告 Z16 单位时，
输出明确标记单位为仿真 fallback。

```python
camera = VirtualD405(noise_profile='d405_noise_profile.json', seed=42)
```

后续拟合建议采集：不同距离的纹理平板、无纹理平板、不同角度叶片/枝条、前后两层遮挡、
静态序列及不同速度的手腕运动序列。先用可靠内部区域拟合视差 sigma/bias，再用连通域尺寸
拟合 correlation/stride、边缘/细结构分组拟合各 gain、带已知背景的遮挡拟合 outcome 权重，
最后用静态序列/运动序列分别拟合 persistence 和 motion gains。保留验证场景，比较统计与
形态，不能用一张平板图唯一确定全部参数。

## 柑橘树验证与产物

```powershell
python -m examples.agriculture.virtual_d405_noise
python -m tools.validate_d405_noise --width 640 --height 480 --frames 32 --output build/d405_noise_640
python -m tools.validate_d405_noise --width 320 --height 240 --frames 32 --output build/d405_noise_320
python -m unittest discover -s tests -p test_d405_noise.py -v
python -m unittest discover -s tests -p test_virtual_depth_camera.py -v
```

使用既有柑橘树生成器：1,528 片叶、274 段枝、3 个果实，约 91,336 三角形。
相机位于第一个果实前约 30 cm，示例内参随图像尺寸缩放，未冒充实机标定。
浏览器显示 clean/noisy/RGB 与彩色点云，可勾选小幅手腕平移观察时序变化。

离线产物：`comparison.png`、`clean_depth.png`、`noisy_depth.png`、`corruption_reason.png`、
`temporal.gif`、`clean.ply`、`noisy.ply`、`sequence.npz`、`debug.npz`、
`report.json`、`d405_noise_profile.json`。PLY 使用世界坐标 m 和配准 RGB。
原因图：灰=有效、黑=无几何/量程外、白=空洞、红=背景接管、青=前景扩张、紫=错误视差。

报告中的 fill/invalid 以有真值且在量程内的像素为 ROI；另列全画面 fill，防止背景空白
影响噪声评估。离群阈值默认 1 cm；边缘指标分母为真值边缘带内像素，离群率含无效像素在
分母中。MAE/RMSE 只在有测量且有参考的像素计算。temporal std 含有效离群值，另列排除
大误差后的 inlier residual std。运动验证单列，不混入静态时间噪声统计。

有真实录制时可增加：

```powershell
python -m tools.validate_d405_noise --real-recording recording.npz --real-fx 600 --real-fy 600 --real-baseline-m .018 --real-reference reference.npy --real-mask roi.npy
```

用同一真实参考场景生成模拟序列，并列真实/模拟的填充率、时间噪声、边缘和离群分布。
未提供真实数据时只输出仿真结果；不能把合成记录的工具测试当作实机验证。
对比时采用录制文件或 `--real-depth-unit-m` 报告的单位；只有未报告单位的 metric 记录
才使用标注过的仿真 fallback。范围参数仍来自当前 preset，应按真实 stream 设置另行调整。

## 本次实测（2026-09-17）

原有 `test_virtual_depth_camera.py` 的 18 项测试与新增 `test_d405_noise.py` 的 12 项测试
全部通过。新测试覆盖距离平方误差增长、标定/偏置、相关空洞、四类同行错配、细结构、
RGB/法线/语义/运动线索、AR/Markov 状态、种子/快照/克隆、软量程、最终量化和统计拟合。
浏览器演示实际运行，确认三路图像和点云持续更新、运动控件生效，未发现浏览器脚本错误。
真实录制工具的读写/初始拟合/加载/real-vs-sim 分支用合成平板序列完成了端到端检查；
本任务没有真实录制，尚未验证 sim-to-real 一致性。

`agriculture_foliage`、seed=42，静态 32 帧，真值量程内 ROI：

| 指标 | 320×240 | 640×480 |
| --- | ---: | ---: |
| fill / invalid | 76.51% / 23.49% | 83.26% / 16.74% |
| 有效像素 MAE / RMSE（包括离群值） | 7.14 / 32.04 mm | 3.49 / 21.81 mm |
| temporal std（包括离群值） | 15.38 mm | 8.12 mm |
| inlier temporal residual std | 1.84 mm | 1.02 mm |
| gross outlier（误差 > 10 mm） | 4.84% | 2.35% |
| edge invalid / outlier | 28.05% / 5.89% | 24.44% / 3.68% |
| interior invalid / outlier | 3.42% / 0.20% | 4.09% / 0.15% |
| 空洞连通域面积 p50 / p95 | 4 / 56 px | 3 / 44 px |
| 离群误差幅度 p50 / p95 | 54.99 / 299.31 mm | 59.12 / 292.96 mm |
| 实测 invalid persistence | 80.88% | 79.60% |

错配距离大是因为 donor 来自树冠内真实的前后表面；它们不是大幅 Gaussian Z 扰动。
少数细轮廓出现单像素空洞，但邻域/连通域测试确认主要空洞有空间相关性。
同一 preset 的像素窗口没有随分辨率缩放，因此低分辨率下细枝和叶缘更脆弱。
640×480 的 clean/noisy PLY 分别为 183,034 / 153,034 个带 RGB 的世界坐标点。

独立 CPU 噪声基准使用缓存的同一 clean/RGB 输入，5 帧预热、40 帧测量，包含噪声状态
更新和最终量化，不包含渲染/XYZ/UI。环境为 Windows 11、Python 3.12.0、NumPy 2.5.2、
SciPy 1.17.0；完整 CPU 标识和逐帧样本见 `build/d405_noise_runtime.json`。

| 分辨率 | CPU 噪声均值 | 中位数 | p95 |
| --- | ---: | ---: | ---: |
| 320×240 | 24.71 ms | 23.18 ms | 32.09 ms |
| 640×480 | 142.08 ms | 139.13 ms | 162.46 ms |

320×240 的完整 capture 均值为 50.77 ms。640×480 完整验证运行有明显时延波动：末次均值
418.18 ms、此前约 181 ms；所有分阶段均值/p95 保存在对应 `report.json`，不能用独立
噪声基准冒充整条管线帧率。目前 640×480 不满足 30 Hz 或大量并行环境的高吞吐要求。
实现采用已有 NumPy/SciPy，不引入学习框架，但后续高吞吐训练仍需将这些算子移到 GPU
或降低传感器采样频率/分辨率。演示默认 320×240、15 Hz 调度。

产物目录：`build/d405_noise_320/` 和 `build/d405_noise_640/`。另生成了 640×480
`pointcloud_comparison.png` 供侧视比较；`build/d405_noise_320/live_browser.png` 是浏览器检查截图。
图中的果实内部总体完整，叶片轮廓缺测、局部枝条断裂和前后表面错配可见。

## 本任务文件清单

| 文件 | 改动 |
| --- | --- |
| `wrs/sensor/d405_noise_config.py` | 新增配置、预设与 JSON profile |
| `wrs/sensor/d405_noise.py` | 新增 hole/value 模型、时序状态、诊断与组合管线 |
| `wrs/sensor/d405_noise_analysis.py` | 新增录制读取、统计、初始拟合、彩色 PLY |
| `wrs/sensor/virtual_d405.py` | 接入可选模型/profile，兼容旧 API，更新噪声点云 |
| `wrs/sensor/virtual_depth_camera.py` | DepthFrame 追加可选 noise_debug 字段 |
| `wrs/sensor/depth_sensor_model.py` | 新模型启用时跳过旧 GPU 噪声，避免重复处理 |
| `wrs/sensor/__init__.py` | 导出新配置/模型/状态 |
| `tools/analyze_d405_noise.py` | 新增录制分析 CLI |
| `tools/validate_d405_noise.py` | 新增场景验证、统计、产物和真实/仿真对比 CLI |
| `examples/agriculture/virtual_d405_noise.py` | 新增无参数浏览器演示 |
| `tests/test_d405_noise.py` | 新增 12 项测试 |
| `docs/tutorials/d405_noise.md` | 本文：参数、限制、拟合与结果 |
| `docs/tutorials/virtual_depth_camera.md` | 加入新模型教程入口 |
| `mkdocs.yml` | 加入文档导航 |
| `docs/API_INDEX.md`、`docs/api/index.md`、`docs/api/sensor.md` | 重新生成 API 索引 |
| `docs/api/robots.md` | 同次索引生成同步已存在的 FAFU API，无机器人实现改动 |

## 小步加速与机器人可视化

`python -m examples.agriculture.robot_citrus_interaction` 现在默认启用农业噪声。
右侧显示 RGB 与单个深度图，按钮 `Switch Clean / Noise` 在 raw-like/clean 之间切换，
默认 Noise；切换使用同一帧及相同 Jet 色标。可叠加测量点云。sigma 只调节正常精度，关闭噪声
开关才返回理想几何。复位重置 RNG/时序状态；显示点云仍从相机采样中排除。

几何渲染、旧版噪声及理想 Z16/XYZ 已使用现有 WGPU。新模型目前仍是 NumPy/SciPy。
本轮最小优化只针对关闭噪声的路径：复用 GPU 已有结果，避免 CPU 特征和重复点云计算。
启用时保留相同的 hole/value、AR/Markov 算法和随机数序列，未添加另一套 GPU 噪声实现。

后续适合移入 WGPU compute 的运算包括：

- 深度跳变、窄边缘带、局部纹理方差、同行 min/max donor 搜索。
- 相关随机场滤波、视差扰动、概率判断和 AR/Markov 更新。
- 最终量化与噪声深度的 XYZ 生成，复用既有 LUT/输出格式。

像素并行不等于只把几个 NumPy 调用换成 GPU 就能提速。当前 RGB/depth 已回读 CPU，
零散迁移会增加上传、下载和队列同步；尤其是 320×240，小算子的 GPU 启动成本可能抵消收益。
下一轮若需要 GPU，应直接复用现有设备与深度纹理，将中间特征和时序缓冲留在 GPU，
最终统一回读。先做有限边缘带/局部窗口即可，无需 CUDA、PyTorch 或新后端抽象。
这仍需独立验证随机场分布、时序确定性和 CPU/GPU 的量化差异，本轮不宣称已完成迁移。

机器人集成后实测：320×240、静止姿态、各 4 帧预热 + 24 帧测量，无浏览器编码/物理步进：

| 模式 | CPU 噪声 | 噪声 XYZ | 完整 capture |
| --- | ---: | ---: | ---: |
| 关闭噪声，复用 GPU 结果 | 0 ms | 0 ms | 16.82 ms |
| agriculture_foliage | 14.71 ms | 2.39 ms | 34.73 ms |

这与上面的独立柑橘场景不同，不能直接当成启用噪声的前后加速比。
逐项均值/p95 在 `build/robot_d405_noise/timings.json`。实际浏览器运行完成 505 帧，验证
RGB/raw-like/clean 三路图像、点云开关、噪声开关和 Cartesian jog；无浏览器脚本错误。
截图为 `build/robot_d405_noise/browser.png`。相关测试为旧相机 18 项、新噪声 13 项、
法兰相机集成 7 项，共 38 项通过；新增覆盖关闭噪声的 GPU 复用、sigma=0 仍有空洞、
时序复位和调参不重建渲染器。

## 参考

- [RealSense Depth Testing Methodology](https://dev.realsenseai.com/download/15283/)：区分覆盖率、空间误差、时间噪声，使用适合被动双目的纹理目标。
- [RealSense stereo tuning](https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/)：曝光、纹理、匹配阈值与应用条件影响原始深度。
- [ByteCameraDepth 数据卡](https://huggingface.co/datasets/ByteDance-Seed/ByteCameraDepth)：发布的数据包含 D405；请求中对应 GitHub 地址无法读取，实际公开数据位于 Hugging Face。
- [manip-as-in-sim-suite / CDM](https://github.com/ByteDance-Seed/manip-as-in-sim-suite/blob/main/cdm/README.md)：训练流程明确分开 hole/value noise。这里只采用分层思路，不引入其模型、权重或推理栈。

## 全部默认值

概率和 gain 无单位；sigma/bias/window/correlation/edge band 为像素或图中注明的单位。
gain 与对应 [0,1] cue 相乘后相加，再裁剪到概率范围；混合权重在使用前归一化。

```json
{
  "enabled": true,
  "disparity_sigma_px": 0.06,
  "disparity_bias_px": 0.0,
  "spatial_correlation_px": 0.6,
  "temporal_rho": 0.75,
  "base_invalid_prob": 0.002,
  "edge_invalid_gain": 0.16,
  "thin_structure_invalid_gain": 0.16,
  "low_texture_invalid_gain": 0.025,
  "grazing_invalid_gain": 0.06,
  "near_range_invalid_gain": 0.3,
  "far_range_invalid_gain": 0.25,
  "lighting_invalid_gain": 0.025,
  "material_invalid_gain": 0.08,
  "occlusion_invalid_gain": 0.2,
  "base_outlier_prob": 0.0005,
  "edge_outlier_gain": 0.09,
  "thin_structure_outlier_gain": 0.08,
  "low_texture_outlier_gain": 0.008,
  "grazing_outlier_gain": 0.02,
  "lighting_outlier_gain": 0.01,
  "material_outlier_gain": 0.05,
  "far_range_outlier_gain": 0.03,
  "background_takeover_weight": 0.45,
  "foreground_bleed_weight": 0.2,
  "wrong_disparity_weight": 0.2,
  "reject_to_invalid_weight": 0.15,
  "depth_edge_threshold_m": 0.005,
  "edge_band_width_px": 3,
  "thin_structure_width_px": 4,
  "epipolar_search_px": 12,
  "horizontal_edge_weight": 0.75,
  "hole_correlation_px": 1.4,
  "failure_field_stride": 2,
  "epipolar_correlation_ratio": 1.8,
  "texture_window_px": 5,
  "texture_threshold": 0.035,
  "texture_gain": 1.0,
  "dark_threshold": 0.08,
  "saturated_threshold": 0.97,
  "grazing_cos_threshold": 0.35,
  "invalid_persistence": 0.7,
  "outlier_persistence": 0.55,
  "temporal_depth_threshold_m": 0.01,
  "temporal_reset_motion_px": 4.0,
  "motion_invalid_gain": 0.0,
  "motion_outlier_gain": 0.0,
  "motion_reference_px": 2.0,
  "near_degradation_width_m": 0.025,
  "far_degradation_width_m": 0.5,
  "far_disparity_noise_gain": 1.5,
  "occlusion_tolerance_m": 0.002,
  "depth_unit_m": 0.0001,
  "min_depth_m": 0.07,
  "preferred_max_depth_m": 0.5,
  "max_depth_m": 2.0,
  "semantic_multipliers": {}
}
```
