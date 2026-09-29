# Virtual Depth Camera

`wrs.sensor.VirtualDepthCamera` 在 Python 侧独立进行 WGPU 离屏渲染，
`VirtualD405` 继承它并提供 D405 名义参数。运行不需要浏览器、相机硬件、
Open3D、CUDA 工具链或其他仿真器，也不读取 Viewer 的 depth buffer。

需要原始双目深度的成片空洞、错配、细结构退化和跨帧状态时，参见
[D405 raw-like 噪声模型](d405_noise.md)。通过 `noise_config='agriculture_foliage'`
启用；本页的 `StereoDepthNoise` 保留为兼容旧 GPU 路径。

## 快速运行

在仓库根目录、安装了项目依赖的 Python 环境中运行：

```powershell
.venv\Scripts\python.exe -m examples.virtual_depth_camera
.venv\Scripts\python.exe -m examples.virtual_depth_camera --mode d405_fast --distortion --show
.venv\Scripts\python.exe -m examples.virtual_depth_camera --benchmark 100
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

只有 `--show` 会启动 WRS Viewer；默认只采集并打印结果。
示例使用三个球形水果、一根树枝及背景，未将这些几何体的颜色视为真实果皮纹理。

## 简单显示示例

下面三个脚本各演示一件事，没有命令行参数；直接修改脚本里的相机、物体和显示参数即可。
在仓库根目录运行：

```powershell
.venv\Scripts\python.exe -m examples.virtual_d405_rgbd
.venv\Scripts\python.exe -m examples.virtual_d405_point_cloud
.venv\Scripts\python.exe -m examples.virtual_d405_model_overlay
```

| 脚本 | 演示内容 |
| --- | --- |
| `examples/virtual_d405_rgbd.py` | 浏览器 UI 实时显示 RGB、深度图；用按钮或键盘移动虚拟相机，同时观察 XYZ 坐标、相机模型和坐标轴 |
| `examples/virtual_d405_point_cloud.py` | 在 WRS Viewer 中查看带 RGB 颜色的点云；`stride=3` 控制像素采样间隔 |
| `examples/virtual_d405_model_overlay.py` | 青色测量点与原始模型的半透明副本叠加；`ghost.alpha` 控制模型透明度，`stride=6` 方便观察点与表面的位置关系 |

三个示例都使用项目已有依赖。逐个运行，在终端按 Ctrl+C 结束当前脚本。
鼠标拖动可从不同方向观察三维场景。

RGBD 示例通过 `base.ui.add_image()` 创建图像控件，以 20 Hz 目标频率采集并更新；
RGB 和深度来自同一次 `capture()`。深度使用 **Jet（彩虹色图）**，固定映射 0.07–0.50 m，
由近到远为深蓝、蓝、青、绿、黄、红、深红。无效深度为黑色；显示用的伪彩色不改变
`frame.depth_m` 和 `frame.depth_raw`。RGB 图中没有物体覆盖的区域也默认是黑色，
这是离屏渲染的清屏颜色，与浏览器三维 Viewer 的背景色独立。
可用 `rgb_background` 指定背景色；若希望采集实际背景模型（如墙面），使其位于
`near..far` 渲染范围内即可。RGB 可见范围与 `min_depth..max_depth` 有效测量范围独立。

先点击三维画面，再按住以下按键移动传感器；也可以点击左侧对应按钮，每次移动 1 cm：

| 按键 | 传感器移动方向 |
| --- | --- |
| A / D | 左 / 右，即 X− / X+ |
| Q / E | 上 / 下，即 Y− / Y+ |
| W / S | 前 / 后，即 Z+ / Z− |
| R | 恢复初始位置 |

左侧分别显示相机光学原点的世界坐标 X、Y、Z，单位 m。示例保持相机旋转为单位阵，
因此 optical 轴与 world 轴一致。机身方盒位于光学原点后方，圆柱表示镜头；
红、绿、蓝三根 7 cm 箭头分别表示相机 X、Y、Z 正方向。该外形仅为示意模型。
相机模型与坐标轴只用于显示，不参与传感器采集。鼠标拖动改变观察场景的 Viewer 视角，
按钮和上述按键改变实际采集图像的 VirtualD405 位置。

三个示例都使用 `mode='ideal'`，便于先理解图像、点云和模型的关系；改成 `'d405_fast'`
可以观察近似双目遮挡造成的缺测。示例内参仅用于演示。RGB 是当前模型的无光照颜色。
相机只采集可见表面，因此点云不会覆盖物体背面。叠加示例使用采集时的世界坐标点云，
即使相机不在世界原点，点也能与模型对齐。半透明副本仅添加到显示场景，不参与采集。

## 接口与坐标

### 可选 RGB 光照与实时噪声

```python
from wrs.sensor import VirtualD405, RGBLighting, StereoDepthNoise

camera = VirtualD405(
    far=5.0,                          # RGB / 几何可见至 5 m
    min_depth=.07, max_depth=.50,      # 只有 7–50 cm 输出有效测量点
    rgb_background=(242/255, 242/255, 240/255),
    rgb_lighting=RGBLighting(),       # 两盏随光学相机移动的方向光 + 环境光
    seed=42,
)
camera.noise = StereoDepthNoise(disparity_std_px=.05, disparity_step_px=1/32)
frame = camera.capture(scene)
camera.noise = None                   # 关闭随机噪声/视差量化，仍保留 fast 几何遮挡
camera.rgb_lighting = None            # 恢复基础色；无需重建 GPU buffers
camera.close()
```

`RGBLighting` 使用面法线、half-Lambert 主光、Lambert 补光和线性空间颜色计算，
RGB 输出再编码为 sRGB。双面叶片按可见面着色；它不会改变几何 Z、碰撞或法兰位姿。
光照、背景色独立于深度有效性；超量程表面可以出现在 RGB 中，但不会产生有效点云。
公共相机默认仍为无光照、黑背景，以兼容已有脚本；Citrus 示例显式开启光照和浅色背景。
没有投射阴影、viewer 的装饰性黑描边、曝光或完整 ISP。

`noise` 可在 `d405_fast` 模式下实时替换，不重建相机、不重置 RNG。
同一 seed、同一调用序列可复现；`clone()` 复制 RNG 状态和光照设置。
噪声先作用于视差，再得到深度和点云，RGB 与 `depth_gt` 保持不变。
在固定视差标准差下，`sigma_Z ≈ Z² * sigma_disparity / (fx * baseline)`；
这是双目深度的不确定性，与镜头的 Brown–Conrady 畸变是不同设置。
该距离关系也见 [RealSense 深度后处理说明](https://dev.realsenseai.com/docs/depth-post-processing-for-intel-realsense-depth-camera-d400-series/)。

```python
import numpy as np
from wrs import wss, wssop
from wrs.sensor import VirtualDepthCamera, VirtualD405, CameraModel, StereoDepthNoise

scene = wss.Scene()
scene.add(wssop.box(pos=(0, 0, .32), xyz_lengths=(.1, .1, .04)))

# Example values only: replace with intrinsics from the active real stream.
with VirtualDepthCamera(
    width=640, height=480, fps=30,
    fx=430, fy=435, cx=317.2, cy=241.1,
    near=.001, far=1, min_depth=.07, max_depth=.5,
    depth_scale=1e-4,
    distortion_model="none", distortion_coeffs=(0, 0, 0, 0, 0),
    T_mount_camera=np.eye(4), mode="ideal",
) as camera:
    frame = camera.capture(scene, T_world_mount=np.eye(4))
    depth_m = frame.depth_m
    depth_raw = frame.depth_raw
    valid_mask = frame.valid_mask
    points_cam = frame.points_cam
    points_world = frame.points_world
    K = frame.intrinsics
    T_world_camera = frame.T_world_camera
```

采用 optical frame：X 向图像右侧，Y 向下，Z 向前；长度统一为米。
整数 `(u, v)` 表示像素中心，左上像素为 `(0, 0)`。
`depth_m` 是相机坐标 Z，不是点到相机的欧氏距离。
WGPU 投影显式使用四个内参，并补偿光栅化的半像素中心约定。

| 输出 | 类型与含义 |
| --- | --- |
| `depth_gt` | H×W float32，裁剪范围内的可见几何 Z，经过输出畸变查表，尚未加入传感器误差 |
| `depth_m`（也可用 `depth`） | H×W float32，测量 Z，单位 m，无效为 0 |
| `depth_raw` | H×W uint16；`round(depth_m / depth_scale)`，无效为 0 |
| `valid_mask` | H×W bool |
| `points_cam` / `points_world` | N×3 float32，只包含有效像素，按图像行顺序排列 |
| `points_cam_image` / `points_world_image` | H×W×3，GPU 已计算的稠密 XYZ，无效为零 |
| `rgb` | H×W×3 uint8，当前 visual model 的颜色，可选 matte 光照，与深度同一视角 |
| `confidence` | H×W float32；启用纹理项时是 Sobel 亮度梯度启发式分数，不能当作标定概率 |
| `camera_model` / `intrinsics` | 完整内参及畸变模型 / 3×3 K |
| `T_world_camera` | 采集瞬间的 world-from-optical 变换快照 |
| `timings_ms` | CPU 编码、提交及同步回读等分阶段墙钟时间 |

深度单位量化与视差量化独立。超出有效范围或 uint16 表示范围的像素置零，
不会溢出回绕；构造时拒绝无法覆盖 `max_depth` 的 `depth_scale`。
帧拥有自己的回读内存，后续采集或相机移动不会改变已有帧。
稠密 XYZ 在 GPU 上生成，N×3 点云在首次访问时在 CPU 上筛除无效点并缓存。
`frame.get_point_cloud(world_frame=True, stride=2)` 返回采样后的点与 `[0, 1]` RGB，
可以直接交给 `wssop.point_cloud`。

`fps` 是供调用端调度的目标频率，`capture()` 本身不等待下一帧，也不保证实时帧率。
可由现有 World 调度器按 `1 / camera.fps` 调用。参数在构造时固定；修改标定需新建相机。
`close()` 释放相机自己的 GPU 资源，不销毁 WRS 共享 device；后续 capture 可重新初始化。

## 安装到机械臂末端

两种方式选一种：

1. 在每次 capture 传入 `T_world_mount`，内部计算
   `T_world_camera = T_world_mount @ T_mount_camera`。
2. 相机继承 `SceneObject`，可以使用现有
   `robot.mount(camera, link, loc_tf=T_mount_camera, update=True)`。
   此后调用 `camera.capture(scene)`，由 WRS 的挂载机制更新光学坐标系。

第二种方式应把安装变换传给 `robot.mount`，不再向 capture 传外参，避免重复变换。
`T_mount_camera` 必须包含机械臂安装轴到 optical frame 的旋转；它不是相机外壳的几何中心。
相机默认没有可见外壳。额外坐标轴、调试 mesh 等可用 `exclude=[object, ...]` 排除。

## WRS 中的文件与数据流

| 新文件 | 职责 |
| --- | --- |
| `wrs/sensor/camera_model.py` | 标定、投影/反投影、一次性求解畸变逆映射 LUT |
| `wrs/sensor/offscreen_renderer.py` | 共享 GPU device、缓存 visual mesh buffer、深度测试、metric Z/RGB 输出 |
| `wrs/sensor/depth_sensor_model.py` | GPU 单视图遮挡、传感器处理、LUT、编码、XYZ、一次回读 |
| `wrs/sensor/depth_noise.py` | 误差参数与可独立运行的 NumPy 参考处理 |
| `wrs/sensor/virtual_depth_camera.py` | 相机姿态、采集 API、生命周期、帧输出 |
| `wrs/sensor/virtual_d405.py` | D405 子类及默认参数 |

Scene 的迭代器已包含普通物体和机器人的 runtime links。逐个读取 `visuals`，使用
`T_camera_world @ object.tf @ model.loc_tf`，传入原有 `geom.vs / geom.fs`。
不使用碰撞简化网格，不采集点云图元；共享几何只上传一次，物体姿态和颜色每帧更新。
网格被替换、物体增删都会被识别；若原地修改顶点数组，先 `camera.close()` 使缓存失效。
没有修改原有 Scene、RenderModel、Viewer、机器人或 GPU device 实现。

```text
WRS visual triangles
  -> independent WGPU rasterization
     depth32float: Z-test only
     r32float: camera-space metric Z
     rgba8unorm: sRGB colour (unlit or matte-lit)
  -> optional synthetic-right reprojection / sensor model
  -> inverse-distortion LUT, nearest source sample
  -> range + uint16 encoding + camera/world XYZ (GPU)
  -> one packed buffer readback
  -> optional CPU compaction of valid points
```

GPU 输出使用显式布局的 storage buffer，不逐行复制纹理，因此没有 256 字节纹理行距假设；
31、33、41 等非对齐宽度已有测试。WGPU clip Z 为 `[0, 1]`；near/far 与传感器
min/max 独立，默认 near 比 min 小，避免过近的物体被直接裁掉后错误露出背景。
几何裁剪面以外的表面不可见，这是标准光栅裁剪的限制。

## 畸变

支持 `none` 和标准 `brown_conrady`，系数顺序为 `(k1, k2, p1, p2, k3)`。
首次创建 GPU 资源时缓存每个输出像素的逆畸变射线及源像素索引；不每帧迭代。
必要时扩展无畸变渲染目标，覆盖所有输出射线，避免仅因源视野不足产生黑边。
运行时在 GPU 上 nearest 查表，不对前景、背景深度做 bilinear 混合。
点云用输出像素对应的逆畸变射线计算。

这是离散重采样，倾斜面或边界会有最多约半个源像素的采样误差，不等同于逐条畸变射线求交。
不可逆或迭代不收敛的系数会被拒绝。不能把 RealSense 的 inverse/modified Brown 或
fisheye 参数不加转换地当作标准 Brown 输入。

## D405 fast 模型

```python
calibration = CameraModel(640, 480, 430, 435, 317.2, 241.1)
camera = VirtualD405(
    calibration, seed=42,
    noise=StereoDepthNoise(
        disparity_std_px=.08, disparity_step_px=1/32,
        stereo_occlusion=True,
        confidence_threshold=0, texture_dropout_strength=0,
        edge_dropout_strength=.1, edge_threshold_m=.005,
        dropout_rate=.005, occlusion_tolerance_m=.001,
    ),
)
```

默认 `mode="d405_fast"`，支持 `mode="ideal"`；未知模式直接报错。
默认开启几何遮挡，随机误差和纹理淘汰默认关闭。上面的误差参数仅用于演示，未用实机拟合。
`baseline_m`、`disparity_std_px`、`disparity_step_px`、`dropout_rate`、`seed`
分别对应计划中的 baseline、disparity_noise_std、disparity_quantization、
random_dropout_probability、random_seed；其余强度和置信度参数保留原含义。

`d = fx * baseline / Z`；量化及扰动在视差域进行，再用 `Z = fx * baseline / d` 恢复。
单视图重投影用 atomic 深度比较生成虚拟右视图，拒绝右视野外及被较近样本遮挡的点。
置信度取 rectified RGB 的 Sobel 梯度，边缘丢点取四邻域 Z 跳变。
处理在 rectified 源域定义，随后通过畸变 LUT 采样输出。
相同 seed、配置和采集调用序列在同一后端可复现；连续帧噪声不同。
CPU `process_depth()` 采用 NumPy RNG，和 GPU 的随机序列不逐像素相同。

D405 默认基线取 **18 mm**，深度原点采用左侧光学坐标系；数据手册中的 **9 mm**
是安装中心线到左成像器的偏移，不能当作双目基线。D405 没有内置投影器，并使用
IR-cut 滤光片。[D400 数据手册，表 4-1 / 4-20](https://www.realsenseai.com/download/21345/?tmstv=1770190762)

默认图像为 640×480；87°×58° 的 FOV 只生成便于启动的近似内参，不代表这个分辨率下的
工厂标定。官方产品页标注约 7–50 cm 的适用距离，RGB 来自左侧成像器；因此这里共用左视角，
没有虚构独立 RGB 相机。`depth_scale=1e-4` 是仿真编码选择，不代表精度或固件默认值。
需要拟合时，应读取真实设备当前 stream 的内参、畸变模型、基线及 depth scale。
[D405 官方产品信息](https://www.realsenseai.com/product-family/d405-series/)

## SimSense 与 Isaac：分别拆解，再决定迁移范围

SimSense 本身接收左右图像；场景成像是 SAPIEN 等上游渲染器的工作。
它采用 CSCT 特征与四方向 SGBM，并包含唯一性、左右一致性、滤波和深度转换。
这与 D405 的被动双目原理相符，但不是 D4 固件的开源复刻。
[SimSense 作者仓库与 pipeline](https://github.com/angli66/simsense)

| SimSense 环节 | 对 D405 / WRS 的判断 | 当前实现 |
| --- | --- | --- |
| 上游左右图像与材质/照明 | 高保真阶段需要真实纹理及双视角 | 单视角 RGB / Z，可选 matte 光照 |
| 图像噪声、rectification | 双目匹配前有价值，需实机参数 | 后续；当前直接在 rectified 几何上工作 |
| CSCT、Hamming cost、四方向 SGBM | 适合未来 `d405_stereo`，计算和存储成本明显增加 | 未实现 |
| uniqueness、左右一致性 | 真实匹配可信度与空洞来源，应随 matcher 一起实现 | fast 仅做几何重投影遮挡，不能等同匹配一致性 |
| median、disparity→Z、注册 | 深度转换现在复用；滤波应可选，避免抹平小枝叶 | 视差转换已实现；不做中值或空洞填充 |
| IR 投影、激光散斑模型 | 不适合原生 D405 | 不迁入 D405 默认模型 |

Isaac 的 single-view depth sensor 从渲染深度构造视差和右视图，加入视差误差及图像梯度置信度，
不运行完整双目 matcher。这里借鉴其算法分层，使用 WGPU 重写，不依赖 Omniverse/RTX/CUDA。
[NVIDIA single-view pipeline 文档](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.camera/latest/camera_extension.html#single-view-depth-camera)

| Isaac 环节 | WRS 迁移决定 |
| --- | --- |
| 几何 Z 与显示相机分离 | 已实现独立离屏管线；保持共享 GPU device |
| 视差、右视图重投影、遮挡 | 已实现轻量近似；无法恢复仅右相机可见的遮挡物 |
| 视差量化、噪声 | 已实现，参数可调，默认不冒充实机拟合值 |
| Sobel confidence、丢点 | 已实现可选项；当前 RGB 无真实纹理，默认关闭纹理淘汰 |
| outlier removal / 空洞处理 | 暂缓，先保留小目标与真实缺测边界 |
| USD、RTX 渲染框架、ISP | 不引入；当前任务只复用算法思路 |

## 测试与性能

测试覆盖正面平面、倾斜/偏轴物体、非对称内参、visual 局部变换、遮挡、量程、动态场景、
安装外参、真实 Lite6 挂载与 FK、帧快照、点云重投影、Brown LUT、CPU raycast 对照、
视差遮挡参考、GPU 随机可复现、误差随距离增长、纹理淘汰、量程边界、深度编码及资源重建。
相机测试还覆盖线性光照、双面着色、远处 RGB/深度量程分离、空背景不产生点云、
实时噪声切换以及 clone 的设置/随机序列继承。
GPU 对 CPU raycast 的低分辨率深度误差断言为 20 µm；
三角形恰好穿过像素中心的轮廓处可能因光栅覆盖规则与 raycast 判定不同。

2026-09-16，本机 NVIDIA GeForce RTX 5060 Laptop GPU / Vulkan，640×480，5 个物体、
3916 个三角形，预热后 100 帧；包含完整 Python 深度数据和两套有效 N×3 点云：

| 阶段（平均 ms） | ideal | d405_fast + Brown |
| --- | ---: | ---: |
| render command encoding（CPU） | 0.430 | 0.423 |
| sensor + XYZ command encoding（CPU） | 0.074 | 0.084 |
| GPU submit（CPU） | 0.236 | 0.243 |
| GPU 执行等待 + 回读 + unpack | 8.810 | 8.537 |
| 有效点压缩（CPU） | 6.756 | 6.676 |
| 完整帧循环 | 17.615 | 17.278 |
| 完整帧循环 p95 | 20.321 | 20.003 |
| 平均 FPS | 56.8 | 57.9 |

此处达到了 30 FPS 的时间预算；不保证复杂橘子树场景有相同速度。
fast 有较少有效点，CPU 压缩量不同，不能由这个结果推断 fast 的 GPU 运算比 ideal 更快。
GPU 传感器处理和 XYZ 合并计算；没有为独立 GPU 时间戳改变共享 device 的能力配置。
因此 command encoding 不是 GPU 执行耗时，GPU 时间包含在同步回读项中，不能据此拆分每个
kernel 的耗时。benchmark 也打印每项 p95，可直接在目标场景重测。

## 已知限制与 `d405_stereo` 扩展

当前是几何与可调误差模型，没有模拟果皮纹理、镜面反射、透明叶片、真实照明或 ISP。
alpha>0 的 mesh 视为不透明，alpha=0 跳过。单视图近似无法观察右相机独有遮挡物，
也不能产生真实 stereo matching 的误匹配；置信度不是 D4 ASIC 的置信度。
未实现异步帧队列、传感器曝光/运动模糊，且 GPU 输出回读仍有显著成本。

后续 `d405_stereo` 保留 `CameraModel / capture / DepthFrame` 接口，在 render 与编码之间
增加接受 `(left_image, right_image, left_calibration, right_calibration, T_left_right)` 的 matcher，
输出左侧 rectified Z、validity 与 confidence，再接现有畸变、深度单位与点云模块。
右 optical pose 为左 pose 沿本地 X 轴平移 baseline；非理想标定时使用实际左右外参。
届时再分步加入图像噪声、rectification、CSCT、SGBM、uniqueness、左右一致性和可选滤波。
当前传入 `mode="d405_stereo"` 会明确报错，不会静默退回理想深度。
