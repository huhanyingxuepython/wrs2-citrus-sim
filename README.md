# WRS2 柑橘采摘仿真

带 FAFU 机械臂、可断果梗和履带底盘小车的柑橘采摘仿真。小车用逆可达图（IRM）停到果实前，再沿预抓取点闭合夹爪并后拉摘果。

浏览器打开 http://127.0.0.1:8000/ 。需要支持 WebGPU 的 Chrome。

## 环境

Python 3.12+。在仓库根目录：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

## 运行

```bash
# 树 + 机械臂 + 小车。左侧 Cart / Actions
.venv/bin/python -m examples.agriculture.robot_citrus_interaction

# 只看小车和机械臂，没有树
.venv/bin/python -m examples.agriculture.show_cart

# 果梗拉脱示例
.venv/bin/python -m examples.agriculture.fruit_harvest
```

无界面检查：

```bash
.venv/bin/python -m examples.agriculture.robot_citrus_interaction --headless
```

Chrome 若打不开 WebGPU，用：

```bash
google-chrome --enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan,WebGPUService --use-angle=vulkan http://127.0.0.1:8000/
```

## 场景里怎么操作

`robot_citrus_interaction` 读 `agriculture/configs/presets/lab_citrus_robot.json`。

- **IRM to orange_000**：按当前果实的抓取位姿查询小车停点，再旋转、直行、旋转开过去。地面上的点是小车 CAD 原点，不是箱体中心。绿色是选中的停点，红色是这条抓取路径失败的候选，蓝色是其余候选。
- **Grasp orange_000**：张开夹爪，走到预抓取，再靠近果实并闭合。闭合后沿世界 −Y 后拉，果梗拉力超过阈值后断开。
- **Arm home**：关节空间回到停放姿态。后拉逆解失败时用它收回机械臂。
- **Show IRM candidates**：开关候选点显示。

当前按钮只使用配置里的 `robot_demo.fruit_id`（默认 `orange_000`）。改成 `orange_001` 或 `orange_002` 后重启，同一套 IRM 会查那颗果。IRM 缓存是相对机械臂底座的，挪树不用重算。

## 小车和 IRM

| 文件 | 内容 |
| --- | --- |
| `agriculture/bunker_cart.py` | 平面小车，状态是 `(x, y, yaw)` |
| `agriculture/cart_irm.py` | 采样 IRM、查询停点、独轮车轨迹 |
| `agriculture/assets/bunker_cart/cart.stl` | 碰撞网格 |
| `agriculture/assets/bunker_cart/cart_viz.stl` | 显示网格 |
| `agriculture/assets/bunker_cart/cart_meta.json` | 安装位和包围盒 |
| `agriculture/assets/bunker_cart/irm_tcp.npz` | 12 万条末端采样缓存 |

植物几何、果梗和夹爪接触的更细说明在 [agriculture/README.md](agriculture/README.md) 和 [agriculture/HARVEST.md](agriculture/HARVEST.md)。

## WRS 库

下面是这个仓库所基于的 WRS 查看器和运动学库。

# The Workman Robot System (WRS)

A self-contained robotics library — kinematics, scene, mesh geometry, collision,
grasp planning, motion planning, and a viewer.

## Lineage

WRS is the third generation of this codebase:

| Generation | Renderer | Repository |
|---|---|---|
| 1st | Panda3D | [wanweiwei07/wrs](https://github.com/wanweiwei07/wrs) |
| 2nd | Pyglet / OpenGL | [wanweiwei07/one](https://github.com/wanweiwei07/one) |
| **3rd (this one)** | **WGPU** | — |

The rewrite is motivated by OpenGL's inherent limitations in how the CPU and the
GPU interact: state is global and implicit, buffer uploads and draw submission
are hard to separate, and validation happens per call rather than once. WGPU
replaces that with explicit pipelines, command buffers, and up-front validation,
so the renderer builds its state once and the per-frame work is just recording
draws.

## Architecture: one server, two clients

```
   Python client                      Web browser client
   (your script)                      (visualization)
        │                                     │
        │  publishes the scene                │  fetches the page,
        │  as it changes                      │  renders with WebGPU
        ▼                                     ▼
        └────────►  server (port 8000)  ◄─────┘
```

**The server** owns port 8000. It serves the viewer page and relays scenes; it
holds no simulation state of its own, only the latest scene it was told about.

**The Python client** is where you work. Building the scene is ordinary library
code; calling `run()` connects to the server and starts publishing.

```python
from wrs import wvw, wssop, wsso

base = wvw.World(cam_pos=(.3, .3, .3))
wssop.frame().add_to_scene(base.scene)
wsso.SceneObject.from_file("bunny.stl").add_to_scene(base.scene)
base.run()
```

**The web browser client** is the visualization. It connects to the local
service on port 8000, pulls the scene from the server, and draws it.

## It sets itself up

`run()` checks whether a server is already listening. If not, it starts one; if
so, it simply connects and begins sending. When no browser client is open, a web
page is opened for you, connected to the server, and showing the scene.

So the two-client mechanism underneath is **transparent to you**. In practice
there is one thing to know:

> Call `run()`, then look at **http://127.0.0.1:8000/** in your browser.

The rest is handled:

- **Several Python clients at once** — the newest one takes the page over, and
  the one it displaced stops rather than fighting for it.
- **Everything closed** — with no page and no script left, the server shuts
  itself down after a short grace period, so nothing is left running.
- **The page outlives your script** — rerun as often as you like; the page stays
  open and picks up whatever the next run publishes.

You can also start the server yourself, which is useful on a headless machine or
when you want it to stay put:

```
py -3.12 -m wrs.viewer.server
```

## Getting started

```
py -3.12 examples/test_bunny.py
```

More examples live in [`examples/`](examples/). For the library itself, see
[`docs/API_INDEX.md`](docs/API_INDEX.md) — a generated map of every public
function, so in-house utilities are easy to find before reaching for an external
one.

## Interactive viewer controls

Define buttons, sliders, dropdowns, and status text from Python with `base.ui`. Browser
actions run callbacks in the World loop, and Python updates both the scene and
the panels. Use `base.ui.add_panel()` for independently positioned panels with
configurable sizes. The UI uses native HTML/CSS with no frontend dependencies.

```
py -3.12 examples/viewer_ui.py
```

See [Viewer controls](docs/tutorials/viewer_ui.md) for the API, lifecycle, and
design notes.
