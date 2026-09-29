# Viewer controls

Define buttons, sliders, dropdowns, checkboxes, images and read-only text from Python. Native HTML
and CSS draw the panels above the WebGPU canvas, without a frontend framework,
external fonts or a build step.

## Example

Run from the repository root with WRS installed:

```sh
python -m examples.viewer_ui
```

The example opens three panels: six UR3 joint sliders at top left, a node
selector with X/Y/Z controls at top right, and Workspace status at bottom left.
The coordinate axes follow the selected node; toggle **Show coordinate axes**
to hide or show them. Drag a panel heading to move it;
close the joint/node panels with × and reopen them with **Show controls**.
Joint and position sliders update the scene while dragging.
Hold the Joint 1 buttons or the left/right arrow keys to move that joint in
one-degree steps. Press H to restore the home pose.

## Python API

```python
from wrs import wssop, wvw
import wrs.viewer.web_ui as wvui

base = wvw.World(cam_pos=(0.8, 0.8, 0.5))
block = wssop.box()
block.add_to_scene(base.scene)
panel = base.ui.add_panel('position', title='Position',
                          anchor=wvui.Anchor.TOP_RIGHT, width=260, font_size=12,
                          closable=True, movable=True)


def move_x(value):
    block.pos = [value, 0, 0]
    panel.set_value('status', f'X = {value:.2f} m')


panel.add_slider('x', label='X position', unit='m',
                  min_value=-0.25, max_value=0.25, step=0.01,
                  value=0, on_change=move_x)
panel.add_label('status', label='Last action', value='Ready')
base.run()
```

`base.ui.add_*()` addresses the default panel. `add_panel(id, **options)` returns
an independent panel; `remove_panel(id)` removes a named panel. IDs are nonempty
strings of up to 128 characters, unique within their panel. Empty panels stay hidden.

| Method | Behavior |
| --- | --- |
| `add_button(id, label=..., on_click=...)` | Callback with no arguments |
| `add_slider(id, ..., on_change=...)` | Callback with a float when the value is committed |
| `add_select(id, options=[...], value=..., on_change=...)` | Callback with the selected string |
| `add_checkbox(id, value=False, on_change=...)` | Callback with a bool when toggled |
| `add_label(id, label=..., value=...)` | Read-only text |
| `add_image(id, image=..., format='png', max_fps=20)` | Read-only image; returns an `UIImage` handle |
| `set_image(id, image, color_order='rgb')` | Replace an image; `None` clears it |
| `set_value(id, value)` | Update a value without invoking its callback |
| `set_enabled(id, enabled)` | Enable or disable interaction |
| `remove(id)` | Remove a control |

Controls accept `group` to start a visual section. Declare controls in the same
group together. Slider units are display text; values are not converted.
Checkbox values must be Python booleans (`True` or `False`). For example:

```python
panel.add_checkbox('axes', label='Show coordinate axes', value=True,
                   on_change=lambda checked: print('Show axes:', checked))
panel.set_value('axes', False)  # Uncheck without invoking the callback.
```

## Images and live previews

Create an image once, then call `update()` whenever new pixels are available:

```python
panel = base.ui.add_panel('vision', title='Vision', width=420, movable=True)
preview = panel.add_image('rgb', label='Camera RGB', format='jpeg', max_fps=20)
result = panel.add_image('result', label='Algorithm result')

result.update(result_rgb)  # One-off update; defaults to lossless PNG.

def refresh(dt):
    frame = camera.capture(base.scene)
    preview.update(frame.rgb)

base.schedule_interval(refresh, interval=1 / 20)
# base.run() starts publishing and scheduled callbacks.
```

`add_image()` is also available on `base.ui` for its default panel. It accepts
`label` and `group`, an optional initial `image`, `format='png'` or `'jpeg'`,
JPEG `quality=85` (integer 1–95), and positive finite `max_fps=20`.
The image fills the panel's content width and preserves its aspect ratio.

Supported inputs are NumPy `uint8` arrays of shape `(H, W)` (gray), `(H, W, 3)`
(RGB), `(H, W, 4)` (RGBA, PNG only), or a PNG/JPEG filesystem path. For OpenCV
arrays use `preview.update(bgr_image, color_order='bgr')`; files already carry
RGB colors. Inputs are copied, so callers may reuse their array after `update()`
returns. The call queues a frame; it does not wait for browser display.

`panel.set_image('rgb', pixels)` addresses the same control by ID. Use
`preview.clear()` or `panel.set_image('rgb', None)` to show the empty placeholder.
Removing the control or its panel invalidates its handle; subsequent updates
raise `RuntimeError`. Images accept no browser input, and `set_value()` is for
the other control types.

For metric or raw depth, convert to display colors using explicit fixed bounds:

```python
from wrs.viewer.web_ui import colorize_depth

depth_view = panel.add_image('depth', label='Depth · meters')
depth_view.update(colorize_depth(
    frame.depth_m, value_range=(0.07, 0.5), valid_mask=frame.valid_mask))
```

`colorize_depth()` uses Jet: dark blue, blue, cyan, green, yellow, red, then dark red
from near to far.
Nonpositive/nonfinite or masked pixels are black; finite positive values outside
the bounds are clipped. Bounds use the input's units, and input data is unchanged.

Updates retain only the newest pending frame and are encoded on a worker. No
image bytes enter ordinary UI state or callback replies. `max_fps` caps sends;
the effective rate also depends on `World(hz=...)`, capture, encoding and network
time. The final queued frame is sent even if no more updates follow. Capture
callbacks themselves remain synchronous; expensive capture can delay other
main-loop callbacks.

The hub caches one latest frame per image, replays it after page reload, and
allows one unacknowledged frame per image/browser. Slow decoders skip intermediate
frames. Disconnects preserve the last displayed image and mark the panel offline;
new scripts, removed controls and cleared images cannot replay stale pictures.
Browser blob URLs are released when replaced or destroyed. This is an image
preview channel over the existing WebSocket, not a guaranteed video frame rate.

Run the NumPy animation, snapshot button and scene slider together:

```sh
python -m examples.viewer_images
python -m examples.viewer_images --camera
```

The second command uses the existing `VirtualD405` and requires its GPU renderer.
Image encoding uses Pillow, included in the project's dependencies.

## Button repeat and shortcuts

`on_click` is the activation callback for both ordinary and repeating buttons.
It takes no arguments. Defaults are `repeat=False`, `repeat_hz=10` and
`shortcut=None`.

```python
panel.add_button('step_right', label='Step right', on_click=step_right,
                  repeat=True, repeat_hz=10, shortcut='ArrowRight')
panel.add_button('reset', label='Reset', on_click=reset, shortcut='h')
```

With `repeat=True`, pressing the button calls `on_click` immediately, then
repeats while held. `repeat_hz` is a positive finite rate in callbacks per second;
busy ticks are skipped while waiting for Python. Release stops future ticks,
without a second click or a queue to drain. An already-sent callback may still
complete. Pointer cancellation, focus loss, a hidden tab/control, disabling,
disconnecting and callback errors also stop repetition.

`shortcut` uses a single [KeyboardEvent.key](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key)
such as `'w'`, `'ArrowRight'`, `'Enter'`, or `' '` for Space. Letters match
regardless of case. Tab and modifier-only bindings are reserved; key combinations
are not supported. Ctrl/Alt/Meta shortcuts and text composition are left alone.

A shortcut activates once on keydown, or repeats at `repeat_hz` when enabled.
Operating-system key repeat does not add extra activations. Focused repeating
buttons also accept Enter and Space. Ordinary buttons keep native click behavior.
The implementation uses `pointerdown`/`pointerup` and `keydown`/`keyup`; no new
wire event type is needed.

Shortcuts work only for visible, enabled controls. They do not intercept typing
or keys directed at an input or dropdown. Enter and Space on another button keep
that button's native behavior. A claimed shortcut is not
also sent to the scene's key handlers. Assign distinct shortcuts; if two visible
buttons share one, the first registered button handles it. The key hint is shown
on the button and exposed through `aria-keyshortcuts`.

Browser-only `Button` widgets use `onClick` with the same `repeat`, `repeat_hz`
and `shortcut` options. Destroying a widget removes its keyboard listeners.

### Keypad layout and styling

Use `columns` for the panel grid and `column_span` for each button. Other controls
and section headings keep the full width. Defaults (`columns=1`,
`column_span=None`, `variant='default'`) keep the usual stacked layout.

```python
panel = base.ui.add_panel('jog', title='Jog', columns=3, width=280)
panel.add_button('forward', label='Forward +Y', shortcut='w',
                  repeat=True, repeat_hz=10, on_click=forward,
                  column_span=1, variant='keycap')
```

`variant='keycap'` gives a button a dark outline, a large key hint and a small
action label. `variant` only selects appearance via `data-variant` on the widget;
callbacks, shortcuts and hold behavior are unchanged. Add a CSS selector such as
`.wrs-ui-panel [data-variant='custom'] .ui-action` for another style, without
changing the button class or transport. Keycap colors use `--ui-key-border` and
`--ui-key-background`. Browser-only `Button` accepts the same options;
`Panel({ columns: 3 })` sets its grid.

For a working robot example, run:

```sh
python -m examples.agriculture.robot_citrus_interaction
```

Its move keys form two rows: **Q W E / A S D**. W/S move forward/back along Y,
A/D move left/right along X, and Q/E move up/down along Z. Hold the mouse button
or key to repeat at up to 10 Hz.
I/K, J/L and U/O rotate about world X, Y and Z at the TCP. Sliders set the move
and rotation increments. Each jog starts from the current command, keeping the
target at most one step ahead; releasing stops repeats, and that last step can
finish. R opens the gripper and F closes it; both move toward their target
opening with a speed limit. Escape holds the current servo commands.

The example uses two independent finger servos with matching targets, so contact
can leave the fingers at different positions. This is an example controller,
not a physical mimic constraint. The shared robot definition is unchanged.

## Slider updates

Sliders default to `continuous=False`: dragging previews the value in the browser,
and releasing commits it to Python. Set `continuous=True` to run the same
`on_change` callback during dragging too:

```python
panel.add_slider('live_x', label='X position', unit='m',
                  min_value=-0.25, max_value=0.25, step=0.01,
                  value=0, on_change=move_x, continuous=True, update_hz=30)
```

`update_hz` defaults to 30 and must be a positive finite number. It caps drag
updates per slider; it is not a guaranteed callback rate. Each slider waits for
its previous reply and keeps only the latest queued position. Release sends the
final value without waiting for the rate limit, or as soon as the previous reply
arrives. An already-sent, unchanged value is not sent again. Errors, timeouts,
disconnects and control removal discard queued input.

Browser-only `Slider` widgets accept the same `continuous` and `update_hz`
options with their `onChange` callback.

| Layer | Default behavior |
| --- | --- |
| Browser slider | Native input events update the local preview; continuous sends are capped by `update_hz` |
| Python event loop | `World(tick_hz=60)` checks queued events about every 16.7 ms |
| Scene and UI publishing | `World(hz=30)` checks for updates about every 33.3 ms and sends changed state; UI replies are also drained on this cycle |
| Browser rendering | Uses `requestAnimationFrame`, independently of the network rates |

The hub forwards messages as they arrive. Callback cost, network latency and
publishing time can reduce the effective rate. `World(hz=60, tick_hz=60)` raises
the publishing rate as well; a slider setting alone does not change it. These
settings control polling intervals, not real-time deadlines.

## Panel layout

Pass these options to `add_panel()` or change them with `panel.configure(...)`:

| Option | Default | Meaning |
| --- | --- | --- |
| `title`, `description` | `'Scene controls'`, `''` | Heading and optional description |
| `anchor` | `wvui.Anchor.TOP_RIGHT` | Also `TOP_LEFT`, `BOTTOM_LEFT`, `BOTTOM_RIGHT` |
| `offset` | `24` | Distance from the anchored edges, in CSS pixels |
| `width`, `height` | `296`, `None` | CSS pixels; `None` gives automatic height |
| `columns` | `1` | Positive integer; buttons can share a row with `column_span` |
| `font_size` | `13` | Base font size in CSS pixels; headings scale relative to it |
| `closable` | `False` | Show a close button |
| `movable` | `False` | Allow heading dragging within the viewport or container |
| `visible` | `True` | Show or hide the panel |

Text size is configurable; it does not shrink automatically with panel width.
Content scrolls when it exceeds the available height. The example uses 12px
text, 260px widths and 16px offsets. Mouse resizing is not implemented.

Closing and dragging affect only the current browser. Ordinary value updates
preserve those local choices. `panel.show()` and `panel.hide()` control visibility
in all viewers, retaining controls and callbacks. Repeated `show()` calls can
reopen a locally closed panel. Reloading restores the Python configuration.
Changing layout dimensions or anchor resets the dragged position; supplying the
same layout preserves it. Panels do not automatically avoid each other.

## Components and files

```text
wrs/viewer/
  web_ui/
    __init__.py       exports Anchor, UIPanel, UIManager, UIImage, colorize_depth
    constant.py       Anchor constants
    manager.py        default/named panels and event routing
    panel.py          control definitions, state and callbacks
    protocol.py       JSON message contract and value validation
    image.py          latest image storage, encoding and depth colorization
  web/ui/
    index.js          browser exports
    controls.js       Button, Slider, Select, Checkbox, Text, ImageView
    panel.js          Panel layout, collapse, close and drag
    python_panel.js   Python state/event binding
    styles.css        shared appearance
```

Import Python components from `wrs.viewer.web_ui`. Browser widgets own their DOM,
updates and cleanup; the small shared `Control` base handles listeners and
interaction state. They can also be used independently of Python:

```javascript
import { Panel, Button, Text } from './ui/index.js';

const panel = new Panel({ title: 'Local controls', container: document.body,
  anchor: 'bottom-left', width: 260, fontSize: 12, movable: true });
const status = panel.add(new Text({ label: 'Status', value: 'Ready' }));
panel.add(new Button({ label: 'Run',
  onClick: () => status.update({ value: 'Done' }) }));
```

Load `./ui/styles.css` once. `Panel` defaults to inline layout and can be mounted
later with `mount(element)`. Floating panels in a custom container require that
container to have `position: relative` and a defined size. `setLayout(...)`
changes placement, and `update(...)` changes heading, font or behavior options.
After resizing a custom container, call `setLayout()` to constrain its dragged
panel again. `show()` / `hide()` change visibility; `destroy()` removes a panel
and its owned controls. Individual widgets can use `element.hidden`.

## State and callbacks

Callbacks run on the `World.run()` thread. Sliders send on release, or also while
dragging when `continuous=True`. Keep callbacks short. Python validates values
and publishes the resulting state to all connected viewers.

`web_ui.protocol` defines `ui_state` (a collection of panels), `ui_event`,
`ui_result` and `ui_reset`. Events require a panel ID, session ID, event ID and
control ID. The hub relays messages and caches state for page reloads; the
manager routes events and the panel runs callbacks. These JSON messages are
separate from the binary scene protocol in `viewer.protocol`. Images use binary
`ui_image` messages with panel/session/control IDs, a unique stream ID, monotonic
sequence, MIME type and encoded bytes. `ui_image_ack` JSON messages terminate at
the hub after browser decoding; they never invoke Python control callbacks.

Session IDs reject stale events when a script or panel is replaced. Disconnected
controls are disabled, and actions are not replayed on reconnect. Callback errors
appear in the panel and restore that slider/select/checkbox value; other scene changes
made by the callback are not rolled back.
