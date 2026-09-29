"""One Python-owned panel: control state, snapshots and main-loop callbacks."""
from collections import OrderedDict
from copy import deepcopy
import logging
import math
import threading
import uuid

from .constant import Anchor
from . import protocol
from .image import UIImage

_UNSET = object()


class UIPanel:
    """Define a small control panel and publish immutable state snapshots.

    Callbacks run on the World main loop. Keep them short; scheduled callbacks
    can apply results from long-running background work. Programmatic updates
    never invoke control callbacks.
    """

    def __init__(self, *, title='Scene controls', description='',
                 anchor=Anchor.TOP_RIGHT, offset=24, width=296, height=None, font_size=13,
                 closable=False, movable=False, visible=True, columns=1):
        self._session = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._revision = 0
        self._title = 'Scene controls'
        self._description = 'Adjust your scene and inspect its state.'
        self._font_size = 13
        self._closable = False
        self._movable = False
        self._visible = True
        self._visibility_revision = 0
        self._controls = {}
        self._images = {}
        self._callbacks = {}
        self._results = OrderedDict()
        self._panel_id = 'default'
        self._layout = {}
        self.configure(title=title, description=description, anchor=anchor,
                       offset=offset, width=width, height=height, font_size=font_size,
                       closable=closable, movable=movable, visible=visible, columns=columns)

    def configure(self, *, title=None, description=None, anchor=None,
                  offset=None, width=None, height=_UNSET, font_size=None,
                  closable=None, movable=None, visible=None, columns=None):
        """Update heading and layout; sizes are CSS pixels, height=None is auto.

        Anchors are top-left, top-right, bottom-left and bottom-right.
        font_size is the base text size in CSS pixels, independent of width.
        columns sets the grid width; buttons can opt into a column_span.
        closable adds a local hide button; movable enables header dragging.
        Explicit visible=True reopens locally closed panels. Omitted fields keep
        their current values; browser dragging/closing does not change Python state.
        """
        layout = {}
        if columns is not None:
            if type(columns) is not int or columns < 1:
                raise ValueError('columns must be a positive integer')
            layout['columns'] = columns
        if anchor is not None:
            if anchor not in Anchor.ALL:
                raise ValueError('unknown panel anchor')
            layout['anchor'] = anchor
        for name, value in [('offset', offset), ('width', width), ('height', height)]:
            if name == 'height' and value is None:
                layout[name] = None
            elif value is not None and value is not _UNSET:
                value = protocol.finite_number(value)
                if value < 0 or (name != 'offset' and value == 0):
                    raise ValueError('offset must be nonnegative; sizes must be positive')
                layout[name] = value
        if font_size is not None:
            font_size = protocol.finite_number(font_size)
            if font_size <= 0:
                raise ValueError('font_size must be positive')
        for name, value in [('closable', closable), ('movable', movable), ('visible', visible)]:
            if value is not None and not isinstance(value, bool):
                raise ValueError(f'{name} must be a bool')
        with self._lock:
            if title is not None:
                self._title = str(title)
            if description is not None:
                self._description = str(description)
            if font_size is not None:
                self._font_size = font_size
            if closable is not None:
                self._closable = closable
            if movable is not None:
                self._movable = movable
            if visible is not None:
                self._visible = visible
                self._visibility_revision += 1
            self._layout.update(layout)
            self._revision += 1

    def show(self):
        """Show in all viewers, including browsers that locally closed the panel."""
        self.configure(visible=True)

    def hide(self):
        """Hide in all viewers without removing controls or callbacks."""
        self.configure(visible=False)

    def add_button(self, control_id, *, label=None, on_click=None,
                   repeat=False, repeat_hz=10, shortcut=None,
                   variant='default', column_span=None,
                   group='', enabled=True):
        """Add a button whose on_click callback takes no arguments.

        repeat=True activates on press and repeats while held, capped at
        repeat_hz. shortcut is a single KeyboardEvent.key, e.g. 'w' or
        'ArrowRight'. Ordinary buttons still activate on click.
        variant selects CSS appearance. column_span=None fills the panel width;
        a positive span places buttons side by side within the panel's columns.
        """
        if not isinstance(repeat, bool):
            raise ValueError('repeat must be a bool')
        repeat_hz = protocol.finite_number(repeat_hz)
        if repeat_hz <= 0:
            raise ValueError('repeat_hz must be positive')
        if not isinstance(variant, str) or not variant or len(variant) > 64:
            raise ValueError('variant must be a nonempty CSS variant name')
        if column_span is not None and (type(column_span) is not int or column_span < 1):
            raise ValueError('column_span must be a positive integer or None')
        self._add(control_id, 'button', label, group, enabled, on_click,
                  repeat=repeat, repeat_hz=repeat_hz,
                  shortcut=protocol.shortcut_key(shortcut),
                  variant=variant, column_span=column_span)

    def add_slider(self, control_id, *, min_value=0, max_value=1, step=0.01,
                   value=0, label=None, unit='', on_change=None,
                   continuous=False, update_hz=30,
                   group='', enabled=True):
        """Add a numeric slider; the callback receives a float.

        Bounds and step must be finite, with min_value < max_value and step > 0.
        Values are validated against the bounds and snapped to the nearest
        step from min_value. Units are for display only, with no conversion.
        continuous=True also sends values during dragging, capped at update_hz
        per slider. Release flushes the final value; slow replies coalesce updates.
        The default continuous=False sends only committed changes.
        """
        low, high, step = map(protocol.finite_number, (min_value, max_value, step))
        if low >= high or step <= 0 or not math.isfinite((high - low) / step):
            raise ValueError('slider requires finite min < max and step > 0')
        if not isinstance(continuous, bool):
            raise ValueError('continuous must be a bool')
        update_hz = protocol.finite_number(update_hz)
        if update_hz <= 0:
            raise ValueError('update_hz must be positive')
        spec = dict(min=low, max=high, step=step, unit=str(unit),
                    continuous=continuous, update_hz=update_hz)
        spec['value'] = protocol.slider_value(spec, value)
        self._add(control_id, 'slider', label, group, enabled, on_change, **spec)

    def add_label(self, control_id, *, value='', label=None, group=''):
        """Add read-only text, updated with set_value()."""
        self._add(control_id, 'label', label, group, True, None, value=str(value))

    def add_image(self, control_id, *, image=None, label=None, group='',
                  format='png', quality=85, max_fps=20):
        """Add a read-only image and return its update()/clear() handle.

        Accepts uint8 gray/RGB/RGBA arrays or PNG/JPEG paths. max_fps caps
        publishing; updates coalesce to the latest frame. Use PNG for RGBA.
        """
        handle = UIImage(self._panel_id, self._session, control_id,
                         format=format, quality=quality, max_fps=max_fps)
        if image is not None:
            handle.update(image)
        with self._lock:
            self._add(control_id, 'image', label, group, True, None,
                      stream=handle._identity['stream'])
            self._images[control_id] = handle
        return handle

    def set_image(self, control_id, image, *, color_order='rgb'):
        """Update an image by ID; None clears it. Programmatic only."""
        with self._lock:
            self._controls[control_id]  # Preserve KeyError for unknown IDs.
            if control_id not in self._images:
                raise ValueError('control is not an image')
            handle = self._images[control_id]
        if image is None:
            handle.clear()
        else:
            handle.update(image, color_order=color_order)

    def add_checkbox(self, control_id, *, value=False, label=None,
                     on_change=None, group='', enabled=True):
        """Add a checkbox; on_change receives a bool when toggled."""
        self._add(control_id, 'checkbox', label, group, enabled, on_change,
                  value=protocol.checkbox_value(value))

    def add_select(self, control_id, *, options, value=None, label=None,
                   on_change=None, group='', enabled=True):
        """Add a dropdown of strings; on_change receives the selected string."""
        if isinstance(options, (str, bytes)):
            raise ValueError('options must be a sequence of unique strings')
        options = list(options)
        if (not options or any(not isinstance(v, str) or not v for v in options)
                or len(set(options)) != len(options)):
            raise ValueError('options must be nonempty, unique strings')
        value = options[0] if value is None else value
        protocol.select_value({'options': options}, value)
        self._add(control_id, 'select', label, group, enabled, on_change,
                  options=options, value=value)

    def set_value(self, control_id, value):
        """Update a slider, select, checkbox or label without invoking its callback.

        Raises KeyError for unknown IDs and ValueError for invalid values or
        buttons. Slider values use the same validation as browser events.
        """
        with self._lock:
            control = self._controls[control_id]
            if control['kind'] == 'image':
                raise ValueError('use set_image() to update images')
            if control['kind'] == 'button':
                raise ValueError('buttons do not have a value')
            if control['kind'] == 'slider':
                value = protocol.slider_value(control, value)
            elif control['kind'] == 'select':
                value = protocol.select_value(control, value)
            elif control['kind'] == 'checkbox':
                value = protocol.checkbox_value(value)
            else:
                value = str(value)
            if control['value'] != value:
                control['value'] = value
                self._revision += 1

    def set_enabled(self, control_id, enabled):
        """Enable or disable input in both the browser and Python dispatcher."""
        with self._lock:
            control = self._controls[control_id]
            if control['enabled'] != bool(enabled):
                control['enabled'] = bool(enabled)
                self._revision += 1

    def remove(self, control_id):
        """Remove a control; queued events for it will return an error."""
        with self._lock:
            del self._controls[control_id]
            image = self._images.pop(control_id, None)
            if image is not None:
                image._close()
            self._callbacks.pop(control_id, None)
            self._revision += 1

    def _add(self, control_id, kind, label, group, enabled, callback, **fields):
        if not protocol.valid_id(control_id):
            raise ValueError('control_id must be a nonempty string of at most 128 characters')
        if callback is not None and not callable(callback):
            raise TypeError('callback must be callable')
        with self._lock:
            if control_id in self._controls:
                raise ValueError(f'duplicate control_id: {control_id}')
            self._controls[control_id] = dict(
                id=control_id, kind=kind,
                label=control_id if label is None else str(label),
                group=str(group), enabled=bool(enabled), **fields)
            self._callbacks[control_id] = callback
            self._revision += 1

    def _snapshot(self, after_revision=None):
        with self._lock:
            if after_revision == self._revision:
                return None
            return dict(type='ui_state', session=self._session, id=self._panel_id,
                        revision=self._revision, title=self._title,
                        description=self._description, font_size=self._font_size,
                        closable=self._closable, movable=self._movable,
                        visible=self._visible, visibility_revision=self._visibility_revision,
                        layout=dict(self._layout),
                        controls=deepcopy(list(self._controls.values())))

    def _handle_event(self, payload):
        """Validate and execute on the main thread, deduplicating recent IDs."""
        if not protocol.is_ui_event(payload) or payload['session'] != self._session:
            return None
        event_id, control_id = payload.get('event_id'), payload.get('id')
        if event_id in self._results:
            return self._results[event_id]
        result = dict(type='ui_result', session=self._session,
                      panel_id=self._panel_id, event_id=event_id, id=control_id, ok=False)
        previous = None
        control = None
        try:
            with self._lock:
                control = self._controls.get(control_id)
                if control is None:
                    raise ValueError('unknown control')
                if not control['enabled'] or control['kind'] in ('label', 'image'):
                    raise ValueError('control does not accept input')
                callback = self._callbacks[control_id]
                if control['kind'] in ('slider', 'select', 'checkbox'):
                    value = payload.get('value')
                    previous = control['value']
                    self.set_value(control_id, value)
                    value = control['value']
                elif payload.get('value') is not None:
                    raise ValueError('buttons do not accept a value')
            # Do not hold the snapshot lock while user code is running.
            if callback is not None:
                if control['kind'] in ('slider', 'select', 'checkbox'):
                    callback(value)
                else:
                    callback()
            result['ok'] = True
        except Exception as exc:
            if previous is not None:
                with self._lock:
                    if self._controls.get(control_id) is control:
                        self.set_value(control_id, previous)
            result['error'] = str(exc) or type(exc).__name__
            logging.getLogger(__name__).warning('UI control %s: %s', control_id, exc)
        result['state'] = self._snapshot()
        self._results[event_id] = result
        if len(self._results) > 256:
            self._results.popitem(last=False)
        return result
