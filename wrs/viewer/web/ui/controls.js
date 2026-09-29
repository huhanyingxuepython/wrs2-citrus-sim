/** Native controls: DOM and callbacks only, independent of the viewer transport. */
let nextControlId = 0;

class Control {
  constructor(kind, props) {
    this.element = document.createElement('div');
    this.element.className = `ui-control ui-${kind}`;
    this.control = { id: '', label: '', enabled: true, ...props, kind };
    this.pending = false;
    this.editing = false;
    this._events = new AbortController();
  }

  _listen(element, event, callback, options = {}) {
    element.addEventListener(event, callback, { ...options, signal: this._events.signal });
  }

  _bindBusyGuard(allowPending = () => false) {
    this._listen(this.input, 'pointerdown', (event) => {
      if (this.pending && !allowPending()) event.preventDefault();
    });
    this._listen(this.input, 'keydown', (event) => {
      if (this.pending && !allowPending() && event.key !== 'Tab') event.preventDefault();
    });
  }

  _updateInput(allowPending = false) {
    // aria-disabled preserves keyboard focus while a request is pending.
    this.input.disabled = !this.control.enabled;
    this.input.setAttribute('aria-disabled', String(this.input.disabled || (this.pending && !allowPending)));
    this.input.setAttribute('aria-busy', String(this.pending));
  }

  destroy() {
    this._panel?._children.delete(this);
    this._panel = null;
    this._events.abort();
    this.element.remove();
  }
}

export class Button extends Control {
  constructor({ onClick = () => {}, ...props } = {}) {
    super('button', { repeat: false, repeat_hz: 10, shortcut: null,
      variant: 'default', column_span: null, ...props });
    this._onClick = onClick;
    this.input = document.createElement('button');
    this.input.type = 'button';
    this.input.className = 'ui-action';
    this.caption = document.createElement('span');
    this.caption.className = 'ui-action-label';
    this.shortcut = document.createElement('kbd');
    this.shortcut.className = 'ui-shortcut';
    this.shortcut.setAttribute('aria-hidden', 'true');
    this.input.append(this.caption, this.shortcut);
    this.element.appendChild(this.input);
    this._listen(this.input, 'click', event => {
      // Pointer repeat already activated on press; release must not click again.
      if (event.detail > 0 && this._skipClick) { this._skipClick = false; return; }
      if (this._available() && !this.pending) this._onClick();
    });
    this._listen(this.input, 'pointerdown', event => {
      this._skipClick = false;
      if (!this.control.repeat || !this._available() || this._held
          || event.button !== 0 || !event.isPrimary) return;
      event.preventDefault();
      this._skipClick = true;
      this.input.focus({ preventScroll: true });
      this._held = { type: 'pointer', id: event.pointerId, target: this.input };
      this.input.setPointerCapture(event.pointerId);
      this.input.dataset.held = 'true';
      this._repeat();
    });
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      this._listen(this.input, type, event => {
        if (this._held?.type === 'pointer' && this._held.id === event.pointerId) this.cancelPending();
      });
    }
    this._listen(document, 'keydown', event => this._keyDown(event), { capture: true });
    this._listen(document, 'keyup', event => {
      if (this._held?.type !== 'key' || this._held.id !== (event.code || event.key)) return;
      event.preventDefault();
      this.cancelPending();
    }, { capture: true });
    this._listen(window, 'blur', () => this.cancelPending());
    this._listen(document, 'visibilitychange', () => {
      if (document.hidden) this.cancelPending();
    });
    this._listen(document, 'focusin', event => {
      if (this._held && event.target !== this._held.target) this.cancelPending();
    });
    this._bindBusyGuard(() => this.control.repeat);
    this.update();
  }

  update(props = {}) {
    const next = { ...this.control, ...props };
    if (typeof next.variant !== 'string' || !next.variant) throw new TypeError('variant must be a CSS variant name');
    if (next.column_span !== null && (!Number.isInteger(next.column_span) || next.column_span < 1)) {
      throw new RangeError('column_span must be a positive integer or null');
    }
    if (typeof next.repeat !== 'boolean') throw new TypeError('repeat must be a boolean');
    if (!Number.isFinite(next.repeat_hz) || next.repeat_hz <= 0) throw new RangeError('repeat_hz must be positive');
    if (next.shortcut !== null && (typeof next.shortcut !== 'string' || !next.shortcut)) {
      throw new TypeError('shortcut must be a KeyboardEvent.key or null');
    }
    if (!next.enabled || next.repeat !== this.control.repeat || next.shortcut !== this.control.shortcut) {
      this.cancelPending();
    }
    Object.assign(this.control, next);
    this.element.dataset.variant = next.variant;
    this.element.style.gridColumn = next.column_span === null ? '' : `span ${next.column_span}`;
    this._updateInput(next.repeat);
    this.caption.textContent = this.pending && !next.repeat ? `${next.label}…` : next.label;
    this.shortcut.hidden = !next.shortcut;
    const key = next.shortcut;
    this.shortcut.textContent = ({ ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓', ' ': 'Space' })[key]
      || (key?.length === 1 ? key.toUpperCase() : key);
    if (key) this.input.setAttribute('aria-keyshortcuts', key === ' ' ? 'Space' : key === '+' ? 'plus' : key);
    else this.input.removeAttribute('aria-keyshortcuts');
  }

  _available() {
    return this.control.enabled && !document.hidden && this.input.isConnected
      && this.input.getClientRects().length > 0 && getComputedStyle(this.input).visibility !== 'hidden';
  }

  _keyDown(event) {
    if (event.defaultPrevented || event.isComposing || event.ctrlKey || event.altKey || event.metaKey
        || !this._available()) return;
    const target = event.target;
    if (target instanceof Element && target.closest('input, textarea, select, [contenteditable]')) return;
    if (target instanceof Element && target.closest('button') && target !== this.input
        && ['Enter', ' '].includes(event.key)) return;
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    const shortcut = this.control.shortcut;
    const matches = shortcut && key === (shortcut.length === 1 ? shortcut.toLowerCase() : shortcut);
    const focused = target === this.input && this.control.repeat && ['Enter', ' '].includes(event.key);
    if (!matches && !focused) return;
    event.preventDefault(); // Keep this activation out of the scene's key handlers.
    if (event.repeat || this._held) return;
    this._held = { type: 'key', id: event.code || event.key, target };
    this.input.dataset.held = 'true';
    this._repeat();
  }

  _repeat() {
    if (!this._held || !this._available()) { this.cancelPending(); return; }
    // Skip busy ticks instead of queuing clicks to run after release.
    if (!this.pending) this._onClick();
    if (this._held && this.control.repeat) {
      this._repeatTimer = setTimeout(() => this._repeat(), Math.min(1000 / this.control.repeat_hz, 2147483647));
    }
  }

  cancelPending() {
    clearTimeout(this._repeatTimer);
    const held = this._held;
    this._held = null;
    delete this.input.dataset.held;
    if (held?.type === 'pointer' && this.input.hasPointerCapture(held.id)) this.input.releasePointerCapture(held.id);
  }

  destroy() {
    this.cancelPending();
    super.destroy();
  }
}

export class Slider extends Control {
  constructor({ onChange = () => {}, ...props } = {}) {
    super('slider', { min: 0, max: 1, step: 0.01, value: 0, unit: '',
      continuous: false, update_hz: 30, ...props });
    this._onChange = onChange;
    this._lastEmitAt = -Infinity;
    const top = document.createElement('div');
    top.className = 'ui-slider-top';
    this.label = document.createElement('label');
    this.input = document.createElement('input');
    this.input.type = 'range';
    this.input.id = `wrs-slider-${++nextControlId}`;
    this.label.htmlFor = this.input.id;
    this.output = document.createElement('output');
    this.output.htmlFor = this.input.id;
    top.append(this.label, this.output);
    const bounds = document.createElement('div');
    bounds.className = 'ui-bounds';
    this.min = document.createElement('span');
    this.max = document.createElement('span');
    bounds.append(this.min, this.max);
    this.element.append(top, this.input, bounds);
    this._listen(this.input, 'input', () => {
      if (!this.control.enabled) return;
      this.editing = true;
      this._display(this.input.valueAsNumber);
      if (this.control.continuous) this._queueValue(false);
    });
    this._listen(this.input, 'change', () => {
      this.editing = false;
      if (!this.control.enabled || (this.pending && !this.control.continuous)) return;
      this._queueValue(true);
      this.update();
    });
    this._listen(this.input, 'blur', () => {
      this.editing = false;
      this.update();
    });
    this._bindBusyGuard(() => this.control.continuous);
    this.update();
  }

  update(props = {}) {
    const next = { ...this.control, ...props };
    if (typeof next.continuous !== 'boolean') throw new TypeError('continuous must be a boolean');
    if (!Number.isFinite(next.update_hz) || next.update_hz <= 0) {
      throw new RangeError('update_hz must be positive');
    }
    const control = Object.assign(this.control, next);
    if (!control.enabled) this.cancelPending();
    this._updateInput(control.continuous);
    this.label.textContent = control.label;
    this.input.min = control.min;
    this.input.max = control.max;
    this.input.step = control.step;
    this.min.textContent = this._format(control.min);
    this.max.textContent = this._format(control.max);
    this._flushQueued();
    if (!this.editing && !this.pending && this._queuedValue === undefined) {
      this.input.value = control.value;
      this._display(this.input.valueAsNumber);
    }
  }

  _queueValue(final) {
    // Keep only the latest position while throttled or waiting for Python.
    this._queuedValue = this.input.valueAsNumber;
    this._flushImmediately = final;
    this._flushQueued();
  }

  _flushQueued() {
    clearTimeout(this._timer);
    if (this._queuedValue === undefined || this.pending) return;
    if (this._queuedValue === this.control.value) {
      this._queuedValue = undefined;
      return;
    }
    const delay = 1000 / this.control.update_hz - (performance.now() - this._lastEmitAt);
    if (!this._flushImmediately && delay > 0) {
      this._timer = setTimeout(() => this._flushQueued(), Math.min(delay, 2147483647));
      return;
    }
    const value = this._queuedValue;
    this._queuedValue = undefined;
    this._flushImmediately = false;
    this._lastEmitAt = performance.now();
    this.control.value = value;
    this._onChange(value);
  }

  cancelPending() {
    clearTimeout(this._timer);
    this._queuedValue = undefined;
    this._flushImmediately = false;
    this.editing = false;
  }

  destroy() {
    this.cancelPending();
    super.destroy();
  }

  _format(value) {
    return `${Number(value.toPrecision(8))}${this.control.unit ? ` ${this.control.unit}` : ''}`;
  }

  _display(value) {
    this.output.textContent = this._format(value);
    const percent = 100 * (value - this.control.min) / (this.control.max - this.control.min);
    this.input.style.setProperty('--fill', `${percent}%`);
    this.input.setAttribute('aria-valuetext', this.output.textContent);
  }
}

export class Select extends Control {
  constructor({ onChange = () => {}, ...props } = {}) {
    super('select', { options: [], value: '', ...props });
    this.label = document.createElement('label');
    this.input = document.createElement('select');
    this.input.id = `wrs-select-${++nextControlId}`;
    this.label.htmlFor = this.input.id;
    this.element.append(this.label, this.input);
    this._listen(this.input, 'change', () => {
      if (!this.control.enabled || this.pending) return;
      this.control.value = this.input.value;
      onChange(this.control.value);
    });
    this._bindBusyGuard();
    this.update();
  }

  update(props = {}) {
    Object.assign(this.control, props);
    this._updateInput();
    this.label.textContent = this.control.label;
    const signature = JSON.stringify(this.control.options);
    if (signature !== this.signature) {
      this.signature = signature;
      this.input.replaceChildren(...this.control.options.map(value => {
        const option = document.createElement('option');
        option.value = option.textContent = value;
        return option;
      }));
    }
    if (!this.pending) this.input.value = this.control.value;
  }
}

export class Checkbox extends Control {
  constructor({ onChange = () => {}, ...props } = {}) {
    super('checkbox', { value: false, ...props });
    this.input = document.createElement('input');
    this.input.type = 'checkbox';
    this.input.id = `wrs-checkbox-${++nextControlId}`;
    this.label = document.createElement('label');
    this.label.htmlFor = this.input.id;
    this.element.append(this.input, this.label);
    this._listen(this.input, 'click', (event) => {
      // Label clicks also activate the input while an earlier change is pending.
      if (!this.control.enabled || this.pending) event.preventDefault();
    });
    this._listen(this.input, 'change', () => {
      if (!this.control.enabled || this.pending) return;
      this.control.value = this.input.checked;
      onChange(this.control.value);
    });
    this._bindBusyGuard();
    this.update();
  }

  update(props = {}) {
    Object.assign(this.control, props);
    this._updateInput();
    this.label.textContent = this.control.label;
    if (!this.pending) this.input.checked = this.control.value;
  }
}

export class Text extends Control {
  constructor(props = {}) {
    super('label', { value: '', ...props });
    this.label = document.createElement('span');
    this.label.className = 'ui-label-name';
    this.output = document.createElement('span');
    this.output.className = 'ui-label-value';
    this.element.append(this.label, this.output);
    this.update();
  }

  update(props = {}) {
    Object.assign(this.control, props);
    this.label.textContent = this.control.label;
    this.output.textContent = this.control.value;
  }
}

/** Read-only image. Decode before swapping, and always release owned blob URLs. */
export class ImageView extends Control {
  constructor(props = {}) {
    super('image', props);
    this.label = document.createElement('span');
    this.label.className = 'ui-label-name';
    this.viewport = document.createElement('div');
    this.viewport.className = 'ui-image-viewport';
    this.placeholder = document.createElement('span');
    this.placeholder.textContent = 'No image';
    this.viewport.appendChild(this.placeholder);
    this.element.append(this.label, this.viewport);
    this._sequence = -1;
    this._generation = 0;
    this._urls = new Set();
    this._disposed = false;
    this.update();
  }

  update(props = {}) {
    const stream = this.control.stream;
    Object.assign(this.control, props);
    if (stream !== this.control.stream) {
      this._clear();
      this._sequence = -1;
    }
    this.label.textContent = this.control.label;
    if (this.output) this.output.alt = this.control.label;
  }

  async setFrame(header, bytes) {
    if (this._disposed || header.stream !== this.control.stream || header.sequence <= this._sequence) return;
    this._sequence = header.sequence;
    if (!header.mime) {
      this._clear();
      return;
    }
    const generation = ++this._generation;
    let url;
    try {
      if (!['image/png', 'image/jpeg'].includes(header.mime)) throw new Error('Unsupported image');
      url = URL.createObjectURL(new Blob([bytes], { type: header.mime }));
      this._urls.add(url);
      const img = document.createElement('img');
      img.alt = this.control.label;
      img.draggable = false;
      img.src = url;
      await img.decode();
      if (this._disposed || generation !== this._generation) return;
      const previousURL = this._visibleURL;
      this.output = img;
      this._visibleURL = url;
      this.viewport.replaceChildren(img);
      this.element.removeAttribute('data-error');
      this._release(previousURL);
    } catch (error) {
      if (!this._disposed && generation === this._generation) {
        this.element.dataset.error = 'true';
        this.placeholder.textContent = 'Image could not be displayed';
        if (!this.output) this.viewport.replaceChildren(this.placeholder);
      }
    } finally {
      if (url !== this._visibleURL) this._release(url);
    }
  }

  _release(url) {
    if (url && this._urls.delete(url)) URL.revokeObjectURL(url);
  }

  _clear() {
    ++this._generation;
    for (const url of this._urls) this._release(url);
    this._visibleURL = null;
    this.output = null;
    this.placeholder.textContent = 'No image';
    this.viewport.replaceChildren(this.placeholder);
    this.element.removeAttribute('data-error');
  }

  destroy() {
    this._disposed = true;
    this._clear();
    super.destroy();
  }
}
