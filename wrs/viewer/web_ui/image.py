"""Latest-frame image storage, separate from JSON control snapshots."""
from io import BytesIO
from os import PathLike
import threading
import uuid

import numpy as np
from PIL import Image

from . import protocol


class UIImage:
    """A read-only image handle returned by UIPanel.add_image().

    update() copies uint8 gray/RGB/RGBA arrays (or reads a PNG/JPEG path).
    Encoding happens on the publisher worker, only for the latest queued frame.
    Removed handles reject updates rather than addressing a replacement control.
    """

    def __init__(self, panel_id, session, control_id, *, format='png',
                 quality=85, max_fps=20):
        if format not in ('png', 'jpeg'):
            raise ValueError("format must be 'png' or 'jpeg'")
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 95:
            raise ValueError('quality must be an integer from 1 to 95')
        max_fps = protocol.finite_number(max_fps)
        if max_fps <= 0:
            raise ValueError('max_fps must be positive')
        self._identity = dict(panel_id=panel_id, session=session, id=control_id,
                              stream=uuid.uuid4().hex)
        self._format, self._quality, self._max_fps = format, quality, max_fps
        self._lock = threading.RLock()
        self._sequence = 0
        self._pixels = None
        self._encoded = None
        self._closed = False

    def update(self, image, *, color_order='rgb'):
        """Replace the pending image; BGR/BGRA input requires color_order='bgr'."""
        if color_order not in ('rgb', 'bgr'):
            raise ValueError("color_order must be 'rgb' or 'bgr'")
        if isinstance(image, (str, PathLike)):
            with Image.open(image, formats=('PNG', 'JPEG')) as source:
                image = np.asarray(source.convert('RGBA' if 'A' in source.getbands()
                                                 or 'transparency' in source.info else 'RGB'))
            if color_order != 'rgb':
                raise ValueError('file images already use RGB color order')
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError('image must be a uint8 numpy array or a PNG/JPEG path')
        if (image.ndim not in (2, 3) or min(image.shape[:2]) == 0
                or (image.ndim == 3 and image.shape[2] not in (3, 4))):
            raise ValueError('image shape must be (H, W), (H, W, 3) or (H, W, 4)')
        if self._format == 'jpeg' and image.ndim == 3 and image.shape[2] == 4:
            raise ValueError('RGBA images require format="png"')
        pixels = image.copy(order='C')
        if color_order == 'bgr' and pixels.ndim == 3:
            pixels[..., :3] = pixels[..., 2::-1].copy()
        with self._lock:
            self._check_open()
            self._pixels = pixels
            self._sequence += 1
            self._encoded = None

    def clear(self):
        """Display the empty placeholder, including after browser reloads."""
        with self._lock:
            self._check_open()
            self._pixels = None
            self._sequence += 1
            self._encoded = None

    def _check_open(self):
        if self._closed:
            raise RuntimeError('image control has been removed')

    def _close(self):
        with self._lock:
            self._closed = True
            self._pixels = self._encoded = None

    def _frame(self, previous, now):
        """Encode outside locks; previous is (last sent sequence, monotonic time)."""
        with self._lock:
            if self._closed or (previous is not None and (
                    previous[0] == self._sequence or now - previous[1] < 1 / self._max_fps)):
                return None
            sequence, pixels, encoded = self._sequence, self._pixels, self._encoded
        if encoded is None:
            data = b''
            if pixels is not None:
                output = BytesIO()
                Image.fromarray(pixels).save(output, format=self._format.upper(),
                                             **({'quality': self._quality} if self._format == 'jpeg' else {}))
                data = output.getvalue()
            encoded = protocol.image_message(
                **self._identity, sequence=sequence,
                mime=f'image/{self._format}' if pixels is not None else '', data=data)
            with self._lock:
                if self._closed:
                    return None
                if sequence == self._sequence:
                    self._encoded = encoded
        return sequence, encoded


def colorize_depth(depth, *, value_range, valid_mask=None):
    """Map a 2-D depth array to uint8 RGB using Jet; invalid pixels are black.

    value_range uses the input's units: meters for DepthFrame.depth_m. Fixed
    bounds make colors comparable across frames. Does not modify source data.
    Near to far: dark blue, blue, cyan, green, yellow, red, dark red.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.dtype.kind not in 'uif':
        raise ValueError('depth must be a 2-D numeric array')
    low, high = map(protocol.finite_number, value_range)
    if low >= high:
        raise ValueError('value_range requires min < max')
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != depth.shape or mask.dtype != np.bool_:
            raise ValueError('valid_mask must be a boolean array matching depth')
        valid &= mask
    t = np.clip((np.where(valid, depth, low).astype(np.float64) - low) / (high - low), 0, 1)
    # Piecewise-linear Jet, evaluated directly without a plotting dependency.
    rgb = np.clip(1.5 - np.abs(4 * t[..., None] - np.array([3., 2., 1.])), 0, 1)
    rgb[~valid] = 0
    return np.rint(rgb * 255).astype(np.uint8)
