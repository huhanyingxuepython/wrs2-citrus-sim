"""Image contracts, independent UI state, coalescing and hub lifecycle."""
import asyncio
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import websockets

from wrs.viewer import protocol as wire
from wrs.viewer.server import Hub
from wrs.viewer.web_ui import UIManager, colorize_depth
from wrs.viewer.world import World


def pixels(message):
    header, blob = wire.unpack(message)
    return header, np.array(Image.open(BytesIO(blob))) if blob else None


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.ui = UIManager()
        self.panel = self.ui.add_panel('camera')
        self.image = self.panel.add_image('rgb')

    def test_array_is_copied_and_frames_do_not_change_json_state(self):
        state = self.ui._snapshot_all()
        source = np.full((3, 4, 3), (230, 10, 20), dtype=np.uint8)
        self.image.update(source)
        source[:] = 0
        header, result = pixels(self.image._frame(None, 0)[1])
        np.testing.assert_array_equal(result[0, 0], [230, 10, 20])
        self.assertEqual(header['panel_id'], 'camera')
        self.assertIsNone(self.ui._snapshot_all(state['revision']))
        self.assertNotIn('value', state['panels'][1]['controls'][0])
        json.dumps(state, allow_nan=False)

    def test_gray_rgba_noncontiguous_and_bgr(self):
        for source in (np.arange(12, dtype=np.uint8).reshape(3, 4),
                       np.full((3, 4, 4), (10, 20, 30, 40), dtype=np.uint8)):
            source = source[:, ::-1]
            self.image.update(source)
            np.testing.assert_array_equal(pixels(self.image._frame(None, 0)[1])[1], source)
        source = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        self.panel.set_image('rgb', source, color_order='bgr')
        np.testing.assert_array_equal(pixels(self.image._frame(None, 0)[1])[1][0, 0], [30, 20, 10])

    def test_files_and_jpeg(self):
        source = np.full((16, 16, 3), (210, 40, 20), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.png'
            Image.fromarray(source).save(path)
            self.image.update(path)
            np.testing.assert_array_equal(pixels(self.image._frame(None, 0)[1])[1], source)
        image = self.ui.add_image('jpeg', format='jpeg', image=source)
        header, result = pixels(image._frame(None, 0)[1])
        self.assertEqual(header['mime'], 'image/jpeg')
        self.assertLess(np.abs(result.astype(int) - source).max(), 5)

    def test_latest_frame_rate_limit_and_final_update(self):
        source = np.zeros((2, 2), dtype=np.uint8)
        self.image.update(source)
        sequence, _ = self.image._frame(None, 10)
        for value in (10, 20, 30):
            self.image.update(source + value)
        self.assertIsNone(self.image._frame((sequence, 10), 10.01))
        last_sequence, message = self.image._frame((sequence, 10), 11)
        np.testing.assert_array_equal(pixels(message)[1], source + 30)
        self.assertIsNone(self.image._frame((last_sequence, 11), 12))
        # A reconnect can replay the same last frame without a new update().
        self.assertEqual(self.image._frame(None, 12)[1], message)
        self.panel.set_image('rgb', None)
        header, result = pixels(self.image._frame((last_sequence, 11), 12)[1])
        self.assertEqual(header['mime'], '')
        self.assertIsNone(result)

    def test_invalid_input_does_not_replace_image(self):
        self.image.update(np.zeros((2, 2, 3), dtype=np.uint8))
        previous = self.image._frame(None, 0)
        for invalid in (np.zeros((2, 2)), np.zeros((0, 2), dtype=np.uint8),
                        np.zeros((2, 2, 2), dtype=np.uint8), [], None):
            with self.subTest(shape=getattr(invalid, 'shape', None)), self.assertRaises(ValueError):
                self.image.update(invalid)
        self.assertEqual(self.image._frame(None, 0), previous)
        with self.assertRaises(ValueError):
            self.panel.set_value('rgb', 'invalid')
        with self.assertRaises(ValueError):
            self.ui.add_image('alpha', format='jpeg', image=np.zeros((2, 2, 4), dtype=np.uint8))
        for options in ({'max_fps': 0}, {'max_fps': True}, {'max_fps': float('nan')},
                        {'format': 'gif'}, {'quality': 0}, {'quality': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.ui.add_image('invalid', **options)

    def test_removed_handles_and_browser_events(self):
        event = dict(type='ui_event', panel_id='camera', session=self.panel._session,
                     id='rgb', event_id='click', value=None)
        with self.assertLogs('wrs.viewer.web_ui.panel', level='WARNING'):
            self.assertFalse(self.ui._handle_event(event)['ok'])
        old_stream = self.image._identity['stream']
        self.panel.remove('rgb')
        with self.assertRaises(RuntimeError):
            self.image.update(np.zeros((2, 2), dtype=np.uint8))
        replacement = self.panel.add_image('rgb')
        self.assertNotEqual(replacement._identity['stream'], old_stream)
        self.ui.remove_panel('camera')
        with self.assertRaises(RuntimeError):
            replacement.clear()
        self.assertEqual(self.ui._image_frames({}), [])

    def test_depth_range_invalid_values_and_mask(self):
        depth = np.array([[.1, .3, .5], [0, np.nan, np.inf]])
        before = depth.copy()
        rgb = colorize_depth(depth, value_range=(.1, .5))
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 128])
        np.testing.assert_array_equal(rgb[0, 2], [128, 0, 0])
        np.testing.assert_array_equal(rgb[1], 0)
        np.testing.assert_array_equal(depth, before)
        np.testing.assert_array_equal(colorize_depth(depth, value_range=(.1, .5),
                                                     valid_mask=np.zeros(depth.shape, dtype=bool)), 0)
        with self.assertRaises(ValueError):
            colorize_depth(depth, value_range=(1, 0))
        # Recognizable Jet landmarks, also checking clipping beyond the range.
        ramp = np.array([[.5, 1, 1.375, 1.625, 2, 3]])
        np.testing.assert_array_equal(colorize_depth(ramp, value_range=(1, 2)),
                                      [[[0, 0, 128], [0, 0, 128], [0, 255, 255],
                                        [255, 255, 0], [128, 0, 0], [128, 0, 0]]])


class FakeViewer:
    def __init__(self):
        self.messages = []

    async def send(self, message):
        self.messages.append(message)


class HubImageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.hub = Hub(auto_open=False)
        self.ui = UIManager()
        self.image = self.ui.add_image('rgb')
        await self.hub._relay(json.dumps(self.ui._snapshot_all()))

    def viewer(self):
        viewer = FakeViewer()
        self.hub.viewers.add(viewer)
        self.hub._image_pending[viewer] = {}
        self.hub._image_sent[viewer] = {}
        self.hub._image_tasks[viewer] = set()
        return viewer

    async def publish(self, value):
        self.image.update(np.full((2, 2, 3), value, dtype=np.uint8))
        sequence, message = self.image._frame(None, 0)
        await self.hub._relay(message)
        await asyncio.sleep(0)
        return sequence, message

    async def test_slow_viewer_keeps_one_in_flight_and_latest_next(self):
        slow, fast = self.viewer(), self.viewer()
        sequence, _ = await self.publish(10)
        stream = self.image._identity['stream']
        self.hub._ack_image(fast, dict(stream=stream, sequence=sequence))
        sequence, _ = await self.publish(20)
        self.hub._ack_image(fast, dict(stream=stream, sequence=sequence))
        sequence, message = await self.publish(30)
        self.assertEqual(len(slow.messages), 1)
        self.assertEqual(len(fast.messages), 3)
        self.assertEqual(self.hub.images[stream], (sequence, message))
        self.hub._ack_image(slow, dict(stream=stream, sequence=999))
        await asyncio.sleep(0)
        self.assertEqual(len(slow.messages), 1)
        first, _ = wire.unpack(slow.messages[0])
        self.hub._ack_image(slow, first)
        await asyncio.sleep(0)
        self.assertEqual(slow.messages[-1], message)
        self.assertEqual(len(slow.messages), 2)

    async def test_removal_and_recreation_reject_old_frames(self):
        _, old_message = await self.publish(20)
        self.ui.remove('rgb')
        self.ui.add_image('rgb')
        await self.hub._relay(json.dumps(self.ui._snapshot_all()))
        self.assertFalse(self.hub.images)
        await self.hub._relay(old_message)
        self.assertFalse(self.hub.images)
        self.hub._sync_images({})
        self.assertFalse(self.hub.image_streams)

    async def test_real_websocket_publisher_reload_and_clear(self):
        hub = Hub(auto_open=False)

        async def route(ws):
            if ws.request.path == '/publish':
                await hub.on_publish(ws)
            else:
                await hub.on_view(ws)

        async def receive_image(ws):
            async with asyncio.timeout(5):
                while True:
                    message = await ws.recv()
                    if isinstance(message, bytes) and wire.unpack(message)[0]['type'] == 'ui_image':
                        header, _ = wire.unpack(message)
                        await ws.send(json.dumps(dict(type='ui_image_ack', stream=header['stream'],
                                                      sequence=header['sequence'])))
                        return message

        async with websockets.serve(route, '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            url = f'ws://127.0.0.1:{port}'
            world = World(auto_start_hub=False)
            image = world.ui.add_image('rgb', image=np.full((3, 4, 3), 40, dtype=np.uint8))
            async with websockets.connect(url + '/publish') as publisher:
                task = asyncio.create_task(world._send_ui(publisher))
                try:
                    async with websockets.connect(url + '/view') as viewer:
                        first = await receive_image(viewer)
                        self.assertEqual(pixels(first)[1][0, 0, 0], 40)
                    async with websockets.connect(url + '/view') as viewer:
                        self.assertEqual(await receive_image(viewer), first)
                        image.clear()
                        self.assertIsNone(pixels(await receive_image(viewer))[1])
                    async with websockets.connect(url + '/view') as viewer:
                        self.assertIsNone(pixels(await receive_image(viewer))[1])
                finally:
                    world.close()
                    await task


if __name__ == '__main__':
    unittest.main()
