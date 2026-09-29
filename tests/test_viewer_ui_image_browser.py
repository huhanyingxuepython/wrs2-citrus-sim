"""Optional browser regression: python -m unittest discover -s tests -p '*image_browser.py'.

Requires Playwright and its Chromium, or installed Microsoft Edge on Windows.
Exercises the actual UI components without requiring WebGPU for the 3-D canvas.
"""
import asyncio
import json
import os
import unittest

import numpy as np
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

from wrs.viewer.server import Hub, _static_response
from wrs.viewer.world import World


HARNESS = b'''<!doctype html><link rel="stylesheet" href="/ui/styles.css">
<script type="module">
import { UIManager } from '/ui/index.js';
import { decode } from '/wire.js';
const ws = new WebSocket(`ws://${location.host}/view`);
ws.binaryType = 'arraybuffer';
window.ui = new UIManager(m => { ws.send(JSON.stringify(m)); return true; },
  { container: document.body });
ws.onmessage = async ({data}) => {
  if (typeof data === 'string') {
    const p = JSON.parse(data);
    if (p.type === 'ui_state') ui.apply(p);
    if (p.type === 'ui_result') ui.receiveResult(p);
    if (p.type === 'ui_reset') ui.reset();
    if (p.type === 'status') ui.setConnected(p.publisher);
  } else {
    const {header, view} = decode(data);
    if (header.type === 'ui_image') {
      await ui.receiveImage(header, view(header.data, Uint8Array));
      ws.send(JSON.stringify({type: 'ui_image_ack', stream: header.stream, sequence: header.sequence}));
    }
  }
};
window.pixel = () => {
  const image = document.querySelector('.ui-image img');
  if (!image) return null;
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0);
  return [...ctx.getImageData(0, 0, 1, 1).data];
};
</script>'''


@unittest.skipUnless(async_playwright, 'Playwright is optional')
class BrowserImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_pixels_reload_clear_replacement_and_blob_cleanup(self):
        hub = Hub(auto_open=False)

        async def route(ws):
            await (hub.on_publish(ws) if ws.request.path == '/publish' else hub.on_view(ws))

        def http(connection, request):
            if request.headers.get('Upgrade', '').lower() == 'websocket':
                return None
            if request.path == '/test':
                return Response(200, 'OK', Headers({'Content-Type': 'text/html',
                                                    'Content-Length': str(len(HARNESS))}), HARNESS)
            return _static_response(request.path)

        async with async_playwright() as playwright:
            options = {'headless': True}
            if not os.path.exists(playwright.chromium.executable_path) and os.name == 'nt':
                options['channel'] = 'msedge'
            browser = await playwright.chromium.launch(**options)
            try:
                async with websockets.serve(route, '127.0.0.1', 0, process_request=http) as server:
                    port = server.sockets[0].getsockname()[1]
                    world = World(auto_start_hub=False)
                    panel = world.ui.add_panel('images', title='Image regression', width=320)
                    image = panel.add_image('rgb', image=np.full((48, 64, 3), (230, 10, 20), dtype=np.uint8))
                    changes = []
                    panel.add_checkbox('check', label='Interactive', on_change=changes.append)
                    async with websockets.connect(f'ws://127.0.0.1:{port}/publish') as publisher:
                        async def dispatch():
                            while not world._closed:
                                world._drain_events()
                                await asyncio.sleep(.01)

                        tasks = [asyncio.create_task(world._send_ui(publisher)),
                                 asyncio.create_task(world._recv_events(publisher)),
                                 asyncio.create_task(dispatch())]
                        try:
                            page = await browser.new_page()
                            errors = []
                            page.on('pageerror', lambda error: errors.append(str(error)))
                            await page.add_init_script('''window.blobs = new Set();
                              const create = URL.createObjectURL.bind(URL), revoke = URL.revokeObjectURL.bind(URL);
                              URL.createObjectURL = b => { const u=create(b); blobs.add(u); return u; };
                              URL.revokeObjectURL = u => { blobs.delete(u); revoke(u); };''')
                            await page.goto(f'http://127.0.0.1:{port}/test')
                            await page.wait_for_function('window.pixel?.()?.[0] === 230')
                            bounds = await page.locator('.ui-image img').bounding_box()
                            self.assertAlmostEqual(bounds['width'] / bounds['height'], 64 / 48, places=2)
                            await page.get_by_label('Interactive', exact=True).check()
                            await page.wait_for_function("document.querySelector('[data-panel-id=images] .ui-feedback').textContent === 'Applied'")
                            self.assertEqual(changes, [True])
                            for value in range(50):
                                image.update(np.full((48, 64, 3), value, dtype=np.uint8))
                                await asyncio.sleep(.003)
                            await page.wait_for_function('window.pixel?.()?.[0] === 49')
                            self.assertEqual(await page.evaluate('blobs.size'), 1)
                            await page.reload()
                            await page.wait_for_function('window.pixel?.()?.[0] === 49')
                            image.clear()
                            await page.wait_for_function("!document.querySelector('.ui-image img')")
                            self.assertEqual(await page.evaluate('blobs.size'), 0)
                            # Recreate the same ID between state snapshots: stream identity must reset it.
                            panel.remove('rgb')
                            image = panel.add_image('rgb', image=np.full((32, 32, 4), (10, 220, 30, 128), dtype=np.uint8))
                            await page.wait_for_function('window.pixel?.()?.[1] === 219 || window.pixel?.()?.[1] === 220')
                            self.assertEqual((await page.evaluate('pixel()'))[3], 128)
                            world.ui.remove_panel('images')
                            await page.wait_for_function("!document.querySelector('.ui-image')")
                            self.assertEqual(await page.evaluate('blobs.size'), 0)
                            self.assertEqual(errors, [])
                        finally:
                            world.close()
                            for task in tasks:
                                task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await browser.close()
