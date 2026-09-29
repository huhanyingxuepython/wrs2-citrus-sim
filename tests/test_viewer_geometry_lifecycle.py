"""Repeated point-cloud replacement must release geometry without breaking sharing."""
import asyncio
import unittest

import numpy as np

from wrs import wssop, wvw
from wrs.viewer import protocol
from wrs.viewer.server import Hub


class GeometryLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_geometry_survives_until_last_model_is_removed(self):
        first = wssop.point_cloud(np.ones((3, 3)), np.ones((3, 3)))
        second = first.clone()
        models, geometry = protocol.describe([('first', first.visuals[0]),
                                              ('second', second.visuals[0])], set())
        self.assertEqual(len(geometry), 1)
        hub = Hub(auto_open=False)
        await hub._relay(protocol.scene_message('scene_init', models, geometry))
        await hub._relay(protocol.transform_message(['first', 'second'], np.tile(np.eye(4).ravel(), (2, 1))))
        await hub._relay(protocol.scene_message('scene_delta', [], [], remove=['first']))
        self.assertEqual(len(hub.geoms), 1)
        self.assertEqual(set(hub.transforms), {'second'})
        await hub._relay(protocol.scene_message('scene_delta', [], [], remove=['second']))
        self.assertFalse(hub.geoms)
        self.assertFalse(hub.transforms)

    async def test_publisher_resends_evicted_geometry_when_object_returns(self):
        world = wvw.World(auto_start_hub=False, hz=100)
        cloud = wssop.point_cloud(np.ones((3, 3)), np.ones((3, 3)))
        cloud.add_to_scene(world.scene)
        known = set()
        entries = [(mid, model) for mid, model, _ in protocol.iter_scene_models(world.scene)]
        models, geometries = protocol.describe(entries, known)
        hub = Hub(auto_open=False)
        await hub._relay(protocol.scene_message('scene_init', models, geometries))
        cloud.remove_from_scene(world.scene)
        deltas = []

        class Connection:
            async def send(self, message):
                await hub._relay(message)
                if isinstance(message, bytes):
                    header, _ = protocol.unpack(message)
                    if header['type'] == 'scene_delta':
                        deltas.append(header)
                        if len(deltas) == 1:
                            cloud.add_to_scene(world.scene)
                        else:
                            world.close()

        await asyncio.wait_for(world._send_scene(Connection(), {entry['id'] for entry in models}, known), 3)
        self.assertEqual(len(deltas), 2)
        self.assertEqual(len(deltas[1]['geometries']), 1)
        self.assertEqual(len(hub.geoms), 1)
        self.assertEqual(known, set(hub.geoms))


if __name__ == '__main__':
    unittest.main()
