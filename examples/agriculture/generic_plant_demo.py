"""Different seeds, same PlantSpec and WRS builder; no lab-specific IDs."""
from pathlib import Path
import argparse
import json
import sys
import math
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agriculture.config import GENERIC_CONFIG, load_config
from agriculture.generator import generate
from agriculture.static import StaticPlantBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be finite and positive')
    config = load_config(GENERIC_CONFIG)
    plant = StaticPlantBuilder(config).build(generate(config, args.seed))
    print(json.dumps(plant.summary(), indent=2), flush=True)
    if args.headless:
        return
    from wrs import wvw, wssop
    base = wvw.World(cam_pos=config['demo']['camera_pos'], cam_lookat_pos=config['demo']['camera_lookat'], port=args.port)
    plant.add_to_scene(base.scene)
    wssop.plane(size=(2, 2)).add_to_scene(base.scene)
    if args.duration is not None:
        base.schedule_once(lambda dt: base.close(), args.duration)
    base.run()


if __name__ == '__main__':
    main()
