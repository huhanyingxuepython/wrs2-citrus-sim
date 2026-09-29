"""Species defaults and scene presets compose into plain JSON recipes."""
from copy import deepcopy
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).with_name('configs')
LAB_CONFIG = CONFIG_DIR / 'presets/lab_citrus_v2.json'
GENERIC_CONFIG = CONFIG_DIR / 'presets/generic_plant.json'


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path=LAB_CONFIG, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError('cycle in config extends')
    seen.add(path)
    config = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('config must be a JSON object')
    parent = config.pop('extends', None)
    if parent:
        parent_path = CONFIG_DIR / parent[8:] if parent.startswith('package:') else path.parent / parent
        config = merge(load_config(parent_path, seen), config)
    if config.get('schema_version') != 1 or config.get('units') != 'metre':
        raise ValueError("expected config schema_version=1 and units='metre'")
    return config
