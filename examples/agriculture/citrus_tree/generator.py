"""V1 compatibility entry; generation is now engine-independent."""
from pathlib import Path
from agriculture.config import load_config as _load_config
from agriculture.reference import generate_tree, _intersects_foliage_gap
DEFAULT_CONFIG = Path(__file__).with_name('configs') / 'lab_tree_v1.json'
GENERIC_CONFIG = DEFAULT_CONFIG.with_name('generic_citrus_v1.json')
def load_config(path=DEFAULT_CONFIG):
    return _load_config(path)
