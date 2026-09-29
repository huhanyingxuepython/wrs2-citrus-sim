"""Optional pure-geometry transformations; never edit physics/runtime state."""
from copy import deepcopy
import numpy as np


def apply_pipe_model(skeleton, terminal_radius, beta=2.0, tip_taper=.7):
    """Return a copy with radii inferred bottom-up from terminal radius.

    r_start**beta = sum(child.r_start**beta). r_end is the largest child
    radius (or a tapered terminal tip). Explicit reference radii need no call.
    For interior attachments this is a qualitative sizing rule, not mechanics.
    """
    skeleton.validate()
    if not np.isfinite([terminal_radius, beta, tip_taper]).all() or terminal_radius <= 0 or beta <= 0 or not 0 < tip_taper <= 1:
        raise ValueError('invalid pipe model parameters')
    result = deepcopy(skeleton)
    children, by_id = result.children, result.by_id
    pending = [(key, False) for key in result.roots]
    while pending:
        key, ready = pending.pop()
        if not ready:
            pending.append((key, True))
            pending.extend((child, False) for child in children[key])
            continue
        stem = by_id[key]
        if children[key]:
            radii = [by_id[child].radius_start for child in children[key]]
            stem.radius_start = float(sum(r ** beta for r in radii) ** (1 / beta))
            stem.radius_end = max(radii)
        else:
            stem.radius_start, stem.radius_end = terminal_radius, terminal_radius * tip_taper
    return result.validate()
