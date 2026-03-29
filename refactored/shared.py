"""Shared utility functions for the DreamerV3 Snake project.

Centralizes duplicated code (upscale, config loading, JSON encoding)
to ensure consistency across scripts and analysis modules.
"""

import argparse
import json
import pathlib

import cv2
import numpy as np

from constants import DISPLAY_SIZE


DREAMER_DIR = pathlib.Path(__file__).parent / "dreamerv3-torch"


def upscale(frame: np.ndarray) -> np.ndarray:
    """Upscale a 64x64 frame to display size using nearest-neighbor.

    Args:
        frame: (64, 64, 3) uint8 RGB image.

    Returns:
        (DISPLAY_SIZE, DISPLAY_SIZE, 3) uint8 RGB image.
    """
    size = (DISPLAY_SIZE, DISPLAY_SIZE)
    return cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)


def load_config(config_name="snake", dreamer_dir=None):
    """Load and merge default + named config from configs.yaml.

    Args:
        config_name: Config section name (e.g. 'snake', 'snake_32x32').
        dreamer_dir: Path to dreamerv3-torch directory. Defaults to
            the sibling 'dreamerv3-torch/' directory.

    Returns:
        argparse.Namespace with merged configuration values.
    """
    import sys  # pylint: disable=import-outside-toplevel
    if dreamer_dir is None:
        dreamer_dir = DREAMER_DIR
    dreamer_dir = pathlib.Path(dreamer_dir)

    # Ensure dreamerv3-torch is on path for tools import
    dreamer_str = str(dreamer_dir)
    if dreamer_str not in sys.path:
        sys.path.insert(0, dreamer_str)

    from ruamel.yaml import YAML as _YAML  # pylint: disable=import-outside-toplevel
    import tools  # pylint: disable=import-outside-toplevel,import-error

    yaml = _YAML(typ="safe", pure=True)
    config_path = dreamer_dir / "configs.yaml"
    configs = yaml.load(config_path.read_text())

    def recursive_update(base, update):
        """Merge update dict into base dict recursively."""
        for key, value in update.items():
            if (isinstance(value, dict) and key in base
                    and isinstance(base[key], dict)):
                recursive_update(base[key], value)
            else:
                base[key] = value

    merged = dict(configs["defaults"])
    recursive_update(merged, configs[config_name])

    parsed = {}
    for key, value in merged.items():
        if isinstance(value, str):
            parsed[key] = tools.args_type(value)(value)
        else:
            parsed[key] = value
    config = argparse.Namespace(**parsed)

    from constants import NUM_ACTIONS  # pylint: disable=import-outside-toplevel
    config.num_actions = NUM_ACTIONS
    return config


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types.

    Converts numpy integers, floats, bools, and arrays to their
    native Python equivalents for JSON serialization.
    """

    def default(self, o):
        """Encode numpy types as native Python types."""
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
