# This file is absed on https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/constants.py

from __future__ import annotations

import numpy as np

TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]), 
    'orange': np.array([255, 165, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]), 
    # Appended color
    'black': np.array([0, 0, 0]),
    'white': np.array([255, 255, 255])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {}
for key, value in COLORS.items():
    COLOR_TO_IDX[key] = len(COLOR_TO_IDX)

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    # can overlap objects
    "unseen": 0,
    "empty": 1, 
    'goal': 2,

    "wall": 3, 

    'fire': 4,
    'fireextinguisher': 5,
    'debris': 6,
    
    'survivor': 7,

    'door': 8,
    'key': 9, 
    'start': 10, 
    'lava': 11,
    'agent': 12
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2, 

    'unknown': 3,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]
