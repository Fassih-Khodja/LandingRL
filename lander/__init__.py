"""Lander simulation core. No UI dependencies - safe to import in a trainer."""
from .config import LanderConfig
from .env import (STATUS_CRASHED, STATUS_FLYING, STATUS_LANDED_OFF,
                  STATUS_LANDED_PAD, STATUS_OUT_OF_BOUNDS, STATUS_TIMEOUT,
                  LanderEnv, Snapshot)
from .physics import ACTION_NAMES, LEFT, MAIN, NOOP, RIGHT, LanderState
from .terrain import Terrain

__all__ = [
    "LanderConfig", "LanderEnv", "LanderState", "Snapshot", "Terrain",
    "NOOP", "LEFT", "MAIN", "RIGHT", "ACTION_NAMES",
    "STATUS_FLYING", "STATUS_LANDED_PAD", "STATUS_LANDED_OFF",
    "STATUS_CRASHED", "STATUS_OUT_OF_BOUNDS", "STATUS_TIMEOUT",
]
