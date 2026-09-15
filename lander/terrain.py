"""Procedural ground with a fixed, flat landing pad at the bottom centre."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from .config import LanderConfig


@dataclass
class Terrain:
    xs: List[float]           # sample x positions, ascending, 0 .. world_width
    ys: List[float]           # ground height at each sample
    pad_x0: float             # pad left edge
    pad_x1: float             # pad right edge
    pad_y: float              # pad altitude

    @property
    def pad_center_x(self) -> float:
        return 0.5 * (self.pad_x0 + self.pad_x1)

    def height_at(self, x: float) -> float:
        """Piecewise-linear ground height at world x (clamped at the edges)."""
        xs, ys = self.xs, self.ys
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        # xs are evenly spaced, so the segment index is direct
        seg = (x - xs[0]) / (xs[1] - xs[0])
        i = int(seg)
        if i >= len(xs) - 1:
            i = len(xs) - 2
        t = seg - i
        return ys[i] + (ys[i + 1] - ys[i]) * t

    def on_pad(self, x: float) -> bool:
        return self.pad_x0 <= x <= self.pad_x1


def generate_terrain(cfg: LanderConfig, rng: random.Random) -> Terrain:
    """Random rolling hills, smoothed, with the centre flattened into a pad.

    The pad itself is exactly one flat stretch; the neighbouring samples are
    also set to pad height so the ground rises gently away from it instead of
    forming a cliff right at the pad edge.
    """
    n = cfg.terrain_points
    step = cfg.world_width / (n - 1)
    xs = [i * step for i in range(n)]
    raw = [rng.uniform(cfg.terrain_min_height, cfg.terrain_max_height) for _ in range(n)]

    # 3-tap smoothing so the hills roll instead of spike
    ys = []
    for i in range(n):
        a = raw[max(i - 1, 0)]
        b = raw[i]
        c = raw[min(i + 1, n - 1)]
        ys.append((a + b + c) / 3.0)

    # flatten the middle. The pad edges are snapped onto terrain samples so
    # height_at() is exactly flat across the pad.
    center = cfg.pad_center_x
    pad_x0 = center - cfg.pad_half_width
    pad_x1 = center + cfg.pad_half_width
    for i, x in enumerate(xs):
        if pad_x0 - step <= x <= pad_x1 + step:
            ys[i] = cfg.pad_y

    return Terrain(xs=xs, ys=ys, pad_x0=pad_x0, pad_x1=pad_x1, pad_y=cfg.pad_y)
