"""LanderEnv - the UI-independent simulation core.

Gym-style interface::

    env = LanderEnv(seed=0)
    obs = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Observation (8 floats, roughly in [-1, 1]):
    0  horizontal offset from the pad centre   (x - pad_x) / (W/2)
    1  altitude above the pad                  (y - pad_y - leg_height) / (H/2)
    2  horizontal velocity                     vx / velocity_scale
    3  vertical velocity                       vy / velocity_scale
    4  angle (radians, 0 = upright, +ccw)
    5  angular velocity (rad/s)
    6  left leg touching ground  (0/1)
    7  right leg touching ground (0/1)

Actions: 0 noop, 1 fire left engine, 2 fire main engine, 3 fire right engine.

Reward: potential-based shaping (closer / slower / more upright is better,
legs on the ground is better), minus a small fuel cost per engine burn, plus
a terminal bonus/penalty: crash -100, landed on pad +100, landed off pad +20.

The renderer never calls step(); it only reads `env.snapshot()`.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import LanderConfig
from .physics import (ACTION_NAMES, LEFT, MAIN, NOOP, RIGHT, LanderState,
                      StepResult, step_physics)
from .terrain import Terrain, generate_terrain

Observation = Tuple[float, ...]

STATUS_FLYING = "flying"
STATUS_LANDED_PAD = "landed_on_pad"
STATUS_LANDED_OFF = "landed_off_pad"
STATUS_CRASHED = "crashed"
STATUS_OUT_OF_BOUNDS = "out_of_bounds"
STATUS_TIMEOUT = "timeout"


@dataclass
class Snapshot:
    """Everything a renderer needs to draw one frame. Read-only by contract."""
    config: LanderConfig
    terrain: Terrain
    state: LanderState                 # a copy - mutate freely
    action: int                        # action applied in the last step
    step_index: int
    status: str
    crash_reason: str
    last_reward: float
    episode_reward: float
    settle_progress: float             # 0..1, how close to counting as landed
    leg_compression: Tuple[float, float]
    fuel_used: float
    observation: Observation
    trail: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.status != STATUS_FLYING


class LanderEnv:
    observation_size = 8
    action_size = 4
    action_names = ACTION_NAMES
    NOOP, LEFT, MAIN, RIGHT = NOOP, LEFT, MAIN, RIGHT

    def __init__(self, config: Optional[LanderConfig] = None,
                 seed: Optional[int] = None, trail_length: int = 400):
        self.cfg = config or LanderConfig()
        self.rng = random.Random(seed)
        self.trail_length = trail_length
        self.terrain: Terrain = generate_terrain(self.cfg, self.rng)
        self._state: Optional[LanderState] = None
        self._reset_bookkeeping()

    # ------------------------------------------------------------------
    def _reset_bookkeeping(self) -> None:
        self.step_index = 0
        self.status = STATUS_FLYING
        self.crash_reason = ""
        self.last_action = NOOP
        self.last_reward = 0.0
        self.episode_reward = 0.0
        self.fuel_used = 0.0
        self._settle_count = 0
        self._prev_shaping: Optional[float] = None
        self._last_result = StepResult()
        self._trail: List[Tuple[float, float]] = []

    def seed(self, seed: Optional[int]) -> None:
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self.seed(seed)
        cfg, rng = self.cfg, self.rng
        self.terrain = generate_terrain(cfg, rng)
        self._state = LanderState(
            x=cfg.pad_center_x + rng.uniform(-cfg.start_x_spread, cfg.start_x_spread),
            y=cfg.start_y,
            vx=rng.uniform(*cfg.start_vx),
            vy=rng.uniform(*cfg.start_vy),
            angle=rng.uniform(*cfg.start_angle),
            angular_velocity=rng.uniform(*cfg.start_angular_velocity),
        )
        self._reset_bookkeeping()
        self._prev_shaping = self._shaping(self._observe())
        self._trail.append((self._state.x, self._state.y))
        return self._observe()

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[Observation, float, bool, bool, Dict]:
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        if self.status != STATUS_FLYING:
            raise RuntimeError("episode is over; call reset()")
        action = int(action)
        if not 0 <= action < self.action_size:
            raise ValueError(f"invalid action {action}")

        cfg, st = self.cfg, self._state
        result = step_physics(st, action, self.terrain, cfg)
        self._last_result = result
        self.last_action = action
        self.step_index += 1
        self._trail.append((st.x, st.y))
        if len(self._trail) > self.trail_length:
            del self._trail[0]

        obs = self._observe()

        # ---- shaping reward ------------------------------------------
        shaping = self._shaping(obs)
        reward = shaping - (self._prev_shaping if self._prev_shaping is not None else shaping)
        self._prev_shaping = shaping
        if action == MAIN:
            reward -= cfg.fuel_cost_main
            self.fuel_used += cfg.fuel_cost_main
        elif action in (LEFT, RIGHT):
            reward -= cfg.fuel_cost_side
            self.fuel_used += cfg.fuel_cost_side

        # ---- termination ---------------------------------------------
        terminated = False
        truncated = False
        if result.crashed:
            self.status = STATUS_CRASHED
            self.crash_reason = result.crash_reason
            reward += cfg.reward_crash
            terminated = True
        elif (st.x < -cfg.out_of_bounds_margin
              or st.x > cfg.world_width + cfg.out_of_bounds_margin
              or st.y > cfg.world_height + cfg.out_of_bounds_margin):
            self.status = STATUS_OUT_OF_BOUNDS
            self.crash_reason = "left the play area"
            reward += cfg.reward_out_of_bounds
            terminated = True
        else:
            still = (st.left_contact and st.right_contact
                     and st.speed < cfg.settle_speed
                     and abs(st.angular_velocity) < cfg.settle_angular_speed)
            self._settle_count = self._settle_count + 1 if still else 0
            if self._settle_count >= cfg.settle_steps:
                terminated = True
                if self._legs_on_pad():
                    self.status = STATUS_LANDED_PAD
                    reward += cfg.reward_landed_on_pad
                else:
                    self.status = STATUS_LANDED_OFF
                    reward += cfg.reward_landed_off_pad
            elif self.step_index >= cfg.max_steps:
                self.status = STATUS_TIMEOUT
                truncated = True

        self.last_reward = reward
        self.episode_reward += reward
        info = {
            "status": self.status,
            "step": self.step_index,
            "fuel_used": self.fuel_used,
            "on_pad": self._legs_on_pad(),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _legs_on_pad(self) -> bool:
        from .physics import leg_tips_world
        return all(self.terrain.on_pad(px) for px, _ in leg_tips_world(self._state, self.cfg))

    def _observe(self) -> Observation:
        cfg, st, t = self.cfg, self._state, self.terrain
        leg_height = -cfg.leg_tips[0][1]
        return (
            (st.x - t.pad_center_x) / (cfg.world_width / 2.0),
            (st.y - t.pad_y - leg_height) / (cfg.world_height / 2.0),
            st.vx / cfg.obs_velocity_scale,
            st.vy / cfg.obs_velocity_scale,
            st.angle,
            st.angular_velocity,
            1.0 if st.left_contact else 0.0,
            1.0 if st.right_contact else 0.0,
        )

    def _shaping(self, obs: Observation) -> float:
        cfg = self.cfg
        return (-cfg.shaping_distance * math.hypot(obs[0], obs[1])
                - cfg.shaping_velocity * math.hypot(obs[2], obs[3])
                - cfg.shaping_angle * abs(obs[4])
                + cfg.shaping_leg_contact * obs[6]
                + cfg.shaping_leg_contact * obs[7])

    # ------------------------------------------------------------------
    @property
    def state(self) -> LanderState:
        """A copy of the current lander state."""
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state.copy()

    @property
    def done(self) -> bool:
        return self.status != STATUS_FLYING

    def snapshot(self) -> Snapshot:
        """Frame data for a renderer. Contains copies, safe to hold on to."""
        if self._state is None:
            raise RuntimeError("call reset() first")
        r = self._last_result
        return Snapshot(
            config=self.cfg,
            terrain=self.terrain,
            state=self._state.copy(),
            action=self.last_action,
            step_index=self.step_index,
            status=self.status,
            crash_reason=self.crash_reason,
            last_reward=self.last_reward,
            episode_reward=self.episode_reward,
            settle_progress=min(1.0, self._settle_count / self.cfg.settle_steps),
            leg_compression=(r.left_compression, r.right_compression),
            fuel_used=self.fuel_used,
            observation=self._observe(),
            trail=list(self._trail),
        )
