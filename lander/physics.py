"""Rigid-body physics for the lander. Pure functions over LanderState.

Model
-----
* The lander is a single rigid body: position, velocity, angle, angular
  velocity.  Angle 0 is upright; positive angles are counter-clockwise.
* The main engine pushes along the body's +y axis through the centre of
  mass (no torque).
* Side engines are mounted above the centre of mass, one on each flank, and
  push sideways. Because they sit above the CoM they also spin the body:
      LEFT  engine: thrust toward body -x  => lander drifts left,  turns CCW
      RIGHT engine: thrust toward body +x  => lander drifts right, turns CW
  (This mirrors the classic Gym LunarLander action semantics.)
* Each leg is a point contact with a spring-damper against the ground, plus
  Coulomb friction. Legs report contact independently. A leg touching down
  faster than `max_landing_speed`, or compressed past `leg_max_compression`,
  snaps -> crash. Any hull vertex touching the ground -> crash.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Tuple

from .config import LanderConfig
from .terrain import Terrain

# discrete action ids
NOOP, LEFT, MAIN, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ("noop", "left", "main", "right")

Vec = Tuple[float, float]


@dataclass
class LanderState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    left_contact: bool = False
    right_contact: bool = False

    def copy(self) -> "LanderState":
        return replace(self)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass
class StepResult:
    """What happened inside one physics step, for the env and the renderer."""
    crashed: bool = False
    crash_reason: str = ""
    left_contact: bool = False
    right_contact: bool = False
    left_compression: float = 0.0
    right_compression: float = 0.0


# --------------------------------------------------------------------------
# geometry helpers (also used by the renderer)
# --------------------------------------------------------------------------
def rotate(p: Vec, angle: float) -> Vec:
    c, s = math.cos(angle), math.sin(angle)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def body_to_world(state: LanderState, p: Vec) -> Vec:
    rx, ry = rotate(p, state.angle)
    return (state.x + rx, state.y + ry)


def hull_world(state: LanderState, cfg: LanderConfig) -> List[Vec]:
    return [body_to_world(state, p) for p in cfg.hull]


def leg_tips_world(state: LanderState, cfg: LanderConfig) -> List[Vec]:
    return [body_to_world(state, p) for p in cfg.leg_tips]


def leg_roots_world(state: LanderState, cfg: LanderConfig) -> List[Vec]:
    return [body_to_world(state, p) for p in cfg.leg_roots]


def up_vector(state: LanderState) -> Vec:
    return (-math.sin(state.angle), math.cos(state.angle))


def right_vector(state: LanderState) -> Vec:
    return (math.cos(state.angle), math.sin(state.angle))


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------
def step_physics(state: LanderState, action: int, terrain: Terrain,
                 cfg: LanderConfig) -> StepResult:
    """Advance `state` in place by cfg.dt using `cfg.substeps` sub-iterations.

    Returns a StepResult describing contacts / crashes. The state's contact
    flags are updated as well.
    """
    dt = cfg.dt / cfg.substeps
    result = StepResult()

    for _ in range(cfg.substeps):
        fx, fy, torque = 0.0, -cfg.gravity * cfg.mass, 0.0

        # ---- engines --------------------------------------------------
        if action == MAIN:
            ux, uy = up_vector(state)
            fx += ux * cfg.main_thrust
            fy += uy * cfg.main_thrust
        elif action == LEFT or action == RIGHT:
            # body-frame force and application point
            direction = -1.0 if action == LEFT else 1.0
            f_body = (direction * cfg.side_thrust, 0.0)
            r_body = (-direction * cfg.side_engine_x, cfg.side_engine_y)
            fwx, fwy = rotate(f_body, state.angle)
            fx += fwx
            fy += fwy
            # cross product is rotation invariant -> compute in body frame
            torque += r_body[0] * f_body[1] - r_body[1] * f_body[0]

        # ---- leg contacts ---------------------------------------------
        was_touching = (state.left_contact, state.right_contact)
        contacts = []
        compressions = []
        for i, tip in enumerate(cfg.leg_tips):
            rx, ry = rotate(tip, state.angle)          # lever arm (world)
            px, py = state.x + rx, state.y + ry         # tip position
            ground = terrain.height_at(px)
            pen = ground - py
            if pen <= 0.0:
                contacts.append(False)
                compressions.append(0.0)
                continue
            contacts.append(True)
            compressions.append(pen)
            # velocity of the tip = v + omega x r
            tvx = state.vx - state.angular_velocity * ry
            tvy = state.vy + state.angular_velocity * rx
            if not was_touching[i] and -tvy > cfg.max_landing_speed:
                result.crashed = True
                result.crash_reason = "hard landing"
            fn = cfg.leg_spring * pen - cfg.leg_damping * tvy
            if fn < 0.0:
                fn = 0.0
            ft = -cfg.ground_tangent_damping * tvx
            limit = cfg.ground_friction * fn
            if ft > limit:
                ft = limit
            elif ft < -limit:
                ft = -limit
            fx += ft
            fy += fn
            torque += rx * fn - ry * ft

        # ---- semi-implicit Euler ---------------------------------------
        state.vx += fx / cfg.mass * dt
        state.vy += fy / cfg.mass * dt
        state.angular_velocity += torque / cfg.inertia * dt
        state.x += state.vx * dt
        state.y += state.vy * dt
        state.angle += state.angular_velocity * dt

        result.left_contact, result.right_contact = contacts
        result.left_compression, result.right_compression = compressions
        state.left_contact, state.right_contact = contacts

        # ---- crash checks -------------------------------------------
        if result.crashed:
            break
        if max(compressions) > cfg.leg_max_compression:
            result.crashed = True
            result.crash_reason = "leg snapped"
            break
        for hx, hy in hull_world(state, cfg):
            if hy < terrain.height_at(hx):
                result.crashed = True
                result.crash_reason = "hull hit the ground"
                break
        if result.crashed:
            break

    state.left_contact = result.left_contact
    state.right_contact = result.right_contact
    return result
