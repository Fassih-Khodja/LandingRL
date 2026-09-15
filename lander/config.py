"""All tunable constants for the lander simulation.

Everything the physics, the reward function and the episode rules depend on
lives here, so a training script can build variants of the environment
(harder gravity, weaker engines, ...) by passing a modified LanderConfig.

Units: the world is measured in abstract "units" (think metres) and seconds.
The renderer decides how many pixels a unit is; the core never knows.
Angles are radians, 0 = upright, positive = counter-clockwise.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LanderConfig:
    # ---- world -----------------------------------------------------------
    world_width: float = 20.0
    world_height: float = 13.333
    gravity: float = 5.0                      # units / s^2, pulls -y
    fps: int = 50                             # env.step() advances 1 / fps seconds
    substeps: int = 4                         # physics sub-iterations per step
    max_steps: int = 1000                     # truncation limit per episode

    # ---- terrain ---------------------------------------------------------
    terrain_points: int = 11                  # samples across the world width
    pad_y: float = 2.4                        # altitude of the landing pad
    pad_half_width: float = 2.0               # pad spans pad_center +- this
    terrain_min_height: float = 0.4
    terrain_max_height: float = 5.5

    # ---- lander body -----------------------------------------------------
    mass: float = 1.0
    inertia: float = 0.45                     # rotational inertia (kg * units^2)
    body_half_width: float = 0.45
    body_half_height: float = 0.32
    # polygon of the hull in body frame (x right, y up); used for crash tests
    # and offered to the renderer. Pointy nose on top, flat-ish bottom.
    hull: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.62),
        (0.42, 0.28),
        (0.45, -0.32),
        (-0.45, -0.32),
        (-0.42, 0.28),
    )
    # leg tips in body frame; where the leg meets the hull for drawing
    leg_tips: Tuple[Tuple[float, float], ...] = ((-0.70, -0.80), (0.70, -0.80))
    leg_roots: Tuple[Tuple[float, float], ...] = ((-0.30, -0.25), (0.30, -0.25))

    # ---- engines ---------------------------------------------------------
    main_thrust: float = 9.0                  # force, along body +y
    side_thrust: float = 1.6                  # force, along body +-x
    side_engine_x: float = 0.45               # side engines mounted at (+-x, y)
    side_engine_y: float = 0.30               # above the centre => produces torque

    # ---- leg / ground contact (spring-damper) ---------------------------
    leg_spring: float = 220.0
    leg_damping: float = 18.0
    leg_max_compression: float = 0.30         # deeper than this = leg snapped
    max_landing_speed: float = 3.0            # leg touching down faster = snapped
    ground_friction: float = 0.9              # Coulomb friction coefficient
    ground_tangent_damping: float = 25.0

    # ---- episode rules ---------------------------------------------------
    settle_speed: float = 0.08                # |v| below this counts as "still"
    settle_angular_speed: float = 0.08
    settle_steps: int = 25                    # consecutive still steps => landed
    out_of_bounds_margin: float = 1.0

    # ---- initial conditions (uniform random ranges) ---------------------
    start_x_spread: float = 4.0               # start x = centre +- spread
    start_y: float = 12.0
    start_vx: Tuple[float, float] = (-1.5, 1.5)
    start_vy: Tuple[float, float] = (-1.0, 0.5)
    start_angle: Tuple[float, float] = (-0.25, 0.25)
    start_angular_velocity: Tuple[float, float] = (-0.3, 0.3)

    # ---- rewards ---------------------------------------------------------
    reward_crash: float = -100.0
    reward_out_of_bounds: float = -100.0
    reward_landed_on_pad: float = 100.0
    reward_landed_off_pad: float = 20.0
    fuel_cost_main: float = 0.30              # per step the main engine fires
    fuel_cost_side: float = 0.03
    shaping_distance: float = 100.0           # weights of the potential-based
    shaping_velocity: float = 100.0           # shaping term (see env.py)
    shaping_angle: float = 100.0
    shaping_leg_contact: float = 10.0
    obs_velocity_scale: float = 5.0           # velocities / this in the obs

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def pad_center_x(self) -> float:
        return self.world_width / 2.0
