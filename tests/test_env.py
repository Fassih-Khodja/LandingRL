"""Core-only tests: no pygame needed.  Run with  python -m pytest  or  python tests/test_env.py"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lander import (LEFT, MAIN, NOOP, RIGHT, STATUS_CRASHED, STATUS_LANDED_OFF,
                    STATUS_LANDED_PAD, STATUS_OUT_OF_BOUNDS, LanderConfig,
                    LanderEnv)
from lander.physics import LanderState, step_physics
from lander.terrain import generate_terrain


def run_episode(env, policy):
    obs = env.reset()
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        done = term or trunc
    return env


def autopilot(obs):
    """Hand-tuned proportional controller. Not RL - just proves the env is landable."""
    x, y, vx, vy, ang, angv, l, r = obs
    angle_targ = max(-0.4, min(0.4, x * 0.5 + vx * 1.0))
    hover_targ = 0.55 * abs(x)
    angle_todo = (angle_targ - ang) * 0.5 - angv * 1.0
    hover_todo = (hover_targ - y) * 0.5 - vy * 0.5
    if l or r:
        angle_todo = 0.0
        hover_todo = -vy * 0.5
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return MAIN
    if angle_todo < -0.05:
        return RIGHT
    if angle_todo > 0.05:
        return LEFT
    return NOOP


# ---------------------------------------------------------------- terrain
def test_pad_is_flat_and_centred():
    cfg = LanderConfig()
    t = generate_terrain(cfg, random.Random(0))
    assert abs(t.pad_center_x - cfg.world_width / 2) < 1e-9
    for i in range(50):
        x = t.pad_x0 + (t.pad_x1 - t.pad_x0) * i / 49
        assert abs(t.height_at(x) - cfg.pad_y) < 1e-9
    assert t.on_pad(t.pad_center_x) and not t.on_pad(t.pad_x1 + 0.1)


# ---------------------------------------------------------------- physics
def test_gravity_pulls_down_and_main_engine_pushes_up():
    cfg = LanderConfig()
    t = generate_terrain(cfg, random.Random(0))
    st = LanderState(x=10, y=10, vx=0, vy=0, angle=0, angular_velocity=0)
    step_physics(st, NOOP, t, cfg)
    assert st.vy < 0
    st = LanderState(x=10, y=10, vx=0, vy=0, angle=0, angular_velocity=0)
    step_physics(st, MAIN, t, cfg)
    assert st.vy > 0 and abs(st.vx) < 1e-9 and abs(st.angular_velocity) < 1e-9


def test_side_engines_push_and_rotate_in_opposite_directions():
    cfg = LanderConfig()
    t = generate_terrain(cfg, random.Random(0))
    left = LanderState(x=10, y=10, vx=0, vy=0, angle=0, angular_velocity=0)
    right = LanderState(x=10, y=10, vx=0, vy=0, angle=0, angular_velocity=0)
    step_physics(left, LEFT, t, cfg)
    step_physics(right, RIGHT, t, cfg)
    assert left.vx < 0 < right.vx
    assert left.angular_velocity > 0 > right.angular_velocity   # left => ccw


def test_legs_detect_contact_independently():
    cfg = LanderConfig()
    t = generate_terrain(cfg, random.Random(0))
    leg_h = -cfg.leg_tips[0][1]
    # tilted so that only the right leg reaches the pad
    st = LanderState(x=t.pad_center_x, y=cfg.pad_y + leg_h + 0.05, vx=0, vy=0,
                     angle=-0.25, angular_velocity=0)
    step_physics(st, NOOP, t, cfg)
    assert st.right_contact and not st.left_contact


# ---------------------------------------------------------------- env
def test_observation_shape_and_reset_determinism():
    a = LanderEnv(seed=42).reset()
    b = LanderEnv(seed=42).reset()
    assert len(a) == LanderEnv.observation_size == 8
    assert a == b
    assert LanderEnv(seed=43).reset() != a


def test_free_fall_crashes():
    env = run_episode(LanderEnv(seed=1), lambda obs: NOOP)
    assert env.status == STATUS_CRASHED
    assert env.episode_reward < 0


def test_full_throttle_leaves_the_world():
    env = run_episode(LanderEnv(seed=1), lambda obs: MAIN)
    assert env.status == STATUS_OUT_OF_BOUNDS


def test_step_after_done_raises():
    env = run_episode(LanderEnv(seed=1), lambda obs: NOOP)
    try:
        env.step(NOOP)
    except RuntimeError:
        return
    raise AssertionError("step() after termination should raise")


def test_autopilot_lands_safely_most_of_the_time():
    outcomes = {}
    for seed in range(30):
        env = run_episode(LanderEnv(seed=seed), autopilot)
        outcomes[env.status] = outcomes.get(env.status, 0) + 1
    safe = outcomes.get(STATUS_LANDED_PAD, 0) + outcomes.get(STATUS_LANDED_OFF, 0)
    assert safe >= 27, outcomes
    assert outcomes.get(STATUS_LANDED_PAD, 0) >= 15, outcomes


def test_snapshot_is_a_copy():
    env = LanderEnv(seed=0)
    env.reset()
    snap = env.snapshot()
    snap.state.x = -999
    assert env.state.x != -999


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {exc!r}")
    sys.exit(1 if failed else 0)
