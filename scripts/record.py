#!/usr/bin/env python3
"""Render an agent's episode to an animated GIF, without opening a window.

    python scripts/record.py runs/dqn-v1/checkpoints/best.pt docs/best.gif
    python scripts/record.py --random docs/random.gif          # untrained baseline
    python scripts/record.py CKPT OUT.gif --seed 10003 --max-seconds 12

Uses pygame's dummy video driver, so it works over SSH / in CI.  Frames are
drawn by the same `ui.Renderer` the interactive app uses, downscaled, and
sub-sampled to keep the file small.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame                    # noqa: E402
from PIL import Image            # noqa: E402

from lander import LanderConfig, LanderEnv   # noqa: E402
from ui.renderer import Renderer             # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", help="path to a .pt from train.py")
    p.add_argument("out", help="output .gif")
    p.add_argument("--random", action="store_true", help="random policy instead of a checkpoint")
    p.add_argument("--seed", type=int, default=10_000, help="episode seed (terrain + start)")
    p.add_argument("--episodes", type=int, default=1, help="episodes to chain into one gif")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--every", type=int, default=2, help="keep every Nth frame (50 fps source)")
    p.add_argument("--max-seconds", type=float, default=20.0, help="cap per episode")
    p.add_argument("--hold", type=float, default=1.5, help="seconds to hold the final frame")
    args = p.parse_args()
    if not args.random and not args.checkpoint:
        p.error("give a checkpoint or --random")

    if args.random:
        rng = random.Random(args.seed)
        policy = lambda obs: rng.randrange(4)               # noqa: E731
    else:
        from dqn import DQNAgent
        agent = DQNAgent.load(args.checkpoint)
        policy = lambda obs: agent.act(obs, greedy=True)    # noqa: E731

    pygame.init()
    # Render at full size for crisp text, then downscale each kept frame.
    render_w, render_h = 960, 640
    surface = pygame.Surface((render_w, render_h))
    renderer = Renderer(render_w, render_h)
    env = LanderEnv(config=LanderConfig())

    frames = []
    frame_ms = int(1000 / 50 * args.every)
    max_steps = int(args.max_seconds * 50)

    def grab():
        raw = pygame.image.tobytes(surface, "RGB")
        img = Image.frombytes("RGB", (render_w, render_h), raw)
        frames.append(img.resize((args.width, args.height), Image.LANCZOS))

    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        step = 0
        while True:
            renderer.draw(surface, env.snapshot())
            if step % args.every == 0:
                grab()
            if env.done or step >= max_steps:
                break
            obs, _, term, trunc, _ = env.step(policy(obs))
            step += 1
        renderer.draw(surface, env.snapshot())
        for _ in range(int(args.hold * 1000 / frame_ms)):
            grab()
        print(f"episode {ep}: {env.status} after {env.step_index} steps, "
              f"return {env.episode_reward:.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Quantise to a shared palette so the gif doesn't flicker between frames.
    first = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    rest = [f.quantize(palette=first, dither=Image.Dither.NONE) for f in frames[1:]]
    first.save(out, save_all=True, append_images=rest, duration=frame_ms, loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
