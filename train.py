#!/usr/bin/env python3
"""Train the DQN on the lander.

    .venv/bin/python train.py                          # 1000 episodes, runs/<timestamp>/
    .venv/bin/python train.py --episodes 300 --out runs/test

Produces, under --out:
    train_log.csv        one row per training episode
    eval_log.csv         one row per evaluation (greedy policy, fixed seeds)
    checkpoints/epNNNN.pt  every --checkpoint-every episodes (+ final.pt, best.pt)

=========================================================================
 What to log in RL, and why
=========================================================================

* episode return  - sum of rewards.  THE number, but noisy: it depends on the
                    random start AND on epsilon (a random action mid-descent
                    can wreck an otherwise good episode).  Look at a rolling
                    mean, never a single episode.
* outcome         - crashed / landed_on_pad / landed_off_pad / timeout /
                    out_of_bounds.  Return can rise a lot before the first
                    landing (thanks to shaping), so "pad rate" is the honest
                    progress bar.
* episode length  - early: short (crash fast).  Later: long (hovers).  Then
                    shorter again (lands decisively).  That arc is normal.
* epsilon         - to correlate behaviour changes with the exploration schedule.
* loss            - NOT a progress signal in DQN.  The targets move as the net
                    learns, so loss can go UP while the agent gets better.
                    Log it only to catch explosions (NaN, 1e6).
* mean Q          - the net's own estimate of "how good is life".  Should rise
                    and then plateau near the true return.  If it keeps
                    climbing past what any episode ever earns, Q is
                    over-estimating (the known DQN weakness).
* eval return     - same net, epsilon = 0, same fixed seeds every time.  This
                    is the agent's actual skill, uncontaminated by exploration
                    and start-state luck.  The checkpoints and "best" are
                    picked on this, not on the training return.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import deque
from pathlib import Path

from dqn import DQNAgent, Hyperparams
from lander import STATUS_LANDED_PAD, LanderConfig, LanderEnv

# Seeds for evaluation episodes.  Fixed => every evaluation flies the SAME
# set of terrains / starts, so eval numbers are comparable across time.
EVAL_SEED_BASE = 10_000


def evaluate(agent: DQNAgent, n_episodes: int, config: LanderConfig) -> dict:
    """Greedy rollouts (epsilon = 0) on fixed seeds.  Never trains."""
    env = LanderEnv(config=config)
    returns, lengths, statuses = [], [], []
    for i in range(n_episodes):
        obs = env.reset(seed=EVAL_SEED_BASE + i)
        total, steps = 0.0, 0
        while True:
            obs, r, term, trunc, info = env.step(agent.act(obs, greedy=True))
            total += r
            steps += 1
            if term or trunc:
                break
        returns.append(total)
        lengths.append(steps)
        statuses.append(info["status"])
    return {
        "eval_return_mean": statistics.mean(returns),
        "eval_return_min": min(returns),
        "eval_return_max": max(returns),
        "eval_len_mean": statistics.mean(lengths),
        "eval_pad_rate": statuses.count(STATUS_LANDED_PAD) / n_episodes,
        "eval_crash_rate": sum(s in ("crashed", "out_of_bounds") for s in statuses) / n_episodes,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="DQN training for the lander")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="run directory (default runs/<timestamp>)")
    p.add_argument("--checkpoint-every", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=50, help="evaluate every N episodes")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--print-every", type=int, default=10)
    args = p.parse_args()

    out = Path(args.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hp = Hyperparams(seed=args.seed)
    agent = DQNAgent(hp)
    config = LanderConfig()
    # Random start (x offset, velocity, tilt) and random terrain come from the
    # env's own RNG on every reset() - see LanderConfig.start_*  and
    # LanderEnv.reset().  Seeding once here makes the whole run reproducible.
    env = LanderEnv(config=config, seed=args.seed)

    train_fields = ["episode", "steps", "total_steps", "return", "status", "on_pad",
                    "fuel", "epsilon", "loss_mean", "q_mean", "return_avg100",
                    "pad_rate100", "seconds"]
    eval_fields = ["episode", "total_steps", "eval_return_mean", "eval_return_min",
                   "eval_return_max", "eval_len_mean", "eval_pad_rate", "eval_crash_rate"]
    train_f = open(out / "train_log.csv", "w", newline="")
    eval_f = open(out / "eval_log.csv", "w", newline="")
    train_w = csv.DictWriter(train_f, fieldnames=train_fields); train_w.writeheader()
    eval_w = csv.DictWriter(eval_f, fieldnames=eval_fields); eval_w.writeheader()

    recent_returns: deque = deque(maxlen=100)
    recent_pad: deque = deque(maxlen=100)
    best_eval = float("-inf")
    t0 = time.time()
    print(f"run dir: {out}\n{hp}\n")

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        ep_return, ep_steps = 0.0, 0
        losses, q_means = [], []

        # ------------------------------------------------------------------
        # One episode = act -> step -> store/learn, until crash / landing /
        # timeout.  The env enforces max_steps=1000 so it can never loop
        # forever.
        # ------------------------------------------------------------------
        while True:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # done = terminated ONLY.  See DQNAgent.observe for why truncation
            # (timeout) must NOT cut the Bellman bootstrap.
            loss = agent.observe(obs, action, reward, next_obs, terminated)

            if loss is not None:
                losses.append(loss)
            if ep_steps % 20 == 0:        # cheap sample of the net's own outlook
                q_means.append(float(agent.q_values(obs).mean()))

            obs = next_obs
            ep_return += reward
            ep_steps += 1
            if terminated or truncated:
                break

        status = info["status"]
        on_pad = int(status == STATUS_LANDED_PAD)
        recent_returns.append(ep_return)
        recent_pad.append(on_pad)
        row = {
            "episode": ep, "steps": ep_steps, "total_steps": agent.total_steps,
            "return": round(ep_return, 2), "status": status, "on_pad": on_pad,
            "fuel": round(info["fuel_used"], 2), "epsilon": round(agent.epsilon, 4),
            "loss_mean": round(statistics.mean(losses), 4) if losses else "",
            "q_mean": round(statistics.mean(q_means), 3) if q_means else "",
            "return_avg100": round(statistics.mean(recent_returns), 2),
            "pad_rate100": round(sum(recent_pad) / len(recent_pad), 3),
            "seconds": round(time.time() - t0, 1),
        }
        train_w.writerow(row); train_f.flush()

        if ep % args.print_every == 0 or ep == 1:
            print(f"ep {ep:5d} | steps {ep_steps:4d} | ret {ep_return:8.2f} | avg100 {row['return_avg100']:8.2f} "
                  f"| pad100 {row['pad_rate100']:.2f} | eps {agent.epsilon:.3f} | loss {row['loss_mean'] or '-':>7} "
                  f"| Q {row['q_mean']:7.2f} | {status:14s} | {row['seconds']:6.0f}s")

        # ---- evaluation ------------------------------------------------
        if ep % args.eval_every == 0 or ep == args.episodes:
            ev = evaluate(agent, args.eval_episodes, config)
            eval_w.writerow({"episode": ep, "total_steps": agent.total_steps, **ev}); eval_f.flush()
            print(f"   [eval @ {ep}] return {ev['eval_return_mean']:8.2f} "
                  f"(min {ev['eval_return_min']:.0f} / max {ev['eval_return_max']:.0f}) "
                  f"| pad {ev['eval_pad_rate']:.2f} | crash {ev['eval_crash_rate']:.2f} "
                  f"| len {ev['eval_len_mean']:.0f}")
            if ev["eval_return_mean"] > best_eval:
                best_eval = ev["eval_return_mean"]
                agent.save(ckpt_dir / "best.pt", episode=ep, eval=ev)

        # ---- checkpoint ------------------------------------------------
        if ep % args.checkpoint_every == 0:
            agent.save(ckpt_dir / f"ep{ep:04d}.pt", episode=ep)

    agent.save(ckpt_dir / "final.pt", episode=args.episodes)
    train_f.close(); eval_f.close()
    print(f"\ndone in {time.time() - t0:.0f}s. best eval return {best_eval:.2f}. "
          f"watch with:  python play.py --checkpoint {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
