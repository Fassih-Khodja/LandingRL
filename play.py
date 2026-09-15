#!/usr/bin/env python3
"""Fly the lander yourself, or watch a trained agent fly it.

    python play.py                                   # keyboard, random episodes
    python play.py --seed 3                          # keyboard, reproducible
    python play.py --checkpoint runs/X/checkpoints/ep0150.pt        # watch the agent
    python play.py --checkpoint runs/X/checkpoints/best.pt --epsilon 0.05
"""
import argparse

from lander import LanderConfig, LanderEnv
from ui import App


def main() -> None:
    parser = argparse.ArgumentParser(description="Lander - human play or agent replay")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for terrain + start state")
    parser.add_argument("--auto-reset", type=float, default=0.0,
                        help="seconds after an episode ends before restarting (0 = wait for R)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to a .pt saved by train.py; the agent flies instead of you")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="with --checkpoint: probability of a random action (0 = pure greedy)")
    args = parser.parse_args()

    env = LanderEnv(config=LanderConfig(), seed=args.seed)
    policy = None
    auto_reset = args.auto_reset
    if args.checkpoint:
        import random
        from dqn import DQNAgent
        agent = DQNAgent.load(args.checkpoint)
        print(f"loaded {args.checkpoint}  (trained for {agent.total_steps} steps, "
              f"{agent.learn_steps} updates)")

        def policy(obs):
            if args.epsilon > 0 and random.random() < args.epsilon:
                return random.randrange(env.action_size)
            return agent.act(obs, greedy=True)

        if auto_reset == 0.0:
            auto_reset = 2.0        # keep the show going when nobody is at the keyboard
    App(env, policy=policy, auto_reset_delay=auto_reset).run()


if __name__ == "__main__":
    main()
