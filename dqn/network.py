"""Deep Q-Network for the lander: the function approximator + its hyperparameters.

=========================================================================
 What is the network actually computing?
=========================================================================

Q-learning wants a table Q(s, a) = "expected total future reward if I am
in state s, take action a, and act optimally afterwards".  Our state is 8
continuous floats, so a table is impossible (infinitely many states).  A
*deep* Q-network replaces the table with a neural net:

        state (8 floats)  --->  [ network ]  --->  4 numbers = Q(s, a) for a in 0..3

One forward pass gives the Q-value of *every* action at once.  That is the
standard DQN trick (Mnih et al. 2015): the input is only the state, the
output layer has one neuron per action.  The alternative (input = state +
action, output = one number) would need 4 forward passes per decision and
is never used for discrete actions.

The policy is then simply:  action = argmax_a Q(s, a)   (plus exploration).

=========================================================================
 The interface we sit on top of  (all implemented in lander/env.py)
=========================================================================

STATE  - `env.reset()` / `env.step()` return a tuple of 8 floats, all roughly
         in [-1, 1] (the env already normalises them - see LanderEnv._observe).
         Normalisation matters: a raw x of 20.0 next to a leg flag of 1.0 would
         make the first layer's weights fight over very different scales.

           0  horizontal offset from pad centre       (x - pad_x) / (W/2)
           1  altitude above the pad                  (y - pad_y - leg) / (H/2)
           2  horizontal velocity                     vx / 5
           3  vertical velocity                       vy / 5
           4  angle          (rad, 0 = upright)       NOTE: tilt, not "velocity angle"
           5  angular velocity (rad/s)                NOTE: spin rate - separate from 4
           6  left  leg touching ground               0.0 or 1.0
           7  right leg touching ground               0.0 or 1.0

ACTIONS - 4 discrete ids, `lander.physics.NOOP/LEFT/MAIN/RIGHT` = 0/1/2/3.

REWARD  - computed by the env, per step:

           r = Phi(s') - Phi(s)                       potential-based shaping
               - fuel  (0.30 main, 0.03 side)         so hovering is never free
               + terminal bonus                       crash / OOB -100,
                                                      landed on pad +100,
                                                      landed off pad  +20
           with
           Phi(s) = -100*dist_to_pad - 100*speed - 100*|angle|
                    + 10*left_leg + 10*right_leg

         The "+10 per leg" is INSIDE Phi on purpose.  Because reward is the
         *difference* of Phi, a leg touching gives +10 and a leg lifting off
         gives -10: the bonus is borrowed, not given, so it cannot be farmed
         by bouncing.  (Ng, Harada & Russell 1999 prove shaping of this form
         leaves the optimal policy unchanged.)

The DQN never recomputes any of the above - it just reads (obs, reward, done)
from the env.  Keeping one source of truth avoids the classic bug where the
trainer and the simulator disagree about what the state means.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Sizes come from the environment, not from here.  Importing them (instead
# of hard-coding 8 and 4) means the network can never silently disagree
# with the simulator about the shape of the problem.
from lander import LanderEnv

STATE_SIZE: int = LanderEnv.observation_size    # 8
ACTION_SIZE: int = LanderEnv.action_size        # 4


# =========================================================================
#  Hyperparameters
# =========================================================================
@dataclass
class Hyperparams:
    """Every knob of the DQN in one place.

    Defaults are the well-worn values for LunarLander-sized problems.  They
    are a starting point, not gospel - we will revisit them once we can see
    training curves.
    """

    # ---- network -------------------------------------------------------
    # Two hidden layers is enough here.  The state is 8 numbers and the
    # value landscape is smooth; a wider/deeper net mostly adds instability
    # (Q-learning already has enough of that) without adding capacity we need.
    hidden_sizes: tuple[int, ...] = (128, 128)

    # ---- optimiser -----------------------------------------------------
    # Adam is the default for DQN.  5e-4 is on the cautious side: Q-targets
    # move while we train (they depend on the network itself), so a large
    # step size chases a moving target and diverges.  1e-3 also works; 1e-2
    # almost never does.
    learning_rate: float = 5e-4

    # ---- the Bellman equation ------------------------------------------
    # gamma = how much the future is worth.  Q(s,a) = r + gamma * max Q(s',.)
    # 0.99 means a reward 100 steps away is worth 0.99^100 ~ 37% of an
    # immediate one.  At 50 fps an episode is a few hundred steps, so the
    # +100 landing bonus still "reaches back" to early decisions.  With
    # gamma=0.9 it would be 0.9^100 ~ 0.003% - invisible.
    gamma: float = 0.99

    # ---- experience replay ---------------------------------------------
    # Consecutive steps are nearly identical (same episode, 20 ms apart).
    # Training on them in order violates the i.i.d. assumption SGD relies
    # on and the net overfits to "whatever just happened".  We store
    # transitions (s, a, r, s', done) in a big buffer and sample random
    # minibatches from it instead.
    buffer_size: int = 100_000
    batch_size: int = 64
    # Don't start learning until the buffer has at least this many samples,
    # otherwise the first minibatches are all from one or two episodes.
    learn_start: int = 1_000
    # Do one gradient step every N env steps.  Every step is wasteful
    # (neighbouring samples barely change the buffer); 4 is the DQN classic.
    train_every: int = 4

    # ---- target network ------------------------------------------------
    # The TD target r + gamma * max Q(s',.) uses the *same* network we are
    # updating, so each gradient step moves the target we were aiming at.
    # Fix: keep a frozen copy ("target net") for the max Q(s',.) part and
    # only overwrite it every `target_update_every` steps.  The learner then
    # regresses toward something that holds still for a while.
    target_update_every: int = 1_000

    # ---- exploration (epsilon-greedy) ----------------------------------
    # With prob epsilon take a random action, else argmax Q.  Start fully
    # random (we know nothing), decay linearly to a small floor and keep a
    # bit of randomness forever so the agent never stops discovering.
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000       # env steps to go from eps_start to eps_end

    # ---- misc ----------------------------------------------------------
    # Huber loss = MSE near zero, L1 far away.  A single huge TD error (the
    # first time the agent crashes, |error| ~ 100) would give MSE a gradient
    # of ~200 and blow the weights up; Huber caps the gradient at 1.
    huber_delta: float = 1.0
    # Clip the gradient norm as a second safety belt against exploding updates.
    max_grad_norm: float = 10.0
    seed: int = 0


# =========================================================================
#  The Q-network
# =========================================================================
class QNetwork(nn.Module):
    """Maps a batch of states to a batch of Q-value vectors.

        input : (batch, 8)  float32
        output: (batch, 4)  float32   - one Q-value per action
    """

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE,
                 hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            # ReLU: cheap, no saturation (unlike sigmoid/tanh, whose gradient
            # dies for large inputs), and the de-facto choice for DQN MLPs.
            layers.append(nn.ReLU())
            in_features = h
        # Output layer has NO activation.  Q-values are real numbers that can
        # be anything from about -100 (certain crash) to +100 (certain landing)
        # or beyond; squashing them with sigmoid/tanh would make those targets
        # unreachable, and softmax would be wrong because the 4 outputs are
        # independent values, not a probability distribution.
        layers.append(nn.Linear(in_features, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


if __name__ == "__main__":
    # Smoke test: build the net, push one real observation through it, look
    # at the shapes.  The Q-values are garbage (untrained) - that's expected.
    hp = Hyperparams()
    net = QNetwork(hidden_sizes=hp.hidden_sizes)
    print(net)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\ntrainable parameters: {n_params:,}")

    env = LanderEnv(seed=hp.seed)
    obs = env.reset()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)   # (8,) -> (1, 8)
    q = net(x)
    print(f"\nobservation  {tuple(round(v, 3) for v in obs)}")
    print(f"input shape  {tuple(x.shape)}")
    print(f"Q-values     {q.squeeze(0).tolist()}")
    print(f"greedy action{q.argmax(dim=1).item()!r:>4}  ({LanderEnv.action_names[q.argmax(dim=1).item()]})")
