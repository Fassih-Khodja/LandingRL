"""DQNAgent - ties the network, target network, optimizer, loss and replay
buffer together.  The training loop only ever calls:

    agent.act(obs)                       -> action  (epsilon-greedy)
    agent.observe(s, a, r, s2, done)     -> stores the transition and, when
                                            it is time, does one learn() step
    agent.save(path) / DQNAgent.load(path)

=========================================================================
 The loss - the one equation that IS deep Q-learning
=========================================================================

Q-learning's fixed point (the Bellman optimality equation) says that for
the true Q*:

        Q*(s, a)  =  r  +  gamma * max_a' Q*(s', a')          (if s' is not terminal)
        Q*(s, a)  =  r                                        (if s' is terminal)

The right-hand side is called the TD target ("temporal difference").  We
don't know Q*, so we plug our current network into the right-hand side and
treat the result as the label for a regression problem:

        prediction = Q_online(s, a)                          <- has gradient
        target     = r + gamma * (1 - done) * max_a' Q_target(s', a')
                                                             <- NO gradient
        loss       = Huber(prediction - target)

Two details decide whether this converges or explodes:

  * The target uses Q_TARGET, a frozen copy of the online net that we only
    refresh every `target_update_every` steps.  If we used Q_online on both
    sides, every gradient step would move the label we were regressing
    toward, and the whole thing chases its own tail.

  * We wrap the target in torch.no_grad().  Even with a separate target
    network this matters: we want "move prediction toward target", never
    "move target toward prediction".  Gradient flows through ONE side only.

(1 - done) is the "if s' is terminal" branch written as arithmetic: when
done == 1 the whole future term is multiplied by zero.
"""
from __future__ import annotations

import random
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .network import ACTION_SIZE, STATE_SIZE, Hyperparams, QNetwork
from .replay import ReplayBuffer


class DQNAgent:
    def __init__(self, hp: Optional[Hyperparams] = None, device: str = "cpu"):
        self.hp = hp or Hyperparams()
        self.device = torch.device(device)
        self._seed(self.hp.seed)

        # Two networks with identical architecture.
        self.online = QNetwork(STATE_SIZE, ACTION_SIZE, self.hp.hidden_sizes).to(self.device)
        self.target = QNetwork(STATE_SIZE, ACTION_SIZE, self.hp.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())   # start identical
        self.target.eval()                                      # never trained directly
        for p in self.target.parameters():
            p.requires_grad_(False)

        # ---- optimizer -------------------------------------------------
        # Adam: per-parameter adaptive step sizes.  Only the ONLINE net's
        # parameters are registered - the target net is updated by copying,
        # never by gradient.
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.hp.learning_rate)

        self.buffer = ReplayBuffer(self.hp.buffer_size, STATE_SIZE, seed=self.hp.seed)
        self.total_steps = 0      # env steps seen (drives epsilon + schedules)
        self.learn_steps = 0      # gradient updates done

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ------------------------------------------------------------------
    #  Exploration
    # ------------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        """Linear decay from eps_start to eps_end over eps_decay_steps, then flat."""
        hp = self.hp
        frac = min(1.0, self.total_steps / hp.eps_decay_steps)
        return hp.eps_start + frac * (hp.eps_end - hp.eps_start)

    def act(self, obs, greedy: bool = False) -> int:
        """Epsilon-greedy.  greedy=True (evaluation) = always argmax Q."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        with torch.no_grad():      # acting is inference; no graph needed
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(x).argmax(dim=1).item())

    def q_values(self, obs) -> np.ndarray:
        """Q(s, .) for logging / display."""
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.online(x).squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    #  Learning
    # ------------------------------------------------------------------
    def observe(self, s, a: int, r: float, s2, done: bool) -> Optional[float]:
        """Store one transition; learn / sync target on schedule.

        `done` must be the TERMINATED flag (crash / landed), NOT truncated
        (timeout).  On a timeout the lander is still mid-air - there is a
        future, we just stopped watching - so the Bellman target must keep
        bootstrapping from Q(s').  Passing done=True there would teach the
        net that "being alive at step 1000 is worth 0", which is false.

        Returns the loss if a learn step happened this call, else None.
        """
        hp = self.hp
        self.buffer.push(s, a, r, s2, done)
        self.total_steps += 1

        loss = None
        if len(self.buffer) >= hp.learn_start and self.total_steps % hp.train_every == 0:
            loss = self.learn()
        if self.total_steps % hp.target_update_every == 0:
            # Hard update: overwrite the frozen copy with the current weights.
            self.target.load_state_dict(self.online.state_dict())
        return loss

    def learn(self) -> float:
        """One gradient step on one random minibatch.  Returns the loss."""
        hp = self.hp
        s, a, r, s2, done = self.buffer.sample(hp.batch_size)
        s, a, r, s2, done = (t.to(self.device) for t in (s, a, r, s2, done))

        # ---- prediction: Q_online(s, a) ---------------------------------
        # online(s) is (batch, 4) - the Q-value of every action.  We only
        # have a label for the action that was actually taken, so gather()
        # picks column a[i] from row i  ->  (batch, 1)  ->  squeeze  ->  (batch,)
        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # ---- target: r + gamma * (1 - done) * max_a' Q_target(s', a') --
        with torch.no_grad():                       # label, not a variable
            q_next_max = self.target(s2).max(dim=1).values
            q_target = r + hp.gamma * (1.0 - done) * q_next_max

        # ---- loss ---------------------------------------------------------
        # Huber (smooth L1): quadratic for |error| < delta, linear beyond.
        # The gradient is therefore capped at +-delta, so one wildly wrong
        # sample (the first crash, error ~ 100) cannot blow up the weights
        # the way MSE's gradient of 2*error would.
        loss = F.smooth_l1_loss(q_pred, q_target, beta=hp.huber_delta)

        # ---- the update ---------------------------------------------------
        self.optimizer.zero_grad()                  # clear last step's gradients
        loss.backward()                             # dloss/dW for the ONLINE net only
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), hp.max_grad_norm)
        self.optimizer.step()                       # W <- W - lr * adam(grad)
        self.learn_steps += 1
        return float(loss.item())

    # ------------------------------------------------------------------
    #  Checkpoints
    # ------------------------------------------------------------------
    def save(self, path, **extra) -> None:
        torch.save({
            "hp": asdict(self.hp),
            "online": self.online.state_dict(),
            "total_steps": self.total_steps,
            "learn_steps": self.learn_steps,
            **extra,
        }, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "DQNAgent":
        ckpt = torch.load(path, map_location=device)
        hp = Hyperparams(**ckpt["hp"])
        hp.hidden_sizes = tuple(hp.hidden_sizes)    # json/torch may give a list
        agent = cls(hp, device=device)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["online"])
        agent.total_steps = ckpt.get("total_steps", 0)
        agent.learn_steps = ckpt.get("learn_steps", 0)
        return agent
