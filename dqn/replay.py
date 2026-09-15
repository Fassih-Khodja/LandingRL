"""Experience replay buffer.

Stores transitions (s, a, r, s', done) in a fixed-size ring and hands back
uniformly random minibatches.  See the `experience replay` comment in
network.py for *why* we do this instead of learning from steps in order.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, seed: int = 0):
        self.capacity = capacity
        # Pre-allocate one flat array per field.  Faster than a list of
        # tuples (no per-sample Python objects) and sampling becomes fancy
        # indexing: states[idx] is one vectorised copy, not 64 lookups.
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        # `done` is stored as a float so it can be used directly as a
        # multiplier in the Bellman target:  r + gamma * (1 - done) * max Q(s')
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0        # next write position (wraps around)
        self.size = 0       # how many valid entries we hold (<= capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, s, a: int, r: float, s2, done: bool) -> None:
        i = self.pos
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s2
        self.dones[i] = float(done)
        # Ring: after `capacity` pushes we start overwriting the oldest
        # transition.  The buffer never grows past capacity and is never
        # emptied - the agent always keeps its last `capacity` memories.
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        # Uniform random indices over the VALID part of the buffer.  This is
        # the "random, not consecutive" part: idx may mix a step from this
        # episode with one from 300 episodes ago.
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self) -> int:
        return self.size
