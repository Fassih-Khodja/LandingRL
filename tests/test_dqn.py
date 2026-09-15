"""Unit tests for the DQN pieces. Run: .venv/bin/python tests/test_dqn.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from dqn import ACTION_SIZE, STATE_SIZE, DQNAgent, Hyperparams, QNetwork, ReplayBuffer
from lander import LanderEnv


def test_network_shapes():
    net = QNetwork()
    x = torch.zeros(5, STATE_SIZE)
    assert net(x).shape == (5, ACTION_SIZE)


def test_buffer_is_a_ring():
    buf = ReplayBuffer(capacity=4, state_size=STATE_SIZE)
    for i in range(6):
        buf.push(np.full(STATE_SIZE, i), i % ACTION_SIZE, float(i), np.full(STATE_SIZE, i + 1), False)
    assert len(buf) == 4
    assert sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0, 5.0]   # 0 and 1 were overwritten


def test_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=100, state_size=STATE_SIZE)
    for i in range(50):
        buf.push(np.zeros(STATE_SIZE), 1, 0.0, np.zeros(STATE_SIZE), i % 2 == 0)
    s, a, r, s2, d = buf.sample(16)
    assert s.shape == (16, STATE_SIZE) and s2.shape == (16, STATE_SIZE)
    assert a.dtype == torch.int64 and a.shape == (16,)
    assert r.dtype == torch.float32 and d.dtype == torch.float32


def test_terminal_transitions_do_not_bootstrap():
    """With done=1 the target must equal the reward, whatever Q(s') says."""
    hp = Hyperparams(learn_start=1, batch_size=8, gamma=0.99)
    agent = DQNAgent(hp)
    # Make Q_target(s') large so bootstrapping would be obvious.
    with torch.no_grad():
        agent.target.net[-1].bias.fill_(50.0)
    for _ in range(8):
        agent.buffer.push(np.zeros(STATE_SIZE), 0, -100.0, np.zeros(STATE_SIZE), True)
    s, a, r, s2, d = agent.buffer.sample(8)
    with torch.no_grad():
        q_next = agent.target(s2).max(dim=1).values
        target = r + hp.gamma * (1.0 - d) * q_next
    assert torch.allclose(target, torch.full((8,), -100.0))


def test_learn_moves_only_online_weights():
    hp = Hyperparams(learn_start=1, batch_size=8, target_update_every=10 ** 9)
    agent = DQNAgent(hp)
    env = LanderEnv(seed=0)
    obs = env.reset()
    for _ in range(64):
        a = agent.act(obs)
        obs2, r, term, trunc, _ = env.step(a)
        agent.buffer.push(obs, a, r, obs2, term)
        obs = env.reset() if (term or trunc) else obs2
    before_online = [p.clone() for p in agent.online.parameters()]
    before_target = [p.clone() for p in agent.target.parameters()]
    loss = agent.learn()
    assert np.isfinite(loss)
    assert any(not torch.equal(b, p) for b, p in zip(before_online, agent.online.parameters()))
    assert all(torch.equal(b, p) for b, p in zip(before_target, agent.target.parameters()))


def test_epsilon_schedule():
    hp = Hyperparams(eps_start=1.0, eps_end=0.1, eps_decay_steps=100)
    agent = DQNAgent(hp)
    assert agent.epsilon == 1.0
    agent.total_steps = 50
    assert abs(agent.epsilon - 0.55) < 1e-9
    agent.total_steps = 1000
    assert abs(agent.epsilon - 0.1) < 1e-9


def test_save_load_roundtrip(tmp_path=Path("/tmp")):
    agent = DQNAgent(Hyperparams(hidden_sizes=(32, 32)))
    path = tmp_path / "dqn_roundtrip_test.pt"
    agent.save(path)
    clone = DQNAgent.load(path)
    x = torch.randn(3, STATE_SIZE)
    assert torch.allclose(agent.online(x), clone.online(x))
    assert clone.hp.hidden_sizes == (32, 32)
    path.unlink()


if __name__ == "__main__":
    import inspect
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
