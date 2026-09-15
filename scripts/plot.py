#!/usr/bin/env python3
"""Learning curves from a training run's CSV logs.

    python scripts/plot.py runs/dqn-v1                 # -> runs/dqn-v1/curves.png
    python scripts/plot.py runs/dqn-v1 docs/curves.png
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SOFT, GRID = "#1a1a19", "#5f5e58", "#e6e5e0"
OUTCOMES = [("landed_on_pad", "landed on pad", BLUE),
            ("landed_off_pad", "landed off pad", AQUA),
            ("timeout", "timeout (hovering)", YELLOW),
            ("crashed", "crashed / out of bounds", ORANGE)]


def rolling(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    for i in range(min(n - 1, len(x))):        # expanding mean for the head
        out[i] = x[: i + 1].mean()
    return out


def main() -> None:
    run = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "curves.png"
    train = list(csv.DictReader(open(run / "train_log.csv")))
    evals = list(csv.DictReader(open(run / "eval_log.csv")))

    ep = np.array([int(r["episode"]) for r in train])
    ret = np.array([float(r["return"]) for r in train])
    q = np.array([float(r["q_mean"]) if r["q_mean"] else np.nan for r in train])
    status = [("crashed" if r["status"] == "out_of_bounds" else r["status"]) for r in train]
    ev_ep = np.array([int(r["episode"]) for r in evals])
    ev_ret = np.array([float(r["eval_return_mean"]) for r in evals])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
                         "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True, constrained_layout=True)

    # ---- 1. return ----------------------------------------------------
    ax = axes[0]
    ax.plot(ep, ret, color=BLUE, lw=0.8, alpha=0.22)
    ax.plot(ep, rolling(ret, 100), color=BLUE, lw=2, label="training return (100-episode mean)")
    ax.plot(ev_ep, ev_ret, color=ORANGE, lw=2, marker="o", ms=4, label="evaluation return (greedy, 10 fixed seeds)")
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set_ylabel("episode return")
    ax.set_title("Return")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=GRID, lw=0.8)

    # ---- 2. outcome share per 50-episode block -----------------------
    ax = axes[1]
    block = 50
    centers, shares = [], {k: [] for k, _, _ in OUTCOMES}
    for i in range(0, len(status), block):
        c = Counter(status[i:i + block]); n = sum(c.values())
        centers.append(i + n / 2)
        for k, _, _ in OUTCOMES:
            shares[k].append(100 * c.get(k, 0) / n)
    ys = [shares[k] for k, _, _ in OUTCOMES]
    ax.stackplot(centers, *ys, colors=[c for _, _, c in OUTCOMES], labels=[l for _, l, _ in OUTCOMES],
                 alpha=0.9, edgecolor="white", lw=1)
    ax.set_ylim(0, 100); ax.set_ylabel("% of episodes (50-episode blocks)")
    ax.set_title("How episodes end")
    ax.legend(frameon=False, loc="center right", ncol=1, fontsize=9)

    # ---- 3. mean Q ----------------------------------------------------
    ax = axes[2]
    ax.plot(ep, q, color=BLUE, lw=0.8, alpha=0.22)
    ax.plot(ep, rolling(np.nan_to_num(q), 50), color=BLUE, lw=2)
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set_ylabel("mean Q(s, ·) along trajectory")
    ax.set_xlabel("episode")
    ax.set_title("Value estimate (50-episode mean)")
    ax.grid(axis="y", color=GRID, lw=0.8)

    # epsilon floor marker on every panel
    eps = np.array([float(r["epsilon"]) for r in train])
    floor_ep = int(ep[np.argmax(eps <= eps.min() + 1e-9)])
    for ax in axes:
        ax.axvline(floor_ep, color=INK_SOFT, lw=1, ls=(0, (4, 4)), zorder=0)
    axes[0].annotate(f"ε reaches 0.05\n(episode {floor_ep})", xy=(floor_ep, axes[0].get_ylim()[0]),
                     xytext=(6, 6), textcoords="offset points", color=INK_SOFT, fontsize=8.5, va="bottom")

    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
