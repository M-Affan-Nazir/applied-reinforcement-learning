# Bandit Agent — Reinforcement Learning Playground

## Overview
A compact **Python/NumPy** implementation of a *k-armed bandit agent* for practicing action-value estimation and action selection. Supports **ε-greedy** and **UCB** strategies, with both **sample-average** and **constant step-size** updates—perfect for stationary and non-stationary reward settings.

> File: `bandit_agent.py`  
> Class: `BanditAgent`

---

## Features

- **Action Selection**
  - **ε-greedy** (explore with probability ε, exploit otherwise)
  - **UCB** (upper confidence bounds) via parameter `c > 0`
- **Value Updates**
  - **Sample-average** (when `alpha=None`) for stationary problems  
  - **Constant step-size** (when `alpha∈(0,1]`) for non-stationary problems
- **Optimistic Starts** via `init` to encourage early exploration
- **Utilities**: `reset()` to start fresh, `estimate()` to inspect current Q

---

## Algorithms (brief)

- **Incremental action-value update**
  
  \[
  Q_{t+1}(a)=Q_t(a) + \alpha_t\,[R_t - Q_t(a)]
  \]
  with \(\alpha_t=\tfrac{1}{N_t(a)}\) (sample-average) or constant \(\alpha\).

- **UCB action selection**
  
  \[
  A_t=\arg\max_a \left[\, Q_t(a) + c\,\sqrt{\frac{\ln t}{N_t(a)}} \,\right]
  \]

---

## Class API

```python
class BanditAgent:
    def __init__(self, k, alpha=None, c=0, init=0, ε=0.1)
```

- `k`: number of arms  
- `alpha`: `None` → sample-average; float in (0,1] → constant step-size  
- `c`: UCB exploration constant (`0` → use ε-greedy)  
- `init`: optimistic initial value for all actions  
- `ε`: epsilon for ε-greedy exploration

**Methods**

- `select_action() -> int` — choose an arm (ε-greedy or UCB)
- `update(a: int, r: float)` — update estimates with observed reward
- `reset()` — reinitialize `Q`, `N`, and `t`
- `estimate() -> np.ndarray` — copy of current action-value estimates

---

## Quickstart

### Requirements
- Python 3.8+
- NumPy

```bash
pip install numpy
```

### Usage Example

```python
import numpy as np
from bandit_agent import BanditAgent

# Simple Gaussian k-armed bandit environment
class GaussianBandit:
    def __init__(self, means):
        self.means = np.array(means)

    def pull(self, a):
        return np.random.normal(self.means[a], 1.0)

# Setup
k = 10
true_means = np.random.normal(0, 1, size=k)
env = GaussianBandit(true_means)

# Try ε-greedy with sample-average updates
agent = BanditAgent(k=k, alpha=None, c=0, init=0.0, ε=0.1)

steps = 1000
rewards = []
for _ in range(steps):
    a = agent.select_action()
    r = env.pull(a)
    agent.update(a, r)
    rewards.append(r)

print("Estimated Q:", agent.estimate())
print("Average reward:", np.mean(rewards))
```

---

## Experiments to Try

- **Stationary vs Non-stationary**
  - Stationary: `alpha=None`
  - Non-stationary: `alpha=0.1` (or similar constant)
- **UCB vs ε-greedy**
  - UCB: set `c` to 1–2 and keep `ε` small
  - ε-greedy only: `c=0`, tune `ε` (e.g., 0.1 or 0.01)
- **Optimistic Initialization**
  - Use `init > 0` to boost early exploration even with small `ε`

---

## Project Structure (suggested)

```
.
├─ bandit_agent.py           # The agent implementation (this repo)
├─ examples/
│  └─ quickstart.py          # Minimal run script (optional)
└─ tests/                    # Unit tests (optional)
```

---

## Notes

- UCB branch explores each arm at least once (infinite bonus when `N[a]==0`).
- `np.argmax` breaks ties by the first index; for randomized tie-breaking, add noise.
- For reproducibility, set `np.random.seed(seed)` in your scripts.

---

## Acknowledgements
- Concepts and formulas inspired by **Sutton & Barto – Reinforcement Learning: An Introduction**.
