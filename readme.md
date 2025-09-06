# Applied Reinforcement Learning

A compact, professional collection of **reinforcement learning** implementations, agents, and experiments. Each subfolder is a self-contained project with its own README, and code.

## Structure
```
applied-reinforcement-learning/
├─ bandit-agent/            # k-armed bandits (ε-greedy, UCB, sample-average, constant α)
├─ <next-project>/          # e.g., tabular Q-learning, SARSA, policy gradients, DQN, etc.
└─ README.md                # (this file)
```

## Getting Started
- Clone the repo and enter a project folder:
  ```bash
  git clone https://github.com/<you>/applied-reinforcement-learning.git
  cd applied-reinforcement-learning/bandit-agent
  ```
- Follow the project’s local `README.md` (each project declares its own Python/C++ deps and run steps).

## Projects
- **bandit-agent/** — k-armed bandit agent in Python/NumPy with ε-greedy and UCB, supports stationary and non-stationary settings.

*(More agents coming: Gridworld (MDPs), Tabular Q-learning/SARSA, Policy Gradient (REINFORCE), DQN, Actor-Critic.)*

## Goals
- Clean, well-documented reference implementations.
- Reproducible, minimal experiments.
- Easy to extend for interviews, research, and production prototypes.

## Contributing
Small, focused PRs welcome (tests, docs, benchmarks). Keep code idiomatic and commented.

## License
MIT (unless overridden in a subproject).
