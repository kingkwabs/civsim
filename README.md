# CivSim

CivSim is a Python board-game simulation inspired by Catan. It includes a core game engine, multiple AI agents, tournament evaluation helpers, reinforcement-learning experiments, and terminal/matplotlib renderers for demos.

## Setup

Use Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[dev,gui]"
python -m pip install -e ".[dev,rl]"
```

`gui` installs matplotlib for the visual renderer. `rl` installs PyTorch for RL training and RL-specific tests.

## Run Tests

```bash
pytest
```

When PyTorch is not installed, RL-specific tests are skipped. The core engine, agents, evaluation, and visualization tests still run.

## Run A Demo

Terminal demo:

```bash
python -m civsim.demo --agents greedy,greedy,random --seed 42 --pause 0
```

GUI demo:

```bash
python -m civsim.demo --gui --agents greedy,greedy,random --seed 42 --pause 0.2
```

Use `--agents` with `random`, `greedy`, `mcts`, or `rl`. The `rl` agent expects a trained checkpoint such as `rl_model.pt`.

## Train RL

```bash
python -m civsim.rl_train --episodes-random 300 --episodes-greedy 200 --episodes-selfplay 300 --eval-games 30
```

Training requires the `rl` extra.

## Repository Hygiene

Generated outputs, Python caches, local reports, and model checkpoints are ignored by default. Keep source changes focused on files under `civsim/`, tests, and project metadata unless a generated artifact is explicitly needed.
