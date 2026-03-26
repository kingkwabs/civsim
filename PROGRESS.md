# CivSim — Progress Report

## Project Overview

CivSim is a Catan-style board game engine built for CS 4701 (AI). The engine simulates a multiplayer resource-trading board game with custom mechanics (maintenance, divine intervention, unique development cards) and provides an environment for training and evaluating AI agents.

---

## Completed

### Core Game Engine (8 modules, 60+ functions, 101 tests passing)

**`data_types.py`** — All enums, dataclasses, and constants
- 6 resource types (wood, stone, metal, wheat, water, cow)
- 3 build types, 6 action types, 5 dev card types, 7 port types
- Full action hierarchy: Build, TradeProposal, BuyDevCard, PlayDevCard, DivineIntervention, EndTurn
- Observation and OpponentView for partial observability
- All cost tables: building costs, maintenance costs, dev card cost, divine cost, re-establishment costs
- Divine event probability table (6 events)

**`board.py`** — Hex board with axial coordinates
- Standard 19-tile Catan layout generation with randomized resources and dice numbers
- Intersection derivation (vertices shared by 3 mutually adjacent tiles)
- Edge derivation (connections between intersections sharing 2 tiles)
- Port placement on border edges
- All spatial queries: valid settlements (distance rule + road adjacency), valid roads, valid cities, draft positions, abandoned buildings, port access
- Board cloning for MCTS simulations

**`game_state.py`** — Core state management
- Dice rolling (2d6) and resource production for all players
- Bank depletion handling (partial production when bank runs low)
- Maintenance resolution: settlements cost 1 water, cities cost 1 water + 1 cow per turn — failure means loss of ownership
- Game over detection: progress point threshold (10), last player standing, or turn cap (200)
- Partial observability: observations hide opponent resource details and show vague bank supply
- State cloning via deepcopy

**`actions.py`** — Valid action enumeration
- Master dispatcher that collects all legal actions for current player
- Build validation (affordability + valid board positions)
- Bank/port trade generation (4:1 default, 3:1 generic port, 2:1 specific port)
- Dev card purchase and play validation (one card per turn rule)
- Divine intervention check (requires 2 cows)
- RL encoding helpers (action_to_index, index_to_action)

**`action_executors.py`** — Execution logic for every action type
- Build: deduct resources, place structure, award progress points
- Trade: bank trades with ratio validation
- Buy dev card: deduct cost, draw from shuffled deck
- Play dev card (5 distinct effects):
  - Expansionist: 2 free roads (agent picks positions)
  - Espionage: steal 2 random resources from a chosen opponent
  - Maintenance: covers all upkeep for the turn
  - Invention: agent picks 2 resources from bank
  - Plunder: take all of 1 resource type from every opponent
- Divine intervention: 6 probabilistic outcomes (blessing 30%, famine 20%, earthquake 15%, prosperity 20%, plague 10%, miracle 5%)

**`environment.py`** — Game loop orchestration
- Full game lifecycle: reset, snake draft, turn loop, game over
- Snake draft: players place 2 settlements + 2 roads in snake order (1-2-3-3-2-1), then receive starting resources
- Turn structure: dice roll, resource production, action loop (agent picks repeatedly until EndTurn), maintenance resolution, advance player
- Gymnasium-style step interface returning (observation, reward, done, info)

**`evaluation.py`** — Tournament and metrics infrastructure
- GameRunner: run single games or multi-game tournaments
- MetricsTracker: win rates, average game length, agent comparison
- ReplayLogger: turn-by-turn logging with JSON save/load

### AI Agents

**`agents.py`** — 4 agent implementations

| Agent | Strategy | Strength |
|-------|----------|----------|
| **RandomAgent** | Uniform random over valid actions | Baseline |
| **GreedyAgent** | Heuristic scoring (cities > settlements > roads, pip-counting for drafts) | Simple but effective |
| **MCTSAgent** | Monte Carlo Tree Search with UCB1 selection, random rollouts, and heuristic leaf evaluation | Strong lookahead |
| **RLAgent** | Neural network policy trained via REINFORCE with baseline | Learned strategy |

**MCTSAgent** — Full MCTS implementation
- 4 phases: selection (UCB1), expansion, simulation (biased random rollout), backpropagation
- Configurable: n_simulations (default 100), rollout_depth (default 30), exploration_constant (default sqrt(2))
- Heuristic leaf evaluation based on relative progress points
- State reconstruction from partial observations for simulation
- Multi-player perspective handling during backpropagation

**RL System** (`civsim/rl/` package) — Complete training pipeline
- `features.py`: State encoder (36-dim) and action encoder (14-dim) with normalized features
- `network.py`: Actor-Critic neural network (PyTorch) with shared trunk, dot-product action scoring, and action masking
- `rl_agent.py`: Training mode (stochastic sampling + trajectory storage) and inference mode (greedy)
- `trainer.py`: REINFORCE with baseline — discounted returns, advantage estimation, entropy bonus, gradient clipping, temperature annealing

### Test Suite — 101 tests across 7 test files

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_board.py | 14 | Board construction, spatial queries, mutations, cloning |
| test_game_state.py | 11 | Init, dice, production, maintenance, game over, observations |
| test_actions.py | 9 | Valid action enumeration, affordability checks |
| test_executors.py | 9 | Build, trade, dev cards, divine intervention, helpers |
| test_environment.py | 8 | Reset, snake draft, step, full game loops, determinism |
| test_full_game.py | 3 | End-to-end games, 2-player variant, tournaments |
| test_mcts.py | 13 | UCB1, node selection, MCTS agent gameplay, evaluation |
| test_rl.py | 24 | Feature encoding, network forward/masking, agent modes, training |

---

## Remaining Features

### 3. Visualization
The `visualization/` directory is empty. A board renderer would help with debugging agent behavior and creating demos for the course presentation. Options range from terminal-based ASCII art to a browser-based hex grid using something like matplotlib or a simple HTML canvas.

### 4. Richer PlayerStats Tracking
The `PlayerStats` dataclass has fields for `buildings_lost`, `trades_made`, `divine_interventions`, `total_resources_earned`, and `total_resources_spent`, but these aren't being accumulated during gameplay. The action executors and environment need to increment these counters as actions are executed, giving detailed per-player analytics for evaluation and agent comparison.

### 5. Player-to-Player Trade Negotiation
Currently `get_valid_actions` only generates bank/port trades. The spec includes AI-to-AI trade proposals where one agent offers resources to another, and the target agent's `respond_to_trade` method decides whether to accept. This requires:
- Generating trade proposals toward specific opponents
- Calling the target agent's `respond_to_trade` during execution
- Handling the resource swap when accepted
