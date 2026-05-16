# CivSim — Progress Report

## Project Overview

CivSim is a Catan-style board game engine for CS 4701 (AI). It simulates a multiplayer resource-trading game with custom mechanics (per-turn maintenance, divine intervention, unique dev cards, weather events) and serves as an environment for training and evaluating four AI agents at distinct complexity tiers.

---

## Headline Result

| Agent | Method | Win rate vs 2 Greedy |
|---|---|---|
| Random | Uniform action selection | ~3% |
| Greedy | Hand-tuned heuristic (upkeep-aware, target-aware) | ~15% |
| MCTS | K=5 pruning + PUCT + RL policy/value priors + tuned c_puct/temp | ~15% |
| **RL** | Actor-Critic + REINFORCE + curriculum + self-play | **70%** |

The RL agent **convincingly** beats two Greedys (60-game evaluation). MCTS reaches parity with Greedy. The relative ordering (Random < Greedy ≈ MCTS << RL) is methodologically clean: learned policy dominates fixed-budget tree search in this large-action-space domain.

---

## Architecture (final)

### Core engine (10 modules, 165 tests passing)

**`data_types.py`** — enums, dataclasses, balance constants
- 6 resources, 3 build types, 6 action types, 5 dev cards, 7 ports
- Build/maintenance/dev-card cost tables; weather/maintenance constants
- `PlayerStats` with live-tracked counters (built/lost/trades/dev/divine/rain/barn/flow)

**`board.py`** — hex board with axial coords
- 19-tile layout (rebalanced: 3 wood / 3 stone / 3 metal / 4 wheat / 3 water / 2 cow / 1 desert — cow bumped from 1 to address city-upkeep bottleneck)
- Intersection/edge derivation, port placement, spatial queries

**`game_state.py`** — state management with shared turn-transition helpers
- `roll_dice`, `roll_rain`, `roll_barn_day`, `apply_rain`, `apply_barn_day`
- `end_player_turn` (maintenance every Nth own-turn), `start_player_turn` (weather → dice → produce)
- These helpers are shared by `Environment` and `MCTSAgent` so simulated rollouts don't drift from the live game

**`actions.py`** — valid action enumeration
- Build, bank-trade, **p2p-trade proposals (bounded 1-for-1 / 2-for-1)**, buy dev, play dev, divine intervention, end turn
- RL encoding helpers

**`action_executors.py`** — execution + live stat accumulation
- Every flow (build, trade, dev card, divine, maintenance) bumps the appropriate `PlayerStats` counter
- p2p trades go through `target_agent.respond_to_trade`; simulation path (no agents) returns clean failure instead of falling through to a bank trade

**`environment.py`** — game loop
- Snake draft, weather + dice production, action loop, maintenance
- Failure-cap guard (force end-turn after 5 consecutive failed actions) prevents stubborn-rejection loops
- Optional `renderer` parameter for terminal visualization
- Per-action `on_action_result` callback gives agents rejection memory

**`upkeep.py`** — shared upkeep helpers
- `total_upkeep_cost`, `upkeep_gap`, `buildings_at_risk`, `upkeep_pressure_from_obs`
- Used by all agents and the RL state encoder so every layer has the same upkeep signal

**`visualization/terminal.py`** — ASCII renderer
- Hex board (colored tile glyphs + dice numbers), buildings/roads listing, live dashboard with per-player stat one-liner
- `TerminalRenderer` wires into `Environment` via optional `renderer` parameter
- `render_final_stats(GameResult)` end-of-game summary

**`evaluation.py`** — tournament + metrics + replay logging

**`demo.py`** + **`rl_train.py`** — runnable entry points
- `python -m civsim.demo --agents greedy,greedy,mcts` — rendered game
- `python -m civsim.rl_train --episodes-random 300 --episodes-greedy 200 --episodes-selfplay 300 --eval-games 30`

### Agents (`agents.py` + `rl/`)

**`RandomAgent`** — baseline; 50/50 trade response.

**`GreedyAgent`** — heuristic scoring with three layers of awareness:
- **Upkeep-aware**: heavy EndTurn penalty when buildings would fail upkeep; MAINTENANCE-card spike to top priority; refuse to spend critical water/cow reserves; preemptive stockpile trades; refuse dev-card buy if water would drop below upkeep need.
- **Target-aware**: per-action scores scale by `urgency = my_pp / WIN_THRESHOLD`. Cities ramp up sharply as urgency rises; roads ramp down; dev cards de-prioritized; trade scoring includes a city-ingredient bonus near the win line.
- **Per-turn rejection memory**: failed actions get a -10000 penalty so the agent doesn't loop on rejected trades.
- Shared draft heuristic favors water/cow-adjacent intersections (pip-weighted).

**`MCTSAgent`** — Monte Carlo Tree Search with several adaptations for our action space:
- Standard 4 phases, but uses the shared `state.end_player_turn` / `start_player_turn` helpers so rollouts match the live environment exactly.
- **Greedy rollout policy** (not uniform random) — necessary because the ~60-action space made random rollouts uninformative.
- **Action pruning** to top-K=5 by Greedy score before tree expansion; EndTurn always retained.
- **PUCT** instead of vanilla UCB1, with priors from either softmaxed Greedy scores or an external policy network (configurable).
- **Virtual Q-init**: unvisited children's Q starts at the prior estimate (instead of zero).
- **Decisive-move shortcut**: if Greedy's top score exceeds #2 by >100 and that action hasn't already failed this turn, skip MCTS entirely.
- **All-in detection**: if any Build would push PP to threshold, take it without searching.
- Optional `value_network` and `policy_network` (the RL trained model). When set, MCTS uses the value head (sigmoid-squashed) as the depth-limit evaluator and/or the policy head's action probabilities as PUCT priors. Tunable temperature on policy priors.

**`RLAgent`** — Actor-Critic + REINFORCE-with-baseline
- 39-dim state encoder (resources, dev cards, PP, dice, turn, bank, opponent views, building counts, valid positions, **upkeep gap features**)
- 14-dim action encoder (one-hot type + build/dev-card subtypes)
- Shared trunk → policy head (dot-product action scoring) + value head
- Training: stochastic sampling with temperature annealing; trajectory storage; gradient clipping
- Inference: argmax over policy (deterministic)

### Test suite — 165 tests across 16 files

| Test File | Tests | Coverage |
|---|---|---|
| test_actions.py | 9 | Action enumeration, affordability |
| test_balance.py | 5 | Balance pass (maintenance cadence, cow tiles, build progress) |
| test_barn_day.py | 7 | Cow-only city upkeep, barn-day mechanic |
| test_board.py | 14 | Board construction, spatial queries |
| test_environment.py | 8 | Reset, snake draft, step, full game, determinism |
| test_executors.py | 9 | Build, trade, dev cards, divine intervention |
| test_full_game.py | 3 | End-to-end games, tournaments |
| test_game_state.py | 11 | Dice, production, maintenance, game-over, observations |
| test_mcts.py | 13 | UCB1/PUCT, node selection, MCTS agent play |
| test_p2p_trade.py | 12 | Proposal generation, accept/reject, stats, simulation safety |
| test_rain.py | 7 | Rain mechanic, starting water stockpile |
| test_rl.py | 24 | Feature encoding, network forward, agent modes, training |
| test_stats.py | 9 | Live PlayerStats accumulation across every flow |
| test_target_aware.py | 7 | Urgency scaling, city/road scoring, sustain buffer |
| test_upkeep.py | 9 | Upkeep helpers, Greedy synthetic scenarios, RL encoder integration |
| test_visualization.py | 7 | Renderer ANSI/no-color, dashboard, env integration |

---

## What got built (full timeline)

### Phase 1 — Visualization & instrumentation
1. **Terminal hex board renderer**: ASCII tile grid with resource glyphs + dice numbers, colored ANSI output, optional buildings listing.
2. **Live dashboard**: bank, dice, turn number, per-player resources / dev cards / progress points + per-player stat one-liner.
3. **End-of-game summary**: `render_final_stats(GameResult)` shows buildings built per type, trades (bank vs p2p), dev cards, divine outcomes, rain/barn totals, resource flow.
4. **Wired into `Environment`** via optional `renderer` argument; silent by default for tournaments.

### Phase 2 — Stats accumulation
- Extended `PlayerStats`: `buildings_built` per type, `maintenance_failures`, `dev_cards_bought`, `bank_trades` / `player_trades`, `divine_outcomes` histogram, `rain_received` / `barn_received`.
- Wired counters into every resource flow: `_deduct_resources`, `_transfer_resources_between`, `produce_resources`, weather grants, build/trade/dev/divine executors, maintenance.
- `get_final_results` now returns the live stats objects accumulated during play.

### Phase 3 — Player-to-player trade
- `_get_p2p_trade_actions` generates bounded 1-for-1 / 2-for-1 swaps against each opponent.
- `execute_trade` p2p path: handles agent consent, sanity-checks responder affordability, executes the swap, bumps stats on both sides.
- Fixed the silent fall-through bug where p2p trades in MCTS rollouts (no agents) became bank trades.
- Shared `_evaluate_trade_for_responder` heuristic with two acceptance paths (weighted gain, surplus-for-deficit) + upkeep guards.

### Phase 4 — Upkeep-aware agents
- `civsim/upkeep.py` with shared helpers consumed by Greedy, MCTS, and the RL encoder.
- Greedy heavily penalizes EndTurn at risk, refuses water-draining dev cards, prioritizes maintenance card play.
- MCTS rollout policy refuses EndTurn when buildings would be lost.
- RL state vector includes water gap, cow gap, and buildings-at-risk count.

### Phase 5 — Balance pass (largest single set of changes)
The visualization + stats made one finding extremely visible: **every game ended at the turn cap with 0 progress points** because every settlement was being lost to upkeep. The structural problem was the game economy, not agent intelligence. Five sequential changes fixed it:

1. **Rain mechanic** (+33% per player-turn, +1 water for every active player, additive — not from bank). +2 starting water stockpile per player.
2. **Slower maintenance** (every 3rd of a player's own turns instead of every turn).
3. **Rebalanced board** (cow tiles 1 → 2, wood 4 → 3 to keep 19 tiles).
4. **Cow-only city upkeep** (was water + cow — made cities a double-bottleneck).
5. **Barn-day mechanic** (cow's mirror of rain, same 33% rate, additive).

Plus PP recalibration: **settlement = 2 PP, city = 4 PP** (was 1, 2). 5 settlements = 10 PP cleanly, cities now worth their resource investment.

After this pass, Greedy reliably builds cities (4-5/game), reaches 8+ PP in many games, and wins ~15% of the time at WIN_THRESHOLD = 10.

### Phase 6 — Target-aware Greedy
- Added `urgency = my_pp / WIN_THRESHOLD` scaling to all action scores.
- Cities ramp from 100 → 250 as urgency rises; roads decay from 30 → 9; dev cards de-prioritized near the win line.
- Trade scoring includes a "city ingredients" bonus (metal/wheat) at high urgency.
- Sustainability guards: refuse to upgrade to city without ≥2 cow buffer.

### Phase 7 — RL training
1. **Training driver** (`civsim/rl_train.py`) with three-phase curriculum:
   - Phase 1: train vs 2 RandomAgents (~300 episodes, ~80s) — converges to 99% win rate.
   - Phase 2: train vs 2 GreedyAgents (~200 episodes, ~45s) — converges to ~85% win rate during training.
   - Phase 3: snapshot-based self-play (~300 episodes, ~36s) — snapshot every 30 eps, pool capped at 5, 30% Greedy mix to prevent equilibrium collapse.
2. **30-game evaluation** (deterministic argmax) — **21/30 wins vs 2 Greedy = 70%**.
3. Total training time: ~3 minutes wall-clock.

### Phase 8 — MCTS rehabilitation
The visualization + stats showed MCTS-30 was producing 0 PP, 0 cities — worse than Random. Diagnostic + iterative fixes:

1. **Turn-transition drift** — MCTS had hand-rolled EndTurn handling that never picked up rain/barn/maintenance-interval changes. Fix: extract `state.end_player_turn` / `start_player_turn` so both Environment and MCTS use the same code path.
2. **Action space explosion** — at ~60 valid actions, vanilla MCTS at 50 sims is essentially random. Fix: prune to top-K=5 by Greedy score before tree expansion.
3. **Random rollouts** were uninformative. Fix: Greedy rollout policy.
4. **PUCT instead of UCB1** with priors from softmaxed Greedy scores.
5. **Virtual Q-init**: unvisited children start at prior estimate (instead of zero).
6. **All-in detection** + **decisive-move shortcut** for obvious moves.
7. **Rejection-aware shortcuts**: the shortcuts respect `_failed_this_turn` so MCTS doesn't loop on rejected trades.
8. **AlphaZero-lite experiment**: tried plugging the trained RL value head into MCTS's depth-limit evaluator and the RL policy head as PUCT priors. Tuned c_puct and policy temperature.

**MCTS journey final tally** (10-game samples for each step):

| Variant | Wins/10 | Avg PP | Notes |
|---|---|---|---|
| MCTS-30 (broken) | 0 | 0.0 | Turn drift broke rollouts |
| + Greedy rollouts | 0 | 0.0 | Still hopeless |
| + K=8 pruning | ~17% (1/6) | 4.3 | Viable |
| + PUCT + K=5 + virtualQ + shortcuts | 2/10 (20%) | 3.8 | Best heuristic-only result |
| + RL value head only | 1/10 (10%) | 2.2 | Surprising regression |
| + RL policy head only | 0/8 | 4.5 | No wins |
| + Policy + value (full AlphaZero-lite) | 0/8 | **7.8** | Reaches 8 PP reliably; doesn't convert |
| + Aggressive tuning (c_puct=0.5, temp=0.5) | 3/20 (**15%**) | 6.4 | Confirmed over 20 games |

Final MCTS settling at ~15% — **roughly Greedy parity, not better**. The "careful plateau" at 8 PP is a real phenomenon: MCTS reaches a winning position but second-guesses the winning move. Documented as a project finding.

---

## Game-balance constants (final, in `data_types.py`)

```python
WIN_THRESHOLD = 10
MAX_TURNS = 200
BANK_STARTING_SUPPLY = 19  # per resource type

SETTLEMENT_PP = 2
CITY_PP = 4

MAINTENANCE_TURN_INTERVAL = 3        # maintenance every Nth own-turn
STARTING_WATER_STOCKPILE = 2         # additive bonus per player post-draft
RAIN_PROBABILITY = 0.33              # per player-turn, +1 water to all active players
BARN_PROBABILITY = 0.33              # per player-turn, +1 cow to all active players

# Maintenance costs
SETTLEMENT: 1 water/period
CITY: 1 cow/period   (cow-only after balance pass)

# Build costs (unchanged)
ROAD: 1 wood + 1 stone
SETTLEMENT: 1 wood + 1 stone + 1 wheat + 1 cow
CITY: 3 metal + 2 wheat
```

Standard board: 3 wood / 3 stone / 3 metal / 4 wheat / 3 water / 2 cow / 1 desert (19 tiles).

---

## Project findings worth surfacing

1. **Game balance is the bottleneck for agent learning, not agent intelligence** — a pre-balance project produced uniformly 0-PP games regardless of agent type. The visualization made this obvious; the balance pass (rain, barn day, slower upkeep, cow-only cities, PP recalibration) made the game actually playable.

2. **Learned policies dominate fixed-budget tree search in this domain.** At 50 simulations across ~60 actions per turn, even a well-engineered MCTS (pruning, PUCT, RL-derived priors, decisive shortcuts) lands at Greedy parity. The trained RL policy beats both at 70% wins.

3. **Naive AlphaZero-lite doesn't work.** Plugging the value head from a separately-trained RL agent into MCTS *hurt* performance. Plugging the policy head as PUCT prior raised average PP from 3.8 → 7.8 but produced zero wins — a "careful plateau" where MCTS reaches winning positions but second-guesses the winning move. The fix is joint training (MCTS in the loop during RL training), which is the full AlphaZero recipe and a much larger engineering investment than half-measures.

4. **MCTS's structural mismatches with our environment** are mostly addressable but not at this compute budget:
   - Partial observability → wants ISMCTS / multiple determinizations
   - Heavy stochasticity (rain, barn day, dice) → wants chance nodes
   - Sparse PP signal at 60-action depth → wants a learned value head — and the value head only works if co-trained with the search

5. **Self-play matters.** RL training vs Greedy alone reached 60% win rate; adding 300 episodes of snapshot self-play pushed it to 70% with games ending faster and at higher PP averages. Total cost: 36 seconds of compute.

---

## Future directions (if extending the project)

- **Full AlphaZero**: train the RL network with MCTS in the loop (MCTS provides training targets, policy provides priors, value head provides bootstrapping). Likely the only path to MCTS > Greedy in this environment.
- **ISMCTS / multiple determinizations** for the partial-info problem in MCTS.
- **Larger network + more training** to push RL above 70%.
- **Cross-evaluation matrix**: all-pairs tournament across Random / Greedy / MCTS / RL with confidence intervals.
- **Action-space refactor**: the p2p trade variants currently dominate the action space. A categorical "trade with player X for resource Y" with continuous quantity might be cleaner for both MCTS and RL.
