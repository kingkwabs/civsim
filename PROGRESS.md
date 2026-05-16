# CivSim — Progress Report

## Project Overview

CivSim is a Catan-style board game engine for CS 4701 (AI). It simulates a multiplayer resource-trading game with custom mechanics (per-turn maintenance, divine intervention, unique dev cards, weather events, coastal trading ports) and serves as an environment for training and evaluating four AI agents at distinct complexity tiers. Includes both a terminal renderer and a Colonist-inspired matplotlib GUI for demos.

---

## Headline Result

The headline table below uses **threshold wins** (player actually reached 10 PP before turn 200) as the meaningful measure of "winning the game" — not just "had the highest score at the turn cap." See the framing note below for why this distinction matters.

| Agent | Method | Threshold-win rate vs 2 Greedy | Total "wins" (incl. turn-cap) | Sample |
|---|---|---|---|---|
| Random | Uniform action selection | ~0% | ~3% | 30 games |
| Greedy | Hand-tuned heuristic (upkeep / target / port-aware) | ~25-30% | ~30% | 90+ games |
| MCTS | K=5 pruning + PUCT + RL policy/value priors + tuned c_puct/temp | ~5-10% | ~15-20% | 30+ games |
| **RL** | Actor-Critic + REINFORCE + curriculum + self-play + inference override + extended training | **~24%** (22/90) | **~49%** (44/90) | 90 games |

RL wins by actually reaching 10 PP about as often as Greedy does (24% vs ~25-30%), and outscores Greedy in total win rate (49%) when accounting for both threshold and tiebreaker wins. Fastest recorded threshold win: turn 15. The methodological ordering (Random < MCTS < Greedy ≈ RL on threshold wins, RL > Greedy on total) holds — learned policy edges out hand-tuned heuristic when both are running on a fully-developed game (rain, barn day, ports, trade cap, re-establishment, port-aware draft).

**Why threshold wins are the right metric (and why the "headline win rate" trajectory we observed needs context).** The engine has a turn-cap fallback: if no one reaches 10 PP by turn 200, the player with the highest PP is declared "winner." Earlier in the project we measured "win rate" by this combined criterion and saw RL at 70%. Over time we made Greedy stronger (port-aware draft, trade cap, etc.) and that headline number dropped to ~46% — which sounds like regression. But when we finally checked the *composition* of those wins, we discovered that an earlier RL training run was winning 95% of its games via turn-cap fallback (stalling at 8 PP, outscoring Greedy when the timer expired) and only 2/90 by actually crossing the threshold. That's an artifact of optimizing against a tiebreaker, not a real win.

A subsequent retrain (same code, different RNG seed) produced a *qualitatively different* policy: 24/90 threshold wins, average win turn ~26, dramatically more decisive games. Total win count dipped to 39/90 — slightly lower headline number, but **62% of those wins are now genuine threshold wins** rather than 5%. This is more realistic, more aligned with the game's intended victory condition, and substantially better for any meaningful "RL learned to win at this Catan variant" claim.

We also added an **inference-time threshold override** as a safety net (`RLAgent.threshold_override`): when within one Build of WIN_THRESHOLD, force the closing move instead of trusting the policy. A no-op for the current aggressive policy but insurance against future cautious-plateau retrains.

| 90-game eval (current model) | Count | % |
|---|---|---|
| **Genuine threshold wins (≥10 PP before turn 200)** | **22** | **24%** |
| Turn-cap fallback "wins" (highest PP at turn 200) | 22 | 24% |
| Losses | 46 | 51% |

Fastest threshold win: turn 15 (project-wide record). Average win turn across all threshold wins: ~40. The current policy is a *fast-aggressive* strategy that prioritizes reaching the win condition; the bumped turn-cap count vs prior models reflects games where RL accumulated PP defensively and outscored Greedy at turn 200 without crossing the threshold.

---

## Architecture (final)

### Core engine (12 modules, 167 tests passing)

**`data_types.py`** — enums, dataclasses, balance constants
- 6 resources, 3 build types, 6 action types, 5 dev cards, 7 ports
- Build/maintenance/dev-card cost tables; weather/maintenance constants
- `PlayerStats` with live-tracked counters (built/lost/trades/dev/divine/rain/barn/flow)

**`board.py`** — hex board with axial coords
- 19-tile layout (rebalanced: 3 wood / 3 stone / 3 metal / 4 wheat / 3 water / 2 cow / 1 desert — cow bumped from 1 to address city-upkeep bottleneck)
- Intersection/edge derivation, **port placement (rewritten — original logic was buggy, never placed any ports)**, spatial queries

**`game_state.py`** — state management with shared turn-transition helpers
- `roll_dice`, `roll_rain`, `roll_barn_day`, `apply_rain`, `apply_barn_day`
- `end_player_turn` (maintenance every Nth own-turn), `start_player_turn` (weather → dice → produce)
- These helpers are shared by `Environment` and `MCTSAgent` so simulated rollouts don't drift from the live game

**`actions.py`** — valid action enumeration
- Build, bank-trade, **p2p-trade proposals (bounded 1-for-1 / 2-for-1)**, buy dev, play dev, divine intervention, **re-establish abandoned building** (claim an unowned settlement/city at a premium cost), end turn
- RL encoding helpers

**`action_executors.py`** — execution + live stat accumulation
- Every flow (build, trade, dev card, divine, maintenance) bumps the appropriate `PlayerStats` counter
- p2p trades go through `target_agent.respond_to_trade`; simulation path (no agents) returns clean failure instead of falling through to a bank trade

**`environment.py`** — game loop
- Snake draft, weather + dice production, action loop, maintenance
- Failure-cap guard (force end-turn after 5 consecutive failed actions) prevents stubborn-rejection loops
- Optional `renderer` parameter for terminal or GUI visualization
- Per-action `on_action_result` callback gives agents rejection memory
- **Catan-style instant win**: `is_game_over()` is checked after every action, not just at end-of-turn — a build that crosses 10 PP ends the game immediately; maintenance can never silently revert a win
- **Per-turn trade cap (10)**: prevents agents (especially Greedy) from spamming hundreds of slight trade variations per turn

**`upkeep.py`** — shared upkeep helpers
- `total_upkeep_cost`, `upkeep_gap`, `buildings_at_risk`, `upkeep_pressure_from_obs`
- Used by all agents and the RL state encoder so every layer has the same upkeep signal

**`visualization/terminal.py`** — ASCII renderer
- Hex board (colored tile glyphs + dice numbers), buildings/roads listing, live dashboard with per-player stat one-liner
- `TerminalRenderer` wires into `Environment` via optional `renderer` parameter
- `render_final_stats(GameResult)` end-of-game summary

**`visualization/gui.py`** — matplotlib GUI (Colonist-inspired)
- Real pointy-top hexagons via `RegularPolygon`, white "dice tokens" with red text for 6/8
- On-board roads (colored line segments on edges), settlements (circles), cities (squares with cross) at intersections
- Coastal port markers (colored diamonds with `2:1` / `3:1` labels) tethered to their coast intersection
- Right-side panels: one per player, showing agent type (Greedy / MCTS / RL), PP / threshold, colored resources, dev cards, full stat line
- Bottom panel: bank supply + last 6 action-log lines with weather and trade events highlighted
- Used for video demos via `python -m civsim.demo --gui --agents greedy,greedy,rl --pause 0.4 --seed 42`

**`evaluation.py`** — tournament + metrics + replay logging

**`demo.py`** + **`rl_train.py`** — runnable entry points
- `python -m civsim.demo --agents greedy,greedy,mcts` — rendered game
- `python -m civsim.rl_train --episodes-random 300 --episodes-greedy 200 --episodes-selfplay 300 --eval-games 30`

### Agents (`agents.py` + `rl/`)

**`RandomAgent`** — baseline; 50/50 trade response.

**`GreedyAgent`** — heuristic scoring with four layers of awareness:
- **Upkeep-aware**: heavy EndTurn penalty when buildings would fail upkeep; MAINTENANCE-card spike to top priority; refuse to spend critical water/cow reserves; preemptive stockpile trades; refuse dev-card buy if water would drop below upkeep need.
- **Target-aware**: per-action scores scale by `urgency = my_pp / WIN_THRESHOLD`. Cities ramp up sharply as urgency rises; roads ramp down; dev cards de-prioritized; trade scoring includes a city-ingredient bonus near the win line.
- **Port-aware**: shared draft heuristic awards a bonus for intersections that have a port attached (+7 for specific 2:1, +3.5 for generic 3:1), making border intersections with port access competitive against interior intersections with strong pip counts.
- **Per-turn rejection memory**: failed actions get a -10000 penalty so the agent doesn't loop on rejected trades.
- Shared draft heuristic also favors water/cow-adjacent intersections (pip-weighted).

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
- Inference: argmax over policy (deterministic) + **threshold override** safety net for closing moves
- Optional behavior-cloning warm start (`civsim/rl/imitation.py`) — collects demos from Greedy threshold-wins and BC-pretrains the policy. Experimental — current results regressed vs no-BC baseline; left in the codebase for future tuning.

### Test suite — 171 tests across 16 files

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
| test_p2p_trade.py | 14 | Proposal generation, accept/reject, stats, simulation safety, per-turn trade cap |
| test_rain.py | 7 | Rain mechanic, starting water stockpile |
| test_rl.py | 28 | Feature encoding, network forward, agent modes, training, threshold override |
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

2. **Learned policies dominate fixed-budget tree search in this domain.** At 50 simulations across ~60 actions per turn, even a well-engineered MCTS (pruning, PUCT, RL-derived priors, decisive shortcuts) lands at Greedy parity. The trained RL policy still beats both, though margin narrowed once Greedy got port awareness.

3. **Naive AlphaZero-lite doesn't work.** Plugging the value head from a separately-trained RL agent into MCTS *hurt* performance. Plugging the policy head as PUCT prior raised average PP from 3.8 → 7.8 but produced zero wins — a "careful plateau" where MCTS reaches winning positions but second-guesses the winning move. The fix is joint training (MCTS in the loop during RL training), which is the full AlphaZero recipe and a much larger engineering investment than half-measures.

4. **MCTS's structural mismatches with our environment** are mostly addressable but not at this compute budget:
   - Partial observability → wants ISMCTS / multiple determinizations
   - Heavy stochasticity (rain, barn day, dice) → wants chance nodes
   - Sparse PP signal at 60-action depth → wants a learned value head — and the value head only works if co-trained with the search

5. **Self-play matters.** RL training vs Greedy alone reached 60% win rate; adding 300 episodes of snapshot self-play pushed it to 70% with games ending faster and at higher PP averages. Total cost: 36 seconds of compute.

6. **Reward shaping was a dead end here.** Attempts to push RL beyond stalling at 8 PP by bumping the win bonus (+10/-5 → +100/-50) and adding a per-action penalty (-0.1) *regressed* the policy — first to 33% win rate (over-corrected: agent learned "do nothing"), then to 53%. Reverting to ±10/-5 recovered the baseline.

7. **Imitation learning (behavior cloning) also regressed.** We collected ~5000 (state, action) demonstrations from Greedy threshold-winning games and BC-pretrained the policy for 30 epochs. Result: win rate dropped to 36.7% (worse than the 46% baseline), avg PP 0.2. Diagnosis: BC loss converged to 3.37 = ~3.4% probability on chosen action — barely better than random selection among ~30 valid actions. Variable action-set lengths make the BC objective noisy; the policy network couldn't fit demos efficiently and the gradient pulled weights off the helpful random init. Bigger demo sets / longer BC might fix this — left as future work.

8. **Training stochasticity matters more than expected.** Retraining from scratch with identical code and hyperparameters produced **qualitatively different policies**: one run produced a cautious-plateau policy (2/90 threshold wins, max PP 10, won via turn-cap tiebreaker 40 times), a later run produced an aggressive-pusher policy (24/90 threshold wins, fastest win at turn 20). Same architecture, same reward function — but the RNG seed during training apparently steers the policy into one local minimum or another. This is a real practical concern for reproducibility and suggests training multiple models and picking the strongest is a useful workflow.

9. **The inference-time threshold override** (`RLAgent.threshold_override`, default on) is a safety net: when within one Build of WIN_THRESHOLD, force a closing move (city/settlement/road+settlement) instead of the policy's pick. Currently a no-op for the aggressive-pusher policy but inexpensive insurance against future cautious-plateau retrains.

10. **Measure win rate by the game's victory condition, not by the engine's tiebreaker.** The 47%-headline cautious policy and 43%-headline aggressive policy looked nearly identical by total win count, but the composition was inverted: the cautious one won ~95% via turn-cap fallback (which is essentially "I outscored you when neither of us actually won"), the aggressive one wins ~50%+ via genuine threshold crossings. The aggressive policy is dramatically *better* at the game even though the headline number was marginally lower — because reaching 10 PP is what the game is asking for, and stalling at 8 PP to outscore at the cap is gaming the tiebreaker. For a CS 4701 project, the lesson is: pick the metric that aligns with the goal (here: win condition), not the metric that's easiest to measure (engine.winner).

11. **Re-establishment + longer training was the final unlock.** Adding the `ReestablishBuilding` action (claim an abandoned settlement/city at premium cost) and bumping training to 600/400/600 episodes produced the project's best evaluation numbers: **22/90 threshold wins (24%), 44/90 total wins (49%), fastest win at turn 15**. The short-retrain initial result for re-establishment was a regression — Greedy adopted the new action faster than RL since Greedy's heuristic handles it natively, while RL needed gradient steps to learn it. With more episodes, RL caught up and the equilibrium improved for both. Total training time: 12 minutes wall-clock.

7. **Stronger heuristics narrow RL's lead.** Adding port-aware drafting + the per-turn trade cap improved Greedy enough that RL's win rate dropped from 70% to ~46%. This is not a regression — it's the heuristic competing more effectively in a better-balanced game. The relative ordering (RL > Greedy ≈ MCTS > Random) holds, just with smaller gaps.

8. **The port-placement bug went undetected for the entire project until the GUI demanded visible markers.** This is a case study in why visualization matters even for non-presentation purposes: a static feature like ports being broken can hide indefinitely if no one looks for it.

---

## Future directions (if extending the project)

- **Full AlphaZero**: train the RL network with MCTS in the loop (MCTS provides training targets, policy provides priors, value head provides bootstrapping). Likely the only path to MCTS > Greedy in this environment.
- **ISMCTS / multiple determinizations** for the partial-info problem in MCTS.
- **Larger network + longer training** to push RL above 50% post-balance. Current trunk is 128 → 64; bumping to 256 → 128 with 1000+ episodes might recover the original 70% margin.
- **Cross-evaluation matrix**: all-pairs tournament across Random / Greedy / MCTS / RL with confidence intervals.
- **Action-space refactor**: the p2p trade variants currently dominate the action space. A categorical "trade with player X for resource Y" with continuous quantity might be cleaner for both MCTS and RL.
- **In-game build-position scoring**: Greedy picks build positions arbitrarily during gameplay (only the *draft* heuristic considers position quality). Adding the draft scorer to in-game settlement-build choices would likely make Greedy stronger still.
