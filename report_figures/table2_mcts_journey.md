# Table 2: MCTS iteration sequence

| MCTS variant | Threshold wins / 10 | Avg PP | Key finding |
|---|---:|---:|---|
| Vanilla MCTS-30 | 0 | 0.0 | Hand-rolled turn transitions drifted from real game |
| + shared turn helpers | 0 | 0.0 | Rollouts correct; uniform random still uninformative |
| + Greedy rollout policy | 0 | 0.0 | Better signal; action space (~60) still drowns selection |
| + K=8 pruning by Greedy score | 1 | 4.3 | Viable: builds cities, occasional wins |
| + K=5 + PUCT priors | 2 | 3.8 | Aggression trades reliability for wins |
| + virtualQ + all-in + decisive | 2 | 3.0 | Plateau confirmed at this sim count |
| + RL value head | 1 | 2.2 | Sigmoid-squashed head misled rollouts; regressed |
| + RL policy head as PUCT prior | 0 | 7.8 | Reaches 8 PP; 'careful plateau' |
| + tuned c_puct=0.5, temp=0.5 | 3* | 5.0 | Aggression breaks plateau (* 20-game sample) |

Sample sizes are small (6-20 games per row) due to per-sim cost; trends are directional, not statistical.
