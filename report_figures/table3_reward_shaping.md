# Table 3: Reward-shaping experiments (90-game eval each)

| Config | Win bonus | Per-action penalty | Total wins | Avg PP | Diagnosis |
|---|---:|---:|---:|---:|---|
| Baseline (kept) | +10 / -5 | 0 | ~46% | 7.0 | Sweet spot — stable training |
| Stronger bonus + penalty | +100 / -50 | -0.1 | 33% | 0.1 | Penalty dominated; learned 'do nothing' |
| Stronger bonus only | +100 / -50 | 0 | 53% | 7.1 | Higher variance; gradient noise |
| Imitation pretrain (BC) | +10 / -5 | 0 | 37% | 0.2 | BC loss = random baseline; pulled init |
