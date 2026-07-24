# Baseline walk-forward

Ogni stagione è valutata usando esclusivamente stagioni precedenti. Le quote di mercato sono closing medie private del margine.

| Modello | Log Loss | Brier | Accuracy | ECE |
|---|---:|---:|---:|---:|
| market_closing | 0.9691 | 0.5758 | 54.00% | 0.0184 |
| elo | 0.9967 | 0.5946 | 52.19% | 0.0232 |
| sport_features | 1.0174 | 0.6037 | 51.80% | 0.0489 |
| historical_frequency | 1.0759 | 0.6514 | 42.93% | 0.0170 |
