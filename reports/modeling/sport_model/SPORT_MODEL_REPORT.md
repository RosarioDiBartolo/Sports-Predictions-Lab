# Modello predittivo sport-only

Il candidato usa esclusivamente feature sportive pre-partita. Le quote closing compaiono solo come benchmark esterno.

## Confronto walk-forward

| Modello | Match OOS | Log Loss | Brier | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|
| sport_gradient_boosting | 10707 | 1.0126 | 0.6047 | 50.62% | 0.0294 |
| sport_features | 10707 | 1.0163 | 0.6029 | 51.91% | 0.0464 |
| market_closing | 10707 | 0.9691 | 0.5758 | 54.00% | 0.0184 |

## Garanzie

- Ogni stagione è prevista usando soltanto stagioni precedenti.
- Le feature sono allowlistate; quote e target finali non entrano nel modello.
- La calibrazione usa l’ultima stagione interna al training, mai il test.
- Il modello finale è addestrato su tutto lo storico disponibile.
