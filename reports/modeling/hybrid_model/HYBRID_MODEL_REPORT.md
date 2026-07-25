# Candidato ibrido Dixon–Coles + gradient boosting

Il modello ufficiale resta `sport_gradient_boosting`. Il candidato è promuovibile solo dal gate walk-forward.

## Confronto fuori campione

| Modello | Match | Log Loss | Brier | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|
| dixon_coles_gradient_boosting | 10707 | 0.9924 | 0.5916 | 52.55% | 0.0264 |
| sport_gradient_boosting | 10707 | 1.0065 | 0.6000 | 51.84% | 0.0358 |
| market_closing | 10707 | 0.9691 | 0.5758 | 54.00% | 0.0184 |

## Promotion gate

- IC 95% favorevole: True.
- Stagioni vinte: 6/6 (richieste 4).
- Brier non peggiore: True.
- ECE non peggiore: True.
- Verdetto: PROMOSSO.

Le quote closing sono utilizzate esclusivamente come benchmark.
