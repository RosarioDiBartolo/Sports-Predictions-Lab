# Candidato ibrido Dixon–Coles + gradient boosting

Il modello ufficiale resta `sport_gradient_boosting`. Il candidato è promuovibile solo dal gate walk-forward.

## Confronto fuori campione

| Modello | Match | Log Loss | Brier | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|
| dixon_coles_gradient_boosting | 760 | 1.0129 | 0.6060 | 51.05% | 0.0382 |
| sport_gradient_boosting | 760 | 1.0854 | 0.6458 | 48.95% | 0.0902 |
| market_closing | 760 | 0.9588 | 0.5716 | 54.87% | 0.0434 |

## Promotion gate

- IC 95% favorevole: True.
- Stagioni vinte: 2/2 (richieste 2).
- Brier non peggiore: True.
- ECE non peggiore: True.
- Verdetto: PROMOSSO.

Le quote closing sono utilizzate esclusivamente come benchmark.

## Ablazione feature giocatore

| Variante | Match | Log Loss | Brier | ECE |
|---|---:|---:|---:|---:|
| dixon_coles_gradient_boosting | 760 | 1.0129 | 0.6060 | 0.0382 |
| dixon_coles_gradient_boosting_without_players | 760 | 1.0154 | 0.6074 | 0.0508 |

Differenza Log Loss con giocatori - senza giocatori: -0.0025 (IC 95% -0.0101, 0.0050).

## Stabilità stagionale feature giocatore

| Stagione | Log Loss con | Log Loss senza | Delta | Brier con | Brier senza | ECE con | ECE senza |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2324 | 1.0373 | 1.0413 | -0.0040 | 0.6183 | 0.6216 | 0.0326 | 0.0497 |
| 2425 | 0.9885 | 0.9895 | -0.0010 | 0.5936 | 0.5932 | 0.0437 | 0.0520 |
