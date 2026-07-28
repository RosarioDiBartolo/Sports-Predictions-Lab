# Dixon-Coles + correzione lineup confermata

Il candidato non sostituisce il modello ufficiale senza promotion gate OOS.

## Valutazione walk-forward

| Modello | Match | Log Loss | Brier | RPS | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|---:|
| dixon_coles_confirmed_lineup_pooling | 24950 | 0.9905 | 0.5905 | 0.2017 | 52.23% | 0.0281 |
| dixon_coles_without_confirmed_lineup | 24950 | 0.9894 | 0.5899 | 0.2018 | 52.29% | 0.0261 |
| dixon_coles_gradient_boosting | 24950 | 0.9887 | 0.5893 | 0.2016 | 52.28% | 0.0256 |

## Promotion gate

- IC 95% Log Loss favorevole: False.
- Stagioni vinte: 6/14.
- Brier non peggiore: False.
- RPS non peggiore: True.
- ECE non peggiore: False.
- Verdetto: NON PROMOSSO.
