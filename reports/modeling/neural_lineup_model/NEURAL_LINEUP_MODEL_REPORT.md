# Encoder neurale condiviso + pooling

Variante sperimentale: nessuna promozione automatica.

Feature store temporale: 108 feature individuali con titolari e panchina confermata in pool separati. Valori reali, fallback e copertura sono descritti in `neural_feature_store_audit.json`.

| Modello | Match | Log Loss | Brier | RPS | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|---:|
| dixon_coles_confirmed_lineup_pooling | 24950 | 0.9905 | 0.5905 | 0.2017 | 52.23% | 0.0281 |
| dixon_coles_gradient_boosting | 24950 | 0.9886 | 0.5893 | 0.2016 | 52.28% | 0.0258 |
| dixon_coles_shared_encoder_pooling | 24950 | 0.9894 | 0.5897 | 0.2014 | 52.29% | 0.0330 |
| dixon_coles_without_confirmed_lineup | 24950 | 0.9894 | 0.5899 | 0.2018 | 52.29% | 0.0261 |

## Bootstrap appaiato

- Contro dixon_coles: Δ Log Loss 0.00002, IC 95% [-0.00123, 0.00132].
- Contro linear_pooling: Δ Log Loss -0.00109, IC 95% [-0.00229, 0.00011].
- Contro official: Δ Log Loss 0.00075, IC 95% [-0.00062, 0.00214].
