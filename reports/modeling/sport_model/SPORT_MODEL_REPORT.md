# Modello predittivo sport-only

Il candidato usa esclusivamente feature sportive pre-partita. Le quote closing compaiono solo come benchmark esterno.

## Confronto walk-forward

| Modello | Match OOS | Log Loss | Brier | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|
| sport_gradient_boosting | 10707 | 1.0065 | 0.6000 | 51.84% | 0.0358 |
| sport_features | 10707 | 1.0227 | 0.6044 | 51.97% | 0.0495 |
| market_closing | 10707 | 0.9691 | 0.5758 | 54.00% | 0.0184 |

## Evidenza statistica

Differenza Log Loss candidato − logistica: -0.0162 (IC 95% -0.0234, -0.0091).
Stagioni vinte: 3/6; richieste: 4.
Verdetto di promozione: NON PROMOSSO.

## Stabilità stagionale

| Stagione | Log Loss candidato | Log Loss logistica | Delta |
|---|---:|---:|---:|
| 1920 | 1.0543 | 1.1578 | -0.1035 |
| 2021 | 1.0118 | 1.0153 | -0.0036 |
| 2122 | 1.0081 | 1.0022 | +0.0059 |
| 2223 | 0.9929 | 1.0074 | -0.0145 |
| 2324 | 0.9868 | 0.9735 | +0.0134 |
| 2425 | 0.9860 | 0.9837 | +0.0023 |

## Segmenti più difficili

| Dimensione | Segmento | Match | Log Loss |
|---|---|---:|---:|
| league | D1 | 1836 | 1.0233 |
| result | D | 2700 | 1.3722 |
| experience | 0-4 | 227 | 1.0186 |
| confidence | (-0.001, 0.4] | 1778 | 1.0898 |

## Garanzie

- Ogni stagione è prevista usando soltanto stagioni precedenti.
- Le feature sono allowlistate; quote e target finali non entrano nel modello.
- La calibrazione usa l’ultima stagione interna al training, mai il test.
- Il modello finale è addestrato su tutto lo storico disponibile.
