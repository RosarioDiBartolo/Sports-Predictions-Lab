# Edge Discovery Report

Verdetto: **NON PROMOSSO**.

## Regola congelata

- Confidenza minima: 65.0%
- Edge minimo sul mercato: 1.0%
- Differenza Elo assoluta minima: 0
- Esiti ammessi: A
- Esperienza minima richiesta: False
- Regole valutate esclusivamente nel discovery: 2010

## Risultati

| Periodo | Puntate | ROI | IC 95% | Delta Log Loss | Delta Brier |
|---|---:|---:|---:|---:|---:|
| discovery | 204 | 9.10% | [-2.18%, 20.20%] | +0.0063 | +0.0013 |
| holdout | 65 | -23.43% | [-42.71%, -3.94%] | +0.0730 | +0.0513 |

## Gate di promozione

- ROI holdout con limite inferiore IC 95% > 0: False
- Modello migliore del mercato su Log Loss o Brier: False
- ROI positivo in ogni stagione holdout: False

## Garanzie

- La griglia e la regola sono selezionate senza osservare l’holdout.
- La regola congelata viene valutata una sola volta sull’holdout.
- Una sola puntata argmax per partita, stake fisso.
- Quote Market Average closing; nessuna scelta ex post del bookmaker.
- Il closing è un benchmark di ricerca, non garantisce eseguibilità live.
