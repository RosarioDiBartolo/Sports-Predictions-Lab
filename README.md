# Sports Predictions Lab

Laboratorio di ricerca per costruire e valutare probabilità calcistiche
pre-partita senza leakage temporale.

L'obiettivo corrente è un modello **post-lineup e pre-kickoff**: riceve le
formazioni ufficiali, rappresenta i 22 titolari attraverso feature storiche
disponibili prima della partita e produce `P(1)`, `P(X)` e `P(2)`.

Il progetto dispone già di un dataset validato di **26.670 partite
training-ready** e **731.044 osservazioni giocatore-partita**. Il gate corrente
ha **zero partite in quarantena**; eventuali futuri casi incompleti o ambigui
non verranno forzati e conserveranno una motivazione verificabile.

## Documentazione

La fonte autorevole per architettura, pipeline, contratti, feature, comandi,
artefatti, garanzie temporali e decisioni è
[`docs/README.md`](docs/README.md).

Il README principale è una porta d'ingresso; in caso di divergenza prevalgono
i documenti Markdown indicizzati dalla documentazione autorevole.

## Installazione

Requisiti: Python 3.10 o successivo e PowerShell.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Le chiavi opzionali vanno in `.env` o nelle variabili d'ambiente e non devono
essere versionate. Per il fallback API-Football:

```text
API_FOOTBALL_KEY=...
```

## Avvio rapido

Costruisci il dataset giocatori senza effettuare training:

```powershell
odds-lab players dataset
```

Esegui il collector multi-provider e ricostruisci il dataset:

```powershell
odds-lab players collect --request-budget 100
```

Il collector usa cache e stato persistente, rispetta il budget, riprende il
lavoro interrotto, riconcilia squadre e giocatori e mette in quarantena i casi
ambigui. Gli output principali sono:

- `data/processed/player_training_ready.csv`
- `reports/player_data/dataset/coverage.json`
- `reports/player_data/dataset/quarantine.jsonl`
- `reports/player_data/dataset/PLAYER_DATASET_REPORT.md`

Costruisci l'intera pipeline canonica:

```powershell
odds-lab run
```

Allena l'unico modello operativo, il neurale basato sui giocatori:

```powershell
odds-lab model train --embedding-dim 32 --epochs 80
```

Per tutti i target, i contratti di input e gli artefatti prodotti, consulta la
documentazione vivente invece di affidarti a esempi copiati.

## Qualità e riproducibilità

```powershell
pytest
ruff check src tests
mypy src
```

Le feature pre-match vengono calcolate prima di osservare risultato,
prestazioni e lineup storicizzata della partita corrente. La validazione dei
modelli è temporale e gli artefatti conservano provenienza e diagnostica.

## Struttura essenziale

```text
src/football_odds/   libreria e CLI
tests/               test automatici
data/raw/            copie delle sorgenti
data/cache/          risposte riutilizzabili dei provider
data/processed/      dataset derivati
reports/             copertura, quarantena, metriche e modelli
docs/                documentazione Markdown autorevole
```

## Contribuire

Prima di modificare una pipeline o un contratto, leggi `AGENTS.md`. Ogni
variazione a CLI, stage, feature, output, dipendenze, garanzie temporali o
validazione deve aggiornare nello stesso cambiamento il documento pertinente
indicato da `docs/README.md`; il lavoro è completo solo quando documentazione e
test descrivono il comportamento reale.

Non committare chiavi API, database locali, cache o artefatti generati.
