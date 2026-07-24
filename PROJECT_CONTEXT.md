> **DEPRECATED — historical snapshot.** The maintained source of truth is
> `docs-app/src/docs.ts`, rendered by the living documentation webapp.

# Sports Predictions Lab — Project Context

## Project Goal

The goal is **NOT** to build an AI that predicts sports matches immediately.

The first objective is to build a **research platform** capable of measuring how accurately bookmakers estimate real-world probabilities.

The long-term objective is to create a modular platform that:

* evaluates bookmaker accuracy;
* identifies market biases;
* creates research datasets;
* serves as the foundation for future machine learning and AI models.

The project should be designed as a long-term research framework, not as a single prediction script.

---

# Overall Architecture

The architecture follows a layered approach.

```text
Raw Providers
        ↓
Raw Dataset
        ↓
Cleaning
        ↓
Normalization
        ↓
Master Database
        ↓
Feature Engineering
        ↓
Analytics Dataset
        ↓
Reports & Dashboards
        ↓
Machine Learning Dataset
        ↓
Prediction Models
```

Each layer has a single responsibility.

---

# Current Status

The project is already operational.

Current implementation includes:

* Football-Data provider
* Data download
* Data cleaning
* Odds normalization
* Implied probabilities
* Calibration analysis
* Log Loss
* Brier Score
* Calibration Curve
* CLI
* Tests
* Jupyter notebooks

Current metrics (Serie A 2024/25):

* 380 matches
* Accuracy ≈54%
* Log Loss ≈0.95
* Brier Score ≈0.57
* Calibration Error ≈2.7%

---

# Philosophy

Everything must be modular.

Nothing should depend on a specific provider.

Nothing should depend on a specific bookmaker.

Nothing should depend on football only.

Every component should be replaceable.

The system should eventually support multiple sports.

---

# Data Layers

## Layer 1 — Raw Dataset

Purpose:

Store data exactly as received from providers.

No transformations.

No calculations.

No cleaning.

Example:

Football-Data CSV

API-Football JSON

Sportmonks JSON

---

## Layer 2 — Normalized Database

Purpose:

Transform provider-specific data into a common schema.

This is the source of truth.

Tables include:

matches

teams

leagues

bookmakers

providers

provider_match_mapping

odds

results

Every match has an internal MatchID independent from external providers.

---

# Odds Table

The odds table stores every individual betting quote.

Each row represents ONE selection.

Example:

| Match       | Bookmaker | Market | Selection | Odds |
| ----------- | --------- | ------ | --------- | ---- |
| Inter-Milan | Bet365    | 1X2    | Home      | 2.05 |
| Inter-Milan | Bet365    | 1X2    | Draw      | 3.40 |
| Inter-Milan | Bet365    | 1X2    | Away      | 3.60 |

Fields:

* odds_id
* match_id
* bookmaker_id
* provider_id
* market
* selection
* decimal_odds
* implied_probability_raw
* implied_probability
* overround
* margin
* timestamp
* snapshot_number
* is_opening
* is_closing
* created_at

A bookmaker can have multiple snapshots for the same match.

The system must support:

* opening odds
* closing odds
* intermediate snapshots

---

# Layer 3 — Analytics Dataset

This dataset is generated automatically from the normalized database.

Purpose:

Create research-ready observations.

Example fields:

match_id

bookmaker

market

selection

probability

margin

favorite

favorite_won

prediction_correct

logloss_contribution

brier_contribution

calibration_bin

season

league

home_team

away_team

home_goals

away_goals

This dataset is optimized for:

* research
* dashboards
* reports
* statistical analysis

It is NOT the source of truth.

---

# Layer 4 — Machine Learning Dataset

This layer does not exist yet.

It will be generated from Analytics Dataset.

Purpose:

Train ML models.

Example features:

elo_difference

xg_difference

team_form

rest_days

player_ratings

injuries

market_probabilities

opening_vs_closing_difference

weather

travel_distance

referee

Target:

home_win

draw

away_win

over_under

etc.

---

# Analytics

The project should compute:

Accuracy

Log Loss

Brier Score

Calibration Error

Expected Calibration Error

Overround

Margin

Calibration Curve

Reliability Diagram

Probability Histograms

Sharpness

Bookmaker comparison

League comparison

Opening vs Closing comparison

ROI simulations

Favorite-longshot bias

---

# Future Database

Future tables should include:

players

player_match_stats

lineups

injuries

transfers

player_ratings

team_ratings

elo_history

team_form

weather

referees

schedule

travel_distance

No implementation yet.

Only prepare architecture.

---

# Plugin Architecture

Every provider must implement the same interface.

Examples:

FootballDataProvider

ApiFootballProvider

SportmonksProvider

StatsBombProvider

OddsApiProvider

Providers should be interchangeable.

---

# Design Principles

* Modular architecture
* Clean code
* SOLID principles
* Strong typing
* Unit tests
* Config-driven
* No hardcoded logic
* Extensible
* Reproducible research

---

# Long-Term Vision

The project will eventually become a complete Sports Research Platform capable of:

* evaluating bookmakers;
* measuring market efficiency;
* generating research datasets;
* supporting machine learning;
* supporting AI agents;
* supporting multiple sports;
* identifying value betting opportunities;
* studying market behavior over time.

Machine learning is NOT the first goal.

Research-quality data and architecture come first.
