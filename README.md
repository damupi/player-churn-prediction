# Player churn prediction

Predicting which new players of a mobile game will stop playing, from their first 24 hours of event data.

> **Status: work in progress.** The extraction query is written; the modelling and the A/B test notebook are being built. Results will be published here only once they have been run and checked.

## The question

A player installs the game and plays for a day. Will they come back?

If we can tell early, the team can act early: a better onboarding level, a nudge, a bonus. This project builds that early-warning model and measures how good it really is.

## Data

Google's public Firebase sample for the mobile puzzle game **Flood It!**, exported to BigQuery in the GA4/Firebase event schema:

- `firebase-public-project.analytics_153293282.events_*`
- about 15,000 players and 5.7 million events, June to October 2018

It is a free, public dataset; anyone can query it from the [BigQuery sandbox](https://cloud.google.com/bigquery/docs/sandbox) without a billing account.

## Approach

1. **Extract (BigQuery SQL)** - `sql/01_player_features.sql`
   - One row per player.
   - **Label:** a player has *churned* if they show no activity after their first 24 hours. Players who leave within 10 minutes of their first event (bounces) are excluded: they never really started.
   - **Features:** behaviour in the first 24 hours only (sessions, engagement time, levels started, completed, failed and reset, scores posted, screens viewed) plus context (country, operating system, device language).
   - Using only first-day features avoids leakage: the model never sees what happened after the moment it is meant to predict.
2. **Model (DuckDB + Python)** - the extracted table is saved as Parquet and loaded into DuckDB; pandas for exploration, scikit-learn for a logistic-regression baseline and a gradient-boosted model.
3. **Evaluate** - hold-out test set; ROC-AUC, precision and recall at a chosen threshold, calibration, and which first-day behaviours matter most.
4. **Experimentation notebook** - how to read an A/B test on retention: significance, confidence intervals and the sample size a test needs before it starts.

## Repository layout

```
sql/          BigQuery extraction queries
src/churn/    Python package: loading, features, training, evaluation
notebooks/    exploration, modelling walk-through, A/B test notebook
data/         local Parquet extract (not committed)
reports/      figures and results
```

## Reproduce

```bash
# 1. extract (needs a Google Cloud project; the free sandbox works)
bq query --use_legacy_sql=false --format=csv --max_rows=100000 \
  < sql/01_player_features.sql > data/player_features.csv

# 2. model
uv sync
uv run python -m churn.train
```

## Credits

Churn definition and dataset follow Google's public guide on churn prediction for game developers with GA4/Firebase data. The modelling here is done in Python rather than BigQuery ML.

## License

MIT
