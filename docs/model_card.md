# Model card — interest rate forecaster

## Overview

| Field | Value |
| --- | --- |
| Registered name | `interest_rate_forecaster` |
| Current champion | `linear_regression`, version 1 (alias `champion`) |
| Task | one-step-ahead forecast of the monthly effective federal funds rate |
| Target | `Effective Federal Funds Rate`, percent per annum |
| Frequency | monthly (input), horizon = 1 observation |
| Training data | Federal Reserve interest-rate snapshot 1958-01 → 2017-02 (710 monthly rows) |
| Dataset version | `sha256:091410052c81` (see MLflow run tags) |
| Owner | the repository itself (portfolio project, no institutional owner) |

## Intended use

- Demonstrating an end-to-end MLOps workflow: training, tracking, registry,
  promotion gates, serving, monitoring.
- Comparing three regression families on a genuinely hard time-series target.
- Teaching material for drift-vs-degradation monitoring.

## Out of scope

- Any live trading, risk, pricing or investment decision.
- Forecasting horizons beyond one month.
- Data after 2017-02 (the source snapshot ends there; the FRED adapter is not the
  default source).
- Intra-month policy timing: the model sees monthly aggregates only.

## Features

13 engineered features derived exclusively from the target's own history:

`lag_1, lag_2, lag_3, lag_6, lag_12`, `roll_mean_3/6/12`, `roll_std_3/6/12`,
`diff_1, diff_12`. All are shifted so the value at the origin month is the newest
available information; the target is the next month. Leakage is covered by unit
tests that perturb future values and assert the features do not change.

Macro series (unemployment, inflation, GDP) are present in the panel for EDA and
monitoring but disabled as model inputs (`enabled: false`).

## Evaluation

Chronological 80/20 split — train `1959-01 → 2005-06` (558 rows), test
`2005-07 → 2017-01` (139 rows). Selection uses mean walk-forward CV RMSE on the
training block only.

| Model | MAE (pp) | RMSE (pp) | MAPE (%) | MAPE coverage | CV RMSE | CV RMSE std | CV RMSE std/mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Linear Regression** | **0.1125** | **0.1585** | 6.29 | 30.2% | 0.5642 | 0.4594 | 0.814 |
| Random Forest | 0.7243 | 0.8093 | 12.05 | 30.2% | 1.3265 | 1.1616 | 0.876 |
| XGBoost | 0.8303 | 0.9369 | 15.38 | 30.2% | 1.2083 | 1.0381 | 0.859 |

Metric definitions: MAE and RMSE in percentage points; MAPE excludes months with
|actual| < 0.5 pp and reports the retained share; CV is expanding-window
walk-forward with 5 folds on the training block.

## Why the baseline wins

The test block contains the 2008–2015 zero-lower-bound period. Tree ensembles
predict within the range observed during training, so they cannot produce the
sub-0.5% values the actual series takes; their error is dominated by that bias.
Linear Regression extrapolates and tracks the level. This is a data-regime effect,
not a tuning failure, and it is why the project compares models instead of
assuming XGBoost wins.

## Performance by regime (interpretation guidance)

- CV folds span the high-volatility 1970s–80s, so CV RMSE (0.56) is much larger
  than test RMSE (0.16). Quoting either number alone would mislead.
- MAPE coverage of 30.2% means the percentage error only describes the 30% of
  months with a meaningful denominator.
- Drift of input features across eras is expected and does not by itself imply
  accuracy loss; monitoring reports the two separately.

## Fairness, ethics and safety

- No personal data, no protected attributes, no individual-level predictions.
- The model is a statistical description of a public macro series. Presenting its
  output as a forecast of policy decisions would be a misuse.
- Default promotion gates prevent silently replacing an approved model, but this
  is a local demonstration: no human approval workflow, no audit trail beyond the
  database and MLflow.

## Retraining policy

- Trigger: monitoring flags a performance breach (RMSE degradation ≥ 20% warning,
  ≥ 50% critical) or the configured drift policy, with MAPE only counted when its
  coverage is adequate.
- Process: retrain → register candidate → gates (absolute quality, MAPE sanity, CV
  stability, ≥ 2% RMSE improvement over the champion) → alias moves only on pass.
- Rollback: re-point `champion` to the previous version
  (`python -m mlflow ... set alias`, or rerun promotion for the earlier version);
  every version and its metrics remain in the registry.

## Known limitations

- Univariate; no macro drivers, no policy shocks, no uncertainty intervals.
- Single-step only.
- 2017-02 data cutoff.
- Test block is a single regime; a rolling-origin backtest would give a richer
  picture of behaviour across regimes.
- Linear model assumes a stable relationship over the whole history; regime
  breaks are visible in residuals but are not modelled explicitly.
