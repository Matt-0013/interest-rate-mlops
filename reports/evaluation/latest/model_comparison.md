# Model comparison report

- Evaluation date: `20261118T073605Z`
- Dataset: `federal_reserve_interest_rates` version `sha256:091410052c81`
- Target: `effective_federal_funds_rate` (percent per annum), horizon 1 observation(s)
- Split: chronological, train 1959-01-01..2005-06-01 (558 rows), test 2005-07-01..2017-01-01 (139 rows)
- Selection rule: lowest mean walk-forward CV RMSE on the training block; the held-out test block is reported for every model but never used to pick the winner
- Selected model: **linear_regression**

## Results

| Model | MAE | RMSE | MAPE (%) | MAPE coverage (%) | CV RMSE | CV MAE | Train (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear_regression | 0.1125 | 0.1585 | 6.29 | 30.2 | 0.5642 | 0.3696 | 0.001 |
| random_forest | 0.7243 | 0.8093 | 12.05 | 30.2 | 1.3265 | 0.8587 | 0.397 |
| xgboost | 0.8303 | 0.9369 | 15.38 | 30.2 | 1.2083 | 0.7925 | 0.209 |

## Notes

- MAE and RMSE are in percentage points and are the primary metrics.
- MAPE excludes observations with |actual| < 0.5 pp; coverage is reported.
- The test block is the final chronological 20% and is never used for fitting.
