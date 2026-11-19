# Exploratory data analysis

- Panel window: **1958-01-01 to 2017-02-01**
- Monthly observations: **710**
- Target series: **effective_federal_funds_rate** (percent per annum)
- Peak: **19.10%** in 1981-06-01
- Trough (non-zero): **0.07%** in 2011-07-01
- Largest single-month move: **6.63 pp** in 1980-05-01
- Autocorrelation at lag 1: **0.990**, at lag 12: **0.826**

## Series summary

| series | obs | missing | mean | std | min | max | acf(1) | acf(12) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| effective_federal_funds_rate | 710 | 0.0% | 5.0642 | 3.6547 | 0.07 | 19.1 | 0.9898 | 0.8263 |
| unemployment_rate | 710 | 0.0% | 6.0689 | 1.5614 | 3.4 | 10.8 | 0.993 | 0.7651 |
| inflation_rate | 710 | 0.0% | 3.7334 | 2.5736 | 0.6 | 13.6 | 0.9951 | 0.8282 |
| real_gdp_pct_change | 710 | 0.0% | 3.1127 | 3.5566 | -10.0 | 16.5 | 0.7799 | 0.0722 |

## What this implies for modelling

- The target is highly persistent (lag-1 autocorrelation close to 1), so lag and rolling-window features carry most of the signal; a naive 'carry forward' baseline is strong and must be beaten honestly.
- The series spans very different policy regimes (double-digit rates in the late 1970s/early 1980s, a zero lower bound after 2008). Fold-to-fold error variation is therefore expected and is reported rather than hidden.
- The target is observed monthly, not daily. Lags are expressed in observations (months), never in calendar days.
- `Real GDP (Percent Change)` is quarterly and is carried forward as the last published value; missing months are periods with no scheduled observation, not data errors.
- Rates near zero make MAPE unstable, which is why MAE/RMSE are the primary metrics and MAPE is reported with an explicit coverage figure.

## Figures

![history](target_history.png)

![distribution](target_distribution.png)

![rolling](rolling_statistics.png)

![acf](autocorrelation.png)

![seasonality](seasonality_by_month.png)

Data-quality context: 904 raw rows reduced to 710 monthly observations after dropping 151 intra-month policy-decision rows.
