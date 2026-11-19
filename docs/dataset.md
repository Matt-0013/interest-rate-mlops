# Dataset

Source: [Kaggle — Federal Reserve Interest Rates](https://www.kaggle.com/datasets/federalreserve/interest-rates)
(`federalreserve/interest-rates`, single CSV `index.csv`).

The archive is downloadable without credentials through the public API endpoint
`https://www.kaggle.com/api/v1/datasets/download/federalreserve/interest-rates`,
which is what `scripts/download_data.py` uses. Nothing is invented: every number
below was computed from the downloaded file by
`src/interest_rate_mlops/ingestion/loader.py` and is reproducible with
`make ingest`, which writes `reports/data_quality/<timestamp>/raw_profile.json`.

## Structure

| Property | Value |
| --- | --- |
| Rows × columns | 904 × 10 |
| Date representation | three integer columns `Year`, `Month`, `Day` |
| First / last date | 1954-07-01 / 2017-03-16 |
| Distinct month-start dates | 753 |
| Intra-month (policy-decision) rows | 151 |
| Duplicate dates | none |
| Inferred frequency | monthly, with intra-month rows |
| Unit | percent per annum (all rate series) |

| Column | Type | Non-null | Missing | Min | Max | Mean | First observation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Year` | int | 904 | 0% | 1954 | 2017 | — | 1954-07-01 |
| `Month` | int | 904 | 0% | 1 | 12 | — | 1954-07-01 |
| `Day` | int | 904 | 0% | 1 | 31 | — | 1954-07-01 |
| `Federal Funds Target Rate` | float | 462 | 48.89% | 1.00 | 11.50 | 5.6584 | 1982-09-27 |
| `Federal Funds Upper Target` | float | 103 | 88.61% | 0.25 | 1.00 | 0.3083 | 2008-12-16 |
| `Federal Funds Lower Target` | float | 103 | 88.61% | 0.00 | 0.75 | 0.0583 | 2008-12-16 |
| `Effective Federal Funds Rate` | float | 752 | 16.81% | 0.07 | 19.10 | 4.9110 | 1954-07-01 |
| `Real GDP (Percent Change)` | float | 250 | 72.35% | −10.00 | 16.50 | 3.1380 | — |
| `Unemployment Rate` | float | 752 | 16.81% | 3.40 | 10.80 | 5.9791 | — |
| `Inflation Rate` | float | 710 | 21.46% | 0.60 | 13.60 | 3.7334 | — |

## Findings that shaped the design

1. **There is no date column.** `Year`/`Month`/`Day` are three integers; the
   pipeline builds a timestamp and validates that it is real (no 1970-02-30).
2. **Two kinds of rows.** 753 month-start rows are monthly observations. 151 rows
   fall inside a month (e.g. 1982-09-27, 2008-12-16) and carry only a policy
   target — no effective rate, no macro data. Treating them as monthly
   observations would corrupt lags, so they are dropped, not filled.
3. **Missingness is structural.** The upper/lower target band only exists from
   December 2008 (88.6% missing). GDP is quarterly (72.4% missing). These are
   "no observation scheduled", not data errors, and are not imputed for modelling.
4. **Target availability.** `Effective Federal Funds Rate` is present for 752 of
   753 month-start rows (99.87%); the single gap is 2017-03, the last row of the
   file. The panel therefore ends at 2017-02.
5. **Outliers are real history, not errors.** The validation report flags three
   effective-rate observations above the IQR upper bound (19.04–19.10%) and two
   GDP values (−10.0 and 16.5, quarterly percentage change). Each flagged value
   can be traced to a date in `reports/data_quality/<timestamp>/validation_report.json`.
   They are reported, not clipped: trimming them would remove exactly the
   high-volatility regime the model must be evaluated on.
6. **Multiple rate series exist.** The file contains target, upper, lower and
   effective rates. Only one target is modelled; the others remain in the raw
   data and are documented, not silently dropped.

## Panel construction

`src/interest_rate_mlops/preprocessing/panel.py`:

| Step | Result |
| --- | --- |
| Parse `Year`/`Month`/`Day` → timestamp | 904 valid dates |
| Anchor to month start | 753 distinct months |
| Drop intra-month rows | −151 rows |
| Restrict to `panel.start` (1958-01) | inflation series starts here |
| Require the target at month start | 2017-03 dropped (no effective rate) |
| As-of carry-forward for macro series | 502 GDP fill-forward events, 0 for monthly series |
| Final panel | **710 monthly rows, 1958-01 → 2017-02** |

Target gaps inside the panel: **none** (validated with
`require_contiguous_target: true`). Gaps would raise an error rather than be
interpolated, because a gap silently changes the meaning of every lag.

Sanity checks on the panel (Pandera schema + explicit validators): month-start
index, unique index, no NaN in the target, plausible rate range 0–25%.

## Known caveats

- `Inflation Rate` starts 1958, so the panel discards 1954-07 → 1957-12. This is
  a deliberate trade-off to keep a consistent covariate set; the target itself is
  available earlier, and `configs/data.yaml` can restore the full history by
  setting `panel.start: null`.
- The dataset ends 2017-02. Anything after that requires the FRED adapter
  (`source: fred`, `scripts/download_data.py --source fred --series FEDFUNDS`).
- The raw file's policy-target series is not the modern target range for the whole
  history, which is why it is not used as a feature.
