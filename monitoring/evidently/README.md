# Evidently configuration

The drift and performance thresholds live in `configs/monitoring.yaml`; this
directory documents where the generated reports end up and how to read them.

| Artifact | Path |
| --- | --- |
| HTML drift report | `reports/monitoring/data_drift_<UTC timestamp>.html` |
| JSON summary of the last run | `reports/monitoring/latest_summary.json` |
| Latest status consumed by `GET /drift/status` | `reports/drift_status.json` |

Two independent questions are asked on every run:

1. **Has the input distribution changed?** (data drift - Evidently `DataDriftPreset`
   over the engineered features, per-feature threshold from the config).
2. **Has accuracy degraded?** (model performance - MAE/RMSE/MAPE on rows where the
   realised target is known, compared with the champion's recorded test metrics).

Drift alone never sets `retrain_recommended`; only a performance breach or the
configured drift policy does, and MAPE is ignored when fewer than
`performance.min_mape_coverage_pct` of the window has a meaningful denominator.

Reports are regenerated per run and are not committed to Git (they are large and
time-stamped). The JSON summary and status file are what the API and Prometheus
read.
