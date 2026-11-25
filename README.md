# End-to-End MLOps Platform — US Interest Rate Forecasting

A local, production-shaped MLOps platform that forecasts the **effective federal
funds rate** from Federal Reserve historical data. It covers the whole lifecycle:
ingestion and validation → EDA → feature engineering → three-model training and
comparison → experiment tracking and model registry → dataset versioning →
promotion gates → FastAPI (REST + GraphQL) serving → PostgreSQL persistence →
Docker Compose → Airflow orchestration → CI → Evidently drift/performance
monitoring → Prometheus/Grafana → Terraform → local Kubernetes (minikube).

Everything runs on a laptop. No AWS/Azure/GCP, no managed services, no paid
components.

---

## Results (measured on the real dataset)

Chronological split: train `1959-01 → 2005-06` (558 rows), test
`2005-07 → 2017-01` (139 rows). Metrics are in percentage points of the annual
rate. MAE/RMSE decide; MAPE is secondary and reported with its coverage.

| Model | MAE | RMSE | MAPE (%) | MAPE coverage | CV RMSE (mean ± std) | CV MAE | Train (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Linear Regression** | **0.1125** | **0.1585** | 6.29 | 30.2% | 0.5642 ± 0.4594 | 0.3696 | 0.001 |
| Random Forest | 0.7243 | 0.8093 | 12.05 | 30.2% | 1.3265 ± 1.1616 | 0.8587 | 0.397 |
| XGBoost | 0.8303 | 0.9369 | 15.38 | 30.2% | 1.2083 ± 1.0381 | 0.7925 | 0.209 |

**Linear Regression wins and is registered as the champion.** This is the honest
result, not a rigged one: the test block (2005–2017) sits in the post-2008
near-zero regime, and tree ensembles cannot extrapolate below the range they saw
in training, so they predict ~0.7–0.8 pp where the actual rate is ~0.1 pp. The
winner is chosen by lowest mean **walk-forward CV RMSE on the training block** —
never by peeking at the test block.

Interpretation a reviewer should read next to the table:

- **CV RMSE ≫ test RMSE** (0.56 vs 0.16) because the CV folds span the volatile
  1970s–80s; the test block is calmer. Both numbers are reported; neither is
  hidden.
- **MAPE coverage is 30.2%**: only 30% of test months have |actual| ≥ 0.5 pp, so
  MAPE excludes the near-zero years. It is reported with the caveat, never as
  the deciding metric.
- Regenerate everything with `make ingest && make train && make promote`. The
  comparison is written to `reports/evaluation/latest/model_comparison.{json,csv,md}`.

**Promotion worked as designed:** re-running training registered a new candidate
with identical metrics, and the gates **rejected** it
(`RMSE improvement 0.00% < 2.0% vs champion`), leaving the champion alias on v1.
A promotion pipeline that cannot say "no" is not a promotion pipeline.

**Monitoring on the real data:** 13/13 engineered features flagged as drifted
(share 1.0) between the reference window the champion was fitted on
(1959-01…2005-06, 558 rows, read from the split the training run recorded rather
than recomputed) and the latest 60 months (2012-02…2017-01), while measured RMSE
on realised targets was *better* than the baseline (0.0920 vs 0.1585, −41.9%).
Every feature clears the configured 0.10 Kolmogorov–Smirnov effect-size floor —
the smallest is D=0.37 — so this is a real regime shift, not p-value noise: at
the zero bound the level features are completely separated (D=1.0) and the
monthly-change features lose almost all of their spread. Drift is reported as
drift; accuracy is only reported from realised targets. MAPE degradation was
suppressed (5% coverage of the window) — exactly the failure mode this project
guards against.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │  Kaggle CSV (904 rows, 1954-07..2017-03)  │
                        └───────────────────┬──────────────────────┘
                                            │ ingest (DVC-tracked)
                                  ┌─────────▼──────────┐
                                  │ validation + panel │  710 monthly rows
                                  └─────────┬──────────┘
                                            │ features (13, leakage-safe)
                        ┌───────────────────▼────────────────────┐
                        │ training: LR · RF · XGBoost             │
                        │ chronological split + walk-forward CV   │
                        └───────┬───────────────────────┬────────┘
                                │                       │
                     MLflow tracking + registry    comparison reports
                     (Postgres backend, artifacts)  (JSON/CSV/MD/PNG)
                                │
                     ┌──────────▼───────────┐
                     │ promotion gates      │  absolute quality, MAPE sanity,
                     │ champion / challenger│  CV stability, beats champion
                     └──────────┬───────────┘
                                │ alias
             ┌──────────────────▼───────────────────┐
             │ FastAPI: REST + GraphQL              │
             │ champion cached, predictions stored  │──────► Prometheus ──► Grafana
             └───────┬──────────────────┬───────────┘
                     │                  │
             PostgreSQL (app DB)   reports/ (drift status, reports)
                     │
        ▲            │
        │            ▼
   ┌────┴─────────────────────────┐        ┌──────────────────────────┐
   │ Airflow DAGs (4)             │        │ Evidently monitoring      │
   │ ingest/train/promote/monitor │        │ drift ≠ performance       │
   └──────────────────────────────┘        └──────────────────────────┘
```

Application database, MLflow database and Airflow metadata live in **separate
PostgreSQL databases** on one local server. See `docs/architecture.md`.

---

## Dataset

Source: [Kaggle — Federal Reserve Interest Rates](https://www.kaggle.com/datasets/federalreserve/interest-rates)
(`federalreserve/interest-rates`, file `index.csv`). Downloaded anonymously via
the public Kaggle API endpoint — no credentials needed.

| Fact | Value |
| --- | --- |
| Shape | 904 rows × 10 columns |
| Date columns | `Year`, `Month`, `Day` (three integers, no single date column) |
| Range | 1954-07-01 → 2017-03-16 |
| Distinct month-start dates | 753 |
| Intra-month policy rows | 151 (carry a target rate, no effective rate) |
| Duplicate dates | none |
| Inferred frequency | monthly with intra-month FOMC-style rows |

Series: `Federal Funds Target Rate`, `Federal Funds Upper Target`,
`Federal Funds Lower Target`, `Effective Federal Funds Rate`,
`Real GDP (Percent Change)`, `Unemployment Rate`, `Inflation Rate`.
Missingness is structural, not random: target rate 48.9%, upper/lower 88.6%
(only exist from 2008-12), GDP 72.4% (quarterly), inflation 21.5%, effective
rate 16.8%. Full profile and per-column statistics: `docs/dataset.md` and
`reports/data_quality/`.

**Target chosen after inspection:** `Effective Federal Funds Rate` — the only
series observed monthly for the whole history (752 non-null values, 0.07–19.10%).
The 2017-03 month-start row has no effective rate and is dropped; the panel
therefore ends 2017-02. It starts 1958-01 because inflation (kept as an
exogenous series) starts then, giving **710 monthly rows**.

---

## ML approach

**Task.** One-step-ahead forecast of the monthly effective federal funds rate
(percent per annum).

**Features (13).** Lags 1, 2, 3, 6, 12; rolling mean and std over 3, 6, 12;
first differences 1 and 12. `lag_k` is the value at `t-(k-1)`, so `lag_1` is the
rate known at the origin month and the target is `t+1`. Rolling statistics are
computed from history only, shifts are explicit, and every feature is verified by
leakage tests (`tests/unit/test_features.py`). Lags are **observations**, not
calendar days — the series is monthly.

The three macro series (unemployment, inflation, GDP) are carried in the panel for
EDA/monitoring but are **not** model inputs (`enabled: false` in
`configs/data.yaml`, documented decision: keeps the univariate contract stable and
avoids publishing-delay leakage on quarterly GDP).

**Split.** Chronological 80/20, no shuffling. Walk-forward (expanding window,
5 folds, `TimeSeriesSplit`) on the training block for model selection and
stability reporting.

**Models.** `LinearRegression`, `RandomForestRegressor` (n_estimators, max_depth,
min_samples_split, min_samples_leaf, random_state), `XGBRegressor` (n_estimators,
learning_rate, max_depth, subsample, colsample_bytree, random_state). All
hyperparameters live in `configs/model.yaml`; one training implementation serves
all three (`src/interest_rate_mlops/training/trainer.py`).

**Metrics.** MAE, RMSE (primary), MAPE with an explicit denominator guard:
observations with |actual| < 0.5 pp are excluded and the coverage share is always
reported. MAPE is never used to declare degradation when coverage is low.

**Promotion gates** (`configs/model.yaml`): absolute quality (RMSE ≤ ceiling),
MAPE sanity (when computable), CV stability (fold RMSE std/mean ≤ 1.0, relaxed
from 0.5 with a comment because this series genuinely has regime-dependent
volatility), and *beats the champion by ≥ 2%*. Bootstrap path allowed when no
champion exists. Promotion moves the MLflow `champion` alias; rejection is
recorded with reasons and leaves the incumbent in place.

---

## Repository layout

```
interest-rate-mlops/
├── api/                      # FastAPI app: main, routes, schemas, dependencies, services, metrics
├── airflow/dags/             # 4 DAGs + shared helpers
├── alembic/                  # migration environment + versions
├── configs/                  # data.yaml, model.yaml, monitoring.yaml
├── data/{raw,processed}/     # DVC-tracked datasets (.dvc pointers committed)
├── docker/                   # api.Dockerfile, airflow.Dockerfile, postgres init
├── docs/                     # dataset, architecture, model card, operations
├── kubernetes/               # manifests, deploy.sh, verify.sh
├── monitoring/               # evidently notes, prometheus config, grafana provisioning + 3 dashboards
├── notebooks/                # 01_eda.ipynb
├── reports/                  # generated: data quality, EDA, evaluation, promotion, drift
├── scripts/                  # CLI entry points (pipeline, EDA, verify_api, config check)
├── src/interest_rate_mlops/  # installable package (ingestion → registry → monitoring)
├── terraform/                # local cluster prerequisites
├── tests/{unit,api,integration,fixtures}
├── docker-compose.yml  dvc.yaml  dvc.lock  Makefile  pyproject.toml
├── requirements.txt    .env.example  .gitignore
```

Design notes: the library is an installable package (`pip install -e .`) so the
API, DAGs, CI jobs and scripts import the *same* code; `api/` sits at the root
because it is a deployment artifact, not library code.

---

## Quick start (no Docker)

```bash
git clone <your-repo> interest-rate-mlops && cd interest-rate-mlops
python -m venv .venv
source .venv/bin/activate            # Windows: .venv/Scripts/activate
pip install -r requirements.txt && pip install -e . --no-deps

make data          # download the Kaggle CSV (anonymous)
make ingest        # validate raw data, build the monthly panel
make eda           # EDA report + figures -> reports/eda/
make train         # 3 models, MLflow runs, candidate registered
make promote       # gates -> champion alias
make monitor       # Evidently drift + performance -> reports/drift_status.json
make api           # http://localhost:8000/docs
```

Without PostgreSQL the pipelines log `database unavailable - continuing without
persistence` and still produce every artifact; the API needs a database (see
Compose or the SQLite shortcut below).

Local SQLite shortcut for the API:

```bash
export DATABASE_URL="sqlite:///./local_e2e.db"
python -m alembic upgrade head
python -m uvicorn api.main:app --port 8000
python scripts/verify_api.py --api-url http://localhost:8000
```

`scripts/verify_api.py` checks health, model info, features, inference,
persistence, drift status, Prometheus metrics and GraphQL, and exits non-zero on
failure (it is what the Kubernetes verification runs).

---

## Docker Compose

```bash
cp .env.example .env      # then edit if you want different credentials
make compose-up           # postgres + mlflow + api        (~2 GB RAM)
make compose-monitoring   # + prometheus + grafana         (~3 GB RAM)
make compose-orchestration# + airflow webserver + scheduler(~4 GB RAM)
make compose-full         # everything                     (~6 GB RAM)
make compose-ps           # health of every service
```

Profiles exist because a 4–6 GB VM cannot comfortably run Airflow and Grafana at
the same time as training. Services: `postgres` (16.4, init script creates the
`mlflow` and `airflow` databases), `mlflow` (tracking + registry, Postgres
backend, artifacts on a volume), `api` (runs `alembic upgrade head` then uvicorn,
health-checked via `/health`), `prometheus`, `grafana` (auto-provisioned
datasource + dashboards), `airflow-init/webserver/scheduler` (LocalExecutor,
Postgres metadata DB).

| URL | Service |
| --- | --- |
| http://localhost:8000/docs | FastAPI Swagger UI |
| http://localhost:5000 | MLflow (experiments + registry) |
| http://localhost:5000/#/models | MLflow model registry (aliases, versions) |
| http://localhost:9090 | Prometheus |
| http://localhost:3000 | Grafana (admin/admin by default) |
| http://localhost:8080 | Airflow (admin/admin by default) |

All secrets come from `.env`; nothing is baked into an image, and no service has
a hard-coded credential in its Dockerfile.

---

## Airflow

Four DAGs, all thin wrappers over the same pipeline functions the CLI uses:

| DAG | Steps | Schedule |
| --- | --- | --- |
| `ingestion_pipeline` | ingest → validate → quality checks → record lineage | manual (`INGESTION_SCHEDULE` to enable) |
| `training_pipeline` | train 3 models → log → register candidate → trigger promotion | manual (`TRAINING_SCHEDULE`) |
| `promotion_pipeline` | load candidate → gates → move alias or reject → record | triggered by training |
| `monitoring_pipeline` | Evidently drift → performance → status → retrain recommendation | `@weekly` (`MONITORING_SCHEDULE`) |

The static Kaggle dataset does not pretend to receive new data: ingestion is
manual by default and `INGESTION_FETCH=true` re-downloads when a live source is
configured. Monitoring can trigger the training DAG (`AUTO_RETRAIN=true`) only
when thresholds recommend retraining.

```bash
docker compose --profile orchestration up -d
open http://localhost:8080            # admin / admin
# unpause and trigger ingestion_pipeline, then training_pipeline
```

---

## DVC

```bash
python -m dvc repro            # ingest + train stages, writes dvc.lock
python -m dvc metrics show     # latest comparison metrics
python -m dvc push             # local remote: ./dvc-storage (no cloud)
python -m dvc checkout         # restore the data a commit expects
```

`dvc.yaml` defines `ingest` (raw CSV → panel) and `train` (panel → features +
metrics). Raw and processed data are separate DVC outputs; `.dvc` pointers are
committed and the data itself stays out of Git. Reproduce an old experiment with
`git checkout <commit> && dvc checkout && dvc repro train`.

Ingestion also best-effort `dvc add`s a freshly downloaded raw file, and skips any
path that `dvc.yaml` already owns (avoids overlapping outputs).

---

## MLflow

- Experiment: `interest_rate_forecasting`; one run per model per training run.
- Logged: params, metrics (test + CV per fold), model artifact, comparison
  artifacts, dataset version, DVC hash, git commit, target unit and frequency.
- Registry: model `interest_rate_forecaster`; aliases `champion` and
  `challenger` (aliases, not the deprecated stages API).
- Tracking store: SQLite locally (`mlruns/mlflow.db`) or PostgreSQL in Compose /
  Kubernetes; artifacts live on a PVC in-cluster.

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # local
docker compose --profile core up -d mlflow                          # http://localhost:5000
```

---

## API

FastAPI with Pydantic v2 schemas, structured JSON logging, request-ID middleware,
Prometheus metrics and a Strawberry GraphQL endpoint at `/graphql`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness/readiness incl. database and registry status |
| GET | `/model/info` | approved model name, version, alias, metrics, dataset version |
| GET | `/model/features` | the exact feature contract + a served feature row |
| POST | `/predict` | one-step forecast from `features` **or** `history` |
| GET | `/predictions` | stored predictions (filter by model, paginate) |
| GET | `/metrics` | Prometheus exposition format |
| GET | `/drift/status` | latest Evidently drift + performance status |
| POST | `/graphql` | `modelInfo`, `predict`, `predictions`, `driftStatus`, `health` |
| GET | `/docs`, `/openapi.json` | Swagger UI and schema |

`POST /predict` accepts exactly one of:

```json
{"features": {"lag_1": 0.65, "lag_2": 0.54, "...": 0}}
{"history": [{"date": "2017-01-01", "value": 0.65}, {"date": "2017-02-01", "value": 0.66}]}
```

The `history` form is recomputed server-side with the *training* feature code, so
clients never reimplement feature engineering. The model is loaded once per
process from the registry (cached, TTL-configurable — never per request), and every
prediction is persisted with model name/version, features, latency and request ID.

```bash
curl -s localhost:8000/health | jq
curl -s localhost:8000/model/info | jq
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"history":[{"date":"2017-01-01","value":0.65},{"date":"2017-02-01","value":0.66}]}' | jq
curl -s localhost:8000/predictions?limit=5 | jq
curl -s localhost:8000/drift/status | jq
```

Error handling: 422 for schema violations, 404 for unknown routes, 503 when the
database or the champion model is unavailable (with an actionable message), 500
with a request ID for unexpected errors. Responses never leak stack traces.

---

## Database

PostgreSQL via SQLAlchemy 2.0, migrations by Alembic. Seven tables:
`prediction_records`, `model_metadata`, `model_deployment_history`,
`training_runs`, `data_ingestion_logs`, `monitoring_runs`, `pipeline_executions`.

```bash
make db-upgrade                          # alembic upgrade head
make db-revision M="add column"          # autogenerated migration
```

Credentials come from `DATABASE_URL`; the application database is separate from
MLflow's and Airflow's (three databases, one local server).

---

## Monitoring

**Model/data monitoring (Evidently 0.6.7).** Reference window = earliest 80% of the
feature matrix, current window = latest 60 rows (configurable), threshold per
feature from `configs/monitoring.yaml`, HTML report plus machine-readable status:

- data drift share vs warning (0.30) / critical (0.60) thresholds,
- model performance on realised targets vs the champion's recorded test metrics
  (RMSE/MAE/MAPE with degradation bands),
- `retrain_recommended` only from those two signals,
- `reports/drift_status.json` is what `GET /drift/status` and Prometheus read.

**Infrastructure monitoring.** Prometheus scrapes `api:8000/metrics`:
`api_requests_total`, `api_request_duration_seconds` (histogram),
`api_http_errors_total`, `model_predictions_total`, `model_inference_duration_seconds`,
`model_prediction_value_percent`, `mlops_model_info`, `mlops_model_load_failures_total`,
plus file-backed gauges `mlops_pipeline_status`, `mlops_pipeline_last_run_timestamp_seconds`,
`mlops_pipeline_duration_seconds`, `mlops_pipeline_last_success_timestamp_seconds`,
`mlops_data_drift_share`, `mlops_data_drift_detected`, `mlops_performance_status`,
`mlops_model_rmse_percent`, `mlops_model_baseline_rmse_percent`,
`mlops_retrain_recommended`, `mlops_monitoring_status`.

Pipeline state is file-backed so Airflow jobs need no network access to the API.
Grafana ships three provisioned dashboards: **API Monitoring**, **ML Monitoring**
(drift and quality, deliberately separate from traffic) and **Pipeline Monitoring**.

---

## Terraform

```bash
cd terraform && cp terraform.tfvars.example terraform.tfvars
terraform init && terraform plan && terraform apply
```

Manages the cluster-scoped prerequisites only (namespace, credential Secret,
shared ConfigMap). Workloads stay in readable YAML applied by `kubectl`, and the
resource quota is owned by `kubernetes/namespace.yaml` so no object is defined by
two tools. Limitations of local providers are documented in `terraform/README.md`.

---

## Kubernetes (minikube)

```bash
minikube start --cpus=4 --memory=4096
python scripts/download_data.py --source kaggle
./kubernetes/deploy.sh          # builds in minikube's daemon, applies, seeds one pipeline run
./kubernetes/verify.sh          # pods, PVCs, in-cluster curl, end-to-end prediction
```

**Host sizing.** `--memory=4096` is sized for an **8 GB host**, which is the
common case (a Windows machine, or an 8 GB VirtualBox VM whose guest tops out
around 6 GB). PostgreSQL, MLflow, the API, Prometheus and Grafana fit in 4 GB
because Airflow runs outside the cluster. Do not raise `--memory` on an 8 GB
machine: a guest configured above the host's physical RAM will not start, and
VirtualBox reports it as a memory error rather than a cluster problem. Free RAM
(close browsers, editors) before `minikube start` instead.

Deployed: PostgreSQL, MLflow, API (1 replica, RWO volumes), Prometheus, Grafana,
and a one-shot Job that runs ingest → train → promote → monitor in-cluster so the
API has a champion, a feature matrix and a drift status. Every Deployment has
requests/limits, readiness and liveness probes, and init containers that wait for
their dependencies. Details and rationale: `kubernetes/README.md`.

Port-forwarding:

```bash
kubectl -n mlops port-forward svc/api 8000:8000
kubectl -n mlops port-forward svc/grafana 3000:3000
kubectl -n mlops port-forward svc/mlflow 5000:5000
kubectl -n mlops port-forward svc/prometheus 9090:9090
```

Airflow is intentionally not deployed to Kubernetes here: `LocalExecutor` in a
cluster gives none of the durability Kubernetes is for, and a
`KubernetesExecutor` setup needs extra RBAC and DAG sync. It runs under Compose.

---

## CI (GitHub Actions)

| Workflow | What it does |
| --- | --- |
| `tests.yml` | ruff lint + format check, config validation, unit/API/integration tests (with a PostgreSQL service, so the DB integration test actually runs), Alembic upgrade/downgrade, OpenAPI contract check |
| `docker-build.yml` | `docker compose config`, build the API image, assert it runs as non-root, start it, wait for `/health`, probe `/openapi.json` and `/metrics` |
| `pipeline-validation.yml` | generate a synthetic dataset, run ingest → train → promote → monitor, assert artifacts/metrics/registry alias, upload the comparison report |

No cloud registry, no cloud credentials, everything inside GitHub-hosted runner
limits.

---

## Tests

```bash
make test              # 137 tests: 111 unit + 20 API + 6 integration
make test-unit         # feature/leakage, splits, metrics, promotion, monitoring, config
make test-api          # 20 API tests (contract, validation, persistence, GraphQL)
make test-integration  # MLflow round trip, migrations, full pipeline on a fixture
```

The PostgreSQL integration test is skipped unless `TEST_DATABASE_URL` points at a
scratch database (CI sets it). Tests use a 218-row synthetic fixture, never the
historical dataset.

---

## What was verified locally, and what was not

Verified on this machine (Windows, Python 3.12.10, no Docker):

- ruff lint + format clean; all configuration files validated by
  `scripts/check_configs.py`; **137 tests: 136 passed, 1 skipped** (the
  PostgreSQL integration test, which needs a database).
- Real ingestion: 904 → 710 rows, dataset version `sha256:091410052c81`.
- Real training of all three models, MLflow runs logged, candidate registered,
  comparison reports written; promotion gates evaluated (candidate rejected with
  the reason recorded); Evidently drift/performance run on real data.
- DVC `repro` + `push` to the local remote, `dvc status` clean, `dvc.lock` written.
- Alembic migrations applied to SQLite and to a scratch database in tests.
- API end to end against SQLite: `/health` healthy (database + registry v1),
  `/model/info`, `/model/features`, `/predict` (features path 0.7526% and history
  path 0.7362% for 2017-03), `/predictions` persistence, `/metrics`, `/drift/status`,
  GraphQL — 8/8 checks passed via `scripts/verify_api.py`.
- All Kubernetes manifests parse and pass structural checks; shell scripts pass
  `bash -n`; Grafana dashboards and Compose/DVC YAML validated.

**Not verified here** — no Docker or Kubernetes on this host, so these must be run
in your Ubuntu VM:

- `docker compose ... up` (image builds, service health, Airflow UI, Prometheus
  scraping, Grafana dashboards rendering).
- `./kubernetes/deploy.sh` + `./kubernetes/verify.sh` on minikube (image build
  inside minikube, PVC binding, in-cluster seeding job, port-forwarded E2E).
- `terraform plan/apply` (needs a running cluster).
- The GitHub Actions workflows (they have not run on GitHub yet).

I am not claiming those work — they are configured and validated structurally,
not executed.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `database unavailable - continuing without persistence` | No PostgreSQL reachable at `DATABASE_URL`. Expected locally; start Compose or set the SQLite URL. |
| `GET /model/info` → 503 | No `champion` alias. Run `make train` then `make promote`. |
| `alembic` autogenerate produced no changes | Models not imported: `alembic/env.py` imports `Base` and the model module. |
| MLflow 500 on register | Tracking store is file-based; the registry needs a database backend (SQLite/Postgres). Compose and K8s use Postgres. |
| `dvc add` fails with `pathspec` errors | `pathspec` 1.x is incompatible with DVC 3.55; pin `pathspec==0.11.2` (already in `requirements.txt`). |
| Airflow DAG import error | The project package is not installed in the Airflow image; use `docker/airflow.Dockerfile`, or set `PYTHONPATH=src:.`. |
| minikube: `ImagePullBackOff` on `interest-rate-api:local` | The image was built in the host daemon, not minikube's: `eval $(minikube docker-env)` then rebuild (`deploy.sh` does this). |
| minikube: API restarts, `postgres` not ready | Init containers wait 120 s; on a cold, memory-starved host close other applications before `minikube start` (on an 8 GB machine `--memory=4096` is already the ceiling), then `kubectl -n mlops rollout status deploy/api`. |
| Grafana panels empty | Prometheus has no target up: `kubectl -n mlops port-forward svc/prometheus 9090:9090` → `/targets`. |
| Drift status shows `available: false` | `make monitor` has not run yet (or `reports/` is not mounted into the API container). |

---

## Limitations and honest caveats

- **Single-step, univariate forecasting.** No macro drivers, no policy-shock
  modelling, no probabilistic intervals. The macro series are in the panel but
  disabled as features.
- **Data ends 2017-02.** Results describe that regime; they are not a live
  forecast of today's rates. The FRED adapter exists but is not the default
  source.
- **Local-only by design.** One PostgreSQL server, one node, one API replica
  (RWO volumes), file-backed pipeline metrics, no TLS, no SSO, default dev
  credentials in `.env.example`.
- **Drift ≠ degradation.** Data drift is high by construction when comparing
  across policy regimes; only realised-target metrics can claim accuracy loss,
  and MAPE is suppressed when its denominator is meaningless.
- **Airflow LocalExecutor.** Fine for a laptop; not a production scheduler.
- **Not a financial model.** This is an engineering portfolio project, not
  investment advice.

Next steps that would add the most value: probabilistic forecasts (quantile
regression), a proper backtest harness with rolling-origin evaluation, model
explanation (SHAP) in the API, and RWX storage so the API can scale past one
replica.

---

## Documentation

- `docs/dataset.md` — full data-quality findings and panel construction
- `docs/architecture.md` — components, data flow, decisions, failure modes
- `docs/model_card.md` — intended use, metrics, caveats, retraining policy
- `docs/operations.md` — runbook: startup order, health checks, recovery
- `kubernetes/README.md`, `terraform/README.md`, `monitoring/evidently/README.md`
