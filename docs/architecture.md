# Architecture

## Components

| Layer | Component | Responsibility |
| --- | --- | --- |
| Data | `data/raw` (DVC) | immutable source snapshot |
| Data | `ingestion`, `validation`, `preprocessing` | parse dates, validate schema/ranges, drop intra-month rows, build the monthly panel |
| Features | `features` | leakage-safe lags/rolling/diffs; serving-side recomputation from history |
| Models | `models`, `training` | one training implementation, three estimators, chronological split, walk-forward CV |
| Evaluation | `evaluation` | MAE/RMSE/MAPE (guarded), plots, JSON/CSV/Markdown comparison |
| Registry | `registry` | MLflow logging, aliases, promotion gates, registry reads |
| Serving | `api` | REST + GraphQL, model cache, Pydantic validation, persistence, metrics |
| Persistence | `database` | SQLAlchemy models, session factory, repositories |
| Orchestration | `airflow/dags` | 4 DAGs calling the same pipeline functions as the CLI |
| Monitoring | `monitoring`, `api/metrics.py` | Evidently drift/performance, Prometheus gauges, Grafana dashboards |
| Platform | Docker Compose, Kubernetes, Terraform | local runtime, deployment, cluster prerequisites |

The installable package (`src/interest_rate_mlops`) is the single source of truth.
The API, the DAGs, the CLI scripts and CI import it; no logic is duplicated in a
deployment artifact. `pipelines/*` exists so every entry point (CLI, DAG, Job,
test) calls one function per stage.

## Data flow

```
CSV ──► validate_raw ──► build_panel ──► panel.parquet ──► build_supervised_features
                                                                   │
                                                                   ▼
                                                          features.parquet
                                                                   │
              ┌────────────────────────────────────────────────────┘
              ▼
   ModelTrainer (split + walk-forward CV + 3 estimators)
              │
              ├─► reports/evaluation/<ts>/ (json, csv, md, png) + latest/ copy
              ├─► MLflow runs (params, metrics, model, tags: dataset version, git commit)
              ├─► registry: registered model + `challenger` alias
              └─► app DB: training_runs, model_metadata, pipeline_executions
                        │
                        ▼
                promote_candidate: gates vs `champion`
                        │ pass → alias moves, deployment_history recorded
                        │ fail → reason recorded, incumbent untouched
                        ▼
                     API ──► prediction_records ──► /predictions
                      │
                      └──► /metrics (counters, histograms, file-backed ML gauges)
                                  │
                    Prometheus ────┴──── Grafana (API / ML / Pipeline dashboards)
```

## Why these decisions

**Three databases on one server.** Application data, MLflow tracking and Airflow
metadata have different lifecycles and different failure semantics: a corrupted
MLflow store must not prevent serving predictions, and blowing away the app
database must not erase experiment history. Separate databases make that explicit
and cost nothing locally.

**Aliases instead of stages.** MLflow's stages API is deprecated; `champion` /
`challenger` aliases are the supported mechanism and make the "current approved
model" a single pointer that the API resolves at load time.

**Feature code shared with serving.** `features_from_history` calls the same
builders used in training. If a client had to construct features itself, the
contract would drift and predictions would silently change meaning. The API
exposes `/model/features` so a caller can see the exact contract.

**File-backed ML metrics.** Pipeline and monitoring state are written to
`reports/state/pipeline_status.json` and `reports/drift_status.json` and exposed
by a Prometheus collector. Airflow never needs to reach the API, and a Prometheus
scrape never triggers a database query or an Evidently run.

**Drift and performance are separate signals.** Every monitoring run computes
both, stores both, and only the performance signal (or an explicit drift policy)
can recommend retraining. This is the most common self-inflicted wound in ML
monitoring projects.

**One API replica in Kubernetes.** The data/reports volumes are `ReadWriteOnce`
because minikube's default provisioner is node-local. Scaling out needs RWX or
object storage; pretending otherwise would hide a real constraint.

## Failure modes and how they are handled

| Failure | Behaviour |
| --- | --- |
| Raw schema changes (column renamed) | validation fails with the column list; ingestion stops; DAG fails |
| Duplicate dates / invalid date | validation error before any panel is built |
| Target gap inside the panel | error (never interpolated) |
| PostgreSQL unavailable during training/promotion/monitoring | warning + artifacts still produced; `db_persisted: false` in the outcome |
| PostgreSQL unavailable in the API | `/health` reports the dependency as unhealthy; `/predict` returns 503 |
| No champion registered | `/model/info` and `/predict` return 503 with "run training and promotion" |
| MLflow store unreachable at startup | API starts, health reports the registry unhealthy, no cached model |
| Candidate worse than champion | promotion rejected, reason + gates persisted, alias unchanged |
| Drift without accuracy loss | status reflects drift; `retrain_recommended` stays false unless configured otherwise |
| Monitoring window too short / no actuals | performance status `unknown`, MAPE suppressed when coverage < 50% |

## Configuration surface

- `configs/data.yaml` — source, columns, target, exogenous series, panel window, validation rules.
- `configs/model.yaml` — features, split, CV, three model specs, selection rule, promotion gates.
- `configs/monitoring.yaml` — windows, drift thresholds, performance thresholds, paths, alerting.
- Environment variables (`Settings`) — credentials, URLs, model alias, cache TTL, ports, feature flags.

Configs are parsed by strict Pydantic models, so a typo fails at startup with a
precise message instead of silently falling back to a default.
