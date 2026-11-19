# Operations runbook

## Resource requirements

| Setup | CPU | RAM | Disk |
| --- | --- | --- | --- |
| Local training only (venv) | 2 cores | 2 GB | 1 GB |
| Compose core (postgres + mlflow + api) | 2 cores | 3 GB | 5 GB |
| Compose + observability (+ prometheus, grafana) | 3 cores | 4 GB | 6 GB |
| Compose + orchestration (+ airflow) | 3 cores | 5 GB | 8 GB |
| minikube (all services + pipeline job) | 4 cores | 6 GB | 10 GB |

Training the three models on the full panel takes a few seconds (LR ~0.001 s, RF
~0.4 s, XGB ~0.2 s); the expensive parts locally are image builds and the
Evidently report.

## Startup order

Dependencies are explicit in Compose (`depends_on: condition: service_healthy`)
and in Kubernetes (init containers poll the TCP port):

```
postgres → mlflow → api → prometheus → grafana
postgres, mlflow → seed/airflow
```

Nothing depends on the API except Prometheus scraping; the API depends on both
Postgres (persistence) and MLflow (registry).

## Health checks

| Component | Check | Healthy means |
| --- | --- | --- |
| API | `GET /health` | DB reachable + champion resolved; `dependencies[]` shows both |
| API (readiness) | same endpoint | pod receives traffic only when this passes |
| MLflow | `GET /health` | tracking server up and backend reachable |
| Postgres | `pg_isready -U $POSTGRES_USER -d $POSTGRES_DB` | accepting connections |
| Prometheus | `GET /-/ready`, `GET /-/healthy` | TSDB loaded |
| Grafana | `GET /api/health` | database + HTTP server up |
| Airflow | `GET /health` | scheduler heartbeat fresh |

End-to-end verification (works locally, in Compose and in Kubernetes):

```bash
python scripts/verify_api.py --api-url http://localhost:8000
```

It exits non-zero if any of the 8 checks fail and prints the stored prediction it
created.

## Routine operations

```bash
# pipelines
make ingest train promote monitor        # or scripts/run_pipeline.py --stage <stage>

# registry
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# (in Compose/K8s the server is at :5000)

# database
make db-upgrade
psql "$DATABASE_URL" -c 'select model_name, version, stage, is_active from model_metadata order by created_at desc limit 5;'
psql "$DATABASE_URL" -c 'select count(*), max(created_at) from prediction_records;'

# monitoring artifacts
open reports/monitoring/data_drift_*.html     # newest first
cat reports/drift_status.json | jq '{status, data_drift, performance_status, retrain_recommended}'
```

## Recovery procedures

**Champion alias missing / API returns 503 on `/predict`.**
`GET /model/info` reports the registry state. Fix by running `make train` then
`make promote`; verify with
`python -m mlflow ... ` or `GET /model/info` again. The API caches the champion
for `MODEL_CACHE_TTL_SECONDS` (default 300) — restart or wait TTL after changing
the alias.

**Promotion rejected unexpectedly.**
Read `reports/promotion/latest_promotion.json`: it lists every gate with its
detail and the summary. Common cause: candidate metrics identical to the champion
(`improvement 0.00% < 2.0%`), which is a correct rejection.

**Pipeline ran but nothing was persisted to PostgreSQL.**
Look for `database unavailable - continuing without persistence` in the logs; the
pipeline outcome carries `db_persisted: false`. Check `DATABASE_URL` and that
migrations ran (`alembic current`).

**MLflow returns 500 when registering a model.**
The registry needs a database-backed store. File-store tracking URIs cannot
register models: use `sqlite:///mlruns/mlflow.db` locally, or Postgres in Compose
and Kubernetes.

**Drift status endpoint returns `available: false`.**
`reports/drift_status.json` does not exist in the API's `reports` mount. Run
`make monitor`, or mount the reports volume (Compose and the Kubernetes API
deployment already do).

**Kubernetes: API pod in CrashLoopBackOff.**
`kubectl -n mlops logs deploy/api --previous` — usually PostgreSQL credentials
(Secret regenerated with new values), or `alembic upgrade head` failing because the
database was not initialized (the `postgres-init` ConfigMap creates the `mlflow`
and `airflow` databases on first start only; if the PVC already existed with an
old password, delete the claim or the namespace and redeploy).

**Kubernetes: seed job failed.**
`kubectl -n mlops logs job/seed-pipeline`. The job prints the same structured logs
as the CLI. Re-run with `kubectl -n mlops delete job seed-pipeline` then
`kubectl apply -f kubernetes/jobs/seed-pipeline.yaml`. The job is idempotent:
datasets and reports are overwritten, a new model version is registered, and the
promotion gates decide whether the alias moves.

**Rollback the model.**
Every version keeps its metrics and artifacts. Point the alias back:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient("http://localhost:5000")
client.set_registered_model_alias("interest_rate_forecaster", "champion", "1")
```

**Reset everything locally.**

```bash
docker compose down -v          # deletes volumes: DB, MLflow artifacts, Grafana state
rm -rf mlruns data/processed reports/evaluation reports/monitoring reports/promotion reports/state
./kubernetes/deploy.sh --teardown
```

## Backup

| Data | Where | How |
| --- | --- | --- |
| Application data | PostgreSQL volume | `pg_dump "$DATABASE_URL" > backup.sql` |
| MLflow metadata | PostgreSQL (`mlflow` DB) | `pg_dump -d mlflow > mlflow.sql` |
| Model artifacts | MLflow artifacts volume / `dvc-storage` | copy the volume, or `dvc push` for datasets |
| Datasets | `data/` + DVC | `dvc push` to the local remote |
| Generated reports | `reports/` | copy the directory |

## Security notes (local-only scope)

- Credentials come from environment variables; `.env` and `kubernetes/.generated/`
  are git-ignored. `.env.example` contains obvious development defaults.
- Containers run as non-root where the base image allows it; the API image
  explicitly drops capabilities and forbids privilege escalation in Kubernetes.
- No TLS, no authentication on the API, no SSO. This is a local demonstration;
  exposing it publicly would require auth, TLS and rate limiting first.
