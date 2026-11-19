# ---------------------------------------------------------------------------
# Developer entry points. Every target is a thin wrapper around a script or a
# documented command, so CI can run the same steps.
#
#   make help
#   make PY=python test        # override the interpreter if needed
# ---------------------------------------------------------------------------
SHELL := bash
.DEFAULT_GOAL := help

ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif

PY ?= $(VENV_PY)
COMPOSE ?= docker compose
PROFILE ?= core
API_URL ?= http://localhost:8000

.PHONY: help install data eda ingest train promote monitor pipeline api \
        test test-unit test-api test-integration lint fmt check config \
        dvc-track repro db-upgrade db-revision mlflow-ui \
        compose-up compose-monitoring compose-orchestration compose-full compose-down compose-logs compose-ps \
        k8s-deploy k8s-verify k8s-delete tf-init tf-plan tf-apply

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- environment -----------------------------------------------------------
install: ## Create the venv (if missing) and install pinned dependencies
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

config: ## Validate every configuration file
	$(PY) scripts/check_configs.py

# --- data & pipelines ------------------------------------------------------
data: ## Download the raw Kaggle dataset into data/raw
	$(PY) scripts/download_data.py --source kaggle

eda: ## Generate the EDA report and figures
	$(PY) scripts/run_eda.py

ingest: ## Ingestion + validation (DAG 1)
	$(PY) scripts/run_pipeline.py --stage ingest

train: ## Train, evaluate, log and register (DAG 2)
	$(PY) scripts/run_pipeline.py --stage train

promote: ## Apply promotion gates (DAG 3)
	$(PY) scripts/run_pipeline.py --stage promote

monitor: ## Drift + performance monitoring (DAG 4)
	$(PY) scripts/run_pipeline.py --stage monitor

pipeline: ## Run every stage in order
	$(PY) scripts/run_pipeline.py --stage all

api: ## Run the API locally with reload
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# --- quality ---------------------------------------------------------------
test: ## Run the whole test suite
	$(PY) -m pytest tests -q -p no:warnings

test-unit: ## Unit tests
	$(PY) -m pytest tests/unit -q -p no:warnings

test-api: ## API tests
	$(PY) -m pytest tests/api -q -p no:warnings

test-integration: ## Integration tests (PostgreSQL test is skipped unless TEST_DATABASE_URL is set)
	$(PY) -m pytest tests/integration -q -p no:warnings

lint: ## Ruff lint
	$(PY) -m ruff check .

fmt: ## Ruff format and autofix
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

check: lint config test ## Lint, validate configuration, run tests

# --- versioning & database -------------------------------------------------
dvc-track: ## Version the raw and processed datasets with DVC
	$(PY) -m dvc add data/raw data/processed
	$(PY) -m dvc status

repro: ## Reproduce the DVC pipeline (data -> features -> models)
	$(PY) -m dvc repro

db-upgrade: ## Apply Alembic migrations
	$(PY) -m alembic upgrade head

db-revision: ## Autogenerate a migration (make db-revision M="add table")
	$(PY) -m alembic revision --autogenerate -m "$(M)"

mlflow-ui: ## Local MLflow UI against the SQLite tracking store
	$(PY) -m mlflow ui --backend-store-uri "$(MLFLOW_TRACKING_URI)" --port 5000

# --- docker compose --------------------------------------------------------
compose-up: ## Core stack: postgres + mlflow + api
	$(COMPOSE) --profile $(PROFILE) up -d --build

compose-monitoring: ## Core + prometheus + grafana
	$(COMPOSE) --profile core --profile observability up -d --build

compose-orchestration: ## Core + airflow
	$(COMPOSE) --profile core --profile orchestration up -d --build

compose-full: ## Everything (needs ~6GB RAM)
	$(COMPOSE) --profile core --profile observability --profile orchestration up -d --build

compose-down: ## Stop the stack (volumes are kept)
	$(COMPOSE) --profile core --profile observability --profile orchestration down

compose-logs: ## Follow logs
	$(COMPOSE) logs -f --tail=100

compose-ps: ## Show service health
	$(COMPOSE) --profile core --profile observability --profile orchestration ps

# --- kubernetes ------------------------------------------------------------
k8s-deploy: ## Build in minikube and deploy everything
	./kubernetes/deploy.sh

k8s-verify: ## Verify the cluster and run an end-to-end prediction
	./kubernetes/verify.sh

k8s-delete: ## Delete the namespace
	./kubernetes/deploy.sh --teardown

# --- terraform -------------------------------------------------------------
tf-init: ## terraform init
	cd terraform && terraform init

tf-plan: ## terraform plan
	cd terraform && terraform plan

tf-apply: ## terraform apply (namespace, secret, configmap)
	cd terraform && terraform apply
