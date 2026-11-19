#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deploy the interest-rate MLOps platform to a local minikube cluster.
#
#   ./kubernetes/deploy.sh                 # full deploy + one pipeline run
#   ./kubernetes/deploy.sh --skip-seed     # deploy only, seed manually later
#   ./kubernetes/deploy.sh --teardown      # remove the namespace
#
# Prerequisites: minikube running (docker driver), kubectl, docker, and the raw
# dataset at data/raw/federal_reserve_interest_rates/index.csv.
# ---------------------------------------------------------------------------
set -euo pipefail

NAMESPACE="${NAMESPACE:-mlops}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_SEED=false
for arg in "$@"; do
  case "$arg" in
    --skip-seed) SKIP_SEED=true ;;
    --teardown) kubectl delete namespace "$NAMESPACE" --ignore-not-found; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

for bin in kubectl minikube docker; do
  command -v "$bin" >/dev/null || die "$bin not found in PATH"
done
[[ -f data/raw/federal_reserve_interest_rates/index.csv ]] \
  || die "raw dataset missing - run: python scripts/download_data.py --source kaggle"

log "checking cluster"
minikube status >/dev/null 2>&1 || die "minikube is not running - start it first: minikube start --cpus=4 --memory=6144"
context="$(kubectl config current-context)"
[[ "$context" == "minikube" ]] || die "kubectl context is '$context' - run: kubectl config use-context minikube"

# Local defaults; a .env overrides them. In-cluster URLs always use the
# "postgres"/"mlflow" service names, so the compose .env DATABASE_URL (localhost)
# is deliberately not reused.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
POSTGRES_USER="${POSTGRES_USER:-mlops}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-mlops_local_password}"
POSTGRES_DB="${POSTGRES_DB:-mlops}"
GF_USER="${GF_SECURITY_ADMIN_USER:-admin}"
GF_PASSWORD="${GF_SECURITY_ADMIN_PASSWORD:-admin}"
DATABASE_URL_K8S="${DATABASE_URL_K8S:-postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}}"
MLFLOW_BACKEND_STORE_URI_K8S="${MLFLOW_BACKEND_STORE_URI_K8S:-postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/mlflow}"

log "building the API image inside minikube's Docker daemon"
eval "$(minikube -p minikube docker-env)"
docker build -f docker/api.Dockerfile -t interest-rate-api:local .

log "creating namespace"
kubectl apply -f kubernetes/namespace.yaml

log "generating secrets and configmaps"
mkdir -p kubernetes/.generated

kubectl -n "$NAMESPACE" create secret generic postgres-credentials \
  --from-literal=POSTGRES_USER="$POSTGRES_USER" \
  --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  --from-literal=POSTGRES_DB="$POSTGRES_DB" \
  --from-literal=DATABASE_URL="$DATABASE_URL_K8S" \
  --from-literal=MLFLOW_BACKEND_STORE_URI="$MLFLOW_BACKEND_STORE_URI_K8S" \
  --dry-run=client -o yaml > kubernetes/.generated/postgres-secret.yaml

kubectl -n "$NAMESPACE" create secret generic grafana-admin \
  --from-literal=admin-user="$GF_USER" \
  --from-literal=admin-password="$GF_PASSWORD" \
  --dry-run=client -o yaml > kubernetes/.generated/grafana-admin.yaml

kubectl -n "$NAMESPACE" create configmap postgres-init \
  --from-file=init-databases.sh=docker/postgres/init-databases.sh \
  --dry-run=client -o yaml > kubernetes/.generated/postgres-init.yaml

kubectl -n "$NAMESPACE" create configmap raw-dataset \
  --from-file=index.csv=data/raw/federal_reserve_interest_rates/index.csv \
  --dry-run=client -o yaml > kubernetes/.generated/raw-dataset.yaml

kubectl -n "$NAMESPACE" create configmap prometheus-config \
  --from-file=prometheus.yml=monitoring/prometheus/prometheus.yml \
  --dry-run=client -o yaml > kubernetes/.generated/prometheus-config.yaml

kubectl -n "$NAMESPACE" create configmap grafana-datasources \
  --from-file=prometheus.yml=monitoring/grafana/provisioning/datasources/prometheus.yml \
  --dry-run=client -o yaml > kubernetes/.generated/grafana-datasources.yaml

kubectl -n "$NAMESPACE" create configmap grafana-dashboard-provider \
  --from-file=dashboards.yml=monitoring/grafana/provisioning/dashboards/dashboards.yml \
  --dry-run=client -o yaml > kubernetes/.generated/grafana-dashboard-provider.yaml

kubectl -n "$NAMESPACE" create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards \
  --dry-run=client -o yaml > kubernetes/.generated/grafana-dashboards.yaml

kubectl apply -f kubernetes/.generated/

log "applying manifests"
for kind in configmaps storage deployments services; do
  kubectl apply -R -f "kubernetes/$kind"
done

log "waiting for rollouts"
kubectl -n "$NAMESPACE" rollout status deployment/postgres --timeout=180s
kubectl -n "$NAMESPACE" rollout status deployment/mlflow --timeout=300s
kubectl -n "$NAMESPACE" rollout status deployment/api --timeout=300s
kubectl -n "$NAMESPACE" rollout status deployment/prometheus --timeout=180s
kubectl -n "$NAMESPACE" rollout status deployment/grafana --timeout=180s

if [[ "$SKIP_SEED" == "false" ]]; then
  log "running the pipeline once inside the cluster (ingest, train, promote, monitor)"
  kubectl -n "$NAMESPACE" delete job seed-pipeline --ignore-not-found
  kubectl apply -f kubernetes/jobs/seed-pipeline.yaml
  if ! kubectl -n "$NAMESPACE" wait --for=condition=complete job/seed-pipeline --timeout=1200s; then
    kubectl -n "$NAMESPACE" logs job/seed-pipeline --tail=80 || true
    die "seed pipeline failed - inspect: kubectl -n $NAMESPACE logs job/seed-pipeline"
  fi
  log "restarting the API so it picks up the newly registered champion"
  kubectl -n "$NAMESPACE" rollout restart deployment/api
  kubectl -n "$NAMESPACE" rollout status deployment/api --timeout=300s
fi

log "deployed"
kubectl -n "$NAMESPACE" get pods
cat <<EOF

Next steps:
  kubectl -n $NAMESPACE port-forward svc/api 8000:8000
  python scripts/verify_api.py --api-url http://localhost:8000
  kubectl -n $NAMESPACE port-forward svc/grafana 3000:3000   # http://localhost:3000
  kubectl -n $NAMESPACE port-forward svc/mlflow 5000:5000    # http://localhost:5000
  kubectl -n $NAMESPACE port-forward svc/prometheus 9090:9090

EOF
