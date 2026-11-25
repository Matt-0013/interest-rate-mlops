#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Verify the Kubernetes deployment.
#
#   ./kubernetes/verify.sh
#
# Checks: pods ready, PVCs bound, in-cluster service reachability, and a real
# inference call through a port-forwarded API (health, model info, predict,
# persistence, drift status, metrics, GraphQL).
# ---------------------------------------------------------------------------
set -euo pipefail

NAMESPACE="${NAMESPACE:-mlops}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
API_PORT="${API_PORT:-8000}"
failures=0

section() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mFAIL\033[0m %s\n' "$*"; failures=$((failures + 1)); }

section "pods"
kubectl -n "$NAMESPACE" get pods -o wide
kubectl -n "$NAMESPACE" get pods --no-headers \
  | awk '$3 != "Running" && $3 != "Completed" {print $1, $3}' | while read -r pod status; do
      echo "not ready: $pod ($status)"
    done

section "services and storage"
kubectl -n "$NAMESPACE" get svc
kubectl -n "$NAMESPACE" get pvc

section "persistent volume claims bound"
unbound="$(kubectl -n "$NAMESPACE" get pvc --no-headers | awk '$2 != "Bound" {print $1}')"
if [[ -n "$unbound" ]]; then fail "unbound PVCs: $unbound"; else echo "all PVCs bound"; fi

# The check pod must ask for resources (mlops-quota rejects pods without
# requests/limits) and must use an image already on the node, so the check also
# works without registry access.
section "in-cluster connectivity (pod -> api:8000)"
kubectl -n "$NAMESPACE" delete pod api-curl-check --ignore-not-found >/dev/null 2>&1
if cat <<'POD' | kubectl -n "$NAMESPACE" apply --validate=false -f - >/dev/null 2>&1
apiVersion: v1
kind: Pod
metadata:
  name: api-curl-check
  labels:
    app.kubernetes.io/name: api-curl-check
spec:
  restartPolicy: Never
  containers:
    - name: check
      image: interest-rate-api:local
      imagePullPolicy: IfNotPresent
      command:
        - python
        - -c
        - |
          import urllib.request
          body = urllib.request.urlopen("http://api:8000/health", timeout=10).read()
          print(body.decode())
      resources:
        requests: { cpu: 50m, memory: 64Mi }
        limits: { cpu: 200m, memory: 128Mi }
POD
then
  if kubectl -n "$NAMESPACE" wait --for=jsonpath='{.status.phase}'=Succeeded \
        pod/api-curl-check --timeout=180s >/dev/null 2>&1; then
    kubectl -n "$NAMESPACE" logs api-curl-check
    echo "in-cluster service reachability OK"
  else
    kubectl -n "$NAMESPACE" logs api-curl-check 2>&1 | tail -5
    fail "the API was not reachable from another pod"
  fi
  kubectl -n "$NAMESPACE" delete pod api-curl-check --ignore-not-found >/dev/null 2>&1
else
  fail "could not create the in-cluster connectivity check pod"
fi

section "API end to end via port-forward"
kubectl -n "$NAMESPACE" port-forward svc/api "${API_PORT}:8000" >/tmp/mlops-port-forward.log 2>&1 &
PF_PID=$!
trap 'kill "$PF_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
fi

if ! "$PYTHON" scripts/verify_api.py --api-url "http://localhost:${API_PORT}"; then
  fail "API verification reported failures"
fi

printf '\n'
if [[ "$failures" -gt 0 ]]; then
  printf '\033[1;31m%s check(s) failed\033[0m\n' "$failures"
  exit 1
fi
printf '\033[1;32mKubernetes verification passed\033[0m\n'
