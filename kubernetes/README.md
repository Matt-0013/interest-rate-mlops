# Local Kubernetes deployment (minikube)

Minikube is used instead of Kind because this project needs **persistent
volumes across pod restarts** (Postgres data, MLflow artifacts, pipeline
artifacts). Minikube ships a default `standard` StorageClass with a dynamic
hostPath provisioner and a `docker` driver that works inside VirtualBox; Kind
requires an extra local-path provisioner and image-loading step. Nothing here is
minikube-specific beyond that: the manifests are plain Kubernetes.

## What runs in the cluster

| Component | Kind | Port | Storage | Notes |
| --- | --- | --- | --- | --- |
| PostgreSQL | Deployment + Service | 5432 | `mlops-postgres` 2Gi | creates `mlops`, `mlflow`, `airflow` databases |
| MLflow | Deployment + Service | 5000 | `mlops-mlflow-artifacts` 2Gi | tracking + model registry, artifacts served by the tracking server |
| API | Deployment + Service | 8000 | `mlops-data`, `mlops-reports` (ro) | FastAPI + GraphQL, loads `champion` from MLflow |
| Pipeline job | Job | - | same data/reports PVCs | one-shot ingest → train → promote → monitor |
| Prometheus | Deployment + Service | 9090 | `prometheus-data` | scrapes `api:8000/metrics` |
| Grafana | Deployment + Service | 3000 | `grafana-data` | provisioned datasource + 3 dashboards |

Airflow is **not** deployed here on purpose: its `LocalExecutor` in Kubernetes
gives none of the scheduler durability Kubernetes is meant to provide, and a
`KubernetesExecutor` setup would need a second image, a DAG-syncing mechanism and
RBAC. Orchestration runs in Docker Compose (see `docker-compose.yml`,
`--profile orchestration`); the DAGs call exactly the same pipeline functions the
Job calls.

## Layout

```
kubernetes/
├── namespace.yaml              # namespace + resource quota
├── configmaps/api-config.yaml  # non-secret API/pipeline configuration
├── secrets/postgres-secret.example   # template, not applied by kubectl
├── storage/pvcs.yaml           # six PVCs
├── deployments/{postgres,mlflow,api,prometheus,grafana}.yaml   (+ services)
├── services/api.yaml
├── jobs/seed-pipeline.yaml     # one pipeline run inside the cluster
├── deploy.sh                   # build, generate config, apply, seed
├── node-recovery.sh            # repair a cluster booted with `docker start`
└── verify.sh                   # cluster + API verification
```

Secrets and the ConfigMaps derived from repository files (`raw-dataset`,
`prometheus-config`, Grafana provisioning, dashboards, Postgres init script) are
generated at deploy time by `deploy.sh` into `kubernetes/.generated/`
(git-ignored). That keeps one source of truth for those files instead of
duplicating them in YAML.

## Deploy

```bash
# 1. cluster
minikube start --cpus=4 --memory=4096
kubectl config use-context minikube

# 2. data
python scripts/download_data.py --source kaggle

# 3. build images inside minikube's daemon and deploy
./kubernetes/deploy.sh

# 4. verify
./kubernetes/verify.sh
```

`--cpus=4 --memory=4096` is sized for an **8 GB host** — the common case for a
local machine or a VirtualBox guest, which cannot be given more RAM than the
host physically has. Do not raise `--memory` on an 8 GB machine: the VM will
fail to start. The cluster fits in 4 GB because Airflow is deliberately not
deployed here (see `README.md`), so close browsers and editors before
`minikube start` if the API pods restart while waiting for PostgreSQL.

`deploy.sh` builds `interest-rate-api:local` with minikube's Docker environment,
so the image is available without a registry (`imagePullPolicy: IfNotPresent`).
Postgres/Prometheus/Grafana images are pulled from Docker Hub and therefore need
internet access inside the VM.

## After a host reboot

`minikube start` configures the node itself, so the normal path needs nothing
extra. The problem case is a host (or VirtualBox guest) reboot that leaves the
container stopped by the host: the cluster is then resumed with
`docker start minikube`, and the container entrypoint does **not** re-apply the
prerequisites `minikube start` had set up. The cluster reports `Running` but is
unusable, in three independent ways:

| Missing prerequisite | Symptom |
| --- | --- |
| `control-plane.minikube.internal` in the container's `/etc/hosts` (Docker rewrites that file on start) | kube-proxy cannot resolve its kubeconfig server, crashloops, and takes Service networking with it: CoreDNS never becomes Ready, every ClusterIP times out |
| the `br_netfilter` module plus `bridge-nf-call-iptables=1` | pod → ClusterIP traffic is never DNAT'ed to an endpoint; in-cluster DNS and database connections hang |
| kubelet started by the container entrypoint | no node, so pods are never scheduled |

```bash
./kubernetes/node-recovery.sh
```

It restores all three idempotently (re-pinning the hosts entry instead of
appending to it, so a stale IP from a previous run cannot win) and then
*verifies* each one, waiting up to 60 s for the apiserver. Exit status is
non-zero if anything could not be restored. `deploy.sh` runs it automatically
when the minikube container is already up, so a redeploy after a reboot needs no
manual step.

If kube-proxy had already crashlooped before the repair, restart it once the
apiserver answers:

```bash
kubectl -n kube-system rollout restart ds/kube-proxy
kubectl -n mlops rollout status deploy/postgres --timeout=300s
```

## Inspect

```bash
kubectl -n mlops get pods,svc,pvc
kubectl -n mlops logs deploy/api --tail=50
kubectl -n mlops logs job/seed-pipeline

kubectl -n mlops port-forward svc/api 8000:8000         # Swagger: /docs
kubectl -n mlops port-forward svc/grafana 3000:3000    # admin / admin
kubectl -n mlops port-forward svc/mlflow 5000:5000
kubectl -n mlops port-forward svc/prometheus 9090:9090
```

## Design notes

- **Readiness/liveness** differ from Compose: the API's readiness probe is what
  gates Service endpoints, so a pod without a loaded model is never routed.
- **Init containers** wait for `postgres:5432` and `mlflow:5000` with a bounded
  loop, which is more predictable than CrashLoopBackOff races.
- **Resource requests/limits** are sized for a 4 GB minikube VM (the ceiling on
  an 8 GB host). The Job gets the largest limits because training three models
  is the peak workload.
- **One API replica** because the artifacts volume is `ReadWriteOnce`. Horizontal
  scaling would require RWX storage or object storage; this is a deliberate
  local-only simplification, documented in the README's limitations.
- **No hardcoded credentials**: the Secret is generated from `.env` (or local
  defaults) at deploy time and never committed.
- **Idempotent**: `deploy.sh` can be re-run; it re-applies manifests and re-runs
  the pipeline Job, which overwrites artifacts and re-registers a model version
  (promotion gates decide whether the alias moves).
