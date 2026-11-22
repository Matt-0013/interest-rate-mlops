# Terraform (local Kubernetes prerequisites)

Terraform manages the **cluster-scoped prerequisites** for the local minikube
deployment:

- the `mlops` namespace,
- the `postgres-credentials` Secret (user, password, application URL, MLflow URL),
- the shared non-secret `api-config` ConfigMap.

The resource quota that keeps the stack inside a single-node local VM (4 GB of
minikube memory on an 8 GB host) is **not** managed here:
it lives in `kubernetes/namespace.yaml`, because the kubectl-only path
(`kubernetes/deploy.sh`) must deploy the stack with that guard rail intact. Every
object in this project has exactly one owner, so the quota is never defined twice.

The application workloads (Deployments, Services, PVCs, the pipeline Job) stay in
plain manifests under `kubernetes/`. They are long, imperative and benefit from
being readable YAML that `kubectl diff` and the CI config check can inspect;
wrapping them in HCL templates would add indirection without adding safety.

## Usage

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars     # git-ignored
terraform init
terraform plan
terraform apply
```

Then deploy the workloads (the script re-applies the same Secret and ConfigMap,
so Terraform and the manifests agree):

```bash
./kubernetes/deploy.sh
./kubernetes/verify.sh
```

Destroy only the Terraform-managed resources:

```bash
terraform destroy
```

## Security and state

- `postgres_password` is marked `sensitive` and is never printed in plan output.
  For anything beyond a laptop, pass it as `TF_VAR_postgres_password` from a
  secret store instead of a tfvars file.
- The state file contains the password in plain text. It is git-ignored and must
  stay local. A shared environment would need an encrypted remote backend, which
  this project deliberately does not use (local only).
- The kube context is validated: managed cloud contexts (`gke_`, `eks_`, `aks_`)
  are rejected by `variables.tf`.

## Limitations of local providers

- There is no cloud to provision: the cluster must already be running
  (`minikube start`). Terraform configures what already exists.
- No `terraform destroy` of the cluster itself; use `minikube delete`.
- Drift detection covers only the three resources above - the workloads and the
  resource quota are reconciled by `kubectl apply`, not by Terraform.
- The Kubernetes provider needs a reachable API server; `terraform plan` fails
  when the cluster is stopped. That is expected, not a misconfiguration.
