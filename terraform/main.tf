# ---------------------------------------------------------------------------
# Local Kubernetes prerequisites for the interest-rate MLOps platform.
#
# Terraform owns the *cluster-scoped prerequisites* - namespace, the credential
# Secret and the shared non-secret ConfigMap. The application workloads stay in
# plain Kubernetes manifests under kubernetes/ so they remain readable, diffable
# and usable with kubectl alone.
#
# Ownership rule: each object is defined in exactly one place. The ResourceQuota
# lives in kubernetes/namespace.yaml because the kubectl-only path
# (kubernetes/deploy.sh) must deploy the stack with its guard rail intact, and
# two tools defining the same quota drift apart silently. Terraform therefore
# does not manage it.
#
# Why not Terraform everything? The workload manifests contain many long
# commands, probes and volume mounts; expressing them as HCL templates would add
# indirection without adding safety, and the manifests are already validated in
# CI. Keeping one tool per concern is the simpler engineering choice.
#
# Limitations of local providers (documented, not hidden):
#   * there is no local "cloud" to provision - the cluster already exists;
#   * state is local and must not be shared or committed (contains the password);
#   * drift detection only covers the resources declared here.
# ---------------------------------------------------------------------------

resource "kubernetes_namespace_v1" "mlops" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/part-of"    = "interest-rate-mlops"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }
}

locals {
  # In-cluster service DNS names, never localhost: the API and MLflow reach
  # PostgreSQL through the "postgres" Service in this namespace.
  database_url             = "postgresql+psycopg://${var.postgres_user}:${var.postgres_password}@postgres:5432/${var.postgres_db}"
  mlflow_backend_store_uri = "postgresql://${var.postgres_user}:${var.postgres_password}@postgres:5432/mlflow"
}

resource "kubernetes_secret_v1" "postgres_credentials" {
  metadata {
    name      = "postgres-credentials"
    namespace = kubernetes_namespace_v1.mlops.metadata[0].name
    labels = {
      "app.kubernetes.io/part-of" = "interest-rate-mlops"
    }
  }

  type = "Opaque"

  data = {
    POSTGRES_USER             = var.postgres_user
    POSTGRES_PASSWORD         = var.postgres_password
    POSTGRES_DB               = var.postgres_db
    DATABASE_URL              = local.database_url
    MLFLOW_BACKEND_STORE_URI  = local.mlflow_backend_store_uri
  }
}

resource "kubernetes_config_map_v1" "api_config" {
  metadata {
    name      = "api-config"
    namespace = kubernetes_namespace_v1.mlops.metadata[0].name
    labels = {
      "app.kubernetes.io/part-of" = "interest-rate-mlops"
    }
  }

  data = {
    APP_ENV                 = "production"
    LOG_LEVEL               = var.log_level
    LOG_FORMAT              = "json"
    MLFLOW_TRACKING_URI     = "http://mlflow:5000"
    MODEL_NAME              = var.model_name
    MODEL_ALIAS             = "champion"
    MODEL_CACHE_TTL_SECONDS = "300"
    MLOPS_PROJECT_ROOT      = "/app"
    ENABLE_DB_PERSISTENCE   = "true"
    ENABLE_DVC              = "false"
    DATABASE_POOL_SIZE      = "5"
  }
}
