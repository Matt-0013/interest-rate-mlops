variable "kubeconfig_path" {
  description = "Path to the kubeconfig file for the local cluster."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "kubectl context to target. Must be a local cluster (minikube)."
  type        = string
  default     = "minikube"

  validation {
    condition     = !can(regex("(gke|eks|aks)_", var.kube_context))
    error_message = "This configuration is local-only; managed cloud contexts are refused."
  }
}

variable "namespace" {
  description = "Namespace that hosts the MLOps platform."
  type        = string
  default     = "mlops"
}

variable "postgres_user" {
  description = "Database user created for the application and MLflow."
  type        = string
  default     = "mlops"
}

variable "postgres_password" {
  description = "Database password. Supplied via TF_VAR_postgres_password or a tfvars file that is not committed."
  type        = string
  sensitive   = true
}

variable "postgres_db" {
  description = "Application database name."
  type        = string
  default     = "mlops"
}

variable "model_name" {
  description = "Registered MLflow model name the API serves."
  type        = string
  default     = "interest_rate_forecaster"
}

variable "log_level" {
  description = "Log level for the API and pipeline jobs."
  type        = string
  default     = "INFO"
}
