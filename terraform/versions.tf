terraform {
  required_version = ">= 1.6.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.33"
    }
  }

  # Local state on purpose: this configuration manages a local minikube
  # cluster. The state file contains the generated database password, so it is
  # git-ignored (see .gitignore) - never commit it or move it to a bucket.
}
