output "namespace" {
  description = "Namespace created for the platform."
  value       = kubernetes_namespace_v1.mlops.metadata[0].name
}

output "secret_name" {
  description = "Secret holding the database credentials consumed by the workloads."
  value       = kubernetes_secret_v1.postgres_credentials.metadata[0].name
}

output "config_map_name" {
  description = "ConfigMap with the non-secret runtime configuration."
  value       = kubernetes_config_map_v1.api_config.metadata[0].name
}

output "database_url" {
  description = "In-cluster application database URL (sensitive)."
  value       = local.database_url
  sensitive   = true
}

output "next_steps" {
  description = "What to run after terraform apply."
  value       = <<-EOT
    1. Build the image inside minikube: ./kubernetes/deploy.sh
       (deploy.sh regenerates the same Secret/ConfigMap, so the two stay in sync)
    2. Verify: ./kubernetes/verify.sh
  EOT
}
