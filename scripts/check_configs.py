"""Validate all configuration files.

Runs the same strict Pydantic loaders the pipelines use, then checks the
container, Kubernetes, monitoring and CI configuration for structural problems
that would only surface at deploy time. Exits non-zero on any failure, so CI can
gate on it.

Usage:
    python scripts/check_configs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from interest_rate_mlops.config import (
    PROJECT_ROOT,
    load_data_config,
    load_monitoring_config,
    load_pipeline_config,
)

errors: list[str] = []

REQUIRED_COMPOSE_SERVICES = {"postgres", "mlflow", "api"}
REQUIRED_K8S_DEPLOYMENTS = {"postgres", "mlflow", "api", "prometheus", "grafana"}


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"ok    {name}")
    else:
        errors.append(f"{name}: {detail or 'failed'}")
        print(f"FAIL  {name} {detail}")


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_pydantic_configs() -> None:
    data_cfg = load_data_config()
    pipeline_cfg = load_pipeline_config()
    monitoring_cfg = load_monitoring_config()

    check("configs/data.yaml loads", True)
    check(
        "data.yaml target has a column and slug",
        bool(data_cfg.target.column and data_cfg.target.slug),
    )
    check(
        "data.yaml validation covers the target column",
        data_cfg.target.column in data_cfg.validation.required_columns,
        "target missing from required_columns",
    )
    check("configs/model.yaml loads", True)
    check(
        "model.yaml defines the three compared models",
        {"linear_regression", "random_forest", "xgboost"} <= set(pipeline_cfg.models),
        f"found {sorted(pipeline_cfg.models)}",
    )
    check(
        "model.yaml test size matches the documented 80/20 split",
        abs(pipeline_cfg.split.test_fraction - 0.2) < 1e-9,
    )
    check(
        "model.yaml declares a champion and challenger alias",
        bool(pipeline_cfg.mlflow.champion_alias and pipeline_cfg.mlflow.challenger_alias),
    )
    check("configs/monitoring.yaml loads", True)
    check(
        "monitoring.yaml drift shares are ordered probabilities",
        0 <= monitoring_cfg.drift.warning_share < monitoring_cfg.drift.critical_share <= 1,
    )
    check(
        "monitoring.yaml separates drift from performance thresholds",
        monitoring_cfg.performance.rmse_degradation_warning_pct
        < monitoring_cfg.performance.rmse_degradation_critical_pct,
    )


def validate_compose() -> None:
    path = PROJECT_ROOT / "docker-compose.yml"
    document = _load_yaml(path)
    services = document.get("services", {})
    missing = REQUIRED_COMPOSE_SERVICES - set(services)
    check("docker-compose.yml defines required services", not missing, f"missing {sorted(missing)}")

    for name, service in services.items():
        if not service.get("profiles"):
            errors.append(f"compose service {name} has no profile (would always start)")
        if name.startswith("airflow-"):
            continue
        check(
            f"compose service {name} has a healthcheck",
            "healthcheck" in service,
            "no healthcheck",
        )
    check(
        "compose declares named volumes",
        bool(document.get("volumes")),
        "no top-level volumes",
    )


def validate_kubernetes() -> None:
    deployment_names: set[str] = set()
    kinds: dict[str, int] = {}
    for path in sorted((PROJECT_ROOT / "kubernetes").rglob("*.yaml")):
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if not document:
                continue
            kind = document.get("kind")
            kinds[kind] = kinds.get(kind, 0) + 1
            check(f"{path.name}: {kind} has metadata", "metadata" in document)
            if kind == "Deployment":
                name = document["metadata"]["name"]
                deployment_names.add(name)
                spec = document["spec"]["template"]["spec"]
                container = spec["containers"][0]
                check(f"{name}: resources set", "resources" in container)
                check(f"{name}: readiness probe", "readinessProbe" in container)
                check(f"{name}: liveness probe", "livenessProbe" in container)
            if kind == "Deployment" and document["metadata"]["name"] == "api":
                env = document["spec"]["template"]["spec"]["containers"][0].get("envFrom", [])
                check(
                    "api: configuration comes from ConfigMap/Secret",
                    len(env) >= 2,
                    "expected configMapRef + secretRef",
                )

    check(
        "kubernetes defines the required deployments",
        deployment_names >= REQUIRED_K8S_DEPLOYMENTS,
        f"missing {sorted(REQUIRED_K8S_DEPLOYMENTS - deployment_names)}",
    )
    check(
        "kubernetes declares persistent volume claims", kinds.get("PersistentVolumeClaim", 0) >= 4
    )
    check("kubernetes declares a namespace", kinds.get("Namespace", 0) >= 1)


def validate_monitoring() -> None:
    prometheus = _load_yaml(PROJECT_ROOT / "monitoring" / "prometheus" / "prometheus.yml")
    jobs = {job["job_name"] for job in prometheus.get("scrape_configs", [])}
    check("prometheus scrapes the API", "fastapi" in jobs, f"jobs={sorted(jobs)}")

    dashboards = sorted((PROJECT_ROOT / "monitoring" / "grafana" / "dashboards").glob("*.json"))
    check("three Grafana dashboards exist", len(dashboards) == 3, f"found {len(dashboards)}")
    for path in dashboards:
        document = json.loads(path.read_text(encoding="utf-8"))
        check(
            f"dashboard {path.name}: uid and panels",
            bool(document.get("uid") and document.get("panels")),
        )
        for panel in document.get("panels", []):
            check(
                f"dashboard {path.name}: panel {panel.get('id')} has a target",
                bool(panel.get("targets")),
            )

    datasource = _load_yaml(
        PROJECT_ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    )
    check(
        "grafana datasource points at prometheus",
        any(
            "prometheus" in str(source.get("url", ""))
            for source in datasource.get("datasources", [])
        ),
    )


def validate_workflows() -> None:
    workflows = sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yml"))
    check("three CI workflows exist", len(workflows) == 3, f"found {len(workflows)}")
    for path in workflows:
        document = _load_yaml(path)
        check(f"workflow {path.name} has jobs", bool(document.get("jobs")))
        for job_name, job in document.get("jobs", {}).items():
            check(
                f"workflow {path.name}:{job_name} has steps",
                bool(job.get("steps")),
                "no steps",
            )


def main() -> int:
    validate_pydantic_configs()
    validate_compose()
    validate_kubernetes()
    validate_monitoring()
    validate_workflows()

    print()
    if errors:
        print(f"{len(errors)} configuration problem(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("all configuration files are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
