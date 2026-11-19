"""Interest-rate MLOps platform.

Modules are organised by lifecycle stage so that training, serving and
orchestration share one implementation:

* :mod:`interest_rate_mlops.ingestion`      - dataset discovery, sources, profiling
* :mod:`interest_rate_mlops.validation`     - structural + Pandera schema gates
* :mod:`interest_rate_mlops.preprocessing`  - monthly panel construction
* :mod:`interest_rate_mlops.features`       - leakage-safe feature engineering
* :mod:`interest_rate_mlops.models`         - estimator factory
* :mod:`interest_rate_mlops.training`       - splits, walk-forward CV, training
* :mod:`interest_rate_mlops.evaluation`     - metrics, plots, comparison report
* :mod:`interest_rate_mlops.registry`       - MLflow registry access + promotion gates
* :mod:`interest_rate_mlops.monitoring`     - Evidently drift + performance decay
* :mod:`interest_rate_mlops.database`       - SQLAlchemy models and repositories
* :mod:`interest_rate_mlops.pipelines`      - orchestration entry points (Airflow/CLI)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
