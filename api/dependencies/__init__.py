"""FastAPI dependencies (database, model cache)."""

from api.dependencies.database import Database, get_database
from api.dependencies.model import ModelHandle, ModelService, get_model_service

__all__ = ["Database", "ModelHandle", "ModelService", "get_database", "get_model_service"]
