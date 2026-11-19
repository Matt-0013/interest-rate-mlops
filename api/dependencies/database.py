"""Database lifecycle for the API.

The engine is created once at startup and reused; requests get a short-lived
session that commits on success and rolls back on failure. Predictions cannot be
served without a place to record them, so a missing database is a 503 rather
than a silent audit gap.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi import HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from interest_rate_mlops.config import Settings
from interest_rate_mlops.database.session import create_engine_from_url, create_session_factory
from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)


class Database:
    """Owns the SQLAlchemy engine used by the API process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: Any | None = None
        self.session_factory: Any | None = None

    def connect(self) -> bool:
        """Create the engine and verify connectivity. Never raises at startup."""
        try:
            engine = create_engine_from_url(
                self.settings.database_url,
                echo=self.settings.database_echo,
                pool_size=self.settings.database_pool_size,
            )
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            logger.error(
                "database connection failed at startup: %s",
                str(exc).splitlines()[0][:200],
                extra={"event": "database_startup_failed", "status_code": 503},
            )
            return False
        self.engine = engine
        self.session_factory = create_session_factory(engine)
        logger.info("database connected", extra={"event": "database_connected"})
        return True

    @property
    def healthy(self) -> bool:
        return self.session_factory is not None

    def check(self) -> tuple[bool, str]:
        """Live connectivity check used by the health endpoints."""
        if self.session_factory is None:
            return False, "engine not initialised"
        try:
            with self.session_factory() as session:
                session.execute(text("SELECT 1"))
            return True, "SELECT 1 succeeded"
        except SQLAlchemyError as exc:
            return False, str(exc).splitlines()[0][:200]

    def get_session(self) -> Iterator[Session]:
        """FastAPI dependency yielding a transactional session."""
        if self.session_factory is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable: predictions cannot be persisted",
            )
        session: Session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            self.session_factory = None


def get_database(request: Request) -> Database:
    """Resolve the process-wide database holder from application state."""
    return request.app.state.db


def get_session(request: Request) -> Iterator[Session]:
    """Request-scoped session dependency."""
    database: Database = request.app.state.db
    yield from database.get_session()
