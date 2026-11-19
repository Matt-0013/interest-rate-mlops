"""Engine and session management.

Connections are created lazily from ``DATABASE_URL``; credentials never appear
in code. SQLite is supported only so tests can run without a server - the
application targets PostgreSQL.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)


def create_engine_from_url(
    database_url: str,
    *,
    echo: bool = False,
    pool_size: int = 5,
) -> Engine:
    """Create an engine appropriate for the URL scheme.

    SQLite (tests, local smoke runs) needs ``check_same_thread=False`` and a
    ``StaticPool`` for in-memory databases so FastAPI's threadpool and the test
    thread share one database.
    """
    kwargs: dict[str, Any] = {"echo": echo, "future": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = pool_size
        kwargs["max_overflow"] = max(pool_size, 5)

    logger.debug("creating database engine", extra={"event": "engine_created"})
    return create_engine(database_url, **kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory with ``expire_on_commit=False`` for API responses."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on failure."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
