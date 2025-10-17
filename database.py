from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker


raw_database_url = os.environ.get("DATABASE_URL", "sqlite:///./sora_world.db")

# Automatically upgrade legacy PostgreSQL URLs to use the psycopg driver bundled in
# requirements. This avoids needing the deprecated psycopg2 package at runtime.
if raw_database_url.startswith("postgresql://"):
    raw_database_url = raw_database_url.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )

DATABASE_URL = raw_database_url

# Use check_same_thread=False for SQLite so background threads can access.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = scoped_session(
    sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
)

Base = declarative_base()


def init_db() -> None:
    from models import Scene, SceneMetric  # noqa: F401 - ensure metadata is registered

    Base.metadata.create_all(bind=engine)
    _ensure_schema()


def _ensure_schema() -> None:
    statements = [
        "ALTER TABLE scenes ADD COLUMN progress INTEGER",
        "ALTER TABLE scenes ADD COLUMN progress_updated_at TIMESTAMP",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            try:
                conn.execute(text(stmt))
            except Exception:
                continue


@contextmanager
def session_scope() -> Iterator:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
