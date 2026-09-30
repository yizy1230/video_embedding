#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy engine/session for algo workers.

Mirrors the engine's pipeline_api/db.py but uses the worker config.
"""

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import load_config, sqlalchemy_url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = load_config()
    return create_engine(
        sqlalchemy_url(cfg),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def get_session() -> Session:
    return _session_factory()()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
