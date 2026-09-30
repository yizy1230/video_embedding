#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration loader for algo workers.

Each algo worker loads its connection config from a YAML file. The default
path is ``worker_config.yaml`` in the algo's own directory, overridable via
the ``WORKER_CONFIG_PATH`` environment variable.

Secrets are overridable via environment variables for production deployments.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    broker = cfg.setdefault("broker", {})
    broker["url"] = os.environ.get("CELERY_BROKER_URL", broker.get("url"))

    backend = cfg.setdefault("result_backend", {})
    backend["url"] = os.environ.get("CELERY_RESULT_BACKEND", backend.get("url", "rpc://"))

    pg = cfg.setdefault("postgres", {})
    pg["host"] = os.environ.get("PG_HOST", pg.get("host", "localhost"))
    pg["port"] = int(os.environ.get("PG_PORT", pg.get("port", 5432)))
    pg["db"] = os.environ.get("PG_DB", pg.get("db", "pipeline"))
    pg["user"] = os.environ.get("PG_USER", pg.get("user", "pipeline"))
    pg["password"] = os.environ.get("PG_PASSWORD", pg.get("password", "pipeline"))

    mi = cfg.setdefault("minio", {})
    mi["endpoint"] = os.environ.get("MINIO_ENDPOINT", mi.get("endpoint", "localhost:9000"))
    mi["access_key"] = os.environ.get("MINIO_ACCESS_KEY", mi.get("access_key", "minioadmin"))
    mi["secret_key"] = os.environ.get("MINIO_SECRET_KEY", mi.get("secret_key", "minioadmin"))
    mi["secure"] = os.environ.get("MINIO_SECURE", str(mi.get("secure", False))).lower() in ("true", "1")

    cache = cfg.setdefault("cache", {})
    cache["dir"] = os.environ.get("LOCAL_CACHE_DIR", cache.get("dir", "/data/cache/pipeline"))
    cache["enabled"] = os.environ.get("LOCAL_CACHE_ENABLED", str(cache.get("enabled", True))).lower() in ("true", "1", "yes")
    cache["max_size_gb"] = int(os.environ.get("LOCAL_CACHE_MAX_SIZE_GB", cache.get("max_size_gb", 100)))

    return cfg


@lru_cache(maxsize=1)
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load worker config from YAML + env overrides."""
    if path is None:
        path = os.environ.get("WORKER_CONFIG_PATH", "worker_config.yaml")
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Worker config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _apply_env_overrides(cfg)


def sqlalchemy_url(cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or load_config()
    pg = cfg["postgres"]
    return (
        f"postgresql+psycopg2://{pg['user']}:{pg['password']}"
        f"@{pg['host']}:{pg['port']}/{pg['db']}"
    )
