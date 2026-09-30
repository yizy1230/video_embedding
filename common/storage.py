#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MinIO + local cache storage client for algo workers.

Provides file fetch/upload with local cache support, mirroring the engine's
storage/minio_client.py + storage/local_cache.py.
"""

import io
import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from minio import Minio

from .config import load_config


# ---------------------------------------------------------------------------
# MinIO client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    cfg = load_config()["minio"]
    client = Minio(
        cfg["endpoint"],
        access_key=cfg["access_key"],
        secret_key=cfg["secret_key"],
        secure=bool(cfg.get("secure", False)),
    )
    _ensure_buckets(client, cfg)
    return client


def _ensure_buckets(client: Minio, cfg: dict) -> None:
    for key in ("bucket_raw", "bucket_clips", "bucket_results"):
        name = cfg.get(key)
        if not name:
            continue
        try:
            if not client.bucket_exists(name):
                client.make_bucket(name)
                logger.info("created MinIO bucket: {}", name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bucket check failed for {}: {}", name, exc)


def bucket(name_key: str) -> str:
    """Resolve a bucket name from config key (e.g. 'bucket_raw')."""
    return load_config()["minio"][name_key]


def upload_file(bucket_name: str, object_key: str, local_path: str) -> str:
    client = get_minio_client()
    client.fput_object(bucket_name, object_key, local_path)
    logger.info("uploaded {} -> {}/{}", local_path, bucket_name, object_key)
    return object_key


def download_file(bucket_name: str, object_key: str, local_path: str) -> str:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    get_minio_client().fget_object(bucket_name, object_key, local_path)
    return local_path


def put_json(bucket_name: str, object_key: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    stream = io.BytesIO(payload)
    get_minio_client().put_object(
        bucket_name, object_key, stream, length=len(payload),
        content_type="application/json",
    )
    return object_key


# ---------------------------------------------------------------------------
# Local cache
# ---------------------------------------------------------------------------

def _cache_config() -> dict:
    return load_config().get("cache", {})


def cache_enabled() -> bool:
    return _cache_config().get("enabled", True)


def cache_dir() -> Path:
    return Path(_cache_config().get("dir", "/data/cache/pipeline"))


def _mp4_has_moov(path: Path) -> bool:
    """Reject in-progress ffmpeg writes that have size but no moov atom."""
    try:
        size = path.stat().st_size
        if size <= 0:
            return False
        with path.open("rb") as f:
            head = f.read(min(size, 256 * 1024))
            if b"moov" in head:
                return True
            if size > 256 * 1024:
                f.seek(max(0, size - 256 * 1024))
                if b"moov" in f.read(256 * 1024):
                    return True
        return False
    except OSError:
        return False


def cache_get(job_id: str, filename: str, stage: str = "raw") -> Optional[str]:
    """Return cached local path if exists and non-empty, else None."""
    if not cache_enabled():
        return None
    if stage == "raw":
        path = cache_dir() / job_id / "raw" / filename
    else:
        path = cache_dir() / job_id / "clips" / filename
    if not path.exists() or path.stat().st_size <= 0:
        return None
    if path.suffix.lower() == ".mp4" and not _mp4_has_moov(path):
        logger.warning("cache SKIP incomplete mp4 (no moov): {}", path)
        return None
    path.touch()
    logger.debug("cache HIT: {}", path)
    return str(path)


def cache_put_direct(job_id: str, filename: str, stage: str = "raw") -> str:
    """Return the cache path for direct download. Ensures parent dir exists."""
    if stage == "raw":
        path = cache_dir() / job_id / "raw" / filename
    else:
        path = cache_dir() / job_id / "clips" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def cache_put(job_id: str, filename: str, source_path: str, stage: str = "raw") -> str:
    """Copy a file into local cache. Returns the cached path."""
    if not cache_enabled():
        return source_path
    dest_str = cache_put_direct(job_id, filename, stage)
    if os.path.abspath(source_path) == os.path.abspath(dest_str):
        return dest_str
    dest = Path(dest_str)
    if dest.exists():
        dest.unlink()
    try:
        os.link(source_path, dest_str)
    except OSError:
        shutil.copy2(source_path, dest_str)
    return dest_str


def cache_evict_if_needed() -> None:
    """Remove oldest job directories if total cache exceeds max_size_gb."""
    if not cache_enabled():
        return
    cfg = _cache_config()
    max_bytes = cfg.get("max_size_gb", 100) * 1024 * 1024 * 1024
    root = cache_dir()
    if not root.exists():
        return
    job_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not job_dirs:
        return
    total_size = sum(f.stat().st_size for d in job_dirs for f in d.rglob("*") if f.is_file())
    if total_size <= max_bytes:
        return
    job_dirs.sort(key=lambda d: d.stat().st_mtime)
    for d in job_dirs:
        if total_size <= max_bytes * 0.8:
            break
        dir_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        logger.info("cache EVICT: {} ({:.1f} MB)", d.name, dir_size / 1024 / 1024)
        shutil.rmtree(d, ignore_errors=True)
        total_size -= dir_size


# ---------------------------------------------------------------------------
# Combined fetch: cache-first, then MinIO
# ---------------------------------------------------------------------------

def fetch_input(job_id: str, object_key: str, stage: str = "raw") -> str:
    """Fetch input file: check local cache first, fall back to MinIO download."""
    filename = os.path.basename(object_key)

    cached = cache_get(job_id, filename, stage=stage)
    if cached:
        return cached

    bucket_key = "bucket_clips" if stage == "clips" else "bucket_raw"
    bucket_name = bucket(bucket_key)

    if cache_enabled():
        local_path = cache_put_direct(job_id, filename, stage=stage)
    else:
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="algo_")
        local_path = os.path.join(tmp_dir, filename)

    download_file(bucket_name, object_key, local_path)
    logger.info("fetched from MinIO: {}/{} -> {}", bucket_name, object_key, local_path)
    return local_path
