#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base task execution framework for algo celery workers.

Provides:
    - ``create_worker_app(algo_name)`` : build a Celery app for an algo
    - ``execute_algo_task(task_id, celery_task_id, run_fn, postprocess_fn)`` :
      the standard task lifecycle (fetch → run → store → advance)

Each algo worker only needs to:
    1. Call ``create_worker_app("video_xxx")`` to get a Celery app
    2. Define a task that calls ``execute_algo_task(...)`` with its own ``run_fn``

Example (video_statics_worker.py):

    from algos.common.worker_base import create_worker_app, execute_algo_task
    from video_statics import analyze_video

    app = create_worker_app("video_statics")

    @app.task(name="algo.video_statics", bind=True, max_retries=2)
    def run_task(self, task_id: str):
        def run_fn(local_path, params):
            return analyze_video(local_path, device=params.get("device", "cpu"))
        return execute_algo_task(task_id, self.request.id, run_fn)
"""

import os
import time
import traceback
from typing import Any, Callable, Dict, Optional

from celery import Celery
from loguru import logger

from . import storage
from .config import load_config
from .constants import TaskStatus
from .task_context import (
    load_task_context,
    mark_failed,
    mark_running,
    mark_succeeded,
)


def _source_stage_of(ctx: Dict[str, Any]) -> str:
    """Resolve the storage stage ('raw'|'clips') the input file lives in.

    Prefers the explicit ``_source_stage`` hint the engine writes into the
    task input; falls back to a clip_index heuristic for older records.
    """
    hint = (ctx.get("input") or {}).get("_source_stage")
    if hint in ("raw", "clips"):
        return hint
    return "clips" if ctx.get("clip_index") is not None else "raw"


def _result_part_of(ctx: Dict[str, Any]) -> str:
    """Return the result-key path segment: 'video' or 'clip_<index>'."""
    clip_index = ctx.get("clip_index")
    return "video" if clip_index is None else f"clip_{clip_index}"


def create_worker_app(algo_name: str) -> Celery:
    """Create a Celery app configured for an algo worker."""
    cfg = load_config()

    broker_url = cfg["broker"]["url"]
    # Disable AMQP heartbeat. With --pool=solo the worker process is blocked
    # during long-running tasks and cannot respond to broker heartbeats,
    # causing "Connection reset by peer" and task ack failures.
    # NOTE: Celery ignores ?heartbeat= in broker_url and keeps the default
    # broker_heartbeat=120; must set the conf key explicitly.
    app = Celery(algo_name)
    app.conf.update(
        broker_url=broker_url,
        broker_heartbeat=0,
        result_backend=cfg.get("result_backend", {}).get("url", "rpc://"),
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        timezone="UTC",
        enable_utc=True,
        broker_connection_retry=True,
        broker_connection_max_retries=None,
        worker_enable_remote_control=False,
    )
    return app


def _enqueue_advance(job_id: str, cfg: Dict[str, Any]) -> None:
    """Send orchestrate.advance to the orchestrate queue."""
    advance_app = Celery("advance_sender")
    advance_app.conf.update(
        broker_url=cfg["broker"]["url"],
        result_backend=cfg.get("result_backend", {}).get("url", "rpc://"),
    )
    advance_app.send_task(
        "orchestrate.advance",
        kwargs={"job_id": job_id},
        queue="orchestrate",
    )


def _default_postprocess(
    ctx: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    """Default post-processing: store full result JSON to MinIO results bucket.

    Returns {output, output_ref} to be saved in the task record.
    """
    job_id = ctx["job_id"]
    algo_name = ctx["algo_name"]

    bucket_name = storage.bucket("bucket_results")
    clip_part = _result_part_of(ctx)
    key = f"{job_id}/{clip_part}/{algo_name}.json"
    storage.put_json(bucket_name, key, result)

    summary = {k: v for k, v in result.items()
               if k in ("status", "algo_name", "algo_version")}
    return {"output": summary, "output_ref": key}


# Type alias for the algo run function:
#   run_fn(local_input_path: str, params: dict) -> dict
RunFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]
PostprocessFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def execute_algo_task(
    task_id: str,
    celery_task_id: Optional[str] = None,
    run_fn: Optional[RunFn] = None,
    postprocess_fn: Optional[PostprocessFn] = None,
) -> Dict[str, Any]:
    """Execute an algo task end-to-end.

    Args:
        task_id: The DB task ID to process.
        celery_task_id: Celery-assigned task ID for tracking.
        run_fn: The algo's core function. Signature: (local_path, params) -> result_dict.
        postprocess_fn: Optional custom post-processing. Defaults to storing
                        the full result JSON in MinIO. Signature: (ctx, result) -> {output, output_ref}.
    """
    cfg = load_config()
    ctx = load_task_context(task_id)
    if ctx is None:
        logger.error("task {} not found; skipping", task_id)
        return {"status": "missing"}

    job_id = ctx["job_id"]
    algo_name = ctx["algo_name"]
    if ctx.get("status") == TaskStatus.SUCCEEDED.value:
        logger.info("[{}] task {} already succeeded; skip redelivery", algo_name, task_id)
        _enqueue_advance(job_id, cfg)
        return {"status": "succeeded", "task_id": task_id}

    logger.info("[{}] task {} (job {}) start", algo_name, task_id, job_id)
    mark_running(task_id, celery_task_id)

    t0 = time.time()
    try:
        # 1) Fetch input file from cache/MinIO
        object_key = ctx["input"].get("object_key")
        stage = _source_stage_of(ctx)
        local_input = None
        if object_key:
            local_input = storage.fetch_input(job_id, object_key, stage=stage)

        # 2) Run the algo. Strip transport-only keys (object_key) and any
        #    engine-internal keys (prefixed with '_') so they don't leak into
        #    the algo's function signature.
        params = {
            k: v for k, v in ctx["input"].items()
            if k != "object_key" and not k.startswith("_")
        }
        params["job_id"] = job_id

        if run_fn is None:
            raise RuntimeError(f"[{algo_name}] run_fn not provided")

        raw_result = run_fn(local_input, params)

        # 3) Post-process (store results)
        postprocess = postprocess_fn or _default_postprocess
        post = postprocess(ctx, raw_result)

        # 4) Mark succeeded
        timings = {
            "elapsed": round(time.time() - t0, 3),
            "algo_timings": raw_result.get("timings", {}),
        }
        mark_succeeded(
            task_id,
            output=post["output"],
            output_ref=post["output_ref"],
            timings=timings,
        )
        logger.info("[{}] task {} succeeded in {}s", algo_name, task_id, timings["elapsed"])
        status = "succeeded"

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[{}] task {} FAILED: {}\n{}", algo_name, task_id, exc, tb)
        mark_failed(task_id, error=str(exc), timings={"elapsed": round(time.time() - t0, 3)})
        status = "failed"

    finally:
        _enqueue_advance(job_id, cfg)
        storage.cache_evict_if_needed()

    return {"status": status, "task_id": task_id}
