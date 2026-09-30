#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task context read/write helpers for algo workers.

Mirrors engine's storage/postgres.py - provides the state machine interface
that algo workers use to report progress.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from .constants import TaskStatus
from .db import session_scope
from .models import Job, Task


def load_task_context(task_id: str) -> Optional[Dict[str, Any]]:
    """Load task + job info as a detached dict, or None if not found."""
    with session_scope() as session:
        task = session.get(Task, task_id)
        if task is None:
            return None
        job = session.get(Job, task.job_id)
        return {
            "task_id": task.id,
            "job_id": task.job_id,
            "algo_name": task.algo_name,
            "algo_version": task.algo_version,
            "stage": task.stage,
            "clip_index": task.clip_index,
            "clip_ref": task.clip_ref,
            "status": task.status,
            "input": dict(task.input or {}),
            "video_path": job.video_path if job else None,
            "raw_ref": job.raw_ref if job else None,
        }


def mark_running(task_id: str, celery_task_id: Optional[str] = None) -> None:
    with session_scope() as session:
        task = session.get(Task, task_id)
        if task is None:
            return
        task.status = TaskStatus.RUNNING.value
        if celery_task_id:
            task.celery_task_id = celery_task_id
        task.updated_at = datetime.utcnow()


def mark_succeeded(
    task_id: str,
    output: Optional[Dict[str, Any]] = None,
    output_ref: Optional[str] = None,
    timings: Optional[Dict[str, Any]] = None,
) -> None:
    with session_scope() as session:
        task = session.get(Task, task_id)
        if task is None:
            return
        task.status = TaskStatus.SUCCEEDED.value
        task.output = output
        task.output_ref = output_ref
        task.timings = timings
        task.error = None
        task.updated_at = datetime.utcnow()


def mark_failed(task_id: str, error: str, timings: Optional[Dict[str, Any]] = None) -> None:
    with session_scope() as session:
        task = session.get(Task, task_id)
        if task is None:
            return
        task.status = TaskStatus.FAILED.value
        task.error = error
        task.timings = timings
        task.updated_at = datetime.utcnow()


def set_job_raw_ref(job_id: str, raw_ref: str) -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job and not job.raw_ref:
            job.raw_ref = raw_ref
