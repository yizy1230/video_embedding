#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants for algo workers. Must stay in sync with engine."""

from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Stage(str, Enum):
    """Task stages written by the engine into ``tasks.stage``.

    - PRIMARY: tasks created at job start (infra + primary-level algos)
    - FANOUT:  tasks created per chunk/clip/segment (fan-out or direct mode)

    VIDEO/CLIP are deprecated aliases kept for backward-compatible reads.
    """
    PRIMARY = "primary"
    FANOUT = "fanout"

    VIDEO = "video"
    CLIP = "clip"
