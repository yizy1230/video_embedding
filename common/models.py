#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ORM models shared between engine and algo workers.

These must stay in sync with ``pipeline_api/models.py`` in the engine repository.
Only the minimal subset needed for worker state read/write is defined here.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    video_path = Column(Text, nullable=False)
    raw_ref = Column(Text, nullable=True)
    pipeline_version = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False, default="pending", index=True)
    params = Column(JSONB, nullable=False, default=dict)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    tasks = relationship("Task", back_populates="job", cascade="all, delete-orphan")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    job_id = Column(UUID(as_uuid=False), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)

    algo_name = Column(String(64), nullable=False)
    algo_version = Column(String(32), nullable=False)
    stage = Column(String(16), nullable=False)
    clip_index = Column(Integer, nullable=True)
    clip_ref = Column(Text, nullable=True)

    status = Column(String(32), nullable=False, default="pending", index=True)
    depends_on = Column(ARRAY(String), nullable=False, default=list)

    input = Column(JSONB, nullable=False, default=dict)
    output = Column(JSONB, nullable=True)
    output_ref = Column(Text, nullable=True)
    timings = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    celery_task_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    job = relationship("Job", back_populates="tasks")

    __table_args__ = (
        Index("ix_tasks_job_status", "job_id", "status"),
        Index("ix_tasks_job_stage_clip", "job_id", "stage", "clip_index"),
    )
