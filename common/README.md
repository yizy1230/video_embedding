# common — shared Celery worker library

Shared task runner and infrastructure clients for operator workers.

## Modules

| Module | Role |
|--------|------|
| `worker_base.py` | `create_worker_app` builds the Celery app; `execute_algo_task` is the standard task lifecycle |
| `config.py` | Load `worker_config.yaml` and apply environment overrides |
| `storage.py` | MinIO client and a node-local disk cache with LRU eviction |
| `task_context.py` | Read and write PostgreSQL task state (`load` / `mark_running` / `mark_succeeded` / `mark_failed`) |
| `models.py` | SQLAlchemy ORM models (`Job` / `Task`), kept in sync with the engine |
| `constants.py` | Job, task, and stage status enums |
| `db.py` | SQLAlchemy engine and session factory |
| `worker_config.template.yaml` | Worker config template |
| `video_resize.py` | Aspect-preserving `max_side` downscale for analysis operators |

## Usage

An operator worker is about 30–50 lines:

```python
import os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ALGOS_ROOT = SCRIPT_DIR.parent.parent.resolve()
sys.path.insert(0, str(ALGOS_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("WORKER_CONFIG_PATH", str(SCRIPT_DIR / "worker_config.yaml"))

from common.worker_base import create_worker_app, execute_algo_task
from video_xxx import process_video

app = create_worker_app("video_xxx")

def run_fn(local_input_path, params):
    return process_video(local_input_path, **params)

@app.task(name="algo.video_xxx", bind=True, max_retries=2, acks_late=True)
def run_task(self, task_id):
    return execute_algo_task(task_id, self.request.id, run_fn)
```

## execute_algo_task lifecycle

1. Load the task context from PostgreSQL.
2. Set the task status to `RUNNING`.
3. Fetch the input from MinIO or the local cache.
4. Call `run_fn(local_input_path, params)`.
5. Post-process: store the result in MinIO or Milvus (override with `postprocess_fn`).
6. Set the task status to `SUCCEEDED` or `FAILED`.
7. Send `orchestrate.advance(job_id)` to move the state machine forward.
8. Run the cache LRU eviction check.

## Local cache

- The cache directory is `cache.dir` in `worker_config.yaml`.
- Several workers on the same node can share the cache (for example, several clip tasks of one job).
- LRU eviction deletes the oldest files when the cache exceeds `cache.max_size_gb`.
- Environment overrides: `CACHE_DIR`, `CACHE_ENABLED`, `CACHE_MAX_SIZE_GB`.

## Model preload

A worker that keeps a GPU model resident can hook Celery `worker_process_init`:

```python
from celery.signals import worker_process_init

@worker_process_init.connect
def _preload_and_warmup(**kwargs):
    # 1. Load weights onto the GPU.
    # 2. Run one dummy forward pass to warm CUDA kernels.
    pass
```

Operators that need resident weights should connect this signal in their own worker, load the checkpoint at process start, and run one empty forward pass.
