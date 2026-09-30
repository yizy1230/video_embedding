#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery worker for video_embedding.

Consumes the ``video_embedding`` queue. Loads the SigLIP2 model once at
startup (warm start) and reuses it for all subsequent tasks.

Start command:
    celery -A video_embedding_worker.app worker -Q video_embedding -c 1 -n video_embedding@%h
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
ALGOS_ROOT = SCRIPT_DIR.parent.parent.resolve()
if str(ALGOS_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGOS_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("WORKER_CONFIG_PATH", str(SCRIPT_DIR / "worker_config.yaml"))

from common.worker_base import create_worker_app, execute_algo_task  # noqa: E402
from video_embedding import (  # noqa: E402
    DEFAULT_FPS,
    DEFAULT_MAX_DECODE_FRAMES,
    DEFAULT_MAX_DURATION,
    get_video_embedding,
    load_video_frames,
)
from common.video_resize import DEFAULT_ANALYSIS_MAX_SIDE  # noqa: E402

import time

from celery.signals import worker_process_init
from loguru import logger
from PIL import Image

CKPT_PATH = str(SCRIPT_DIR / "checkpoints" / "siglip2-so400m-patch16-384")

app = create_worker_app("video_embedding")

_model = None
_processor = None
_device = None
_DEFAULT_DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def _ensure_model(device_str: str = "cuda"):
    global _model, _processor, _device
    if _model is None:
        from transformers import AutoModel, AutoProcessor
        _device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        _model = AutoModel.from_pretrained(CKPT_PATH).to(_device).eval()
        _processor = AutoProcessor.from_pretrained(CKPT_PATH)


@worker_process_init.connect
def _preload_and_warmup(**kwargs):
    """Preload SigLIP2 model and run warmup inference on worker startup."""
    t0 = time.time()
    logger.info("[video_embedding] preloading SigLIP2 on device={}", _DEFAULT_DEVICE)

    _ensure_model(_DEFAULT_DEVICE)

    # Warmup: run a dummy forward pass with a small synthetic image
    dummy_images = [Image.new("RGB", (384, 384), color=(128, 128, 128))]
    with torch.no_grad():
        inputs = _processor(images=dummy_images, return_tensors="pt").to(_device)
        _model.get_image_features(**inputs)

    logger.info("[video_embedding] preload + warmup done in {:.2f}s", time.time() - t0)


def _get_video_duration(video_path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            pass
    return 0.0


def _probe_fps(video_path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        raw = result.stdout.strip()
        if "/" in raw:
            num, den = raw.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else float(num)
        return float(raw)
    except Exception:
        return 0.0


def _truncate_video(video_path: str, tmp_dir: str, max_duration: float) -> str:
    """Truncate and remux timestamps. Avoid a pure ``-c copy`` that keeps bad PTS."""
    out_path = os.path.join(tmp_dir, "truncated.mp4")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    fps = _probe_fps(video_path) or 30.0
    remux = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-t", str(max_duration),
        "-c", "copy",
        "-bsf:v", f"setts=pts=N/{fps:.6f}/TB:dts=N/{fps:.6f}/TB",
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        "-movflags", "+faststart",
        out_path,
    ]
    ret = subprocess.run(remux, capture_output=True, text=True)
    if ret.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return out_path

    if os.path.exists(out_path):
        os.remove(out_path)
    encoders = [
        ["-c:v", "libx264", "-preset", "fast", "-crf", "23"],
        ["-c:v", "libopenh264", "-b:v", "2M"],
        ["-c:v", "mpeg4", "-q:v", "5"],
    ]
    last_err = (ret.stderr or "").strip()
    for enc in encoders:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path, "-t", str(max_duration),
        ] + enc + [
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            "-movflags", "+faststart",
            out_path,
        ]
        ret2 = subprocess.run(cmd, capture_output=True, text=True)
        if ret2.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return out_path
        last_err = (ret2.stderr or last_err).strip()
    raise RuntimeError(f"ffmpeg truncate/remux failed: {last_err}")


def run_fn(local_input_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    import time
    timings = {}
    t0_total = time.time()

    device_str = params.get("device", "cuda")
    fps = params.get("fps", DEFAULT_FPS)
    max_duration = params.get("max_duration", DEFAULT_MAX_DURATION)
    max_frames = params.get("max_frames", DEFAULT_MAX_DECODE_FRAMES)
    max_side = params.get("max_side", DEFAULT_ANALYSIS_MAX_SIDE)

    t0 = time.time()
    _ensure_model(device_str)
    timings["load_models"] = round(time.time() - t0, 3)

    video_path = local_input_path
    truncate_tmp_dir = None
    try:
        duration = _get_video_duration(video_path)
        if duration > max_duration:
            truncate_tmp_dir = tempfile.mkdtemp(prefix="video_trunc_")
            video_path = _truncate_video(video_path, truncate_tmp_dir, max_duration)

        t0 = time.time()
        images = load_video_frames(
            video_path, target_fps=fps, max_duration=max_duration, max_frames=max_frames,
            max_side=max_side,
        )
        timings["load_video_frames"] = round(time.time() - t0, 3)

        t0 = time.time()
        embedding = get_video_embedding(_model, _processor, images, _device)
        timings["get_embedding"] = round(time.time() - t0, 3)
    finally:
        if truncate_tmp_dir:
            shutil.rmtree(truncate_tmp_dir, ignore_errors=True)

    timings["total"] = round(time.time() - t0_total, 3)

    return {
        "video_path": local_input_path,
        "embedding": embedding,
        "algo_name": "video_embedding",
        "algo_version": "v1_0",
        "timings": timings,
    }


@app.task(name="algo.video_embedding", bind=True, max_retries=2, default_retry_delay=10, acks_late=True)
def run_task(self, task_id: str) -> Dict[str, Any]:
    return execute_algo_task(task_id, celery_task_id=self.request.id, run_fn=run_fn)
