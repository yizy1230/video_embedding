#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_embedding_server.py

FastAPI server for video embedding using SigLIP2.

Routes:
    GET  /ready              -> Check if service is ready
    GET  /test               -> Run predefined test cases and return results
    POST /video_embedding    -> Process a video embedding request

Usage:
    python video_embedding_server.py --port 8000 --device cuda
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import uvicorn
from decord import VideoReader, cpu
from fastapi import FastAPI, HTTPException
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor

_ALGOS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ALGOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGOS_ROOT))
from common.video_resize import DEFAULT_ANALYSIS_MAX_SIDE, resize_frame_max_side  # noqa: E402

CKPT_PATH = "./checkpoints/siglip2-so400m-patch16-384"
DEFAULT_FPS = 2.0
BATCH_SIZE = 32


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
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


def _truncate_video(video_path, output_dir, max_duration):
    """Truncate and remux timestamps. Avoid a pure ``-c copy`` that keeps bad PTS."""
    stem = Path(video_path).stem
    out_path = Path(output_dir) / f"{stem}_truncated_{max_duration}s.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    fps = _probe_fps(str(video_path)) or 30.0
    remux = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-t", str(max_duration),
        "-c", "copy",
        "-bsf:v", f"setts=pts=N/{fps:.6f}/TB:dts=N/{fps:.6f}/TB",
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        "-movflags", "+faststart",
        str(out_path),
    ]
    ret = subprocess.run(remux, capture_output=True)
    if ret.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
        return str(out_path)

    encoders = [
        ["-c:v", "libx264", "-preset", "fast", "-crf", "23"],
        ["-c:v", "libopenh264", "-b:v", "2M"],
        ["-c:v", "mpeg4", "-q:v", "5"],
    ]
    last_err = ret.stderr.decode(errors="ignore").strip()
    for enc in encoders:
        if out_path.exists():
            out_path.unlink()
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-t", str(max_duration),
        ] + enc + [
            "-c:a", "aac", "-ar", "16000", "-ac", "1",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            "-movflags", "+faststart",
            str(out_path),
        ]
        ret2 = subprocess.run(cmd, capture_output=True)
        if ret2.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
            return str(out_path)
        last_err = ret2.stderr.decode(errors="ignore").strip() or last_err

    raise RuntimeError(f"Failed to truncate video (ffmpeg error): {last_err}")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class VideoEmbeddingRequest(BaseModel):
    video_path: str = Field(..., description="Path or URL to the video")
    target_fps: Optional[float] = Field(None, description="Target FPS for frame sampling (default: server default_fps)")
    max_duration: Optional[float] = Field(None, description="Maximum video duration in seconds; longer videos will be truncated")
    max_side: Optional[int] = Field(None, description="Max longest side for decoded frames (aspect-preserving; 0 disables)")


class VideoEmbeddingResponse(BaseModel):
    video_path: str
    embeddings: dict
    timings: dict
    algo_name: str
    algo_version: str
    cuda_mem: Optional[dict] = None


# ---------------------------------------------------------------------------
# Core helpers (sync – executed in thread pool to avoid blocking the event loop)
# ---------------------------------------------------------------------------
def _decode_frame_cap(claimed, fps, duration, max_duration, max_frames=None):
    fps = fps if fps and fps > 0 else 30.0
    span_sec = max_duration if max_duration and max_duration > 0 else 300.0
    if duration > 0:
        span_sec = min(duration, span_sec)
    cap = int(span_sec * fps) + 1
    if max_frames is not None and max_frames > 0:
        cap = min(cap, int(max_frames))
    if claimed > 0:
        cap = min(cap, claimed)
    return max(1, cap)


def _load_video_frames(
    video_path: str,
    target_fps: float = DEFAULT_FPS,
    max_duration: float = 300.0,
    max_side: Optional[int] = DEFAULT_ANALYSIS_MAX_SIDE,
):
    """Load video and sample frames at target_fps, returning a list of PIL Images.

    Decode span is capped by container duration / max_duration, not the
    claimed frame count from VideoReader.
    """
    container_duration = _get_video_duration(video_path)
    vr = VideoReader(video_path, ctx=cpu(0))
    original_fps = float(vr.get_avg_fps()) or 0.0
    claimed = len(vr)
    total_frames = _decode_frame_cap(claimed, original_fps, container_duration, max_duration)

    if claimed == 0 and container_duration <= 0:
        raise ValueError(f"No frames found in video: {video_path}")

    span_sec = max_duration if max_duration and max_duration > 0 else 300.0
    if container_duration > 0:
        span_sec = min(container_duration, span_sec)
    num_samples = max(1, min(total_frames, int(span_sec * target_fps)))
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C) uint8
    images = [Image.fromarray(resize_frame_max_side(frame, max_side)) for frame in frames]
    return images


def _get_video_embedding(images, model, processor, device):
    """Get video embedding by mean-pooling frame embeddings."""
    embeddings_list = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_embeddings = model.get_image_features(**inputs)
            embeddings_list.append(batch_embeddings.cpu())
        del inputs, batch_embeddings

    all_embeddings = torch.cat(embeddings_list, dim=0)  # (N, D)
    video_embedding = torch.mean(all_embeddings, dim=0)  # (D,)
    video_embedding = torch.nn.functional.normalize(
        video_embedding.unsqueeze(0), p=2, dim=-1
    ).squeeze(0)
    result = video_embedding.numpy().tolist()
    del all_embeddings, video_embedding, embeddings_list
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _resolve_video_path(video_path: str) -> tuple[str, Optional[tempfile.NamedTemporaryFile], Optional[float]]:
    """
    Resolve a local path or URL and return:
        (absolute file path, optional temp file handle, download_time or None)
    """
    parsed = urlparse(video_path)
    is_url = parsed.scheme in ("http", "https")

    if is_url:
        logger.info(f"Downloading video from URL: {video_path}")
        t0 = time.time()
        resp = requests.get(video_path, timeout=300)
        resp.raise_for_status()
        download_time = round(time.time() - t0, 2)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name, tmp, download_time

    # Local file
    if not Path(video_path).is_absolute():
        cwd_candidate = Path.cwd() / video_path
        if cwd_candidate.exists():
            return str(cwd_candidate.resolve()), None, None
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path, None, None


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading SigLIP2 model...")
    device = torch.device(app.state.device)
    app.state.model = AutoModel.from_pretrained(CKPT_PATH).to(device).eval()
    app.state.processor = AutoProcessor.from_pretrained(CKPT_PATH)
    logger.info(f"Model loaded on device: {device}")
    yield


app = FastAPI(title="Video Embedding Server", version="v1_0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/ready")
def ready():
    ready_flag = hasattr(app.state, "model") and hasattr(app.state, "processor")
    if not ready_flag:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ok"}


@app.post("/video_embedding", response_model=VideoEmbeddingResponse)
def video_embedding(req: VideoEmbeddingRequest):
    t0_total = time.time()
    timings = {}

    temp_file = None
    try:
        # Resolve video (download if URL)
        video_file_path, temp_file, download_time = _resolve_video_path(req.video_path)
        if download_time is not None:
            timings["download_video"] = download_time

        # Truncate if needed
        max_duration = req.max_duration if req.max_duration is not None else getattr(app.state, 'default_max_duration', 300.0)
        duration = _get_video_duration(video_file_path)
        truncated_path = None
        truncate_tmp_dir = None
        if duration > max_duration:
            logger.info(f"Video duration {duration:.2f}s exceeds max_duration {max_duration:.2f}s, truncating")
            truncate_tmp_dir = tempfile.mkdtemp(prefix="video_trunc_")
            try:
                truncated_path = _truncate_video(video_file_path, truncate_tmp_dir, max_duration)
                video_file_path = truncated_path
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                logger.warning(f"Truncation failed ({exc}), using original video")

        # Load frames
        t0 = time.time()
        target_fps = req.target_fps if req.target_fps is not None else getattr(app.state, 'default_fps', DEFAULT_FPS)
        max_side = req.max_side if req.max_side is not None else DEFAULT_ANALYSIS_MAX_SIDE
        images = _load_video_frames(
            video_file_path, target_fps=target_fps, max_duration=max_duration,
            max_side=max_side,
        )
        timings["load_video_frames"] = round(time.time() - t0, 2)
        logger.info(f"Loaded {len(images)} frames for {req.video_path}")

        # Extract embedding
        t0 = time.time()
        embedding = _get_video_embedding(
            images, app.state.model, app.state.processor, app.state.model.device
        )
        timings["get_embedding"] = round(time.time() - t0, 2)

        timings["load_models"] = None  # model already loaded at startup
        timings["total"] = round(time.time() - t0_total, 2)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error during video embedding")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file is not None:
            os.unlink(temp_file.name)
        if 'truncate_tmp_dir' in dir() and truncate_tmp_dir is not None:
            shutil.rmtree(truncate_tmp_dir, ignore_errors=True)

    cuda_mem = {}
    if torch.cuda.is_available():
        cuda_mem = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        }
        torch.cuda.empty_cache()
    return VideoEmbeddingResponse(
        video_path=req.video_path,
        embeddings={"siglip2": embedding},
        timings=timings,
        algo_name="video_embedding",
        algo_version="v1_0",
        cuda_mem=cuda_mem if cuda_mem else None,
    )


@app.get("/test")
def test_endpoint():
    test_dir = Path("test")
    test_files = sorted(test_dir.glob("test_*.json"))
    if not test_files:
        raise HTTPException(status_code=404, detail="No test JSON files found in test/")

    results = []
    for tf in test_files:
        with open(tf, "r", encoding="utf-8") as f:
            payload = json.load(f)
        video_path = payload.get("video_path")
        target_fps = payload.get("target_fps")
        try:
            req = VideoEmbeddingRequest(video_path=video_path, target_fps=target_fps)
            resp = video_embedding(req)
            results.append({"test_file": str(tf), "status": "ok", "result": resp.dict()})
        except HTTPException as e:
            results.append({"test_file": str(tf), "status": "error", "detail": e.detail})
        except Exception as e:
            results.append({"test_file": str(tf), "status": "error", "detail": str(e)})

    return {"tests": results}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Video Embedding Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for model inference (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Default target FPS for frame sampling (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=300.0,
        help="Default maximum video duration in seconds; longer videos will be truncated (default: 300)",
    )
    args, _ = parser.parse_known_args()

    app.state.device = args.device
    app.state.default_fps = args.fps
    app.state.default_max_duration = args.max_duration
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
