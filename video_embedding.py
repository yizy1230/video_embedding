#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_embedding.py

CLI script to extract video embeddings using SigLIP2.

Usage:
    python video_embedding.py --input-json test/test_local_1.json --output-json test/result_local_1.json
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoModel, AutoProcessor

CKPT_PATH = "./checkpoints/siglip2-so400m-patch16-384"
DEFAULT_FPS = 2.0
BATCH_SIZE = 32
# Decode-span safety cap: 5 min @ 30 fps is a normal clip.
DEFAULT_MAX_DURATION = 300.0
DEFAULT_MAX_DECODE_FRAMES = 9000

_ALGOS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ALGOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGOS_ROOT))
from common.video_resize import DEFAULT_ANALYSIS_MAX_SIDE, resize_frame_max_side  # noqa: E402


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


def _truncate_video(video_path, output_dir, max_duration):
    """Truncate video+audio to max_duration seconds (from 0) via ffmpeg."""
    stem = Path(video_path).stem
    out_path = Path(output_dir) / f"{stem}_truncated_{max_duration}s.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-t", str(max_duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "16000", "-ac", "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return str(out_path)


def _decode_frame_cap(
    claimed: int,
    fps: float,
    duration: float,
    max_duration: float,
    max_frames=None,
) -> int:
    """Cap decode span by container duration / max_duration / max_frames.

    Never trust a claimed frame count that exceeds the container timeline.
    """
    fps = fps if fps and fps > 0 else 30.0
    span_sec = max_duration if max_duration and max_duration > 0 else DEFAULT_MAX_DURATION
    if duration > 0:
        span_sec = min(duration, span_sec)
    cap = int(span_sec * fps) + 1
    if max_frames is not None and max_frames > 0:
        cap = min(cap, int(max_frames))
    if claimed > 0:
        cap = min(cap, claimed)
    return max(1, cap)


def load_video_frames(
    video_path: str,
    target_fps: float = DEFAULT_FPS,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_frames=None,
    max_side=DEFAULT_ANALYSIS_MAX_SIDE,
):
    """Load video and sample frames at target_fps, returning a list of PIL Images.

    Decode span is capped by container duration / max_duration / max_frames,
    not the claimed frame count (which can be inflated by broken timestamps
    and cause VideoReader seek/read to hang).
    """
    container_duration = _get_video_duration(video_path)
    vr = VideoReader(video_path, ctx=cpu(0))
    original_fps = float(vr.get_avg_fps()) or 0.0
    claimed = len(vr)
    total_frames = _decode_frame_cap(
        claimed, original_fps, container_duration, max_duration, max_frames,
    )

    if claimed == 0 and container_duration <= 0:
        raise ValueError(f"No frames found in video: {video_path}")

    span_sec = max_duration if max_duration and max_duration > 0 else DEFAULT_MAX_DURATION
    if container_duration > 0:
        span_sec = min(container_duration, span_sec)
    num_samples = max(1, int(span_sec * target_fps))
    if max_frames is not None and max_frames > 0:
        num_samples = min(num_samples, int(max_frames))
    num_samples = min(num_samples, total_frames)

    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C) uint8
    images = [Image.fromarray(resize_frame_max_side(frame, max_side)) for frame in frames]
    return images


def get_video_embedding(model, processor, images, device):
    """Get video embedding by mean pooling frame embeddings."""
    embeddings_list = []

    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i : i + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)

        with torch.no_grad():
            # batch_embeddings = model.get_image_features(**inputs).pooler_output
            batch_embeddings = model.get_image_features(**inputs)
            # batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=-1)
            embeddings_list.append(batch_embeddings.cpu())

    all_embeddings = torch.cat(embeddings_list, dim=0)  # (N, D)
    video_embedding = torch.mean(all_embeddings, dim=0)  # (D,)
    video_embedding = torch.nn.functional.normalize(
        video_embedding.unsqueeze(0), p=2, dim=-1
    ).squeeze(0)
    return video_embedding.numpy().tolist()


def main():
    parser = argparse.ArgumentParser(description="Video Embedding with SigLIP2")
    parser.add_argument("--input-json", required=True, help="Path to input JSON file")
    parser.add_argument("--output-json", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Target FPS for sampling (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help="Maximum video duration in seconds; longer videos will be truncated (default: 300)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_ANALYSIS_MAX_SIDE,
        help="Max longest side for decoded frames (aspect-preserving; 0 disables)",
    )
    args = parser.parse_args()

    # Read input JSON
    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)

    video_path = inp.get("video_path")
    if not video_path:
        raise ValueError("Input JSON must contain 'video_path' field")

    # Determine if URL or local file
    parsed = urlparse(video_path)
    is_url = parsed.scheme in ("http", "https")

    temp_file = None
    timings = {}
    t0_total = time.time()
    try:
        if is_url:
            # Download video to a temp file
            print(f"Downloading video from URL: {video_path}")
            t0 = time.time()
            response = requests.get(video_path, timeout=300)
            response.raise_for_status()
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.write(response.content)
            temp_file.close()
            video_file_path = temp_file.name
            timings["download_video"] = round(time.time() - t0, 2)
        else:
            # Resolve relative path against input JSON directory first, then CWD
            if not Path(video_path).is_absolute():
                input_dir = Path(args.input_json).parent
                json_relative = input_dir / video_path
                cwd_relative = Path.cwd() / video_path
                if json_relative.exists():
                    video_file_path = str(json_relative.resolve())
                elif cwd_relative.exists():
                    video_file_path = str(cwd_relative.resolve())
                else:
                    raise FileNotFoundError(f"Video not found: {video_path}")
            else:
                video_file_path = video_path

        # Load models
        t0 = time.time()
        print("Loading SigLIP2 model...")
        device = torch.device(args.device)
        model = AutoModel.from_pretrained(CKPT_PATH).to(device).eval()
        processor = AutoProcessor.from_pretrained(CKPT_PATH)
        timings["load_models"] = round(time.time() - t0, 2)
        print(f"Model loaded on device: {device}")

        # Truncate if video is longer than max_duration
        truncated_path = None
        truncate_tmp_dir = None
        duration = _get_video_duration(video_file_path)
        if duration > args.max_duration:
            print(f"Video duration {duration:.2f}s exceeds max_duration {args.max_duration:.2f}s, truncating")
            truncate_tmp_dir = tempfile.mkdtemp(prefix="video_trunc_")
            truncated_path = _truncate_video(video_file_path, truncate_tmp_dir, args.max_duration)
            video_file_path = truncated_path

        # Load video frames
        t0 = time.time()
        print(f"Loading video frames (target_fps={args.fps})...")
        images = load_video_frames(
            video_file_path, target_fps=args.fps, max_duration=args.max_duration,
            max_side=args.max_side,
        )
        timings["load_video_frames"] = round(time.time() - t0, 2)
        print(f"Loaded {len(images)} frames")

        # Get embedding
        t0 = time.time()
        print("Extracting embeddings...")
        embedding = get_video_embedding(model, processor, images, device)
        timings["get_embedding"] = round(time.time() - t0, 2)

        timings["total"] = round(time.time() - t0_total, 2)

        # Build output
        result = {
            "video_path": video_path,
            "embeddings": {
                "siglip2": embedding,
            },
            "timings": timings,
        }

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Result saved to: {args.output_json}")

    finally:
        if temp_file is not None:
            os.unlink(temp_file.name)
        if truncate_tmp_dir is not None:
            shutil.rmtree(truncate_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
