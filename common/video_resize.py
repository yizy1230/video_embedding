#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aspect-preserving downscale helpers for analysis algos.

Ingest operators should keep the source resolution. Analysis operators may
cap the longest side so large inputs do not explode decode memory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

# Longest-side cap: 1080p (1920x1080 / 1080x1920) stays; 4K/8K shrink.
DEFAULT_ANALYSIS_MAX_SIDE = 1920


def resolve_max_side_size(
    width: int,
    height: int,
    max_side: Optional[int],
) -> Tuple[int, int]:
    """Return (width, height) so the longest side is <= max_side.

    Aspect ratio is preserved. Never upscales. ``max_side <= 0`` or ``None``
    disables the cap. Dimensions are at least 1.
    """
    if width <= 0 or height <= 0:
        return max(1, width), max(1, height)
    if max_side is None or int(max_side) <= 0:
        return width, height
    long_side = max(width, height)
    cap = int(max_side)
    if long_side <= cap:
        return width, height
    scale = cap / float(long_side)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_frame_max_side(frame, max_side: Optional[int]):
    """Downscale a HxWxC numpy frame if the longest side exceeds ``max_side``.

    Returns the original array when no resize is needed.
    """
    if frame is None or max_side is None or int(max_side) <= 0:
        return frame
    height, width = frame.shape[:2]
    target_w, target_h = resolve_max_side_size(width, height, max_side)
    if target_w == width and target_h == height:
        return frame
    import cv2  # local import: common is also used by non-cv2 workers

    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def probe_video_wh(video_path: str) -> Tuple[int, int]:
    """Return (width, height) via ffprobe; (0, 0) if probing fails."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0, 0
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        raw = (result.stdout or "").strip().split("x")
        if len(raw) != 2:
            return 0, 0
        return int(raw[0]), int(raw[1])
    except Exception:
        return 0, 0


def maybe_downscale_video(
    video_path: str,
    max_side: Optional[int],
    output_dir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Re-encode ``video_path`` if its longest side exceeds ``max_side``.

    Returns ``(path_to_use, tmp_dir_or_none)``. Caller must
    ``shutil.rmtree(tmp_dir)`` when the temp dir is not ``None``.
    Audio is copied when possible. Used by VLM algos that upload the
    whole file rather than sampled frames.
    """
    if max_side is None or int(max_side) <= 0:
        return video_path, None
    width, height = probe_video_wh(video_path)
    target_w, target_h = resolve_max_side_size(width, height, max_side)
    if width <= 0 or height <= 0 or (target_w == width and target_h == height):
        return video_path, None

    # yuv420p encoders require even dimensions.
    target_w = max(2, target_w - (target_w % 2))
    target_h = max(2, target_h - (target_h % 2))

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    tmp_dir = tempfile.mkdtemp(prefix="algo_max_side_", dir=output_dir)
    out_path = os.path.join(tmp_dir, "downscaled.mp4")
    scale = f"scale={target_w}:{target_h}:flags=bicubic"
    encoders = [
        ["-c:v", "libx264", "-preset", "fast", "-crf", "23"],
        ["-c:v", "libopenh264", "-b:v", "2M"],
        ["-c:v", "mpeg4", "-q:v", "5"],
    ]
    last_err = ""
    for enc in encoders:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path, "-vf", scale,
        ] + enc + [
            "-c:a", "aac", "-ar", "16000", "-ac", "1",
            "-movflags", "+faststart",
            out_path,
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return out_path, tmp_dir
        last_err = (ret.stderr or last_err).strip()
        if os.path.exists(out_path):
            os.remove(out_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    raise RuntimeError(f"ffmpeg max_side downscale failed ({width}x{height} -> {target_w}x{target_h}): {last_err}")
