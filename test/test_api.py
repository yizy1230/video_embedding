#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_api.py

Test script for the video embedding server.
Reads an input JSON (same format as the CLI), sends it to the server,
and writes the response to the output directory.

Usage:
    python3 test/test_api.py --input-json test/test_url.json --output-dir test/server_results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def ensure_server_ready(base_url: str, timeout: float = 300.0) -> bool:
    """Poll /ready until the server responds 200 or timeout is reached."""
    ready_url = urljoin(base_url, "/ready")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(ready_url, timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1.0)
    return False


def send_analysis_request(base_url: str, video_path: str, target_fps: float = 2.0) -> dict:
    url = urljoin(base_url, "/video_embedding")
    payload = {"video_path": video_path, "target_fps": target_fps}
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Test video embedding server API")
    parser.add_argument("--input-json", required=True, help="Path to input JSON file containing video_path")
    parser.add_argument("--output-dir", required=True, help="Directory to write the response JSON")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Server base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--target-fps", type=float, default=2.0, help="Target FPS for sampling (default: 2.0)")
    parser.add_argument("--skip-ready-check", action="store_true", help="Skip polling /ready")
    args = parser.parse_args()

    input_path = Path(args.input_json).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        inp = json.load(f)

    video_path = inp.get("video_path")
    if not video_path:
        raise ValueError("Input JSON must contain 'video_path' field")

    # If relative local path, resolve against cwd first, then json parent
    if "://" not in video_path and not Path(video_path).is_absolute():
        cwd_candidate = Path.cwd() / video_path
        json_candidate = input_path.parent / video_path
        if cwd_candidate.exists():
            video_path = str(cwd_candidate.resolve())
        elif json_candidate.exists():
            video_path = str(json_candidate.resolve())
        else:
            # fallback to cwd if neither exists
            video_path = str(cwd_candidate.resolve())

    if not args.skip_ready_check:
        print("[INFO] Waiting for server to be ready ...")
        if not ensure_server_ready(args.base_url):
            sys.exit("[ERROR] Server did not become ready in time.")
        print("[INFO] Server is ready.")

    print(f"[INFO] Sending request for: {video_path}")
    t0 = time.time()
    result = send_analysis_request(args.base_url, video_path, target_fps=args.target_fps)
    elapsed = time.time() - t0
    print(f"[INFO] Request completed in {elapsed:.2f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_name = input_path.stem  # e.g. test_url
    output_file = output_dir / f"{test_name}_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Result written to: {output_file}")


if __name__ == "__main__":
    main()
