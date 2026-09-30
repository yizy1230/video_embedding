#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_embedding_webui.py

Gradio WebUI for video embedding.
Calls the video_embedding_server.py API to process videos.

Usage:
    python video_embedding_webui.py --server-url http://127.0.0.1:8000 --port 7860
"""

# Monkey-patch: fix jinja2>=3.1 LRUCache with dict-type keys (starlette/gradio compat)
import jinja2


class _SafeLRUCache:
    __slots__ = ("capacity", "_mapping", "_queue")

    def __init__(self, capacity):
        self.capacity = capacity
        self._mapping = {}
        self._queue = []

    def _normalize(self, key):
        if isinstance(key, dict):
            return tuple(sorted(key.items()))
        return key

    def get(self, key):
        key = self._normalize(key)
        if key in self._mapping:
            self._remove(key)
            self._push(key)
            return self._mapping[key]
        return None

    def set(self, key, value):
        key = self._normalize(key)
        if key in self._mapping:
            self._remove(key)
        elif len(self._mapping) >= self.capacity:
            del self._mapping[self._queue[0]]
            del self._queue[0]
        self._push(key)
        self._mapping[key] = value

    def _remove(self, key):
        try:
            idx = self._queue.index(key)
            del self._queue[idx]
        except ValueError:
            pass

    def _push(self, key):
        self._queue.append(key)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __contains__(self, key):
        return self._normalize(key) in self._mapping

    def __repr__(self):
        return f"<_SafeLRUCache {len(self._mapping)}/{self.capacity}>"


jinja2.utils.LRUCache = _SafeLRUCache
# end patch

import argparse
import json
import os
import tempfile

import gradio as gr
import requests
from loguru import logger

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"


def ensure_server_ready(server_url: str, timeout: float = 300.0) -> bool:
    """Poll the server /ready endpoint until it responds 200."""
    import time
    ready_url = f"{server_url.rstrip('/')}/ready"
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


def process_video(video_file, video_path_text, target_fps, server_url):
    """
    Process a video through the embedding server.
    If a file is uploaded, use it; otherwise use the text path.
    """
    if video_file is not None:
        video_path = video_file
    elif video_path_text and video_path_text.strip():
        video_path = video_path_text.strip()
    else:
        return None, None, None, None, "Error: Please provide a video file or path."

    url = f"{server_url.rstrip('/')}/video_embedding"
    payload = {
        "video_path": video_path,
        "target_fps": float(target_fps),
    }
    logger.info(f"Sending request to {url} for {video_path} (fps={target_fps})")
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()

    embeddings = result["embeddings"]["siglip2"]
    timings = result.get("timings", {})
    algo_name = result.get("algo_name", "video_embedding")
    algo_version = result.get("algo_version", "v1_0")

    # Format embeddings: first 10 values + dimension
    emb_preview = f"{embeddings[:10]} ... (dim={len(embeddings)})"

    timings_str = json.dumps(timings, ensure_ascii=False, indent=2)
    result_json = json.dumps(result, ensure_ascii=False, indent=2)

    return algo_name, algo_version, emb_preview, timings_str, result_json


def process_video_local(video_file, target_fps, server_url):
    """Wrapper for uploaded file input."""
    return process_video(video_file, None, target_fps, server_url)


def process_video_remote(video_path_text, target_fps, server_url):
    """Wrapper for text path / URL input."""
    return process_video(None, video_path_text, target_fps, server_url)


with gr.Blocks(title="Video Embedding Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 Video Embedding Demo
    **Powered by SigLIP2** – extract embeddings from video files or URLs.

    **Note:** Make sure `video_embedding_server.py` is running before using this UI.
    """)

    with gr.Row():
        with gr.Column():
            server_url = gr.Textbox(
                label="Server URL",
                value=DEFAULT_SERVER_URL,
                placeholder="http://127.0.0.1:8000",
            )

            with gr.Tab("Upload Video"):
                video_input = gr.Video(label="Upload Video")
                target_fps_upload = gr.Number(label="Target FPS", value=2.0, minimum=0.1, maximum=30.0)
                submit_upload = gr.Button("Get Embedding", variant="primary")

            with gr.Tab("Video Path / URL"):
                video_path_input = gr.Textbox(
                    label="Video Path or URL",
                    placeholder="Enter local path or URL to video...",
                    value="https://ark-project.tos-cn-beijing.volces.com/doc_video/ark_vlm_video_input.mp4"
                )
                target_fps_path = gr.Number(label="Target FPS", value=2.0, minimum=0.1, maximum=30.0)
                submit_path = gr.Button("Get Embedding", variant="primary")

        with gr.Column():
            algo_name = gr.Textbox(label="Algorithm Name")
            algo_version = gr.Textbox(label="Algorithm Version")
            embeddings_preview = gr.Textbox(label="Embeddings Preview")
            timings = gr.Textbox(label="Timings", lines=6)
            full_result = gr.Textbox(label="Full Result (JSON)", lines=18)

    submit_upload.click(
        fn=process_video_local,
        inputs=[video_input, target_fps_upload, server_url],
        outputs=[algo_name, algo_version, embeddings_preview, timings, full_result],
    )

    submit_path.click(
        fn=process_video_remote,
        inputs=[video_path_input, target_fps_path, server_url],
        outputs=[algo_name, algo_version, embeddings_preview, timings, full_result],
    )


def main():
    parser = argparse.ArgumentParser(description="Video Embedding WebUI")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"Server base URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio UI (default: 7860)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for Gradio UI (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--wait-server",
        action="store_true",
        help="Wait for the server to become ready before launching UI",
    )
    args = parser.parse_args()

    if args.wait_server:
        logger.info("Waiting for server to become ready...")
        if not ensure_server_ready(args.server_url):
            logger.error("Server did not become ready in time.")
            return
        logger.info("Server is ready.")

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
