# Overview

Build video, image, and text embeddings for later deduplication and retrieval.

# v1_0

Embeddings are produced with SigLIP 2.

```
Text  --> siglip2 --> Embedding
Image --> siglip2 --> Embedding
Video --> Sampling (fps 1) --> Video Clips --> siglip2 --> Embeddings --> Mean Pooling --> Embedding
```

Checkpoint: `google/siglip2-so400m-patch16-384`

# Development

## Local environment

```bash
conda create --name video_embedding python=3.11 -y
conda activate video_embedding
pip install -r requirements.txt
# web UI
pip install flask fastapi loguru gradio==4.36.1 gradio_client==1.0.1
# profiling
pip install psutil memory_profiler
# Jupyter
pip install ipykernel && python -m ipykernel install --user --name video_embedding --display-name "video_embedding"
```

## Local scripts

`video_embedding.py`

Input JSON:

```json
{
  "video_path": "xxxx"
}
```

`video_path` may be a local file or a URL.

Output JSON:

```json
{
  "video_path": "xxxx",
  "embeddings": {
    "siglip2": [0.1, 0.2]
  },
  "timings": {
    "load_models": 1.12,
    "load_video_frames": 1.12,
    "get_embedding": 1.12,
    "download_video": 2.50,
    "total": 3.36
  }
}
```

`embeddings.siglip2` is a one-dimensional vector. `download_video` is present only when the input is a URL.

```bash
CUDA_VISIBLE_DEVICES=1
python video_embedding.py --input-json test/test_local_1.json --output-json test/result_local_1.json
python video_embedding.py --input-json test/test_local_2.json --output-json test/result_local_2.json
python video_embedding.py --input-json test/test_local_3.json --output-json test/result_local_3.json
python video_embedding.py --input-json test/test_url.json --output-json test/result_url.json
```

`video_embedding_server.py` starts a FastAPI server.

Routes:

- `GET /ready` — service is ready when the status is 200
- `GET /test` — run the predefined test cases
- `POST /video_embedding` — embed one video

Request body:

```json
{
  "video_path": "path_or_url_to_video"
}
```

Response body:

```json
{
  "video_path": "xxxx",
  "embeddings": {
    "siglip2": [0.1, 0.2]
  },
  "timings": {
    "load_models": null,
    "load_video_frames": 1.12,
    "get_embedding": 1.12,
    "download_video": 2.50,
    "total": 3.36
  },
  "algo_name": "video_embedding",
  "algo_version": "v1_0"
}
```

`load_models` is null because the server loads the model at startup. `download_video` is present only for a URL input.

```bash
python video_embedding_server.py --device cuda --port 8000
```

`test/test_api.py`:

```bash
python3 test/test_api.py --input-json test/test_local_1.json --output-dir test/server_results
python3 test/test_api.py --input-json test/test_url.json --output-dir test/server_results
```

## Parameters

`video_embedding.py` (CLI)

- `--input-json` — input JSON path (required)
- `--output-json` — output JSON path (required)
- `--device` — `cpu` or `cuda` (default: cuda if available, otherwise cpu)
- `--fps` — video sample rate (default: 2.0). It changes the number of frames and the runtime.

`video_embedding_server.py` (FastAPI)

- `--device` — `cpu` or `cuda` (default: cuda if available, otherwise cpu)
- `--host` — bind address (default: `0.0.0.0`)
- `--port` — bind port (default: 8000)
- `--fps` — default sample rate when the request omits it (default: 2.0)

`POST /video_embedding` body

- `video_path` — local path or URL (required)
- `target_fps` — sample rate (optional; defaults to the server `--fps`)

Internal constants shared by the CLI and the server

- `CKPT_PATH = "./checkpoints/siglip2-so400m-patch16-384"` — SigLIP 2 checkpoint
- `BATCH_SIZE = 32` — frames per batch; affects GPU memory and speed
- `DEFAULT_FPS = 2.0` — code default; the server can override it with `--fps`

## Evaluation

## Profiling

## Demo

`video_embedding_webui.py`

## Build

Build the worker from this repository root:

```bash
docker build -f Dockerfile -t algo-video_embedding:v1_0 .
```

## Weights

Weight files are not stored in this repository. Download URLs and destination paths are in `checkpoints/WEIGHTS.txt`.

## Citation

```bibtex
@software{video_embedding,
  title   = {video_embedding},
  year    = {2026},
  author  = {yizy},
  license = {Apache-2.0}
}
```

## License

Code written for this repository is under the Apache License 2.0. See [LICENSE](LICENSE). Third-party directories keep their own licenses. See [NOTICE](NOTICE).

## Release

- **v1.0.0** — First public release. Video, image, and text embeddings with SigLIP 2, plus a Celery worker.

## Future Work

- Support embedding models other than SigLIP 2, selected by configuration.
