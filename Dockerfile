# video_embedding worker
# Python 3.10 | GPU (PyTorch 2.4 + CUDA 12.1) — SigLIP2
#
# Build (from algos/ root):
#   docker build -f video_embedding/v1_0/Dockerfile -t algo-video_embedding:v1_0 .
#
# Run:
#   docker run -d --gpus all --name worker-video_embedding \
#     -e CELERY_BROKER_URL=amqp://guest:guest@<host>:5672// \
#     -e PG_HOST=<host> -e MINIO_ENDPOINT=<host>:9000 \
#     -v /data/cache:/data/cache/pipeline \
#     algo-video_embedding:v1_0
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev \
    libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Worker framework
RUN pip install --no-cache-dir \
    celery sqlalchemy psycopg2-binary minio pyyaml loguru requests

# Algo dependencies (filter out torch*/extra-index-url, provided by base image)
COPY requirements.txt /tmp/reqs.txt
RUN printf "numpy<2\ntorch==2.4.1\ntorchvision==0.19.1\n" > /tmp/constraints.txt \
    && grep -v -E '^(torch==|torchvision==|torchaudio==|--extra-index-url|numpy|opencv-python)' /tmp/reqs.txt > /tmp/reqs_filtered.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/reqs_filtered.txt \
    && pip install --no-cache-dir --no-deps opencv-python==4.13.0.92

COPY common /app/common
COPY . /app/algo
WORKDIR /app/algo

ENV PYTHONPATH=/app:/app/algo
ENV WORKER_CONFIG_PATH=/app/algo/worker_config.yaml
ENV DEVICE=cuda

CMD ["celery", "-A", "video_embedding_worker.app", "worker", \
     "--pool=solo", "-Q", "video_embedding", "-c", "1", "-n", "video_embedding@%h"]
