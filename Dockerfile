# CUDA-enabled image for training; for CPU-only inference swap the base for
# python:3.11-slim and install the CPU torch wheels.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System libs required by OpenCV / image IO.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project.
COPY . .

# Gradio demo port.
EXPOSE 7860

# Default: launch the demo. Override CMD to train, e.g.
#   docker run --gpus all sketch2photo python src/train.py --config configs/default.yaml
CMD ["python", "src/demo.py", "--ckpt", "outputs/checkpoints/best.pt"]
