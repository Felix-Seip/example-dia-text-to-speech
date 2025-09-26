# ---------- Base ----------
FROM python:3.11-slim

# Copy uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Workdir
WORKDIR /app

# System deps:
# - libsndfile1: required by soundfile
# - libgomp1: OpenMP runtime required by PyTorch CPU wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# (Optional) Make Hugging Face cache a volume-friendly path
ENV HF_HOME=/root/.cache/huggingface

# Copy dependency files first for cache-friendly builds
COPY pyproject.toml uv.lock ./

# Install project venv & deps (without project files yet)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy the rest of the project
COPY . .

# Install project itself into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Create output dir
RUN mkdir -p audio_files

# FastAPI port
EXPOSE 8000

# (Optional) Nice defaults for PyTorch CPU perf & log noise
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

# Run the API
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0"]
