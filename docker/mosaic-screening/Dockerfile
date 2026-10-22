# Image of the mosaic worker used by bagel's MosaicScreening mutation protocol.
#
# Mosaic (torch 2.11, JAX) cannot share a Python environment with bagel's boileroom (torch < 2.7), so it runs as its own
# service: this image contains mosaic at a pinned commit, a pinned dependency lock (the upstream repository ships none and
# has unpinned git dependencies), and bagel's standalone worker. Model weights are NOT baked in: they are downloaded at run
# time into the cache directory (/cache), which should be a persistent volume / bind mount.
#
# Build (context = repository root):
#   docker build -f docker/mosaic-screening/Dockerfile -t bagel-mosaic-screening .
FROM python:3.12-slim-bookworm

# Pinned mosaic commit (escalante-bio/mosaic). Changing it requires regenerating docker/mosaic-screening/uv.lock:
#   git clone https://github.com/escalante-bio/mosaic && cd mosaic && git checkout <commit> && uv lock
ARG MOSAIC_COMMIT=1eb4af64655870b45a715a5a7e61570e8b098701
ENV MOSAIC_COMMIT=${MOSAIC_COMMIT} UV_LINK_MODE=copy

LABEL org.opencontainers.image.source="https://github.com/softnanolab/bagel" \
      org.opencontainers.image.description="Mosaic worker for bagel's MosaicScreening mutation protocol" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

RUN git clone https://github.com/escalante-bio/mosaic /opt/mosaic \
    && git -C /opt/mosaic checkout "${MOSAIC_COMMIT}"

COPY docker/mosaic-screening/uv.lock /opt/mosaic/uv.lock
RUN cd /opt/mosaic && uv sync --frozen --group jax-cuda --no-dev

COPY src/bagel/mosaic_screening/worker.py /opt/svc/worker.py

# The mosaic environment is /opt/mosaic/.venv; its interpreter is called by full path (PATH is left alone so that tools that
# expect the system python, such as Modal's image builder, keep working).
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    MOSAIC_CACHE_DIR=/cache \
    HF_HOME=/cache/huggingface

CMD ["/opt/mosaic/.venv/bin/python", "/opt/svc/worker.py"]
