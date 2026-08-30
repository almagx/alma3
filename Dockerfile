ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
FROM ${PYTHON_IMAGE} AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && command -v ps >/dev/null \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==2.7.0" \
    && python -m pip check

RUN python - <<'PY'
import platform

import torch

if torch.version.cuda is not None:
    raise SystemExit(f"runtime torch must be CPU-only, found CUDA {torch.version.cuda}")
if platform.machine().lower() in {"amd64", "x86_64"} and "USE_MKL=ON" not in torch.__config__.show():
    raise SystemExit("runtime torch must use MKL on x86-64")
PY

COPY src ./src
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps .

COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/alma3-entrypoint

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 almauser && chown -R almauser:almauser /app
USER almauser

RUN alma3 --version

ENTRYPOINT ["alma3-entrypoint"]
CMD ["alma3", "--help"]

FROM runtime AS standalone

USER root
COPY --from=alma3_release --chown=almauser:almauser / /opt/alma3/release/
USER almauser

ENV ALMA3_RELEASE=/opt/alma3/release

RUN alma3 verify-release
