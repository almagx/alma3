FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS runtime

WORKDIR /app

COPY pyproject.toml README.md LICENSE requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==2.7.0" \
    && python -m pip check

RUN python - <<'PY'
import torch

if torch.version.cuda is not None:
    raise SystemExit(f"runtime torch must be CPU-only, found CUDA {torch.version.cuda}")
PY

COPY src ./src
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps .

COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/alma3-entrypoint

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 almauser && chown -R almauser:almauser /app
USER almauser

RUN alma3 --help

ENTRYPOINT ["alma3-entrypoint"]
CMD ["alma3", "--help"]
