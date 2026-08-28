# ALMA3 runtime

This repository contains the inference-only ALMA3 runtime. It loads an explicit,
completed ALMA3 release artifact, validates every bound file, and produces the
canonical clinical JSONL result. An optional embedding sidecar is generated from
the same model pass for governed downstream projection.

The repository contains no model weights, training data, model-development
pipeline, map reducer, or implicit artifact download.

## Install from source

ALMA3 requires Python 3.12.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.7.0"
python -m pip install --no-build-isolation --no-deps .
python -m pip check
```

Installing `.` directly lets pip choose the generic CUDA-enabled Torch wheel.
The explicit sequence above keeps the standalone runtime CPU-only, matching the
runtime image.

## Verify a release

```bash
alma3 verify-release --artifact /path/to/alma3-release
```

## Run inference

```bash
alma3 infer \
  --artifact /path/to/alma3-release \
  --input sample.bed \
  --format bedmethyl \
  --output sample.alma3.jsonl \
  --embedding-sidecar sample.alma3.embedding.json \
  --device cpu
```

Array inputs use `--format array-csv`. Missing CpG columns and blank cells are
both treated as unobserved; every sample must still satisfy the calibrated
observed-CpG floor in the release. Docker and Python use the same explicit local
release-artifact contract; the runtime never downloads weights.

The source is licensed under Apache-2.0. Model weights are separate artifacts
and are not licensed by this repository.

## Run with Docker

The image contains the CPU runtime but no weights. Mount a release explicitly:

```bash
docker run --rm \
  -v /path/to/alma3-release:/release:ro \
  alma3-runtime verify-release --artifact /release
```

Commands supplied by orchestration remain supported, including the explicit
`alma3 infer ...` form used by almagx.
