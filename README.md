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
python -m pip install .
```

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

Array inputs use `--format array-csv`. Docker and Python use the same explicit
local release-artifact contract; the runtime never downloads weights.

The source is licensed under Apache-2.0. Model weights are separate artifacts
and are not licensed by this repository.
