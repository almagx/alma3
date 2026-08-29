# ALMA3

ALMA3 turns methylation measurements into hierarchical hematolymphoid tumor
predictions. The package includes the ALMA3 foundation architecture and the
ALMA3-Dx diagnostic model used for inference.

The first inference downloads the fixed ALMA3 3.0.0 model from
`models.almagx.com`, verifies every file, and caches it. The model is about
4.2 GB. Allow at least 5 GB of free disk space and start with batch size 1 on
machines with limited memory.

## Command line

ALMA3 supports Python 3.10, 3.11, and 3.12.

```bash
pip install alma3

alma3 infer \
  --input sample.bed \
  --format bedmethyl \
  --output sample.alma3.jsonl
```

Pass `--input` more than once to process multiple BedMethyl files while loading
the model once:

```bash
alma3 infer \
  --input sample-1.bed \
  --input sample-2.bed \
  --format bedmethyl \
  --batch-size 4 \
  --output cohort.alma3.jsonl
```

An array CSV has one sample per row, a `sample_id` first column, and CpG IDs as
the remaining column names. Missing columns and blank or `NaN` cells are
unobserved. Every sample must meet the observed-CpG minimum stored with the
model.

## Python

```python
from alma3 import ALMA3

model = ALMA3(device="cuda:0")
results = model.predict_bedmethyl(
    ["sample-1.bed", "sample-2.bed"],
    batch_size=4,
)
```

For an in-memory array:

```python
results = model.predict_array(
    beta,                 # [samples, CpGs]; NaN means unobserved
    cpg_ids,
    sample_ids,
    batch_size=4,
)
```

Install a CUDA build of Torch before installing ALMA3 when using a GPU. Use
`device="cpu"`, `device="cuda:0"`, or another CUDA index explicitly.

## Docker

The standalone image is CPU-only and contains the verified model, so it does
not download anything at runtime:

```bash
docker run --rm -v "$PWD:/work" alma3:3.0.0 infer \
  --input /work/sample.bed \
  --format bedmethyl \
  --output /work/sample.alma3.jsonl
```

The internal weight-free image target is `runtime`. Build the standalone target
with the model artifact as a named build context:

```bash
docker build \
  --build-context alma3_release=/path/to/alma3-3.0.0 \
  -t alma3:3.0.0 .
```

## Offline use

Download once to a shared directory, then use that exact copy:

```bash
alma3 download --output /shared/alma3-3.0.0
alma3 verify-release --artifact /shared/alma3-3.0.0
alma3 infer \
  --artifact /shared/alma3-3.0.0 \
  --input sample.bed \
  --format bedmethyl \
  --output sample.alma3.jsonl
```

Model selection order is: explicit `--artifact`, `ALMA3_RELEASE`, verified
cache, then automatic download. An invalid explicit model fails immediately.

## Outputs and maps

The canonical output is JSONL with the ALMA3-Dx status, accepted hierarchy,
scores, and model hashes. `--embedding-sidecar path.json` additionally writes
the stable 1,536-dimensional representation from that same inference pass.

Official two-dimensional diagnostic maps and integrated reports are available
through the ALMAGX platform. The public runtime does not contain training data,
training labels, or map assets.

ALMA3 source and released model weights use the MIT License.
