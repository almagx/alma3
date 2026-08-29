# ALMA3

ALMA3 is a foundation model for DNA methylation. ALMA3-Dx is the supervised
diagnostic model built on that foundation. This package runs the released
ALMA3-Dx model for hierarchical hematolymphoid tumor classification.

The first inference downloads the fixed ALMA3 3.0.0 release from
`models.almagx.com`, verifies every file, and caches it. The model is about
4.2 GB. Allow at least 5 GB of free disk space. Batch size 1 is the safe
default; increase it only when memory permits.

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

Outputs are new-only: ALMA3 will not replace an existing result. Run
`alma3 infer --help` for the complete interface.

### BedMethyl input

- One sample per `.bed` or `.bed.gz` file.
- GRCh38 coordinates with chromosome names such as `chr1` and `chrX`.
- Headerless, tab-delimited rows with at least 11 BedMethyl columns.
- Column 2 is the zero-based genomic start, column 10 is integer coverage, and
  column 11 is the modified fraction from 0 to 100.
- The sample ID is the filename without `.bed` or `.bed.gz`.

### Array CSV input

An array CSV has one sample per row, a `sample_id` first column, and CpG IDs as
the remaining column names. Missing CpG columns and blank or `NaN` cells are
unobserved. Extra CpGs not used by the release are ignored.

```bash
alma3 infer \
  --input cohort.csv \
  --format array-csv \
  --output cohort.alma3.jsonl
```

ALMA3 3.0.0 requires at least 1,500 observed release CpGs per sample.

## Python

```python
from alma3 import ALMA3

model = ALMA3()
results = model.predict_bedmethyl(
    ["sample-1.bed", "sample-2.bed"],
    batch_size=2,
)

for result in results:
    print(result["sample_id"], result["status"], result["accepted"])
```

For an in-memory array:

```python
results = model.predict_array(
    beta,                 # [samples, CpGs]; NaN means unobserved
    cpg_ids,
    sample_ids,
    batch_size=2,
)
```

`device="auto"` uses CUDA when available and otherwise uses the CPU. Use
`device="cpu"`, `device="cuda:0"`, or another CUDA index to select explicitly.

On Linux, install the CPU-only Torch wheel first when a smaller CPU environment
is preferred:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
pip install alma3
```

For GPU use, install the appropriate CUDA build of Torch 2.7.0 before ALMA3.

## Docker

The standalone image is CPU-only and contains the verified model, so it does
not download anything at runtime:

```bash
docker run --rm -v "$PWD:/work" alma3:3.0.0 infer \
  --input /work/sample.bed \
  --format bedmethyl \
  --output /work/sample.alma3.jsonl
```

Build the standalone image with the model artifact as a named build context:

```bash
docker build \
  --build-context alma3_release=/path/to/alma3-3.0.0 \
  -t alma3:3.0.0 .
```

The internal weight-free image target is `runtime`.

## Model download and offline use

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

Model selection order is:

1. Explicit `--artifact`
2. `ALMA3_RELEASE`
3. Verified cache
4. Automatic download

An invalid explicit model fails immediately. The default cache is
`~/.cache/alma3`; set `ALMA3_CACHE_DIR` to move it.

## Results

The canonical output is JSONL with one ordered record per sample:

- `no_call`: tumor presence did not meet its threshold.
- `tumor_not_detected`: tumor absence was accepted.
- `classified`: the deepest available terminal label was accepted.
- `unresolved`: tumor presence was accepted, but a later hierarchy level did
  not meet its threshold.

Each record contains the accepted label, the resolved hierarchy path, scores,
thresholds, and exact model hashes. The contract is defined by the
[result schema](https://github.com/almagx/alma3/blob/main/src/alma3/schemas/dx_result.schema.json).

`--embedding-sidecar path.json` additionally writes the stable 1,536-dimensional
diagnostic representation from the same inference pass. Its contract is the
[embedding schema](https://github.com/almagx/alma3/blob/main/src/alma3/schemas/embedding_sidecar.schema.json).

Official two-dimensional diagnostic maps and integrated reports are available
through the ALMAGX platform. The public runtime does not contain training data,
training labels, or map assets.

ALMA3 source and released model weights use the MIT License.
