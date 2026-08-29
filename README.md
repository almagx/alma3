# ALMA3

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ALMA3 classifies hematolymphoid tumors from DNA methylation data. Its ALMA3-Dx model reports the deepest supported level across tumor presence, lineage, family, type, and subtype.

## Quick start

```bash
pip install alma3
alma3 demo
```

Run your own sample:

```bash
alma3 infer -i sample.bed -o sample.alma3.jsonl
```

Use a `.csv` output name for a flat cohort table:

```bash
alma3 infer -i cohort.csv -o results.csv
```

The first run downloads and verifies the 3.9 GB model. Allow about 5 GB of disk space.

For a shared or offline model installation, download it once and provide its location:

```bash
alma3 download --output /shared/alma3
alma3 infer --artifact /shared/alma3 -i sample.bed -o sample.alma3.jsonl
```

## Inputs

| Input | Filename | Layout |
|---|---|---|
| BedMethyl | `.bed` or `.bed.gz` | One GRCh38 sample per file |
| Array beta matrix | `.csv` or `.csv.gz` | Samples by row, `sample_id` first, CpG IDs as columns |

The format is inferred from the filename. Repeat `-i` to process multiple BedMethyl samples while loading the model once:

```bash
alma3 infer -i sample-1.bed -i sample-2.bed -o cohort.alma3.jsonl
```

Blank and `NaN` array cells remain unobserved. ALMA3 accepts mild beta-value excursions around `[0,1]`, clips accepted excursions with a notice, and rejects low CpG overlap or unrelated value scales. Gene-expression matrices are not accepted.

Request M-value conversion explicitly:

```bash
alma3 infer -i cohort.csv --input-values mvalue -o cohort.alma3.jsonl
```

## Results

CSV provides a clinician-readable summary. JSONL preserves the complete hierarchy, model scores, reporting cutoffs, and release identity.

| Result | Meaning |
|---|---|
| `classified` | The applicable hierarchy was fully resolved. |
| `tumor_not_detected` | No hematolymphoid tumor signal was detected. |
| `partially_resolved` | Earlier levels were resolved, but the next level did not reach its reporting cutoff. |
| `no_call` | Tumor presence did not reach its reporting cutoff. |

A partially resolved result retains the deepest resolved classification and provides a two-class informational differential for the next level. Model scores rank classifications within that branch; they are not individual patient probabilities.

## Python

```python
from alma3 import ALMA3

model = ALMA3()
results = model.predict_bedmethyl(
    ["sample-1.bed", "sample-2.bed"],
    batch_size=2,
)
```

For an in-memory matrix:

```python
results = model.predict_array(
    beta,
    cpg_ids,
    sample_ids,
    input_values="beta",
)
```

The model loads once and preserves sample order. `device="auto"` uses CUDA when available and otherwise uses the CPU.

## Performance

Use CPU for occasional samples and small cohorts. Install the official CPU-only PyTorch wheel before ALMA3:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
pip install alma3
```

On a 30-vCPU x86-64 host, the ten-sample demo completed in 46.99 seconds with the official MKL-enabled wheel. A non-MKL build took more than 20 minutes; ALMA3 warns when it detects that configuration.

Use an NVIDIA GPU for repeated inference or larger cohorts. Install the PyTorch 2.7.0 CUDA build that matches the system, then install ALMA3 and select the GPU:

```bash
alma3 infer -i cohort.csv -o results.csv --device cuda:0
```

The 3.9 GB weights must fit in memory alongside inference tensors. Start with the default batch size of 2 and increase it only after checking GPU memory and throughput. Process multiple samples together so the model is loaded once, and benchmark a representative batch before choosing a larger batch size.

## Docker

The standalone image includes the model and runs on CPU:

```bash
docker run --rm -v "$PWD:/work" alma3:3.0.0 \
  infer -i /work/sample.bed -o /work/sample.alma3.jsonl
```

ALMAGX adds diagnostic maps, integrated reports, and production workflow automation.

## License

ALMA3 source and released model weights use the MIT License.
