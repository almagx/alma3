<div align="center">
  <h1>ALMA3</h1>
  <p><strong>DNA methylation foundation and diagnostic models for hematolymphoid tumors</strong></p>
  <p>
    <a href="LICENSE"><img alt="Research use" src="https://img.shields.io/badge/Use-Research-0891B2"></a>
    <a href="https://www.python.org/"><img alt="Python 3.10–3.12" src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&amp;logoColor=white"></a>
    <a href="LICENSE"><img alt="ALMA3 License 1.0" src="https://img.shields.io/badge/License-ALMA3%201.0-2563EB"></a>
  </p>
  <p>
    <a href="https://app.almagx.com/">🚀 Launch ALMAGX</a> ·
    <a href="mailto:support@almagx.com">✉️ Support</a>
  </p>
</div>

## Model introduction

ALMA3 is a DNA methylation foundation model. ALMA3-Dx uses its learned representation to classify hematolymphoid tumors, reporting the deepest supported conclusion from tumor presence through subtype.

### Key features

- **Hierarchical reporting** — stops at the deepest supported classification instead of forcing a subtype.
- **Flexible methylation inputs** — supports BedMethyl and array beta or M-value matrices.
- **Local and cohort-ready** — runs through Python, Docker, or the command line on CPU or NVIDIA GPU.
- **Clear outputs** — produces clinician-readable CSV and complete JSONL results.

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
| `fully_resolved` | The applicable hierarchy was fully resolved. |
| `heme_tumor_not_detected` | No hematolymphoid tumor signal was detected. |
| `partially_resolved` | Earlier levels were resolved, but the next level did not reach its reporting cutoff. |
| `no_call` | Tumor presence did not reach its reporting cutoff. |

A partially resolved result retains the deepest resolved classification and provides a two-class informational differential for the next level. Model scores rank classifications within that branch; they are not individual patient probabilities.

## Python

```python
from alma3 import ALMA3

model = ALMA3()
results = model.predict_bedmethyl(["sample-1.bed", "sample-2.bed"])
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

CPU inference requires no NVIDIA hardware. Provide at least 16 GB of system memory. ALMA3 stops with a direct reinstall command if an unsupported x86-64 PyTorch build is detected.

Use an NVIDIA GPU for repeated inference or larger cohorts. Install the tested CUDA 12.8 PyTorch wheel before ALMA3:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install alma3
alma3 infer -i cohort.csv -o results.csv
```

Use a GPU with at least 16 GB of memory and start inference with at least 14 GB available. `device="auto"` uses CUDA when available. ALMA3 checks this memory before loading the model and does not switch silently to CPU after a CUDA failure. Process multiple samples in one command, or reuse one `ALMA3` Python object, so the 3.9 GB model is downloaded and loaded only once.

## Deployment

Run ALMA3 locally with the Python package or the standalone CPU image:

```bash
docker run --rm -v "$PWD:/work" alma3:3.0.0 \
  infer -i /work/sample.bed -o /work/sample.alma3.jsonl
```

> [!TIP]
> For diagnostic maps, integrated reports, and managed workflows, [launch ALMAGX](https://app.almagx.com/).

## Support

Questions, feedback, or deployment help? Email [support@almagx.com](mailto:support@almagx.com).

## License

ALMA3 source and released model weights use the ALMA3 License 1.0.
