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

- **Context-aware foundation model:** learns methylation relationships across 65,536 CpGs.
- **Coverage-aware inference:** distinguishes observed CpGs from missing sites and incorporates measurement uncertainty.
- **Hierarchical diagnostic model:** reports the deepest supported result across tumor presence, lineage, family, type, and subtype.

## Quick start

```bash
pip install alma3
alma3 demo
```

Run your own sample:

```bash
alma3 infer -i sample.bed -o sample.alma3.jsonl
```

## Inputs

| Input | Filename | Required preparation |
|---|---|---|
| Array matrix | `.csv` or `.csv.gz` | One sample per row, `sample_id` first, CpG IDs as remaining columns |
| BedMethyl | `.bed` or `.bed.gz` | One GRCh38 sample per file, tab-delimited with at least 11 columns |

- In array CSVs, CpG headers should look like `cg00000029`, not gene names. Supply beta values as decimals near `0` to `1`. Mild corrected excursions are accepted. Use `--input-values mvalue` for M-values. Leave missing measurements blank or `NaN`; do not replace them with zero or transpose the matrix.
- For BedMethyl, column 1 is the chromosome, column 2 is the 0-based start, column 10 is coverage, and column 11 is the modified fraction from `0` to `100`.
- Each sample must contain at least 1,500 recognized ALMA3 CpGs. Raw IDAT and BAM files must first be processed with <a href="https://github.com/zwdzwd/sesame">SeSAMe</a> or <a href="https://nanoporetech.github.io/modkit/intro_pileup.html">modkit</a>, or submitted through <a href="https://app.almagx.com/">ALMAGX</a>.

The input format is inferred from the filename.

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
bed_results = model.predict_bedmethyl(["sample-1.bed", "sample-2.bed"])
array_results = model.predict_array(beta, cpg_ids, sample_ids, input_values="beta")
```

The model loads once and preserves sample order. `device="auto"` uses CUDA when available and otherwise uses the CPU.

## Performance

Use CPU with at least 16 GB RAM for occasional samples:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
pip install alma3
```

Use an NVIDIA GPU with at least 16 GB VRAM and 14 GB available for repeated inference or cohorts:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install alma3
```

The first run downloads the 3.9 GB model. Multiple samples in one command or Python object reuse the loaded model. The standalone Docker image is CPU-only.

## Docker

The standalone image includes the model and runs on CPU:

```bash
docker run --rm -v "$PWD:/work" alma3:3.0.0 \
  infer -i /work/sample.bed -o /work/sample.alma3.jsonl
```

## ALMAGX platform

> [!TIP]
> For point-and-click automatic analysis, diagnostic maps, and integrated reports, use <a href="https://app.almagx.com/">ALMAGX</a>.

## Support

Questions, feedback, or deployment help? Email <a href="mailto:support@almagx.com">support@almagx.com</a>.

## License

ALMA3 source and released model weights use the ALMA3 License 1.0.
