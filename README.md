<div align="center">
  <h1>ALMA3</h1>
  <p><strong>DNA methylation foundation and diagnostic models for hematolymphoid tumors</strong></p>
  <p>
    <img alt="Research Use Only" src="https://img.shields.io/badge/Regulatory%20status-Research%20Use%20Only-0891B2">
    <a href="https://www.python.org/"><img alt="Python 3.10–3.12" src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&amp;logoColor=white"></a>
    <a href="LICENSE"><img alt="ALMA3 License 1.0" src="https://img.shields.io/badge/License-ALMA3%201.0-2563EB"></a>
  </p>
</div>

## Model introduction

ALMA3 is a DNA methylation foundation model. ALMA3-Dx uses its learned representation to classify hematolymphoid tumors, reporting the deepest supported conclusion from tumor presence through subtype.

### Key features

- **Billion-parameter methylation transformer:** 1,044,514,206 parameters, 36 transformer layers, 24 attention heads, and a 1,536-dimensional representation across 65,536 curated CpGs.
- **Large-scale diagnostic training:** fine-tuned with 97,374 supervised training samples.
- **Deep diagnostic hierarchy:** 2 tumor-presence classifications, 5 lineages, 14 families, 34 types, and 102 subtypes.

## Quick start

```bash
pip install alma3
alma3 demo
```

Run your own sample:

```bash
# Complete JSONL
alma3 infer -i sample.bed -o sample.alma3.jsonl

# Clinician-readable CSV
alma3 infer -i sample.bed -o sample.alma3.csv
```

## Inputs

| Input | Filename | Required preparation |
|---|---|---|
| Array matrix | `.csv` or `.csv.gz` | One sample per row, `sample_id` first, CpG IDs as remaining columns |
| BedMethyl | `.bed` or `.bed.gz` | One GRCh38 sample per file, tab-delimited with at least 11 columns |

- In array CSVs, CpG headers should look like `cg00000029`, not gene names. Supply beta values as decimals near `0` to `1`. Mild corrected excursions are accepted. Use `--input-values mvalue` for M-values. Leave missing measurements blank or `NaN`; do not replace them with zero or transpose the matrix.
- For BedMethyl, column 1 is the chromosome, column 2 is the 0-based start, column 10 is coverage, and column 11 is the modified fraction from `0` to `100`.
- Each sample must contain at least 1,500 recognized ALMA3 CpGs. Raw IDAT and BAM files must first be processed with <a href="https://github.com/zwdzwd/sesame">SeSAMe</a> or <a href="https://nanoporetech.github.io/modkit/intro_pileup.html">modkit</a>, or submitted through <a href="https://app.almagx.com/">ALMAGX</a>.

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

## Point-and-click automatic analysis

> [!TIP]
> <a href="https://app.almagx.com/">Open ALMAGX</a> for automated preprocessing, ALMA3 classification, diagnostic maps, and integrated reports.

## Citation

TBD

## Support

Questions or feedback: <a href="mailto:support@almagx.com">support@almagx.com</a>.

## License

ALMA3 source and released model weights use the ALMA3 License 1.0.
