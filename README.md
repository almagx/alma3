<div align="center">
  <picture>
    <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="assets/alma3-a3-signal-monogram-dark.svg">
    <source media="(prefers-reduced-motion: reduce)" srcset="assets/alma3-a3-signal-monogram-light.svg">
    <source media="(prefers-color-scheme: dark)" srcset="assets/alma3-a3-motion-signature-relay-readme-dark.gif">
    <source media="(prefers-color-scheme: light)" srcset="assets/alma3-a3-motion-signature-relay-readme-light.gif">
    <img src="assets/alma3-a3-signal-monogram-light.svg" alt="Animated ALMA3 A3 symbol" width="96" height="96">
  </picture>
  <h1>ALMA3</h1>
  <p><strong>Epigenomic foundation and diagnostic models for hematolymphoid tumors</strong></p>
  <p>
    <img alt="Research Use Only" src="https://img.shields.io/badge/Regulatory%20status-Research%20Use%20Only-orange">
    <a href="https://www.python.org/"><img alt="Python 3.10–3.12" src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&amp;logoColor=white"></a>
    <a href="LICENSE"><img alt="ALMA3 License 1.0" src="https://img.shields.io/badge/License-ALMA3%201.0-2563EB"></a>
  </p>
</div>

## Model introduction

ALMA3 is an epigenomic foundation model. ALMA3-Dx uses its learned representation to classify hematolymphoid tumors, reporting the deepest supported conclusion from tumor presence through subtype.

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
alma3 infer -i sample.bed --bedmethyl-modification-mode 5mc_plus_5hmc -o sample.alma3.jsonl

# Clinician-readable CSV
alma3 infer -i sample.bed --bedmethyl-modification-mode 5mc_plus_5hmc -o sample.alma3.csv
```

## Inputs

| Input | Filename | Required preparation |
|---|---|---|
| Array matrix | `.csv` or `.csv.gz` | One sample per row, `sample_id` first, CpG IDs as remaining columns |
| BedMethyl | `.bed` or `.bed.gz` | One GRCh38 sample per file, tab-delimited with at least 11 columns |

- In array CSVs, CpG headers should look like `cg00000029`, not gene names. Supply beta values as decimals near `0` to `1`. Mild corrected excursions are accepted. Use `--input-values mvalue` for M-values. Leave missing measurements blank or `NaN`; do not replace them with zero or transpose the matrix.
- For BedMethyl, column 1 is the chromosome, column 2 is the 0-based start, column 10 is coverage, and column 11 is the modified fraction from `0` to `100`. Every BedMethyl inference must declare `--bedmethyl-modification-mode 5mc_plus_5hmc` or `5mc`; ALMA3 records the declaration because BedMethyl itself cannot prove which upstream modifications were aggregated.
- Each sample must contain at least 1,500 recognized ALMA3 CpGs. Raw IDAT and BAM files must first be processed with <a href="https://github.com/zwdzwd/sesame">SeSAMe</a> or <a href="https://nanoporetech.github.io/modkit/intro_pileup.html">modkit</a>, or submitted through <a href="https://app.almagx.com/">ALMAGX</a>.
- Sample IDs must be nonempty and unique, with no surrounding whitespace, ASCII control characters, or leading `=`, `+`, `-`, or `@`.

Standard Infinium bisulfite chemistry does not distinguish 5mC from 5hmC. Prepare array-comparable ONT input with pinned Modkit `v0.6.3` and one combined row per CpG:

```bash
modkit pileup input.bam output.bed \
  --modified-bases 5mC 5hmC \
  --combine-mods \
  --cpg \
  --combine-strands \
  --no-filtering \
  --ref GRCh38.fa
```

Use `5mc` only for an explicitly 5mC-only BedMethyl projection, including the supported PacBio route. The mode is required and has no fallback.

## Results

JSONL schema v2 is the canonical result. CSV is its fixed clinician-readable projection and carries the same nonredundant diagnostic and troubleshooting facts: result status, release and runtime identity, resolved device, input interpretation and clipping, CpG support, the complete reached hierarchy with per-level scores and cutoffs, and any unresolved differential. JSONL additionally retains numeric taxonomy indices and component hashes that can be resolved from the release-manifest hash recorded in CSV.

| Result | Meaning |
|---|---|
| `fully_resolved` | The applicable hierarchy was fully resolved. |
| `heme_tumor_not_detected` | No hematolymphoid tumor signal was detected. |
| `partially_resolved` | Earlier levels were resolved, but the next level did not reach its reporting cutoff. |
| `no_call` | Tumor presence did not reach its reporting cutoff. |

Every result includes a concise `result_summary`. Calls display the deepest accepted class and `100 × model_score` from the deepest scored accepted path node to one decimal place as confidence. For a hierarchy-implied class, that confidence carries forward from its deepest scored ancestor; `resolved_basis` and the path status identify the implication, while the implied class's structured score remains null in JSONL and blank in CSV. A partially resolved or no-call result also displays the first-ranked unresolved candidate, its confidence, and the applicable threshold. Only that leading candidate appears in the summary; both ranked candidates remain available in the structured differential fields.

Confidence is `100 ×` the parent-conditioned model score for either an accepted call or the displayed leading candidate. The leading candidate remains unresolved because its confidence is below the displayed threshold. Confidence is not an individual patient probability, positive predictive value, or standalone measure of diagnostic certainty. Raw scores and reporting cutoffs remain unrounded in JSONL and CSV; the summary alone uses the shorter word `threshold` and rounds displayed percentages to one decimal place.

The fixed 45-column CSV is ordered for human review: primary result, unresolved differential, CpG/input context, hierarchy evidence, then contract and runtime provenance. BedMethyl rows record `input_modification_mode`; array rows leave it blank. Each hierarchy level contains classification, status, model score, and reporting cutoff. Scored levels populate all four fields; hierarchy-implied levels have blank score and cutoff fields; unreached levels are entirely blank. Unresolved decision fields are populated only for `partially_resolved` and `no_call` results.

For troubleshooting, send the original generated CSV without opening and resaving it in spreadsheet software, which can alter identifiers or numeric precision.

## Python

```python
from alma3 import ALMA3

model = ALMA3()
bed_results = model.predict_bedmethyl(
    ["sample-1.bed", "sample-2.bed"],
    modification_mode="5mc_plus_5hmc",
)
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

ALMA3 models are free to explore and use for research, education, and other noncommercial or internal projects.
