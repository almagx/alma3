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

### Oxford Nanopore: combined 5mC+5hmC

Infinium arrays cannot distinguish 5mC from 5hmC. For ONT, combine both with pinned Modkit `v0.6.3`. The BAM must be coordinate-sorted and indexed against the same chr-prefixed GRCh38 FASTA, with an adjacent `.fai`.

```bash
alma3 export-bedmethyl-target \
  --reference "$PWD/GRCh38.fa" \
  --output "$PWD/alma3-3.0.0-grch38-cpgs.bed"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  ontresearch/modkit@sha256:d9b966437381cdd61ddf8f0f66a88615d800495ab202e586088f0431e8f56929 \
  modkit pileup /work/input.bam /work/sample.bed \
    --modified-bases 5mC 5hmC \
    --combine-mods \
    --cpg \
    --combine-strands \
    --no-filtering \
    --include-bed /work/alma3-3.0.0-grch38-cpgs.bed \
    --ref /work/GRCh38.fa

alma3 infer \
  -i sample.bed \
  --bedmethyl-modification-mode 5mc_plus_5hmc \
  -o sample.alma3.csv
```

The exporter verifies GRCh38 and writes the 65,535 release CpGs plus a receipt. ONT requires `5mc_plus_5hmc`; use `5mc` only for supported 5mC-only inputs such as PacBio. See Modkit's [targeting documentation](https://nanoporetech.github.io/modkit/intro_include_bed.html) and [v0.6.3 release](https://github.com/nanoporetech/modkit/releases/tag/v0.6.3).

### PacBio HiFi: 5mC only

PacBio 5mC is not equivalent to array 5mC+5hmC. Jasmine estimates 5mC and 5hmC independently, so never add or normalize their probabilities. Require standalone `C+m`; ignore separate `C+h`/`G-h`; reject compound `C+mh` and h-only input. See [Jasmine](https://github.com/PacificBiosciences/jasmine#overview) and [pb-CpG-tools](https://github.com/PacificBiosciences/pb-CpG-tools#input-alignment-file).

Use a coordinate-sorted, indexed BAM aligned with `pbmm2 --preset CCS` to chr-prefixed GRCh38. Preserve `MM`/`ML` tags and prohibit hard clipping. Export the target above, then run:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" \
  quay.io/pacbio/pb-cpg-tools@sha256:afd5468a423fe089f1437d525fdc19c704296f723958739a6fe226caa01fba1c \
  /opt/pb-CpG-tools-v3.0.0-x86_64-unknown-linux-gnu/bin/aligned_bam_to_cpg_scores \
    --bam /work/sample.pbmm2.sorted.bam \
    --pileup-mode model \
    --modsites-mode denovo \
    --output-prefix /work/sample \
    --threads 8 \
    --min-coverage 4 \
    --min-mapq 1

gzip -cd sample.combined.bed.gz | \
awk -v target_file="alma3-3.0.0-grch38-cpgs.bed" '
  BEGIN {
    FS = OFS = "\t"
    while ((getline < target_file) > 0) keep[$1 SUBSEP $2] = 1
    close(target_file)
  }
  $1 !~ /^#/ && $5 == "Total" && (($1 SUBSEP $2) in keep) {
    print $1, $2, $3, "m", 0, ".", $2, $3, "255,0,0", $6, $4
  }
' > sample.5mc.bed

alma3 infer \
  -i sample.5mc.bed \
  --bedmethyl-modification-mode 5mc \
  -o sample.alma3.csv
```

For `type=Total` rows, use column 4 `mod_score` and column 6 coverage. Do not use count mode or `discretized_mod_score`. Filter after pileup because the model uses neighboring CpGs. MethBat is not validated for ALMA3. Declare `5mc`.

## Results

JSONL schema v2 is canonical. CSV is its fixed 45-column human-readable projection with the same nonredundant result, differential, input, coverage, hierarchy, and provenance data. JSONL adds taxonomy indices and component hashes derivable from the CSV's release-manifest hash.

### Status

| Result | Meaning |
|---|---|
| `fully_resolved` | All applicable hierarchy levels resolved. |
| `heme_tumor_not_detected` | No hematolymphoid tumor signal detected. |
| `partially_resolved` | A deeper level missed its threshold. |
| `no_call` | Tumor presence missed its threshold. |

### Summary

`result_summary` states the accepted call and confidence, when present. Unresolved results also state the leading candidate, its confidence, and threshold; both candidates remain in the differential fields.

Confidence is `100 ×` the parent-conditioned model score, not patient probability, predictive value, or diagnostic certainty. Implied calls use their deepest scored ancestor's confidence; their score and cutoff stay blank. Summaries round to one decimal; structured values remain unrounded.

### CSV

Column order is result, differential, CpG/input, hierarchy, provenance. Scored levels fill classification, status, score, and cutoff; implied levels omit score and cutoff; unreached levels are blank. Decision fields are populated only when unresolved. `input_modification_mode` is set for BedMethyl and blank for arrays.

For support, send the original CSV without opening or resaving it; spreadsheet software can alter IDs and precision.

## Python

```python
from alma3 import ALMA3

model = ALMA3()
bed_results = model.predict_bedmethyl(
    ["sample-1.bed", "sample-2.bed"],
    modification_mode="5mc_plus_5hmc",
)
pacbio_results = model.predict_bedmethyl(
    ["sample.5mc.bed"],
    modification_mode="5mc",
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
  infer -i /work/sample.bed --bedmethyl-modification-mode 5mc_plus_5hmc \
  -o /work/sample.alma3.jsonl
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
