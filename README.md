<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/alma3-a3-signal-monogram-dark.svg">
    <img src="assets/alma3-a3-signal-monogram-light.svg" alt="ALMA3 A3 symbol" width="96" height="96">
  </picture>
  <h1>ALMA3</h1>
  <p><strong>Epigenomic classifier for hematolymphoid tumors</strong></p>
  <p>
    <img alt="Research Use Only" src="https://img.shields.io/badge/Regulatory%20status-Research%20Use%20Only-orange">
  </p>
</div>

## Model introduction

ALMA3 classifies hematolymphoid tumors using DNA methylation. It works with methylation data from microarrays, Oxford Nanopore and PacBio HiFi sequencing.

## Quick start

```bash
export ALMA3_IMAGE=ghcr.io/almagx/alma3:3.1.1
docker pull "$ALMA3_IMAGE"
(
  set -e
  results=alma3-demo
  mkdir -m 700 "$results"
  container=$(docker create --network none "$ALMA3_IMAGE" demo -o /tmp/alma3-demo.csv)
  trap 'status=$?; docker rm -f "$container" >/dev/null || { [ "$status" -ne 0 ] || status=1; }; exit "$status"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  docker start --attach "$container"
  docker cp "$container:/tmp/alma3-demo.csv" "$results/"
)
```

The demo generates `alma3-demo/alma3-demo.csv`.

Run your own prepared sample:

```bash
(
  set -e
  input=./sample.bed
  output=sample.alma3.csv
  results=alma3-results
  mkdir -m 700 "$results"
  container=$(docker create --network none --user 0:0 "$ALMA3_IMAGE" \
    infer -i "/tmp/$(basename "$input")" -o "/tmp/$output")
  trap 'status=$?; docker rm -f "$container" >/dev/null || { [ "$status" -ne 0 ] || status=1; }; exit "$status"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  docker cp -L "$input" "$container:/tmp/$(basename "$input")"
  docker start --attach "$container"
  docker cp "$container:/tmp/$output" "$results/"
)
```

Set `input` to your prepared file and `results` to a new folder. Use a `.csv` output for a table or `.jsonl` for complete results. Existing results folders are preserved.

The image includes the model and runs on an x86-64 CPU with at least 16 GB RAM.

## Point-and-click automatic analysis

> [!TIP]
> Use our platform, [app.almagx.com](https://app.almagx.com/), for automated preprocessing, ALMA3 classification, diagnostic maps, and integrated reports.

<details><summary><strong>Inputs</strong></summary>

| Input | Filename | Required preparation |
|---|---|---|
| Array matrix | `.csv` or `.csv.gz` | One sample per row, `sample_id` first, CpG IDs as remaining columns |
| BedMethyl | `.bed` or `.bed.gz` | One GRCh38 sample per file, tab-delimited with at least 11 columns |

- In array CSVs, CpG headers should look like `cg00000029`, not gene names. Supply beta values as decimals near `0` to `1`. Mild corrected excursions are accepted. Use `--input-values mvalue` for M-values. Leave missing measurements blank or `NaN`; do not replace them with zero or transpose the matrix.
- Raw IDAT files must first be processed with <a href="https://github.com/zwdzwd/sesame">SeSAMe</a> or submitted through <a href="https://app.almagx.com/">ALMAGX</a>.
- Each sample must contain at least 5,000 recognized ALMA3 CpGs. Sample IDs must be nonempty and unique, with no surrounding whitespace or leading `=`, `+`, `-`, or `@`.

</details>

<details><summary><strong>Oxford Nanopore</strong></summary>

Infinium arrays cannot distinguish 5mC from 5hmC. For ONT, combine both with pinned Modkit `v0.6.3`. The BAM must be coordinate-sorted and indexed against the same chr-prefixed GRCh38 FASTA, with an adjacent `.fai`.

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" "$ALMA3_IMAGE" \
  export-bedmethyl-target \
    --reference /work/GRCh38.fa \
    --output /work/alma3-3.1.0-grch38-cpgs.bed

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
    --include-bed /work/alma3-3.1.0-grch38-cpgs.bed \
    --ref /work/GRCh38.fa
```

Analyze `sample.bed` using the own-sample instructions in [Quick start](#quick-start).

This Modkit command produces combined 5mC+5hmC input, which ALMA3 detects automatically. See Modkit's [targeting documentation](https://nanoporetech.github.io/modkit/intro_include_bed.html) and [v0.6.3 release](https://github.com/nanoporetech/modkit/releases/tag/v0.6.3).

</details>

<details><summary><strong>PacBio HiFi</strong></summary>

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
awk -v target_file="alma3-3.1.0-grch38-cpgs.bed" '
  BEGIN {
    FS = OFS = "\t"
    while ((getline < target_file) > 0) keep[$1 SUBSEP $2] = 1
    close(target_file)
  }
  $1 !~ /^#/ && $5 == "Total" && (($1 SUBSEP $2) in keep) {
    print $1, $2, $3, "m", 0, ".", $2, $3, "255,0,0", $6, $4
  }
' > sample.5mc.bed
```

Analyze `sample.5mc.bed` using the own-sample instructions in [Quick start](#quick-start).

For `type=Total` rows, use column 4 `mod_score` and column 6 coverage. Do not use count mode or `discretized_mod_score`. Filter after pileup because the model uses neighboring CpGs. The converted file uses `m`, so ALMA3 detects 5mC automatically. MethBat is not validated for ALMA3.

</details>

<details><summary><strong>Results</strong></summary>

| Result | Meaning |
|---|---|
| `fully_resolved` | The tumor classification is complete. |
| `heme_tumor_not_detected` | No hematolymphoid tumor signal was detected. |
| `partially_resolved` | The tumor type was identified; the subtype remains unresolved. |

</details>

<details><summary><strong>Citation</strong></summary>

ALMA Genomics Inc. (2026). ALMA3 (Version 3.1.1) [Computer software]. https://github.com/almagx/alma3

Manuscript in preparation.

</details>

<details><summary><strong>Support</strong></summary>

Questions or feedback: [open an issue](https://github.com/almagx/alma3/issues) or email <a href="mailto:support@almagx.com">support@almagx.com</a>.

</details>

<details><summary><strong>License</strong></summary>

ALMA3 models are free to explore and use for research, education, and other noncommercial or internal projects. For commercial use, see [ALMA3 License 1.0](LICENSE).

</details>
