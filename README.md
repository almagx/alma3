# ALMA3

[![Research Use Only](https://img.shields.io/badge/use-research_only-336699)](#license)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ALMA3 runs hierarchical hematolymphoid tumor classification from DNA methylation data. The released diagnostic model combines the ALMA3 foundation model with its supervised ALMA3-Dx head.

## Quick start

```bash
pip install alma3
alma3 demo
```

Run your own sample:

```bash
alma3 infer -i sample.bed -o sample.alma3.jsonl
```

Use a `.csv` output name when a flat cohort table is more convenient:

```bash
alma3 infer -i cohort.csv -o results.csv
```

The first run downloads the fixed model, verifies every file, and caches it. Allow about 5 GB of disk space.

## Inputs

| Input | Filename | Layout |
|---|---|---|
| BedMethyl | `.bed` or `.bed.gz` | One GRCh38 sample per file |
| Array beta matrix | `.csv` or `.csv.gz` | Samples by row, `sample_id` first, CpG IDs as columns |

The format is inferred from the filename. Use `--format array-csv` or `--format bedmethyl` only when an explicit override is useful.

Repeat `-i` to process multiple BedMethyl samples while loading the model once:

```bash
alma3 infer \
  -i sample-1.bed \
  -i sample-2.bed \
  --batch-size 4 \
  -o cohort.alma3.jsonl
```

Blank and `NaN` array cells are unobserved. Mildly corrected beta matrices are accepted when at least 90% of matched values remain in `[0,1]`, at least 99% remain in `[-0.05,1.05]`, and none exceed `[-0.5,1.5]`. Excursions are bounded to `[0,1]` with a concise notice.

For M-values, request the standard conversion explicitly:

```bash
alma3 infer -i cohort.csv --input-values mvalue -o cohort.alma3.jsonl
```

Matrices without enough recognized CpGs and values that are not beta-like fail before inference. Gene-expression matrices are not accepted.

## Python

```python
from alma3 import ALMA3

model = ALMA3(device="cuda:0")
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
    input_values="beta",  # use "mvalue" for explicit M-value conversion
)
```

The model loads once and preserves sample order. `device="auto"` uses CUDA when available and otherwise uses the CPU.

For a smaller CPU environment, install CPU-only Torch first:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
pip install alma3
```

On the same 30-vCPU x86-64 host, the ten-sample demo took 46.99 seconds with the official MKL-enabled CPU wheel but more than 20 minutes with a non-MKL system build. ALMA3 warns when it detects the slow build. Keep the default batch size of 2 unless you benchmark another value.

## Results

ALMA3 writes one ordered result per sample. A `.jsonl` output preserves the complete resolved hierarchy and exact release hashes. A `.csv` output provides a clinician-readable summary, resolved labels, unresolved differential, CpG support, and model version.

- `classified`: a diagnostic label was accepted.
- `tumor_not_detected`: no hematolymphoid tumor signal was detected.
- `unresolved`: tumor presence was accepted but a later level was not.
- `no_call`: tumor presence did not meet its threshold.

An unresolved result reports the deepest resolved category and the two leading valid candidates at the next level. Those candidates are informational and are not accepted classifications. A model score ranks labels within one hierarchy branch and is compared with the release's reporting cutoff; it is not an individual patient probability.

Every result records the observed ALMA3 CpGs and the minimum required by the release. Outputs are new-only. `--embedding-sidecar path.json` additionally writes the stable same-pass diagnostic representation.

## Offline use

```bash
alma3 download --output /shared/alma3
alma3 verify-release --artifact /shared/alma3
alma3 infer --artifact /shared/alma3 -i sample.bed -o sample.alma3.jsonl
```

Model selection order is: explicit `--artifact`, `ALMA3_RELEASE`, verified cache, then automatic download. An invalid explicit artifact never falls back to another model.

## Docker

The standalone CPU image contains the verified model:

```bash
docker run --rm -v "$PWD:/work" alma3:3.0.0 \
  infer -i /work/sample.bed -o /work/sample.alma3.jsonl
```

Build it from a local release artifact:

```bash
docker build \
  --build-context alma3_release=/path/to/alma3-release \
  -t alma3:3.0.0 .
```

Official diagnostic maps and integrated reports are available through ALMAGX. This runtime contains no training data, training labels, or map assets.

## License

ALMA3 source and released model weights use the MIT License.
