"""Small deterministic fixtures for parsers and publication, without released weights."""

import hashlib
import json
from pathlib import Path

import torch

from alma3.data import CpGManifest
from alma3.dx import DX_TARGETS, load_taxonomy
from alma3.hashes import sha256_file
from alma3.sitewise import MINIMUM_RUNTIME_INPUT_CPGS
from alma3.terminal import SUBTYPE_CUTOFF, taxonomy_payload
from tests.test_clinical_result import taxonomy


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def sha256(path):
    return sha256_file(path)


class FixtureEnsemble:
    def embed(self, beta, observed, *args):
        mean = (beta * observed).sum(1) / observed.sum(1).clamp_min(1)
        p = torch.full((len(beta), 109), 0.001, dtype=torch.float64)
        p[:, 1] = 0.5 + 0.2 * mean
        p[:, 0] = 1 - p[:, 1] - 0.107
        return p.repeat(1, 10)

    def logits_from_embedding(self, embedding):
        return embedding[:, :109].log()

    def predict_with_map_features(self, *args):
        e = self.embed(*args)
        return self.logits_from_embedding(e), e.float()


def create_release(root):
    root.mkdir()
    write_json(root / "ensemble.json", {"kind": "test_ensemble"})
    write_json(root / "taxonomy.json", taxonomy_payload(taxonomy()))
    cpg_ids = [f"cg{index:07d}" for index in range(MINIMUM_RUNTIME_INPUT_CPGS)]
    write_json(
        root / "cpg_manifest.json",
        {
            "kind": "alma3_cpg_manifest",
            "schema_version": 1,
            "bundle_fingerprint": "3" * 64,
            "cpg_ids": cpg_ids,
            "cpg_manifest_sha256": hashlib.sha256(("\n".join(cpg_ids) + "\n").encode()).hexdigest(),
            "source_cpg_manifest_sha256": hashlib.sha256(("\n".join(cpg_ids) + "\n").encode()).hexdigest(),
            "selected_cpg_count": MINIMUM_RUNTIME_INPUT_CPGS,
            "selection_algorithm": "parent_conditioned_tumor_dataset_stability_v3",
            "indices": list(range(MINIMUM_RUNTIME_INPUT_CPGS)),
            "chr_id": [0] * MINIMUM_RUNTIME_INPUT_CPGS,
            "chrom": ["chr1"] * MINIMUM_RUNTIME_INPUT_CPGS,
            "start": list(range(100, 100 + MINIMUM_RUNTIME_INPUT_CPGS)),
            "end": list(range(101, 101 + MINIMUM_RUNTIME_INPUT_CPGS)),
            "pos": [index / (MINIMUM_RUNTIME_INPUT_CPGS - 1) for index in range(MINIMUM_RUNTIME_INPUT_CPGS)],
        },
    )

    write_json(
        root / "thresholds.json",
        {"minimum_observed_cpgs": 5000, "thresholds": {k: SUBTYPE_CUTOFF if k == "subtype" else 0.0 for k in DX_TARGETS}},
    )
    write_json(root / "release.json", {"version": "3.1.0"})
    write_json(root / "SHA256SUMS.json", {p.name: sha256(p) for p in root.iterdir()})
    (root / "RELEASE_COMPLETE").write_text("complete\n")
    return root, FixtureEnsemble()


def validated_release_fixture(root, model):
    return {
        "root": root,
        "hashes": json.loads((root / "SHA256SUMS.json").read_text()),
        "manifest_sha256": sha256(root / "SHA256SUMS.json"),
        "taxonomy": load_taxonomy(root / "taxonomy.json"),
        "cpg": CpGManifest.load(root / "cpg_manifest.json"),
        "thresholds": json.loads((root / "thresholds.json").read_text()),
        "release": json.loads((root / "release.json").read_text()),
        "model": model,
    }


def write_array_csv(
    path: Path,
    *,
    sample_count: int = 3,
    observed: int = MINIMUM_RUNTIME_INPUT_CPGS,
    omit_unobserved_columns: bool = False,
) -> None:
    cpg_ids = [f"cg{index:07d}" for index in range(MINIMUM_RUNTIME_INPUT_CPGS)]
    if omit_unobserved_columns:
        cpg_ids = cpg_ids[:observed]
    rows = [",".join(["sample_id", *cpg_ids])]
    for sample_index in range(sample_count):
        values = ["0.5" if index < observed else "" for index in range(len(cpg_ids))]
        rows.append(",".join([f"sample-{sample_index + 1}", *values]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_bedmethyl(
    path: Path,
    *,
    fraction_modified: float = 50.0,
    modification_code: str = "C",
) -> None:
    rows = [
        f"chr1\t{start}\t{start + 1}\t{modification_code}\t0\t.\t{start}\t{start + 1}\t0\t10\t{fraction_modified}"
        for start in range(100, 100 + MINIMUM_RUNTIME_INPUT_CPGS)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
