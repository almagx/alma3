"""Strict inference-only release contract for ALMA3 3.1.0."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import DxConfig
from .data import CpGManifest
from .dx import DX_TARGETS, DxContractError, Taxonomy, load_taxonomy
from .hashes import sha256_file
from .terminal import SUBTYPE_CUTOFF, terminal_paths

RELEASE_KIND = "alma3_release"
RELEASE_VERSION = "3.1.0"
RELEASE_SCHEMA_VERSION = 2
RELEASE_ENV = "ALMA3_RELEASE"
RELEASE_LICENSE_HEADER = "ALMA3 LICENSE 1.0\n"
SELECTED_ENSEMBLE_SHA256 = "f611e107f38ee42d64f43dc0049d165061badbfad34596736f0c86cdaeff9f79"
RELEASE_PAYLOADS = frozenset(
    {"release.json", "ensemble.json", "taxonomy.json", "cpg_manifest.json", "thresholds.json", "LICENSE"}
)
RELEASE_FILES = RELEASE_PAYLOADS | {"SHA256SUMS.json", "RELEASE_COMPLETE"}


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _is_revision(value):
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def _read_object(path, description):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise DxContractError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise DxContractError(f"{description} must be an object")
    return value


def _artifact_files(root):
    if not root.is_dir() or root.is_symlink():
        raise DxContractError("release must be a real directory")
    paths = list(root.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise DxContractError("release contains a symbolic link")
    return {p.relative_to(root).as_posix() for p in paths if p.is_file()}


def ensemble_asset_paths(manifest):
    """Derive the complete public asset inventory, without research directories."""
    if (
        manifest.get("kind") != "alma3_final_ten_v1"
        or manifest.get("decision_rule") != "type_first_v1"
        or len(manifest.get("members", [])) != 10
    ):
        raise DxContractError("unsupported ensemble contract")
    assets = {"ensemble.json", "taxonomy.json", "cpg_manifest.json"}
    for member in manifest["members"]:
        if member["kind"] == "sparse_templates":
            assets.add(member["templates"])
        else:
            assets.update(str(Path(member["export"]) / name) for name in ("config.json", "model.safetensors"))
        for name in ("residual", "partition"):
            if name in member:
                assets.add(member[name])
    for relative in assets:
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or p.as_posix() != relative:
            raise DxContractError("asset path must remain inside the release")
    return assets


def validate_reporting_policy(value, *, ensemble_sha256, taxonomy_sha256):
    expected = dict(
        kind="alma3_ensemble_reporting",
        schema_version=1,
        minimum_observed_cpgs=5000,
        decision_rule="type_first_v1",
        policy="fixed_subtype_only_v1",
        ensemble_sha256=ensemble_sha256,
        taxonomy_sha256=taxonomy_sha256,
        thresholds={key: SUBTYPE_CUTOFF if key == "subtype" else 0.0 for key in DX_TARGETS},
    )
    if value != expected:
        raise DxContractError("reporting requires the frozen subtype-only cutoff and ensemble binding")
    return value


def revalidate_release_identity(artifact, *, manifest_sha256, hashes):
    root = Path(artifact).resolve()
    if (
        _artifact_files(root) != set(hashes) | {"SHA256SUMS.json", "RELEASE_COMPLETE"}
        or (root / "RELEASE_COMPLETE").read_bytes() != b"complete\n"
        or sha256_file(root / "SHA256SUMS.json") != manifest_sha256
    ):
        raise DxContractError("release inventory changed during inference")
    for relative, expected in hashes.items():
        if sha256_file(root / relative) != expected:
            raise DxContractError(f"release changed during inference: {relative}")


def validate_release(artifact=None, *, verify_hashes=True, device="cpu", load_model=False):
    location = artifact or os.environ.get(RELEASE_ENV)
    if not location:
        raise DxContractError("provide a release directory or set ALMA3_RELEASE")
    root = Path(location).resolve()
    files = _artifact_files(root)
    manifest = _read_object(root / "ensemble.json", "ensemble")
    expected = ensemble_asset_paths(manifest) | RELEASE_FILES
    if files != expected:
        raise DxContractError(
            f"invalid inference payload inventory: missing={sorted(expected - files)}, extra={sorted(files - expected)}"
        )
    if (root / "RELEASE_COMPLETE").read_bytes() != b"complete\n":
        raise DxContractError("release publication is incomplete")
    hashes = _read_object(root / "SHA256SUMS.json", "checksums")
    if set(hashes) != expected - {"SHA256SUMS.json", "RELEASE_COMPLETE"} or not all(
        _is_sha256(v) for v in hashes.values()
    ):
        raise DxContractError("checksum inventory differs from inference assets")
    if verify_hashes:
        for relative, expected_hash in hashes.items():
            if sha256_file(root / relative) != expected_hash:
                raise DxContractError(f"release checksum mismatch: {relative}")
    if hashes["ensemble.json"] != SELECTED_ENSEMBLE_SHA256:
        raise DxContractError("3.1.0 requires the selected Final-ten ensemble")
    release = _read_object(root / "release.json", "release metadata")
    if (
        release.get("kind") != RELEASE_KIND
        or release.get("schema_version") != RELEASE_SCHEMA_VERSION
        or release.get("version") != RELEASE_VERSION
        or not _is_revision(release.get("runtime_git_commit"))
        or release.get("ensemble_sha256") != hashes["ensemble.json"]
        or release.get("representation")
        != dict(name="final_ten_member_probabilities", version=1, dtype="float32", dimensions=1090)
    ):
        raise DxContractError("unsupported release metadata")
    if not (root / "LICENSE").read_text().startswith(RELEASE_LICENSE_HEADER):
        raise DxContractError("release license is missing")
    cpg = CpGManifest.load(root / "cpg_manifest.json")
    if len(cpg.cpg_ids) != 387849 or hashes["cpg_manifest.json"] != manifest["cpg_manifest_sha256"]:
        raise DxContractError("ordered 387,849-CpG contract changed")
    taxonomy = load_taxonomy(root / "taxonomy.json")
    if len(terminal_paths(taxonomy)) != 109:
        raise DxContractError("ordered terminal taxonomy changed")
    for member in manifest["members"]:
        if "export" in member:
            config = DxConfig.from_dict(_read_object(root / member["export"] / "config.json", "member configuration"))
            if Taxonomy.from_dict(config.terminal_taxonomy) != taxonomy:
                raise DxContractError("member taxonomy differs from ensemble taxonomy")
    thresholds = validate_reporting_policy(
        _read_object(root / "thresholds.json", "reporting policy"),
        ensemble_sha256=hashes["ensemble.json"],
        taxonomy_sha256=hashes["taxonomy.json"],
    )
    result = dict(
        root=root,
        release=release,
        hashes=hashes,
        manifest_sha256=sha256_file(root / "SHA256SUMS.json"),
        cpg=cpg,
        taxonomy=taxonomy,
        thresholds=thresholds,
        ensemble=manifest,
    )
    if load_model:
        from .final_ensemble import FinalEnsemble

        result["model"] = FinalEnsemble(root, device=device, verify=False)
    return result


def load_release(artifact=None, *, device="cpu", load_model=True):
    return validate_release(artifact, device=device, load_model=load_model)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact")
    args = p.parse_args(argv)
    result = validate_release(args.artifact)
    print(
        json.dumps(
            dict(
                kind="alma3_release_verification",
                schema_version=2,
                release_version=RELEASE_VERSION,
                manifest_sha256=result["manifest_sha256"],
                minimum_observed_cpgs=5000,
                representation_dimensions=1090,
                **{
                    key + "_sha256": result["hashes"][filename]
                    for key, filename in (
                        ("ensemble", "ensemble.json"),
                        ("taxonomy", "taxonomy.json"),
                        ("cpg_manifest", "cpg_manifest.json"),
                        ("thresholds", "thresholds.json"),
                    )
                },
            )
        )
    )
    return 0
