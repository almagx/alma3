from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256_manifest(path: str | Path) -> dict[str, str]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"SHA256 manifest must be a non-empty object: {manifest_path}")
    result: dict[str, str] = {}
    for raw_name, raw_digest in payload.items():
        name = str(raw_name)
        digest = str(raw_digest)
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"SHA256 manifest contains unsafe path: {name!r}")
        if (
            len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"SHA256 manifest contains invalid digest for {name!r}")
        result[name] = digest
    return result


def verify_sha256_manifest(
    root: str | Path,
    *,
    manifest_name: str = "SHA256SUMS.json",
    required: Iterable[str] = (),
) -> dict[str, str]:
    directory = Path(root)
    manifest_path = directory / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing SHA256 manifest: {manifest_path}")
    manifest = read_sha256_manifest(manifest_path)
    missing = sorted(set(required) - set(manifest))
    if missing:
        raise ValueError(f"SHA256 manifest is missing required file(s): {', '.join(missing)}")
    resolved_root = directory.resolve()
    for name, expected in manifest.items():
        path = directory / name
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"SHA256 manifest path escapes artifact root: {name!r}")
        if not path.is_file():
            raise FileNotFoundError(f"SHA256 manifest references missing file: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {name}: expected {expected}, found {actual}")
    return manifest


def validate_new_external_outputs(
    artifact_root: str | Path,
    outputs: Mapping[str, str | Path | None],
    *,
    inputs: Iterable[str | Path | None] = (),
) -> dict[str, Path]:
    root = Path(artifact_root).resolve()
    resolved_inputs = {Path(path).resolve() for path in inputs if path is not None}
    resolved_outputs: dict[str, Path] = {}
    seen: dict[Path, str] = {}
    for name, raw_path in outputs.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        resolved = path.resolve()
        if resolved == root or resolved.is_relative_to(root):
            raise ValueError(f"{name} must be outside the immutable artifact: {path}")
        if resolved in resolved_inputs:
            raise ValueError(f"{name} aliases an input file: {path}")
        if resolved in seen:
            raise ValueError(f"{name} aliases {seen[resolved]}: {path}")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{name} already exists: {path}")
        seen[resolved] = name
        resolved_outputs[name] = path
    return resolved_outputs


def publish_new_file(temporary: str | Path, output: str | Path) -> tuple[int, int]:
    """Atomically publish a same-filesystem temporary without replacing an existing path."""

    temporary_path = Path(temporary)
    temporary_stat = temporary_path.stat()
    identity = temporary_stat.st_dev, temporary_stat.st_ino
    os.link(temporary_path, Path(output))
    return identity
