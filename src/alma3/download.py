from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Any, BinaryIO

from filelock import FileLock

from .hashes import read_sha256_manifest, sha256_file
from .release import RELEASE_FILES, RELEASE_PAYLOADS, RELEASE_VERSION, validate_release

CATALOG_KIND = "alma3_release_catalog"
CATALOG_SCHEMA_VERSION = 1
DEFAULT_CACHE_ENV = "ALMA3_CACHE_DIR"
RELEASE_ENV = "ALMA3_RELEASE"
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_PROGRESS_MIN_BYTES = 100 * 1024 * 1024


class DownloadError(RuntimeError):
    """Raised when a catalog or model download is invalid."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_catalog() -> dict[str, Any]:
    try:
        payload = json.loads(files("alma3").joinpath("release_catalog.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DownloadError("the packaged ALMA3 release catalog is unreadable") from error
    if not isinstance(payload, dict) or set(payload) != {
        "kind",
        "schema_version",
        "default_version",
        "releases",
    }:
        raise DownloadError("the packaged ALMA3 release catalog is invalid")
    if (
        payload["kind"] != CATALOG_KIND
        or payload["schema_version"] != CATALOG_SCHEMA_VERSION
        or payload["default_version"] != RELEASE_VERSION
        or not isinstance(payload["releases"], list)
    ):
        raise DownloadError("the packaged ALMA3 release catalog is invalid")
    return payload


def _release_spec(version: str = RELEASE_VERSION) -> dict[str, Any]:
    matches = [entry for entry in _load_catalog()["releases"] if isinstance(entry, dict) and entry.get("version") == version]
    if len(matches) != 1:
        raise DownloadError(f"ALMA3 release {version} is not available in the packaged catalog")
    spec = matches[0]
    if set(spec) != {"version", "base_url", "manifest_sha256", "files"}:
        raise DownloadError(f"ALMA3 release {version} catalog entry is invalid")
    base_url = spec["base_url"]
    manifest_sha256 = spec["manifest_sha256"]
    file_specs = spec["files"]
    if (
        not isinstance(base_url, str)
        or not base_url.startswith("https://")
        or not base_url.endswith("/")
        or not _is_sha256(manifest_sha256)
        or not isinstance(file_specs, dict)
        or set(file_specs) != RELEASE_FILES
    ):
        raise DownloadError(f"ALMA3 release {version} catalog entry is invalid")
    for name, file_spec in file_specs.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(file_spec, dict)
            or set(file_spec) != {"sha256", "size"}
            or not _is_sha256(file_spec["sha256"])
            or type(file_spec["size"]) is not int
            or file_spec["size"] < 0
        ):
            raise DownloadError(f"ALMA3 release {version} file catalog is invalid")
    if file_specs["SHA256SUMS.json"]["sha256"] != manifest_sha256:
        raise DownloadError(f"ALMA3 release {version} manifest binding is invalid")
    return spec


def cache_root() -> Path:
    configured = os.environ.get(DEFAULT_CACHE_ENV)
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "alma3"


def cached_release_path(version: str = RELEASE_VERSION) -> Path:
    spec = _release_spec(version)
    return cache_root() / version / spec["manifest_sha256"]


def _format_size(size: int) -> str:
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.1f} GB"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    return f"{size / 1_000:.1f} kB"


def _copy_response(
    response: BinaryIO,
    handle: BinaryIO,
    *,
    name: str,
    initial_size: int,
    expected_size: int,
) -> None:
    transferred = initial_size
    progress = expected_size >= DOWNLOAD_PROGRESS_MIN_BYTES and sys.stderr.isatty()
    last_percent = -1

    def report() -> None:
        nonlocal last_percent
        percent = min(100, int(100 * transferred / expected_size))
        if percent == last_percent:
            return
        last_percent = percent
        print(
            f"\rDownloading {name}: {_format_size(transferred)} / {_format_size(expected_size)} ({percent}%)",
            end="",
            file=sys.stderr,
            flush=True,
        )

    if progress:
        report()
    while True:
        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
        if not chunk:
            break
        handle.write(chunk)
        transferred += len(chunk)
        if progress:
            report()
    if progress:
        print(file=sys.stderr)


def _download_file(url: str, destination: Path, *, expected_size: int, expected_sha256: str) -> None:
    if destination.is_file():
        size = destination.stat().st_size
        if size == expected_size and sha256_file(destination) == expected_sha256:
            return
        if size >= expected_size:
            destination.unlink()
    elif destination.exists() or destination.is_symlink():
        raise DownloadError(f"download path is not a regular file: {destination}")

    offset = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", response.getcode())
            append = offset > 0 and status == 206
            mode = "ab" if append else "wb"
            with destination.open(mode) as handle:
                _copy_response(
                    response,
                    handle,
                    name=destination.name,
                    initial_size=offset if append else 0,
                    expected_size=expected_size,
                )
                handle.flush()
                os.fsync(handle.fileno())
    except (OSError, urllib.error.URLError) as error:
        raise DownloadError(f"failed to download {url}") from error
    if destination.stat().st_size != expected_size:
        raise DownloadError(f"downloaded file has the wrong size: {destination.name}")
    actual = sha256_file(destination)
    if actual != expected_sha256:
        destination.unlink(missing_ok=True)
        raise DownloadError(f"downloaded file failed SHA-256 verification: {destination.name}")


def _validate_catalog_files(root: Path, spec: dict[str, Any]) -> None:
    if not root.is_dir() or root.is_symlink():
        raise DownloadError(f"downloaded release is not a real directory: {root}")
    actual_files = {path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()}
    if actual_files != RELEASE_FILES or any(path.is_dir() or path.is_symlink() for path in root.iterdir()):
        raise DownloadError("downloaded release file set is invalid")
    manifest = read_sha256_manifest(root / "SHA256SUMS.json")
    if set(manifest) != RELEASE_PAYLOADS:
        raise DownloadError("downloaded release SHA-256 manifest entries are invalid")
    for name, expected in spec["files"].items():
        path = root / name
        if path.stat().st_size != expected["size"]:
            raise DownloadError(f"downloaded release file is invalid: {name}")
        if name in RELEASE_PAYLOADS and manifest[name] != expected["sha256"]:
            raise DownloadError(f"downloaded release manifest does not match the catalog: {name}")
    if sha256_file(root / "SHA256SUMS.json") != spec["manifest_sha256"]:
        raise DownloadError("downloaded release manifest is not catalog-bound")
    marker = spec["files"]["RELEASE_COMPLETE"]
    if sha256_file(root / "RELEASE_COMPLETE") != marker["sha256"]:
        raise DownloadError("downloaded release completion marker is not catalog-bound")


def _validate_existing(path: Path, spec: dict[str, Any]) -> Path:
    _validate_catalog_files(path, spec)
    validate_release(path, device="cpu", load_model=False)
    return path.resolve()


def download_release(
    output: str | Path | None = None,
    *,
    version: str = RELEASE_VERSION,
    repair_managed_cache: bool = False,
) -> Path:
    spec = _release_spec(version)
    destination = Path(output).expanduser() if output is not None else cached_release_path(version)
    destination = destination.absolute()
    lock = destination.parent / f".{destination.name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock):
        if destination.exists() or destination.is_symlink():
            try:
                return _validate_existing(destination, spec)
            except (DownloadError, OSError, ValueError):
                if not repair_managed_cache or destination != cached_release_path(version).absolute():
                    raise
                shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.parent / f".{destination.name}.partial"
        if partial.is_symlink() or (partial.exists() and not partial.is_dir()):
            raise DownloadError(f"partial download path is invalid: {partial}")
        partial.mkdir(exist_ok=True)
        try:
            for name in sorted(RELEASE_FILES - {"RELEASE_COMPLETE"}):
                expected = spec["files"][name]
                _download_file(
                    spec["base_url"] + name,
                    partial / name,
                    expected_size=expected["size"],
                    expected_sha256=expected["sha256"],
                )
            marker = spec["files"]["RELEASE_COMPLETE"]
            _download_file(
                spec["base_url"] + "RELEASE_COMPLETE",
                partial / "RELEASE_COMPLETE",
                expected_size=marker["size"],
                expected_sha256=marker["sha256"],
            )
            if sys.stderr.isatty():
                print(f"Verifying ALMA3 {version}...", file=sys.stderr, flush=True)
            _validate_catalog_files(partial, spec)
            validate_release(partial, device="cpu", load_model=False)
            os.rename(partial, destination)
            if sys.stderr.isatty():
                print(f"ALMA3 {version} is ready at {destination}", file=sys.stderr, flush=True)
        except BaseException:
            if partial.exists() and not any(partial.iterdir()):
                partial.rmdir()
            raise
    return destination.resolve()


def load_release(
    explicit: str | Path | None = None,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    if explicit is not None:
        path = Path(explicit).expanduser()
        return validate_release(path, device=device)
    configured = os.environ.get(RELEASE_ENV)
    if configured:
        path = Path(configured).expanduser()
        return validate_release(path, device=device)
    cached = cached_release_path()
    if cached.exists() or cached.is_symlink():
        try:
            return validate_release(cached, device=device)
        except (DownloadError, OSError, ValueError):
            cached = download_release(repair_managed_cache=True)
            return validate_release(cached, device=device)
    cached = download_release()
    return validate_release(cached, device=device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alma3 download",
        description="Download and verify the ALMA3 3.0.0 model.",
    )
    parser.add_argument(
        "--output",
        help="new release directory; omit to use the managed ALMA3 cache",
    )
    args = parser.parse_args(argv)
    print(download_release(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
