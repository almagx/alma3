from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from alma3.download import DownloadError, _copy_response, _download_file, download_release, load_release
from alma3.hashes import sha256_file
from tests.helpers import create_release


def _catalog(release: Path) -> dict[str, object]:
    manifest_sha256 = sha256_file(release / "SHA256SUMS.json")
    return {
        "kind": "alma3_release_catalog",
        "schema_version": 1,
        "default_version": "3.0.0",
        "releases": [
            {
                "version": "3.0.0",
                "base_url": f"https://models.almagx.com/alma3/3.0.0/{manifest_sha256}/",
                "manifest_sha256": manifest_sha256,
                "files": {
                    path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
                    for path in release.iterdir()
                },
            }
        ],
    }


class DownloadTests(unittest.TestCase):
    def test_large_download_reports_progress_only_for_a_terminal(self) -> None:
        target = io.BytesIO()
        terminal = io.StringIO()
        terminal.isatty = lambda: True
        with (
            patch("alma3.download.DOWNLOAD_PROGRESS_MIN_BYTES", 0),
            patch("alma3.download.sys.stderr", terminal),
            redirect_stderr(terminal),
        ):
            _copy_response(
                io.BytesIO(b"payload"),
                target,
                name="model.safetensors",
                initial_size=0,
                expected_size=7,
            )
        self.assertEqual(target.getvalue(), b"payload")
        self.assertIn("Downloading model.safetensors", terminal.getvalue())
        self.assertIn("(100%)", terminal.getvalue())

    def test_resumable_file_download_and_hash_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = b"abcdefgh"
            destination = root / "payload"
            destination.write_bytes(data[:3])

            class Response(io.BytesIO):
                status = 206

                def getcode(self) -> int:
                    return self.status

                def __enter__(self):
                    return self

                def __exit__(self, *_args) -> None:
                    self.close()

            with patch("alma3.download.urllib.request.urlopen", return_value=Response(data[3:])) as opened:
                _download_file(
                    "https://models.almagx.com/payload",
                    destination,
                    expected_size=len(data),
                    expected_sha256=hashlib.sha256(data).hexdigest(),
                )
            self.assertEqual(destination.read_bytes(), data)
            self.assertEqual(opened.call_args.args[0].get_header("Range"), "bytes=3-")

            destination.unlink()
            with (
                patch("alma3.download.urllib.request.urlopen", return_value=Response(b"substitute")),
                self.assertRaisesRegex(DownloadError, "SHA-256"),
            ):
                _download_file(
                    "https://models.almagx.com/payload",
                    destination,
                    expected_size=len(b"substitute"),
                    expected_sha256=hashlib.sha256(b"expected!!").hexdigest(),
                )
            self.assertFalse(destination.exists())

    def test_download_is_atomic_and_concurrent_callers_share_one_copy(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, _ = create_release(root / "source")
            output = root / "downloaded"
            catalog = _catalog(release)
            copies: list[str] = []

            def copy_file(url: str, destination: Path, *, expected_size: int, expected_sha256: str) -> None:
                name = url.rsplit("/", 1)[-1]
                copies.append(name)
                shutil.copyfile(release / name, destination)
                self.assertEqual(destination.stat().st_size, expected_size)
                self.assertEqual(sha256_file(destination), expected_sha256)

            with (
                patch("alma3.download._load_catalog", return_value=catalog),
                patch("alma3.download._download_file", side_effect=copy_file),
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                futures = [pool.submit(download_release, output) for _ in range(2)]
                results = [future.result(timeout=10) for future in futures]
            self.assertEqual(results, [output.resolve(), output.resolve()])
            self.assertEqual(sorted(copies), sorted(path.name for path in release.iterdir()))
            self.assertFalse(output.parent.joinpath(f".{output.name}.partial").exists())

    def test_invalid_explicit_release_never_falls_back(self) -> None:
        explicit = Path("invalid-explicit-release")
        with (
            patch("alma3.download.validate_release", side_effect=ValueError("invalid release")) as validate,
            patch("alma3.download.download_release") as download,
            self.assertRaisesRegex(ValueError, "invalid release"),
        ):
            load_release(explicit)
        validate.assert_called_once_with(explicit, device="cpu")
        download.assert_not_called()

    def test_environment_release_precedes_cache_and_download(self) -> None:
        configured = Path("configured-release")
        validated = {"root": configured.resolve()}
        with (
            patch.dict(os.environ, {"ALMA3_RELEASE": str(configured)}),
            patch("alma3.download.validate_release", return_value=validated) as validate,
            patch("alma3.download.cached_release_path") as cached,
            patch("alma3.download.download_release") as download,
        ):
            self.assertIs(load_release(), validated)
        validate.assert_called_once_with(configured, device="cpu")
        cached.assert_not_called()
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
