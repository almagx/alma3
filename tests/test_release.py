from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alma3.dx import DxContractError
from alma3.hashes import sha256_file
from alma3.release import (
    RELEASE_FILES,
    ensemble_asset_paths,
    load_release,
    revalidate_release_identity,
    validate_release,
)


class NestedPayloadTests(unittest.TestCase):
    def manifest(self):
        return {
            "kind": "alma3_final_ten_v1",
            "decision_rule": "type_first_v1",
            "members": [{"kind": "neural", "export": f"members/{i}"} for i in range(10)],
        }

    def test_nested_assets_and_path_escape(self):
        manifest = self.manifest()
        self.assertIn("members/9/model.safetensors", ensemble_asset_paths(manifest))
        for path in ("../outside", "/outside", "members/../../outside"):
            manifest["members"][0]["export"] = path
            with self.assertRaisesRegex(DxContractError, "inside the release"):
                ensemble_asset_paths(manifest)

    def test_private_files_and_nested_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest()
            names = ensemble_asset_paths(manifest) | RELEASE_FILES
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            (root / "ensemble.json").write_text(json.dumps(manifest))
            (root / "RELEASE_COMPLETE").write_text("complete\n")
            hashes = {name: sha256_file(root / name) for name in names - {"SHA256SUMS.json", "RELEASE_COMPLETE"}}
            (root / "SHA256SUMS.json").write_text(json.dumps(hashes))
            identity = sha256_file(root / "SHA256SUMS.json")
            revalidate_release_identity(root, manifest_sha256=identity, hashes=hashes)
            private = root / "clinical_history.json"
            private.write_text("{}")
            with self.assertRaisesRegex(DxContractError, "inventory"):
                validate_release(root)
            private.unlink()
            (root / "members/0/model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(DxContractError, "checksum mismatch"):
                validate_release(root)
            with self.assertRaisesRegex(DxContractError, "changed during inference"):
                revalidate_release_identity(root, manifest_sha256=identity, hashes=hashes)


class ReleaseResolutionTests(unittest.TestCase):
    def test_explicit_release_precedes_environment(self) -> None:
        validated = {"root": Path("explicit")}
        with (
            patch.dict(os.environ, {"ALMA3_RELEASE": "configured"}),
            patch("alma3.release.validate_release", return_value=validated) as validate,
        ):
            self.assertIs(load_release("explicit", device="cuda:1", load_model=False), validated)
        validate.assert_called_once_with("explicit", device="cuda:1", load_model=False)

    def test_environment_release_is_used_when_explicit_is_absent(self) -> None:
        validated = {"root": Path("configured")}
        with (
            patch.dict(os.environ, {"ALMA3_RELEASE": "configured"}),
            patch("alma3.release.validate_release", return_value=validated) as validate,
        ):
            self.assertIs(load_release(load_model=False), validated)
        validate.assert_called_once_with(None, device="cpu", load_model=False)

    def test_missing_release_fails_without_network_fallback(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(
                DxContractError,
                "provide a release directory or set ALMA3_RELEASE",
            ),
        ):
            load_release(load_model=False)


if __name__ == "__main__":
    unittest.main()
