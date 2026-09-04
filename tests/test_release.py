from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from alma3.dx import DxContractError
from alma3.release import load_release


class ReleaseResolutionTests(unittest.TestCase):
    def test_explicit_release_precedes_environment(self) -> None:
        validated = {"root": Path("explicit")}
        with (
            patch.dict(os.environ, {"ALMA3_RELEASE": "configured"}),
            patch("alma3.release.validate_release", return_value=validated) as validate,
        ):
            self.assertIs(load_release("explicit", device="cuda:1", load_model=False), validated)
        validate.assert_called_once_with(Path("explicit"), device="cuda:1", load_model=False)

    def test_environment_release_is_used_when_explicit_is_absent(self) -> None:
        validated = {"root": Path("configured")}
        with (
            patch.dict(os.environ, {"ALMA3_RELEASE": "configured"}),
            patch("alma3.release.validate_release", return_value=validated) as validate,
        ):
            self.assertIs(load_release(load_model=False), validated)
        validate.assert_called_once_with(Path("configured"), device="cpu", load_model=False)

    def test_missing_release_fails_without_network_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            DxContractError,
            "use --artifact or set ALMA3_RELEASE",
        ):
            load_release(load_model=False)


if __name__ == "__main__":
    unittest.main()
