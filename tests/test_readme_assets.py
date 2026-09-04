from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_HASHES = {
    "alma3-a3-signal-monogram-dark.svg": (
        "0b8ed0d7ea03875900bc724bef50323e9c5528143ec46087d0f27312a9d210ed"
    ),
    "alma3-a3-signal-monogram-light.svg": (
        "56ec3f9c5f8360433b58d80c8a183ab712e409ceadc2628ab7ba1694c87ea41e"
    ),
}


class ReadmeAssetTests(unittest.TestCase):
    def test_assets_contain_only_the_approved_theme_svgs(self) -> None:
        assets = ROOT / "assets"
        self.assertEqual({path.name for path in assets.iterdir() if path.is_file()}, set(ASSET_HASHES))
        for name, expected in ASSET_HASHES.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((assets / name).read_bytes()).hexdigest(), expected)

    def test_readme_uses_the_light_and_dark_theme_svgs(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in ASSET_HASHES:
            self.assertIn(f"assets/{name}", readme)
        self.assertIn('(prefers-color-scheme: dark)', readme)
        self.assertIn('alt="ALMA3 A3 symbol"', readme)
        self.assertNotIn(".gif", readme)
        self.assertNotIn("prefers-reduced-motion", readme)


if __name__ == "__main__":
    unittest.main()
