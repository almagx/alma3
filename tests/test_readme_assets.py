from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "assets/alma3-a3-motion-readme.json"
SOURCE_COMMIT = "69821c8a02a8aec48bf86ad0c7b93ada4340851c"
SOURCE_HASHES = {
    "light": "066232e9534bb8204140910bcdf2c83d3382807d8366c990a9bfdeb69c84f026",
    "dark": "d04e9f8c55492c422aa3baa14be647d22b7d227b5d13d803dc1db0ee78191b81",
}
OUTPUT_HASHES = {
    "light": "c7ed28113181034bd6a085bff42b7f832c97624c2d04959f552ccae318f3a7bf",
    "dark": "9cd0f07c614ad295aac1bc68f2b7ba059de6c2ff134432781f46de312996f174",
}


def _subblocks(data: bytes, position: int) -> tuple[int, list[bytes]]:
    blocks: list[bytes] = []
    while True:
        if position >= len(data):
            raise AssertionError("truncated GIF sub-block stream")
        size = data[position]
        position += 1
        if size == 0:
            return position, blocks
        end = position + size
        if end > len(data):
            raise AssertionError("truncated GIF sub-block")
        blocks.append(data[position:end])
        position = end


def _gif_metadata(data: bytes) -> dict[str, object]:
    if data[:6] != b"GIF89a" or len(data) < 14:
        raise AssertionError("README animation must be a GIF89a file")
    width = int.from_bytes(data[6:8], "little")
    height = int.from_bytes(data[8:10], "little")
    packed = data[10]
    if not packed & 0x80:
        raise AssertionError("README animation must use one global color table")
    position = 13 + 3 * (1 << ((packed & 0x07) + 1))
    pending_control: tuple[int, bool] | None = None
    repeat: int | None = None
    frames: list[dict[str, object]] = []

    while position < len(data):
        marker = data[position]
        if marker == 0x3B:
            if position != len(data) - 1:
                raise AssertionError("GIF has trailing bytes")
            break
        if marker == 0x21:
            if position + 2 >= len(data):
                raise AssertionError("truncated GIF extension")
            label = data[position + 1]
            if label == 0xF9:
                if data[position + 2] != 4 or position + 8 > len(data):
                    raise AssertionError("invalid GIF graphic control extension")
                control = data[position + 3]
                delay = int.from_bytes(data[position + 4 : position + 6], "little")
                if data[position + 7] != 0:
                    raise AssertionError("unterminated GIF graphic control extension")
                pending_control = (delay, bool(control & 0x01))
                position += 8
            else:
                position, blocks = _subblocks(data, position + 2)
                if label == 0xFF and len(blocks) >= 2 and blocks[0] == b"NETSCAPE2.0":
                    loop = blocks[1]
                    if len(loop) != 3 or loop[0] != 1:
                        raise AssertionError("invalid GIF repeat extension")
                    repeat = int.from_bytes(loop[1:3], "little")
            continue
        if marker == 0x2C:
            if pending_control is None or position + 10 > len(data):
                raise AssertionError("GIF image has no complete control extension")
            image_start = position
            frame_width = int.from_bytes(data[position + 5 : position + 7], "little")
            frame_height = int.from_bytes(data[position + 7 : position + 9], "little")
            image_packed = data[position + 9]
            local_color_table = bool(image_packed & 0x80)
            position += 10
            if local_color_table:
                position += 3 * (1 << ((image_packed & 0x07) + 1))
            if position >= len(data):
                raise AssertionError("GIF frame has no LZW code size")
            position, _blocks = _subblocks(data, position + 1)
            delay, transparent = pending_control
            frames.append(
                {
                    "width": frame_width,
                    "height": frame_height,
                    "delay": delay,
                    "transparent": transparent,
                    "localColorTable": local_color_table,
                    "image": data[image_start:position],
                }
            )
            pending_control = None
            continue
        raise AssertionError(f"unsupported GIF block marker: 0x{marker:02x}")
    else:
        raise AssertionError("GIF has no trailer")

    return {"width": width, "height": height, "repeat": repeat, "frames": frames}


class ReadmeAssetTests(unittest.TestCase):
    def test_readme_motion_assets_match_the_approved_seal(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["status"], "approved")
        self.assertEqual(manifest["source"]["commit"], SOURCE_COMMIT)
        self.assertEqual(
            {theme: value["sha256"] for theme, value in manifest["source"]["outputs"].items()},
            SOURCE_HASHES,
        )
        self.assertEqual(manifest["transform"]["kind"], "prepend-terminal-poster-frame")
        self.assertEqual(manifest["transform"]["posterDelayCentiseconds"], 5)

        outputs = {output["theme"]: output for output in manifest["outputs"]}
        self.assertEqual(set(outputs), {"light", "dark"})
        for theme, expected_hash in OUTPUT_HASHES.items():
            output = outputs[theme]
            self.assertEqual(output["sha256"], expected_hash)
            path = ROOT / output["path"]
            data = path.read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected_hash)
            metadata = _gif_metadata(data)
            self.assertEqual((metadata["width"], metadata["height"]), (384, 384))
            self.assertEqual(metadata["repeat"], 0)
            frames = metadata["frames"]
            self.assertEqual(len(frames), 62)
            self.assertEqual([frame["delay"] for frame in frames], [5] * 61 + [100])
            self.assertTrue(all(frame["width"] == 384 and frame["height"] == 384 for frame in frames))
            self.assertTrue(all(not frame["transparent"] for frame in frames))
            self.assertTrue(all(not frame["localColorTable"] for frame in frames))
            self.assertEqual(frames[0]["image"], frames[-1]["image"])

    def test_readme_uses_theme_motion_and_static_reduced_motion_sources(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)", readme)
        self.assertIn("assets/alma3-a3-signal-monogram-dark.svg", readme)
        self.assertIn("assets/alma3-a3-signal-monogram-light.svg", readme)
        for theme in ("light", "dark"):
            self.assertIn(f"assets/alma3-a3-motion-signature-relay-readme-{theme}.gif", readme)
        self.assertIn('alt="Animated ALMA3 A3 symbol"', readme)


if __name__ == "__main__":
    unittest.main()
