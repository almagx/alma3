from __future__ import annotations

import hashlib
import io
import json
import runpy
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GATE = runpy.run_path(str(ROOT / "scripts" / "release-gate"), run_name="alma3_release_gate_tests")


def _add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _write_docker_archive(path: Path) -> str:
    config = b'{"architecture":"amd64","os":"linux"}'
    digest = hashlib.sha256(config).hexdigest()
    manifest = json.dumps(
        [{"Config": f"{digest}.json", "Layers": [], "RepoTags": ["alma3:test"]}],
        separators=(",", ":"),
    ).encode()
    with tarfile.open(path, "w") as archive:
        _add_tar_bytes(archive, "manifest.json", manifest)
        _add_tar_bytes(archive, f"{digest}.json", config)
    return f"sha256:{digest}"


class ReleaseGateCandidateTests(unittest.TestCase):
    def test_copy_workflow_preserves_failures_and_cleans_container(self) -> None:
        for phase in ("input", "analysis", "output", "interrupt", "cleanup", "analysis-and-cleanup"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as raw:
                commands: list[list[str]] = []
                original = subprocess.CalledProcessError(2, [phase])

                def execute(
                    command: list[str], *, phase: str = phase,
                    commands: list[list[str]] = commands,
                    original: subprocess.CalledProcessError = original,
                    **kwargs: object,
                ) -> subprocess.CompletedProcess:
                    commands.append(command)
                    current = (
                        "cleanup" if command[1] == "rm" else
                        "analysis" if command[1] == "start" else
                        "output" if command[2].startswith("container:") else "input"
                    )
                    if current == "analysis" and phase == "interrupt":
                        raise KeyboardInterrupt
                    if current != "cleanup" and (
                        current == phase or (current == "analysis" and phase == "analysis-and-cleanup")
                    ):
                        raise original
                    return subprocess.CompletedProcess(
                        command, 1 if current == "cleanup" and "cleanup" in phase else 0,
                    )

                with (
                    patch.object(subprocess, "check_output", return_value="container\n"),
                    patch.object(subprocess, "run", side_effect=execute),
                    self.assertRaises(KeyboardInterrupt if phase == "interrupt" else subprocess.CalledProcessError) as caught,
                ):
                    GATE["_copy_inference_results"](
                        "image", Path(raw) / "result.jsonl", input_file=Path(raw) / "private.csv.gz",
                    )
                self.assertEqual(commands[-1], ["docker", "rm", "-f", "container"])
                if phase not in {"cleanup", "interrupt"}:
                    self.assertIs(caught.exception, original)
                if phase in {"input", "analysis", "interrupt", "analysis-and-cleanup"}:
                    self.assertFalse(any(command[1:3] == ["cp", "container:/tmp/alma3-results.jsonl"] for command in commands))

    def test_copy_workflow_refuses_existing_output_before_creating_container(self) -> None:
        with tempfile.TemporaryDirectory() as raw, patch.object(subprocess, "check_output") as create:
            output = Path(raw) / "result.jsonl"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                GATE["_copy_inference_results"]("image", output)
            create.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_atomic_candidate_publishes_new_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "candidate"
            with GATE["_atomic_candidate"](output) as temporary:
                (temporary / "payload").write_text("complete\n", encoding="utf-8")
            self.assertEqual((output / "payload").read_text(encoding="utf-8"), "complete\n")

    def test_atomic_candidate_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "candidate"
            output.mkdir()
            with self.assertRaises(FileExistsError), GATE["_atomic_candidate"](output):
                pass

    def test_atomic_candidate_removes_temporary_directory_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "candidate"
            with (
                self.assertRaisesRegex(RuntimeError, "injected"),
                GATE["_atomic_candidate"](output) as temporary,
            ):
                (temporary / "partial").write_text("partial", encoding="utf-8")
                raise RuntimeError("injected")
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".candidate.tmp-*")), [])

    def test_archive_config_id_is_bound_to_config_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            archive = Path(raw) / "image.tar"
            expected = _write_docker_archive(archive)
            self.assertEqual(GATE["_archive_config_id"](archive), expected)

    def test_candidate_verifier_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            candidate = Path(raw)
            image = candidate / "image" / "alma3-3.1.1-cpu.tar"
            image.parent.mkdir()
            image_id = _write_docker_archive(image)
            acceptance = {
                "image": {
                    "archive_config_id": image_id,
                    "archive_sha256": GATE["_sha256"](image),
                    "image_id": image_id,
                },
                "kind": GATE["CANDIDATE_KIND"],
                "schema_version": GATE["CANDIDATE_SCHEMA_VERSION"],
                "status": "passed",
            }
            (candidate / "acceptance.json").write_bytes(GATE["_canonical_json"](acceptance))
            checksums = {
                "acceptance.json": GATE["_sha256"](candidate / "acceptance.json"),
                "image/alma3-3.1.1-cpu.tar": GATE["_sha256"](image),
            }
            (candidate / "SHA256SUMS.json").write_bytes(GATE["_canonical_json"](checksums))
            (candidate / GATE["CANDIDATE_MARKER"]).write_bytes(b"complete\n")

            GATE["_verify_candidate"](candidate)
            extra = candidate / "extra"
            extra.mkdir()
            with self.assertRaisesRegex(RuntimeError, "directory layout"):
                GATE["_verify_candidate"](candidate)
            extra.rmdir()
            image.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                GATE["_verify_candidate"](candidate)

    def test_candidate_name_binds_source_and_release(self) -> None:
        source = {"commit": "a" * 40}
        release = {"manifest_sha256": "b" * 64}
        self.assertEqual(
            GATE["_expected_candidate_name"](source, release),
            "alma3-3.1.1-docker-aaaaaaaa-bbbbbbbb",
        )

    def test_current_citation_matches_software_citation(self) -> None:
        self.assertEqual(
            GATE["_citation_value"](),
            "ALMA Genomics Inc. (2026). ALMA3 (Version 3.1.1) [Computer software]. "
            "https://github.com/almagx/alma3\n\n"
            "Manuscript in preparation.",
        )


if __name__ == "__main__":
    unittest.main()
