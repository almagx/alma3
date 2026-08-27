from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from alma3.clinical_result import validate_result
from alma3.config import DxConfig
from alma3.dx import DX_TARGETS, DiagnosticModel
from alma3.hashes import publish_new_file as publish_new_file_actual
from alma3.infer import (
    InputContractError,
    main as infer_main,
    run_inference,
    validate_embedding_sidecar,
)
from alma3.model import FoundationModel
from alma3.release import RELEASE_FILES, validate_release

from helpers import create_release, foundation_config, sha256, write_array_csv, write_json


class RuntimeContractTests(unittest.TestCase):
    def test_embedding_api_and_forward_are_exactly_equivalent(self) -> None:
        config = foundation_config()
        model = DiagnosticModel(
            FoundationModel(config),
            DxConfig(
                foundation=config,
                targets={target: 2 for target in DX_TARGETS},
                hidden_dim=16,
                dropout=0.0,
            ),
            freeze_foundation=False,
        ).eval()
        inputs = (
            torch.rand(2, config.n_cpgs),
            torch.ones(2, config.n_cpgs, dtype=torch.bool),
            torch.zeros(2, config.n_cpgs),
            torch.zeros(2, config.n_cpgs, dtype=torch.long),
            torch.linspace(0, 1, config.n_cpgs).expand(2, -1),
        )
        with torch.inference_mode():
            embedding = model.embed(*inputs)
            split = model.logits_from_embedding(embedding)
            composed = model(*inputs)
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertEqual(embedding.shape, (2, config.d_model))
        for target in DX_TARGETS:
            self.assertTrue(torch.equal(split[target], composed[target]))
        with self.assertRaisesRegex(ValueError, "embedding must have shape"):
            model.logits_from_embedding(torch.zeros(2, config.d_model + 1))
        with self.assertRaisesRegex(ValueError, "float32"):
            model.logits_from_embedding(torch.zeros(2, config.d_model, dtype=torch.float64))

    def test_release_validation_is_exact_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, _ = create_release(root / "release")
            validated = validate_release(release)
            self.assertEqual(set(path.name for path in release.iterdir()), set(RELEASE_FILES))
            self.assertEqual(validated["thresholds"]["minimum_observed_cpgs"], 1500)
            self.assertEqual(validated["config"].foundation.d_model, 8)

            future = root / "future"
            shutil.copytree(release, future)
            provenance_path = future / "release_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance.update(
                {
                    "schema_version": 8,
                    "training_git_commit": provenance["evaluation_git_commit"],
                    "runtime_git_commit": "a" * 40,
                }
            )
            write_json(provenance_path, provenance)
            manifest_path = future / "SHA256SUMS.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["release_provenance.json"] = sha256(provenance_path)
            write_json(manifest_path, manifest)
            self.assertEqual(validate_release(future)["provenance"]["schema_version"], 8)

            provenance.pop("runtime_git_commit")
            write_json(provenance_path, provenance)
            manifest["release_provenance.json"] = sha256(provenance_path)
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "source provenance"):
                validate_release(future)

            mutations = {
                "marker": lambda path: (path / "RELEASE_COMPLETE").write_text("done\n", encoding="utf-8"),
                "missing": lambda path: (path / "taxonomy.json").unlink(),
                "extra": lambda path: (path / "unexpected.txt").write_text("extra\n", encoding="utf-8"),
                "tampered": lambda path: (path / "model.safetensors").write_bytes(b"tampered\n"),
            }
            for name, mutation in mutations.items():
                with self.subTest(name=name):
                    candidate = root / name
                    shutil.copytree(release, candidate)
                    mutation(candidate)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        validate_release(candidate)

    def test_inference_and_sidecar_share_one_embedding_tensor_and_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, _ = create_release(root / "release")
            input_path = root / "input.csv"
            write_array_csv(input_path, sample_count=3)
            output = root / "result.jsonl"
            sidecar = root / "embedding.json"
            embedded: list[torch.Tensor] = []
            consumed: list[torch.Tensor] = []
            batch_sizes: list[int] = []
            original_embed = DiagnosticModel.embed
            original_logits = DiagnosticModel.logits_from_embedding

            def record_embed(instance, beta, *args, **kwargs):
                batch_sizes.append(int(beta.shape[0]))
                value = original_embed(instance, beta, *args, **kwargs)
                embedded.append(value)
                return value

            def record_logits(instance, embedding, *args, **kwargs):
                consumed.append(embedding)
                return original_logits(instance, embedding, *args, **kwargs)

            with (
                patch.object(DiagnosticModel, "embed", record_embed),
                patch.object(DiagnosticModel, "logits_from_embedding", record_logits),
            ):
                run_inference(
                    release,
                    input_path,
                    "array-csv",
                    output,
                    device="cpu",
                    embedding_sidecar=sidecar,
                )
            self.assertEqual(batch_sizes, [2, 1])
            self.assertEqual(len(embedded), len(consumed))
            self.assertTrue(all(source is target for source, target in zip(embedded, consumed, strict=True)))

            results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            validate_embedding_sidecar(payload)
            self.assertEqual([row["sample_id"] for row in results], ["sample-1", "sample-2", "sample-3"])
            self.assertEqual(
                [row["sample_id"] for row in payload["samples"]],
                ["sample-1", "sample-2", "sample-3"],
            )
            for result in results:
                validate_result(result)
            self.assertEqual(payload["release"]["model_sha256"], results[0]["release"]["model_sha256"])
            self.assertEqual(payload["release"]["taxonomy_sha256"], results[0]["release"]["taxonomy_sha256"])
            self.assertEqual(payload["release"]["thresholds_sha256"], results[0]["release"]["thresholds_sha256"])

    def test_sparse_and_interrupted_inference_publish_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, _ = create_release(root / "release")
            sparse = root / "sparse.csv"
            write_array_csv(sparse, sample_count=1, observed=1499)
            sparse_output = root / "sparse.jsonl"
            sparse_sidecar = root / "sparse.embedding.json"
            with self.assertRaisesRegex(InputContractError, "below calibrated observed-CpG floor"):
                run_inference(
                    release,
                    sparse,
                    "array-csv",
                    sparse_output,
                    device="cpu",
                    embedding_sidecar=sparse_sidecar,
                )
            self.assertFalse(sparse_output.exists())
            self.assertFalse(sparse_sidecar.exists())

            input_path = root / "input.csv"
            write_array_csv(input_path, sample_count=1)
            failed_output = root / "failed.jsonl"
            failed_sidecar = root / "failed.embedding.json"
            calls = 0

            def fail_clinical_publish(temporary, destination):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return publish_new_file_actual(temporary, destination)
                raise OSError("clinical publication failed")

            with patch("alma3.infer.publish_new_file", side_effect=fail_clinical_publish):
                with self.assertRaisesRegex(OSError, "clinical publication failed"):
                    run_inference(
                        release,
                        input_path,
                        "array-csv",
                        failed_output,
                        device="cpu",
                        embedding_sidecar=failed_sidecar,
                    )
            self.assertEqual(calls, 2)
            self.assertFalse(failed_output.exists())
            self.assertFalse(failed_sidecar.exists())

    def test_infer_cli_has_no_research_or_implicit_download_mode(self) -> None:
        arguments = [
            "--artifact",
            "release",
            "--input",
            "input.csv",
            "--format",
            "array-csv",
            "--output",
            "result.jsonl",
        ]
        with patch("alma3.infer.run_inference") as run:
            self.assertEqual(infer_main(arguments), 0)
            run.assert_called_once_with(
                "release",
                "input.csv",
                "array-csv",
                "result.jsonl",
                device="auto",
                embedding_sidecar=None,
            )
        for removed in ("--research", "--download", "--all-probs", "--output-format"):
            with self.subTest(removed=removed), self.assertRaises(SystemExit):
                infer_main([*arguments, removed])


if __name__ == "__main__":
    unittest.main()
