from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from alma3 import ALMA3
from alma3.cli import main as cli_main
from alma3.clinical_result import validate_result
from alma3.config import DxConfig
from alma3.data import CpGManifest
from alma3.dx import DX_TARGETS, DiagnosticModel
from alma3.hashes import publish_new_file as publish_new_file_actual
from alma3.infer import (
    InputContractError,
    load_bed_methyl_with_manifest,
    run_inference,
    validate_embedding_sidecar,
)
from alma3.infer import main as infer_main
from alma3.model import FoundationModel
from alma3.release import RELEASE_FILES, validate_release
from alma3.runtime import _align_array_batch
from tests.helpers import (
    create_release,
    foundation_config,
    sha256,
    validated_release_fixture,
    write_array_csv,
    write_bedmethyl,
    write_json,
)


def _assert_json_close(test: unittest.TestCase, left, right, *, atol: float = 1e-5) -> None:
    test.assertIs(type(left), type(right))
    if isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_json_close(test, left[key], right[key], atol=atol)
    elif isinstance(left, list):
        test.assertEqual(len(left), len(right))
        for left_value, right_value in zip(left, right, strict=True):
            _assert_json_close(test, left_value, right_value, atol=atol)
    elif isinstance(left, float):
        test.assertTrue(math.isclose(left, right, rel_tol=0.0, abs_tol=atol), (left, right))
    else:
        test.assertEqual(left, right)


class RuntimeContractTests(unittest.TestCase):
    def test_bedmethyl_coordinate_index_is_cached_and_preserves_duplicate_probes(self) -> None:
        manifest = CpGManifest(
            cpg_ids=("first", "second"),
            chr_id=torch.tensor([0, 0]),
            pos=torch.tensor([0.1, 0.1]),
            chrom=("chr1", "chr1"),
            start=(100, 100),
        )
        with tempfile.TemporaryDirectory() as raw:
            bed = Path(raw) / "sample.bed"
            bed.write_text("chr1\t100\t101\t.\t0\t.\t100\t101\t0\t10\t50\n", encoding="utf-8")
            _, beta, observed, coverage = load_bed_methyl_with_manifest(bed, manifest)
        self.assertIs(manifest.coordinate_index, manifest.coordinate_index)
        self.assertEqual(beta.tolist(), [[0.5, 0.5]])
        self.assertEqual(observed.tolist(), [[True, True]])
        self.assertEqual(coverage.tolist(), [[10, 10]])

    def test_package_module_entrypoint_and_concise_cli_errors(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "alma3", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "alma3 3.0.0")

        error_output = StringIO()
        with (
            patch("alma3.infer.main", side_effect=InputContractError("invalid test input")),
            redirect_stderr(error_output),
        ):
            self.assertEqual(cli_main(["infer"]), 2)
        self.assertEqual(error_output.getvalue(), "alma3: error: invalid test input\n")

    def test_multiple_bedmethyl_inputs_share_one_batch_and_preserve_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            first = root / "first.bed"
            second = root / "second.bed"
            write_bedmethyl(first)
            write_bedmethyl(second, fraction_modified=25.0)
            output = root / "result.jsonl"
            embedded_batch_sizes: list[int] = []
            original_embed = DiagnosticModel.embed

            def record_embed(instance, beta, *args, **kwargs):
                embedded_batch_sizes.append(int(beta.shape[0]))
                return original_embed(instance, beta, *args, **kwargs)

            with (
                patch("alma3.runtime.load_release", return_value=validated),
                patch.object(DiagnosticModel, "embed", record_embed),
            ):
                run_inference(
                    release,
                    [first, second],
                    "bedmethyl",
                    output,
                    device="cpu",
                    batch_size=2,
                )
            results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(embedded_batch_sizes, [2])
            self.assertEqual([result["sample_id"] for result in results], ["first", "second"])

    def test_invalid_outputs_fail_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            input_path = root / "input.csv"
            input_path.write_text("sample_id,cg0000000\nsample,0.5\n", encoding="utf-8")
            output = root / "existing.jsonl"
            output.write_text("existing\n", encoding="utf-8")
            with (
                patch("alma3.infer.ALMA3") as runtime,
                self.assertRaisesRegex(FileExistsError, "already exists"),
            ):
                run_inference(None, input_path, "array-csv", output)
            runtime.assert_not_called()

    def test_python_api_loads_once_batches_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            cpg_ids = [*validated["cpg"].cpg_ids, "unused-cpg"]
            beta = torch.full((3, len(cpg_ids)), 0.5)
            beta[:, -1] = 2.0
            embedded_batch_sizes: list[int] = []
            original_embed = DiagnosticModel.embed

            def record_embed(instance, values, *args, **kwargs):
                embedded_batch_sizes.append(int(values.shape[0]))
                return original_embed(instance, values, *args, **kwargs)

            with (
                patch("alma3.runtime.load_release", return_value=validated) as load,
                patch.object(DiagnosticModel, "embed", record_embed),
                patch("alma3.runtime._align_array_batch", wraps=_align_array_batch) as align,
            ):
                runtime = ALMA3(release, device="cpu")
                results = runtime.predict_array(
                    beta,
                    cpg_ids,
                    ["first", "second", "third"],
                    batch_size=2,
                )
            load.assert_called_once_with(release, device="cpu")
            self.assertEqual(embedded_batch_sizes, [2, 1])
            self.assertEqual([int(call.args[0].shape[0]) for call in align.call_args_list], [2, 1])
            self.assertEqual([result["sample_id"] for result in results], ["first", "second", "third"])

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
            release, model = create_release(root / "release")
            with patch("alma3.release.load_dx", return_value=model):
                validated = validate_release(release)
            self.assertEqual({path.name for path in release.iterdir()}, set(RELEASE_FILES))
            self.assertEqual(validated["thresholds"]["minimum_observed_cpgs"], 1500)
            self.assertEqual(validated["config"].foundation.d_model, 1536)

            wrong_metadata = root / "wrong-metadata"
            shutil.copytree(release, wrong_metadata)
            metadata_path = wrong_metadata / "release.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["version"] = "3.0.1"
            write_json(metadata_path, metadata)
            manifest_path = wrong_metadata / "SHA256SUMS.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["release.json"] = sha256(metadata_path)
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "metadata"):
                validate_release(wrong_metadata)

            wrong_dimensions = root / "wrong-dimensions"
            shutil.copytree(release, wrong_dimensions)
            config_path = wrong_dimensions / "config.json"
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            config_payload["foundation"]["d_model"] = 8
            config_payload["foundation"]["n_heads"] = 2
            write_json(config_path, config_payload)
            wrong_manifest = json.loads(
                (wrong_dimensions / "SHA256SUMS.json").read_text(encoding="utf-8")
            )
            wrong_manifest["config.json"] = sha256(config_path)
            write_json(wrong_dimensions / "SHA256SUMS.json", wrong_manifest)
            with self.assertRaisesRegex(ValueError, "embedding dimension must be 1536"):
                validate_release(wrong_dimensions)

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

            cpg_mutations = {
                "cpg-kind": lambda payload: payload.__setitem__("kind", "wrong"),
                "cpg-count": lambda payload: payload.__setitem__("selected_cpg_count", 1),
                "cpg-indices": lambda payload: payload["indices"].__setitem__(0, payload["indices"][1]),
                "cpg-algorithm": lambda payload: payload.__setitem__("selection_algorithm", ""),
                "cpg-source-hash": lambda payload: payload.__setitem__("source_cpg_manifest_sha256", "bad"),
            }
            for name, mutation in cpg_mutations.items():
                with self.subTest(name=name):
                    candidate = root / name
                    shutil.copytree(release, candidate)
                    cpg_path = candidate / "cpg_manifest.json"
                    cpg_payload = json.loads(cpg_path.read_text(encoding="utf-8"))
                    mutation(cpg_payload)
                    write_json(cpg_path, cpg_payload)
                    candidate_manifest = json.loads(
                        (candidate / "SHA256SUMS.json").read_text(encoding="utf-8")
                    )
                    candidate_manifest["cpg_manifest.json"] = sha256(cpg_path)
                    write_json(candidate / "SHA256SUMS.json", candidate_manifest)
                    with self.assertRaisesRegex(ValueError, "CpG manifest"):
                        validate_release(candidate)

    def test_inference_and_sidecar_share_one_embedding_tensor_and_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
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
                patch("alma3.runtime.load_release", return_value=validated),
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
            self.assertEqual(payload["representation"]["dimensions"], 1536)
            invalid_sidecar = json.loads(json.dumps(payload))
            invalid_sidecar["representation"]["dimensions"] = 8
            with self.assertRaisesRegex(InputContractError, "representation is invalid"):
                validate_embedding_sidecar(invalid_sidecar)
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
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            sparse = root / "sparse.csv"
            write_array_csv(sparse, sample_count=1, observed=1499)
            sparse_output = root / "sparse.jsonl"
            sparse_sidecar = root / "sparse.embedding.json"
            with (
                patch("alma3.runtime.load_release", return_value=validated),
                self.assertRaisesRegex(InputContractError, "below calibrated observed-CpG floor"),
            ):
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

            with (
                patch("alma3.runtime.load_release", return_value=validated),
                patch("alma3.infer.publish_new_file", side_effect=fail_clinical_publish),
                self.assertRaisesRegex(OSError, "clinical publication failed"),
            ):
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

    def test_absent_array_columns_equal_blank_unobserved_cells(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            complete_input = root / "complete.csv"
            sparse_input = root / "sparse.csv"
            write_array_csv(complete_input, sample_count=1, observed=1500)
            write_array_csv(
                sparse_input,
                sample_count=1,
                observed=1500,
                omit_unobserved_columns=True,
            )
            complete_output = root / "complete.jsonl"
            complete_sidecar = root / "complete.embedding.json"
            sparse_output = root / "sparse.jsonl"
            sparse_sidecar = root / "sparse.embedding.json"

            with patch("alma3.runtime.load_release", return_value=validated):
                run_inference(
                    release,
                    complete_input,
                    "array-csv",
                    complete_output,
                    device="cpu",
                    embedding_sidecar=complete_sidecar,
                )
                run_inference(
                    release,
                    sparse_input,
                    "array-csv",
                    sparse_output,
                    device="cpu",
                    embedding_sidecar=sparse_sidecar,
                )

            _assert_json_close(
                self,
                json.loads(sparse_output.read_text(encoding="utf-8")),
                json.loads(complete_output.read_text(encoding="utf-8")),
            )
            _assert_json_close(
                self,
                json.loads(sparse_sidecar.read_text(encoding="utf-8")),
                json.loads(complete_sidecar.read_text(encoding="utf-8")),
            )

    def test_bedmethyl_and_release_changes_fail_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            bed = root / "sample.bed"
            bed.write_text("chr1\t100\t101\t.\t0\t.\t100\t101\t0\t10\t50\n", encoding="utf-8")
            original_reader = __import__("csv").reader

            def mutating_reader(handle, *args, **kwargs):
                yield from original_reader(handle, *args, **kwargs)
                bed.write_text("changed\n", encoding="utf-8")

            with (
                patch("alma3.infer.csv.reader", mutating_reader),
                self.assertRaisesRegex(InputContractError, "changed while inference was running"),
            ):
                load_bed_methyl_with_manifest(bed, validated["cpg"])

            input_path = root / "input.csv"
            write_array_csv(input_path, sample_count=1)
            output = root / "result.jsonl"
            sidecar = root / "embedding.json"
            original_logits = DiagnosticModel.logits_from_embedding

            def mutate_release(instance, embedding):
                logits = original_logits(instance, embedding)
                (release / "RELEASE_COMPLETE").write_text("changed\n", encoding="utf-8")
                return logits

            with (
                patch("alma3.runtime.load_release", return_value=validated),
                patch.object(DiagnosticModel, "logits_from_embedding", mutate_release),
                self.assertRaisesRegex(ValueError, "release artifact changed during inference"),
            ):
                run_inference(
                    release,
                    input_path,
                    "array-csv",
                    output,
                    device="cpu",
                    embedding_sidecar=sidecar,
                )
            self.assertFalse(output.exists())
            self.assertFalse(sidecar.exists())

    def test_infer_cli_supports_automatic_release_and_batching(self) -> None:
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
                ["input.csv"],
                "array-csv",
                "result.jsonl",
                device="auto",
                embedding_sidecar=None,
                batch_size=2,
            )
        automatic = [item for item in arguments if item not in {"--artifact", "release"}]
        with patch("alma3.infer.run_inference") as run:
            self.assertEqual(infer_main([*automatic, "--batch-size", "8", "--device", "cuda:1"]), 0)
            run.assert_called_once_with(
                None,
                ["input.csv"],
                "array-csv",
                "result.jsonl",
                device="cuda:1",
                embedding_sidecar=None,
                batch_size=8,
            )
        for removed in ("--research", "--download", "--all-probs", "--output-format"):
            with self.subTest(removed=removed), self.assertRaises(SystemExit):
                infer_main([*arguments, removed])


if __name__ == "__main__":
    unittest.main()
