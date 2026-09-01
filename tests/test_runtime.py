from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
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
from alma3.hashes import runtime_contract_sha256
from alma3.infer import (
    _RESULT_CSV_FIELDS,
    InputContractError,
    _array_csv_batches,
    _result_csv_row,
    demo_main,
    demo_path,
    infer_input_format,
    load_bed_methyl_with_manifest,
    run_inference,
    validate_embedding_sidecar,
)
from alma3.infer import main as infer_main
from alma3.model import FoundationModel
from alma3.release import RELEASE_FILES, validate_release
from alma3.runtime import (
    CPU_TORCH_INSTALL_COMMAND,
    MINIMUM_CUDA_AVAILABLE_MEMORY_BYTES,
    MINIMUM_CUDA_TOTAL_MEMORY_BYTES,
    _align_array_batch,
    _prepare_array_values,
    _require_supported_cpu_build,
    _require_supported_cuda_memory,
)
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
    def test_runtime_contract_fingerprint_is_deterministic_and_content_bound(self) -> None:
        package = Path(__file__).resolve().parents[1] / "src" / "alma3"
        expected = runtime_contract_sha256(package)
        self.assertEqual(expected, runtime_contract_sha256(package))
        self.assertEqual(len(expected), 64)

        with tempfile.TemporaryDirectory() as raw:
            copied = Path(raw) / "alma3"
            shutil.copytree(package, copied)
            self.assertEqual(runtime_contract_sha256(copied), expected)

            (copied / "examples" / "ignored.csv").write_text("ignored\n", encoding="utf-8")
            (copied / "examples" / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
            (copied / "__pycache__").mkdir(exist_ok=True)
            (copied / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")
            (copied / "__pycache__" / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
            (copied / "generated-output.txt").write_text("ignored\n", encoding="utf-8")
            self.assertEqual(runtime_contract_sha256(copied), expected)

            source = copied / "clinical_result.py"
            source.write_text(source.read_text(encoding="utf-8") + "# fingerprint change\n", encoding="utf-8")
            self.assertNotEqual(runtime_contract_sha256(copied), expected)

            (copied / "schemas" / "dx_result.schema.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing contract file"):
                runtime_contract_sha256(copied)

    def test_x86_cpu_requires_mkl_before_model_load(self) -> None:
        with (
            patch("alma3.runtime.platform.machine", return_value="x86_64"),
            patch("alma3.runtime.torch.__config__.show", return_value="USE_MKL=OFF"),
            self.assertRaisesRegex(RuntimeError, "requires the official MKL-enabled PyTorch build") as raised,
        ):
            _require_supported_cpu_build(torch.device("cpu"))
        self.assertIn(CPU_TORCH_INSTALL_COMMAND, str(raised.exception))

        with (
            patch("alma3.runtime.platform.machine", return_value="x86_64"),
            patch("alma3.runtime.torch.__config__.show", return_value="USE_MKL=ON"),
        ):
            _require_supported_cpu_build(torch.device("cpu"))

        with (
            patch("alma3.runtime.platform.machine", return_value="x86_64"),
            patch("alma3.runtime.torch.__config__.show", return_value="USE_MKL=OFF"),
        ):
            _require_supported_cpu_build(torch.device("cuda", 0))

    def test_cuda_memory_is_checked_before_model_load(self) -> None:
        gib = 1024**3
        with patch(
            "alma3.runtime.torch.cuda.mem_get_info",
            return_value=(MINIMUM_CUDA_AVAILABLE_MEMORY_BYTES, MINIMUM_CUDA_TOTAL_MEMORY_BYTES),
        ):
            _require_supported_cuda_memory(torch.device("cuda", 0))

        with (
            patch("alma3.runtime.torch.cuda.mem_get_info", return_value=(15 * gib, 15 * gib)),
            self.assertRaisesRegex(RuntimeError, "at least 16 GiB") as too_small,
        ):
            _require_supported_cuda_memory(torch.device("cuda", 0))
        self.assertIn("reports 15.0 GiB", str(too_small.exception))

        with (
            patch("alma3.runtime.torch.cuda.mem_get_info", return_value=(13 * gib, 16 * gib)),
            self.assertRaisesRegex(RuntimeError, "at least 14 GiB of available GPU memory") as too_busy,
        ):
            _require_supported_cuda_memory(torch.device("cuda", 0))
        self.assertIn("has 13.0 GiB available", str(too_busy.exception))

        with patch("alma3.runtime.torch.cuda.mem_get_info") as memory:
            _require_supported_cuda_memory(torch.device("cpu"))
        memory.assert_not_called()

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
            _, beta, observed, coverage = load_bed_methyl_with_manifest(
                bed,
                manifest,
                modification_mode="5mc_plus_5hmc",
            )
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
                    bedmethyl_modification_mode="5mc_plus_5hmc",
                )
            results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(embedded_batch_sizes, [2])
            self.assertEqual([result["sample_id"] for result in results], ["first", "second"])
            self.assertTrue(
                all(
                    result["input"]
                    == {
                        "format": "bedmethyl",
                        "value_mode": "fraction_modified",
                        "modification_mode": "5mc_plus_5hmc",
                        "clipped_value_count": 0,
                    }
                    for result in results
                )
            )

    def test_bedmethyl_modification_mode_is_explicit_and_format_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bed = root / "sample.bed"
            array = root / "samples.csv"
            bed.write_text("chr1\t100\t101\t.\t0\t.\t100\t101\t0\t10\t50\n", encoding="utf-8")
            array.write_text("sample_id,cg0000000\nsample,0.5\n", encoding="utf-8")
            with patch("alma3.infer.ALMA3") as runtime:
                for mode in (None, "unknown"):
                    with self.subTest(mode=mode), self.assertRaisesRegex(
                        InputContractError,
                        "bedmethyl-modification-mode is required",
                    ):
                        run_inference(
                            None,
                            bed,
                            "bedmethyl",
                            root / f"{mode}.jsonl",
                            bedmethyl_modification_mode=mode,
                        )
                with self.assertRaisesRegex(
                    InputContractError,
                    "applies only to bedmethyl",
                ):
                    run_inference(
                        None,
                        array,
                        "array-csv",
                        root / "array.jsonl",
                        bedmethyl_modification_mode="5mc_plus_5hmc",
                    )
            runtime.assert_not_called()

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
            with self.assertRaisesRegex(InputContractError, "must end in .jsonl or .csv"):
                run_inference(None, input_path, "array-csv", root / "result.txt")
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
                )
            load.assert_called_once_with(release, device="cpu")
            self.assertEqual(embedded_batch_sizes, [2, 1])
            self.assertEqual([int(call.args[0].shape[0]) for call in align.call_args_list], [2, 1])
            self.assertEqual([result["sample_id"] for result in results], ["first", "second", "third"])
            self.assertTrue(all(result["input"]["format"] == "array" for result in results))
            self.assertTrue(all(result["input"]["value_mode"] == "beta" for result in results))
            self.assertTrue(all(result["input"]["clipped_value_count"] == 0 for result in results))
            self.assertTrue(all(result["runtime"]["device"] == "cpu" for result in results))
            self.assertTrue(
                all(result["runtime"]["contract_sha256"] == runtime_contract_sha256() for result in results)
            )

    def test_corrected_beta_policy_accepts_only_bounded_beta_like_rows(self) -> None:
        accepted = {
            "lower limits": (torch.tensor([[*([0.5] * 90), *([-0.05] * 9), -0.5]]), 0.0),
            "upper limits": (torch.tensor([[*([0.5] * 90), *([1.05] * 9), 1.5]]), 1.0),
        }
        for name, (values, bound) in accepted.items():
            with self.subTest(name=name):
                prepared, observed, summary = _prepare_array_values(
                    values,
                    [name],
                    input_values="beta",
                )
                self.assertEqual(summary.observed, 100)
                self.assertEqual(summary.clipped, 10)
                self.assertEqual(summary.clipped_by_sample, (10,))
                self.assertTrue(bool(observed.all().item()))
                self.assertEqual(prepared[0, 89].item(), 0.5)
                self.assertEqual(prepared[0, 90:].tolist(), [bound] * 10)

        ordinary = torch.tensor([[0.0, 0.25, 1.0, math.nan]])
        prepared, observed, summary = _prepare_array_values(ordinary, ["ordinary"], input_values="beta")
        self.assertTrue(torch.equal(prepared[observed], ordinary[observed]))
        self.assertTrue(bool(torch.isnan(prepared[~observed]).all().item()))
        self.assertEqual(summary.observed, 3)
        self.assertEqual(summary.clipped, 0)
        self.assertEqual(summary.clipped_by_sample, (0,))

        rejected = {
            "inside": torch.tensor([[*([0.5] * 89), *([-0.01] * 10), -0.1]]),
            "near": torch.tensor([[*([0.5] * 90), *([-0.01] * 8), -0.1, -0.1]]),
            "range": torch.tensor([[*([0.5] * 99), -0.5001]]),
            "upper range": torch.tensor([[*([0.5] * 99), 1.5001]]),
            "percentage": torch.full((1, 100), 50.0),
            "implicit mvalue": torch.tensor([[-10.0, 0.0, 10.0]]),
        }
        for name, values in rejected.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "not beta-like"):
                _prepare_array_values(values, [name], input_values="beta")

    def test_per_sample_clipping_metadata_is_batch_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            cpg_ids = list(validated["cpg"].cpg_ids)
            beta = torch.full((3, len(cpg_ids)), 0.5)
            beta[0, 0] = -0.01
            beta[1, :2] = 1.01

            outputs = []
            with (
                patch("alma3.runtime.load_release", return_value=validated),
                warnings.catch_warnings(record=True) as emitted,
            ):
                warnings.simplefilter("always")
                runtime = ALMA3(release, device="cpu")
                for batch_size in (2, 3):
                    outputs.append(
                        runtime.predict_array(
                            beta,
                            cpg_ids,
                            ["first", "second", "third"],
                            batch_size=batch_size,
                        )
                    )
                mvalue_results = runtime.predict_array(
                    torch.zeros_like(beta),
                    cpg_ids,
                    ["first", "second", "third"],
                    input_values="mvalue",
                )
            self.assertEqual(len(emitted), 2)
            for results in outputs:
                self.assertEqual(
                    [result["input"]["clipped_value_count"] for result in results],
                    [1, 2, 0],
                )
            self.assertEqual(
                [result["input"] for result in outputs[0]],
                [result["input"] for result in outputs[1]],
            )
            self.assertTrue(
                all(
                    result["input"]
                    == {
                        "format": "array",
                        "value_mode": "mvalue",
                        "clipped_value_count": 0,
                    }
                    for result in mvalue_results
                )
            )

    def test_array_csv_clipping_metadata_remains_per_sample(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            input_path = root / "input.csv"
            write_array_csv(input_path, sample_count=2)
            with input_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            rows[1][1] = "-0.01"
            rows[2][1:3] = ["1.01", "1.01"]
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerows(rows)

            output = root / "result.jsonl"
            stderr = StringIO()
            with (
                patch("alma3.runtime.load_release", return_value=validated),
                redirect_stderr(stderr),
            ):
                run_inference(
                    release,
                    input_path,
                    "array-csv",
                    output,
                    device="cpu",
                    batch_size=1,
                )
            results = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [result["input"] for result in results],
                [
                    {
                        "format": "array-csv",
                        "value_mode": "beta",
                        "clipped_value_count": 1,
                    },
                    {
                        "format": "array-csv",
                        "value_mode": "beta",
                        "clipped_value_count": 2,
                    },
                ],
            )
            self.assertIn("clipped 3/3,000 matched values", stderr.getvalue())

    def test_explicit_mvalues_convert_stably_and_missing_stays_unobserved(self) -> None:
        converted, observed, summary = _prepare_array_values(
            torch.tensor([[-1000.0, 0.0, 1000.0, math.nan]]),
            ["mvalues"],
            input_values="mvalue",
        )
        self.assertEqual(converted[0, :3].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(observed.tolist(), [[True, True, True, False]])
        self.assertEqual(summary.observed, 3)
        self.assertEqual(summary.clipped, 0)
        self.assertEqual(summary.clipped_by_sample, (0,))
        with self.assertRaisesRegex(ValueError, "infinity"):
            _prepare_array_values(torch.tensor([[math.inf]]), ["bad"], input_values="mvalue")
        with self.assertRaisesRegex(ValueError, "infinity"):
            _prepare_array_values(torch.tensor([[-math.inf]]), ["bad"], input_values="beta")

    def test_array_csv_rejects_low_cpg_overlap_before_values(self) -> None:
        manifest = CpGManifest(
            cpg_ids=tuple(f"cg{index}" for index in range(1500)),
            chr_id=torch.zeros(1500, dtype=torch.long),
            pos=torch.linspace(0, 1, 1500),
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "expression.csv"
            path.write_text("sample_id,ENSG1,ENSG2\nsample,5,10\n", encoding="utf-8")
            with self.assertRaisesRegex(InputContractError, "Gene-expression matrices are not supported"):
                list(_array_csv_batches(path, manifest, minimum_observed_cpgs=1500))

            invalid = Path(raw) / "invalid.csv"
            invalid.write_text(
                "sample_id," + ",".join(manifest.cpg_ids) + "\nsample,word," + ",".join(["0.5"] * 1499) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(InputContractError, "non-numeric value"):
                list(_array_csv_batches(invalid, manifest, minimum_observed_cpgs=1500))

    def test_sample_id_safety_is_shared_across_input_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)

            array_path = root / "unsafe.csv"
            write_array_csv(array_path, sample_count=1)
            array_path.write_text(
                array_path.read_text(encoding="utf-8").replace("sample-1,", "=formula,"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(InputContractError, "sample ID"):
                list(
                    _array_csv_batches(
                        array_path,
                        validated["cpg"],
                        minimum_observed_cpgs=1500,
                    )
                )

            bed_path = root / "=formula.bed"
            write_bedmethyl(bed_path)
            with self.assertRaisesRegex(InputContractError, "sample ID"):
                load_bed_methyl_with_manifest(
                    bed_path,
                    validated["cpg"],
                    modification_mode="5mc_plus_5hmc",
                )
            with self.assertRaisesRegex(InputContractError, "sample ID"):
                load_bed_methyl_with_manifest(
                    bed_path,
                    validated["cpg"],
                    sample_id="",
                    modification_mode="5mc_plus_5hmc",
                )

            beta = torch.full((1, len(validated["cpg"].cpg_ids)), 0.5)
            with patch("alma3.runtime.load_release", return_value=validated):
                runtime = ALMA3(release, device="cpu")
                for unsafe in ("=formula", " sample", "sample\nname", 7):
                    with self.subTest(unsafe=unsafe), self.assertRaisesRegex(ValueError, "sample ID"):
                        runtime.predict_array(
                            beta,
                            validated["cpg"].cpg_ids,
                            [unsafe],
                        )
                result = runtime.predict_array(
                    beta,
                    validated["cpg"].cpg_ids,
                    ["Tumor α 01"],
                )[0]
            self.assertEqual(result["sample_id"], "Tumor α 01")

    def test_rejected_corrected_input_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            input_path = root / "percent.csv"
            cpg_ids = validated["cpg"].cpg_ids
            input_path.write_text(
                "sample_id," + ",".join(cpg_ids) + "\npercent," + ",".join(["50"] * len(cpg_ids)) + "\n",
                encoding="utf-8",
            )
            output = root / "result.jsonl"
            sidecar = root / "embedding.json"
            with (
                patch("alma3.runtime.load_release", return_value=validated),
                self.assertRaisesRegex(ValueError, "not beta-like"),
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

    def test_corrected_input_notice_and_progress_are_concise(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            input_path = root / "corrected.csv"
            write_array_csv(input_path, sample_count=1)
            input_path.write_text(
                input_path.read_text(encoding="utf-8").replace(",0.5", ",-0.01", 1),
                encoding="utf-8",
            )
            output = root / "result.jsonl"
            stderr = StringIO()
            with (
                patch("alma3.runtime.load_release", return_value=validated),
                redirect_stderr(stderr),
            ):
                run_inference(
                    release,
                    input_path,
                    "array-csv",
                    output,
                    device="cpu",
                    progress=True,
                )
            self.assertEqual(
                stderr.getvalue().splitlines(),
                [
                    "Processed 1 samples.",
                    "Adjusted corrected beta values: clipped 1/1,500 matched values to [0,1].",
                    f"Results saved to {output}",
                ],
            )

    def test_demo_is_the_exact_v2_dataset(self) -> None:
        digest = hashlib.sha256()
        with gzip.open(demo_path(), "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.assertEqual(digest.hexdigest(), "172ddb11f799ccc7952c5f4a86e8babefba99a685e92ec014a46a531b49227a6")
        with gzip.open(demo_path(), "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = list(reader)
        self.assertEqual(len(header) - 1, 331556)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(len(row) == len(header) for row in rows))

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

            wrong_license = root / "wrong-license"
            shutil.copytree(release, wrong_license)
            license_path = wrong_license / "LICENSE"
            license_path.write_text("MIT License\n", encoding="utf-8")
            license_manifest = json.loads(
                (wrong_license / "SHA256SUMS.json").read_text(encoding="utf-8")
            )
            license_manifest["LICENSE"] = sha256(license_path)
            write_json(wrong_license / "SHA256SUMS.json", license_manifest)
            with self.assertRaisesRegex(ValueError, "ALMA3 License 1.0"):
                validate_release(wrong_license)

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
            unsafe_sidecar = json.loads(json.dumps(payload))
            unsafe_sidecar["samples"][0]["sample_id"] = "=formula"
            with self.assertRaisesRegex(InputContractError, "sample ID"):
                validate_embedding_sidecar(unsafe_sidecar)
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

    def test_csv_output_is_a_flat_view_of_the_same_ordered_results(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            input_path = root / "input.csv"
            write_array_csv(input_path, sample_count=3)
            json_output = root / "result.jsonl"
            csv_output = root / "result.csv"

            with patch("alma3.runtime.load_release", return_value=validated):
                run_inference(release, input_path, "array-csv", json_output, device="cpu")
                run_inference(release, input_path, "array-csv", csv_output, device="cpu")

            results = [
                json.loads(line)
                for line in json_output.read_text(encoding="utf-8").splitlines()
            ]
            with csv_output.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames or ()), _RESULT_CSV_FIELDS)
            expected = [
                {
                    field: "" if value == "" else str(value)
                    for field, value in _result_csv_row(result).items()
                }
                for result in results
            ]
            self.assertEqual(rows, expected)
            inspect_paired_results = runpy.run_path(
                str(Path(__file__).resolve().parents[1] / "scripts" / "release-gate"),
                run_name="alma3_release_gate_test",
            )["inspect_paired_results"]
            inspect_paired_results(json_output, csv_output)
            tampered_csv = root / "tampered.csv"
            tampered_rows = [dict(row) for row in rows]
            tampered_rows[0]["observed_cpg_count"] = "999999"
            with tampered_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=_RESULT_CSV_FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(tampered_rows)
            with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                inspect_paired_results(json_output, tampered_csv)
            self.assertEqual([row["sample_id"] for row in rows], ["sample-1", "sample-2", "sample-3"])
            self.assertTrue(all(row["result_summary"] == result["result_summary"] for row, result in zip(rows, results, strict=True)))
            self.assertTrue(all(row["input_format"] == "array-csv" for row in rows))
            self.assertTrue(all(row["input_value_mode"] == "beta" for row in rows))
            self.assertTrue(all(row["input_modification_mode"] == "" for row in rows))
            self.assertTrue(all(row["input_clipped_value_count"] == "0" for row in rows))
            release_hash_files = {
                "model_sha256": "model.safetensors",
                "taxonomy_sha256": "taxonomy.json",
                "cpg_manifest_sha256": "cpg_manifest.json",
                "thresholds_sha256": "thresholds.json",
            }
            target_by_level = {
                "presence": "hematolymphoid_tumor_presence",
                "lineage": "lineage",
                "family": "family",
                "type": "type",
                "subtype": "subtype",
            }
            for result, row in zip(results, rows, strict=True):
                self.assertEqual(row["release_manifest_sha256"], validated["manifest_sha256"])
                for field, filename in release_hash_files.items():
                    self.assertEqual(result["release"][field], validated["hashes"][filename])
                for node in result["path"]:
                    target = target_by_level[node["level"]]
                    self.assertEqual(
                        validated["taxonomy"].classes[target][node["index"]],
                        node["classification"],
                    )
                if result["decision"] is not None:
                    target = target_by_level[result["decision"]["level"]]
                    for entry in result["decision"]["differential"]:
                        self.assertEqual(
                            validated["taxonomy"].classes[target][entry["index"]],
                            entry["classification"],
                        )

    def test_csv_serialization_failure_publishes_neither_result_nor_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, model = create_release(root / "release")
            validated = validated_release_fixture(release, model)
            input_path = root / "input.csv"
            write_array_csv(input_path, sample_count=1)
            output = root / "result.csv"
            sidecar = root / "embedding.json"
            with (
                patch("alma3.runtime.load_release", return_value=validated),
                patch("alma3.infer._result_csv_row", side_effect=OSError("CSV serialization failed")),
                self.assertRaisesRegex(OSError, "CSV serialization failed"),
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
                load_bed_methyl_with_manifest(
                    bed,
                    validated["cpg"],
                    modification_mode="5mc_plus_5hmc",
                )

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
                input_values="beta",
                bedmethyl_modification_mode=None,
                progress=False,
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
                input_values="beta",
                bedmethyl_modification_mode=None,
                progress=False,
            )
        inferred = ["-i", "input.csv.gz", "-o", "result.jsonl", "--input-values", "mvalue"]
        with patch("alma3.infer.run_inference") as run:
            self.assertEqual(infer_main(inferred), 0)
            run.assert_called_once_with(
                None,
                ["input.csv.gz"],
                "array-csv",
                "result.jsonl",
                device="auto",
                embedding_sidecar=None,
                batch_size=2,
                input_values="mvalue",
                bedmethyl_modification_mode=None,
                progress=False,
            )
        bedmethyl = [
            "-i",
            "sample.bed",
            "-o",
            "result.jsonl",
            "--bedmethyl-modification-mode",
            "5mc_plus_5hmc",
        ]
        with patch("alma3.infer.run_inference") as run:
            self.assertEqual(infer_main(bedmethyl), 0)
            run.assert_called_once_with(
                None,
                ["sample.bed"],
                "bedmethyl",
                "result.jsonl",
                device="auto",
                embedding_sidecar=None,
                batch_size=2,
                input_values="beta",
                bedmethyl_modification_mode="5mc_plus_5hmc",
                progress=False,
            )
        self.assertEqual(infer_input_format(["first.bed", "second.bed.gz"]), "bedmethyl")
        self.assertEqual(infer_input_format("cohort.csv.gz"), "array-csv")
        with self.assertRaisesRegex(InputContractError, "mixed formats"):
            infer_input_format(["sample.bed", "cohort.csv"])
        with self.assertRaisesRegex(InputContractError, "cannot infer"):
            infer_input_format("input.tsv")
        terminal = StringIO()
        terminal.isatty = lambda: True
        with patch("alma3.infer.run_inference") as run, redirect_stderr(terminal):
            self.assertEqual(infer_main(["-i", "input.csv", "-o", "result.jsonl"]), 0)
            self.assertTrue(run.call_args.kwargs["progress"])
        self.assertEqual(terminal.getvalue(), "Loading ALMA3...\n")
        for removed in ("--research", "--download", "--all-probs", "--output-format"):
            with self.subTest(removed=removed), self.assertRaises(SystemExit):
                infer_main([*arguments, removed])

    def test_demo_cli_uses_packaged_input_and_new_only_default_output(self) -> None:
        with patch("alma3.infer.run_inference") as run:
            self.assertEqual(demo_main([]), 0)
            run.assert_called_once_with(
                None,
                demo_path(),
                "array-csv",
                "alma3-demo.jsonl",
                device="auto",
                batch_size=2,
                input_values="beta",
                progress=False,
            )


if __name__ == "__main__":
    unittest.main()
