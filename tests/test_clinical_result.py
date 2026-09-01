from __future__ import annotations

import copy
import json
import math
import re
import unittest

import torch

from alma3.clinical_result import (
    RESULT_LEVELS,
    RESULT_SCHEMA_VERSION,
    result_schema_path,
    results_from_logits,
    validate_result,
)
from alma3.dx import DX_TARGETS, DxContractError, Taxonomy
from alma3.infer import _RESULT_CSV_FIELDS, _result_csv_row
from tests.helpers import taxonomy_payload


def _taxonomy() -> Taxonomy:
    return Taxonomy.from_dict(
        {
            "kind": "alma3_taxonomy",
            "levels": {
                "hematolymphoid_tumor_presence": ["absent", "present"],
                "lineage": ["myeloid", "lymphoid"],
                "family": ["m1", "m2", "l1", "l2"],
                "type": ["m1t1", "m1t2", "m2t1", "m2t2", "l1t1", "l1t2", "l2t1", "l2t2"],
                "subtype": [
                    "m1t1s1",
                    "m1t1s2",
                    "m1t2s1",
                    "m1t2s2",
                    "m2t1s1",
                    "m2t1s2",
                    "m2t2s1",
                    "m2t2s2",
                    "l1t1s1",
                    "l1t1s2",
                    "l1t2s1",
                    "l1t2s2",
                    "l2t1s1",
                    "l2t1s2",
                    "l2t2s1",
                    "l2t2s2",
                ],
            },
            "family_by_lineage": {
                "myeloid": ["m1", "m2"],
                "lymphoid": ["l1", "l2"],
            },
            "type_by_family": {
                "m1": ["m1t1", "m1t2"],
                "m2": ["m2t1", "m2t2"],
                "l1": ["l1t1", "l1t2"],
                "l2": ["l2t1", "l2t2"],
            },
            "subtype_by_type": {
                "m1t1": ["m1t1s1", "m1t1s2"],
                "m1t2": ["m1t2s1", "m1t2s2"],
                "m2t1": ["m2t1s1", "m2t1s2"],
                "m2t2": ["m2t2s1", "m2t2s2"],
                "l1t1": ["l1t1s1", "l1t1s2"],
                "l1t2": ["l1t2s1", "l1t2s2"],
                "l2t1": ["l2t1s1", "l2t1s2"],
                "l2t2": ["l2t2s1", "l2t2s2"],
            },
        }
    )


def _release() -> dict[str, str]:
    return {
        "version": "3.0.0",
        "manifest_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "taxonomy_sha256": "3" * 64,
        "cpg_manifest_sha256": "4" * 64,
        "thresholds_sha256": "5" * 64,
    }


def _runtime() -> dict[str, str]:
    return {
        "package_version": "3.0.0",
        "contract_sha256": "6" * 64,
        "device": "cpu",
    }


def _input(*, clipped: int = 0, input_format: str = "array") -> dict[str, object]:
    value = {
        "format": input_format,
        "value_mode": "fraction_modified" if input_format == "bedmethyl" else "beta",
        "clipped_value_count": clipped,
    }
    if input_format == "bedmethyl":
        value["modification_mode"] = "5mc_plus_5hmc"
    return value


def _thresholds(value: float = 0.8) -> dict[str, object]:
    return {
        "temperatures": {target: 1.0 for target in DX_TARGETS},
        "thresholds": {target: value for target in DX_TARGETS},
    }


def _logits(taxonomy: Taxonomy) -> dict[str, torch.Tensor]:
    return {
        target: torch.full((1, len(taxonomy.classes[target])), -8.0)
        for target in DX_TARGETS
    }


def _select(logits: dict[str, torch.Tensor], target: str, index: int) -> None:
    logits[target][0, index] = 8.0


def _result(
    *,
    unresolved_level: str | None = None,
    tumor_absent: bool = False,
    taxonomy: Taxonomy | None = None,
    threshold: float = 0.8,
) -> dict[str, object]:
    taxonomy = taxonomy or _taxonomy()
    logits = _logits(taxonomy)
    if tumor_absent:
        _select(logits, "hematolymphoid_tumor_presence", 0)
    else:
        _select(logits, "hematolymphoid_tumor_presence", 1)
        _select(logits, "lineage", 0)
        _select(logits, "family", 0)
        _select(logits, "type", 0)
        _select(logits, "subtype", 0)

    if unresolved_level is not None:
        target = {
            "presence": "hematolymphoid_tumor_presence",
            "lineage": "lineage",
            "family": "family",
            "type": "type",
            "subtype": "subtype",
        }[unresolved_level]
        logits[target].fill_(-20.0)
        valid = {
            "presence": [0, 1],
            "lineage": [0, 1],
            "family": [0, 1],
            "type": [0, 1],
            "subtype": [0, 1],
        }[unresolved_level]
        logits[target][0, valid[0]] = 0.4
        logits[target][0, valid[1]] = 0.0
        if unresolved_level in {"family", "type", "subtype"}:
            logits[target][0, -1] = 20.0

    return results_from_logits(
        ["sample-1"],
        logits,
        _thresholds(threshold),
        taxonomy,
        _release(),
        runtime=_runtime(),
        input_metadata=[_input()],
        observed_cpg_counts=[2000],
        minimum_observed_cpgs=1500,
    )[0]


class ClinicalResultContractTests(unittest.TestCase):
    def test_json_schema_matches_the_v2_runtime_contract(self) -> None:
        schema = json.loads(result_schema_path().read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], RESULT_SCHEMA_VERSION)
        self.assertEqual(
            set(schema["required"]),
            {
                "kind",
                "schema_version",
                "sample_id",
                "result_summary",
                "release",
                "runtime",
                "input",
                "observed_cpg_count",
                "minimum_observed_cpgs",
                "status",
                "accepted",
                "path",
                "decision",
            },
        )
        self.assertEqual(
            set(schema["$defs"]["runtime"]["required"]),
            {"package_version", "contract_sha256", "device"},
        )
        self.assertEqual(
            set(schema["$defs"]["input"]["required"]),
            {"format", "value_mode", "clipped_value_count"},
        )
        self.assertEqual(
            schema["$defs"]["input"]["properties"]["modification_mode"]["enum"],
            ["5mc", "5mc_plus_5hmc"],
        )

    def test_every_unresolved_level_reports_two_ranked_valid_differential_entries(self) -> None:
        taxonomy = _taxonomy()
        expected_summaries = {
            "presence": (
                "Tumor presence unresolved. Leading candidate: absent "
                "(59.9% confidence; threshold 80.0%)."
            ),
            "lineage": (
                "Tumor detected (100.0% confidence). Lineage unresolved. "
                "Leading candidate: myeloid (59.9% confidence; threshold 80.0%)."
            ),
            "family": (
                "Lineage: myeloid (100.0% confidence). Family unresolved. "
                "Leading candidate: m1 (59.9% confidence; threshold 80.0%)."
            ),
            "type": (
                "Family: m1 (100.0% confidence). Type unresolved. "
                "Leading candidate: m1t1 (59.9% confidence; threshold 80.0%)."
            ),
            "subtype": (
                "Type: m1t1 (100.0% confidence). Subtype unresolved. "
                "Leading candidate: m1t1s1 (59.9% confidence; threshold 80.0%)."
            ),
        }
        for level_index, level in enumerate(RESULT_LEVELS):
            with self.subTest(level=level):
                result = _result(unresolved_level=level, taxonomy=taxonomy)
                self.assertEqual(
                    result["status"],
                    "no_call" if level == "presence" else "partially_resolved",
                )
                self.assertEqual(result["decision"]["level"], level)
                self.assertEqual(len(result["path"]), level_index)
                differential = result["decision"]["differential"]
                self.assertEqual(len(differential), 2)
                self.assertGreater(
                    differential[0]["model_score"], differential[1]["model_score"]
                )
                self.assertLess(
                    differential[0]["model_score"],
                    result["decision"]["reporting_cutoff"],
                )
                self.assertEqual(result["result_summary"], expected_summaries[level])
                self.assertNotIn("—review", result["result_summary"])
                self.assertNotIn("\n", result["result_summary"])
                self.assertTrue(result["result_summary"].endswith("."))
                leading_matches = re.findall(
                    rf"(?<!\w){re.escape(differential[0]['classification'])}(?!\w)",
                    result["result_summary"],
                )
                self.assertEqual(len(leading_matches), 1)
                self.assertIsNone(
                    re.search(
                        rf"(?<!\w){re.escape(differential[1]['classification'])}(?!\w)",
                        result["result_summary"],
                    )
                )
                self.assertEqual(
                    result["result_summary"].count("% confidence"),
                    1 if level == "presence" else 2,
                )
                if level == "presence":
                    self.assertEqual(
                        [entry["classification"] for entry in differential],
                        ["absent", "present"],
                    )
                elif level == "family":
                    self.assertEqual(
                        [entry["classification"] for entry in differential],
                        ["m1", "m2"],
                    )
                elif level == "type":
                    self.assertEqual(
                        [entry["classification"] for entry in differential],
                        ["m1t1", "m1t2"],
                    )
                elif level == "subtype":
                    self.assertEqual(
                        [entry["classification"] for entry in differential],
                        ["m1t1s1", "m1t1s2"],
                    )

    def test_ties_are_broken_by_taxonomy_index(self) -> None:
        taxonomy = _taxonomy()
        logits = _logits(taxonomy)
        logits["hematolymphoid_tumor_presence"].zero_()
        result = results_from_logits(
            ["sample-1"],
            logits,
            _thresholds(),
            taxonomy,
            _release(),
            runtime=_runtime(),
            input_metadata=[_input()],
            observed_cpg_counts=[1500],
            minimum_observed_cpgs=1500,
        )[0]
        self.assertEqual(
            [entry["index"] for entry in result["decision"]["differential"]],
            [0, 1],
        )
        self.assertEqual(
            result["result_summary"],
            "Tumor presence unresolved. Leading candidate: absent "
            "(50.0% confidence; threshold 80.0%).",
        )

    def test_unresolved_summary_rounds_candidate_confidence_and_threshold_independently(self) -> None:
        result = _result(unresolved_level="subtype", threshold=0.81234)
        self.assertEqual(
            result["result_summary"],
            "Type: m1t1 (100.0% confidence). Subtype unresolved. "
            "Leading candidate: m1t1s1 (59.9% confidence; threshold 81.2%).",
        )
        self.assertAlmostEqual(
            result["decision"]["differential"][0]["model_score"],
            0.5986876487731934,
        )
        self.assertEqual(result["decision"]["reporting_cutoff"], 0.81234)

    def test_summary_confidence_uses_only_the_deepest_scored_class(self) -> None:
        taxonomy = _taxonomy()
        logits = _logits(taxonomy)
        requested_scores = {
            "hematolymphoid_tumor_presence": (1, 0.99),
            "lineage": (0, 0.98),
            "family": (0, 0.97),
            "type": (0, 0.96),
            "subtype": (0, 0.91234),
        }
        for target, (selected, score) in requested_scores.items():
            logits[target].fill_(-20.0)
            logits[target][0, selected] = math.log(score / (1.0 - score))
            logits[target][0, 1 - selected] = 0.0
        result = results_from_logits(
            ["sample-1"],
            logits,
            _thresholds(),
            taxonomy,
            _release(),
            runtime=_runtime(),
            input_metadata=[_input()],
            observed_cpg_counts=[1500],
            minimum_observed_cpgs=1500,
        )[0]
        self.assertEqual(result["status"], "fully_resolved")
        self.assertEqual(
            result["result_summary"],
            "Subtype: m1t1s1 (91.2% confidence).",
        )
        self.assertAlmostEqual(result["path"][-1]["model_score"], 0.91234, places=5)
        self.assertNotIn("99.0%", result["result_summary"])

    def test_csv_is_clinician_centered_for_each_status(self) -> None:
        self.assertEqual(
            _RESULT_CSV_FIELDS,
            (
                "sample_id",
                "result_summary",
                "result_status",
                "resolved_level",
                "resolved_classification",
                "resolved_basis",
                "unresolved_level",
                "unresolved_reporting_cutoff",
                "differential_1_classification",
                "differential_1_model_score",
                "differential_2_classification",
                "differential_2_model_score",
                "observed_cpg_count",
                "minimum_observed_cpgs",
                "input_format",
                "input_value_mode",
                "input_modification_mode",
                "input_clipped_value_count",
                "tumor_presence",
                "tumor_presence_status",
                "tumor_presence_model_score",
                "tumor_presence_reporting_cutoff",
                "lineage",
                "lineage_status",
                "lineage_model_score",
                "lineage_reporting_cutoff",
                "family",
                "family_status",
                "family_model_score",
                "family_reporting_cutoff",
                "type",
                "type_status",
                "type_model_score",
                "type_reporting_cutoff",
                "subtype",
                "subtype_status",
                "subtype_model_score",
                "subtype_reporting_cutoff",
                "result_kind",
                "result_schema_version",
                "release_version",
                "release_manifest_sha256",
                "runtime_package_version",
                "runtime_contract_sha256",
                "inference_device",
            ),
        )
        self.assertEqual(len(_RESULT_CSV_FIELDS), 45)
        fully_resolved = _result()
        tumor_absent = _result(tumor_absent=True)
        partially_resolved = _result(unresolved_level="subtype")
        no_call = _result(unresolved_level="presence")

        fully_resolved_row = _result_csv_row(fully_resolved)
        self.assertEqual(
            fully_resolved_row["result_summary"],
            "Subtype: m1t1s1 (100.0% confidence).",
        )
        self.assertEqual(fully_resolved["result_summary"], fully_resolved_row["result_summary"])
        self.assertEqual(fully_resolved_row["resolved_basis"], "scored")
        self.assertEqual(fully_resolved_row["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(fully_resolved_row["input_modification_mode"], "")
        self.assertEqual(fully_resolved_row["subtype_status"], "resolved")
        self.assertNotEqual(fully_resolved_row["subtype_model_score"], "")
        self.assertEqual(fully_resolved_row["differential_1_classification"], "")

        bedmethyl = copy.deepcopy(fully_resolved)
        bedmethyl["input"] = _input(input_format="bedmethyl")
        self.assertEqual(
            _result_csv_row(bedmethyl)["input_modification_mode"],
            "5mc_plus_5hmc",
        )

        absent_row = _result_csv_row(tumor_absent)
        self.assertEqual(
            absent_row["result_summary"],
            "Tumor not detected (100.0% confidence).",
        )
        self.assertEqual(absent_row["resolved_classification"], "absent")
        self.assertEqual(absent_row["lineage"], "")

        unresolved_row = _result_csv_row(partially_resolved)
        self.assertEqual(
            unresolved_row["result_summary"],
            "Type: m1t1 (100.0% confidence). Subtype unresolved. "
            "Leading candidate: m1t1s1 (59.9% confidence; threshold 80.0%).",
        )
        self.assertEqual(unresolved_row["differential_1_classification"], "m1t1s1")
        self.assertEqual(unresolved_row["differential_2_classification"], "m1t1s2")
        self.assertEqual(
            unresolved_row["differential_1_model_score"],
            partially_resolved["decision"]["differential"][0]["model_score"],
        )
        self.assertEqual(
            unresolved_row["unresolved_reporting_cutoff"],
            partially_resolved["decision"]["reporting_cutoff"],
        )
        self.assertEqual(unresolved_row["observed_cpg_count"], 2000)
        self.assertEqual(unresolved_row["minimum_observed_cpgs"], 1500)
        self.assertEqual(unresolved_row["release_version"], "3.0.0")
        self.assertEqual(unresolved_row["runtime_package_version"], "3.0.0")
        self.assertNotIn("model_version", unresolved_row)
        self.assertNotIn("reporting_cutoff", unresolved_row)
        self.assertFalse(any("label" in field for field in unresolved_row))

        no_call_row = _result_csv_row(no_call)
        self.assertEqual(
            no_call_row["result_summary"],
            "Tumor presence unresolved. Leading candidate: absent "
            "(59.9% confidence; threshold 80.0%).",
        )
        self.assertEqual(no_call_row["resolved_classification"], "")
        self.assertEqual(no_call_row["tumor_presence"], "")
        self.assertEqual(no_call_row["differential_1_classification"], "absent")
        self.assertEqual(no_call_row["differential_2_classification"], "present")

    def test_implied_terminal_classification_is_explicit_in_csv(self) -> None:
        taxonomy = Taxonomy.from_dict(taxonomy_payload())
        logits = _logits(taxonomy)
        _select(logits, "hematolymphoid_tumor_presence", 1)
        _select(logits, "lineage", 0)
        result = results_from_logits(
            ["sample-1"],
            logits,
            _thresholds(),
            taxonomy,
            _release(),
            runtime=_runtime(),
            input_metadata=[_input()],
            observed_cpg_counts=[1500],
            minimum_observed_cpgs=1500,
        )[0]
        self.assertEqual(result["status"], "fully_resolved")
        row = _result_csv_row(result)
        self.assertEqual(row["resolved_basis"], "implied_by_hierarchy")
        self.assertEqual(
            result["result_summary"],
            "Subtype: myeloid_subtype (100.0% confidence).",
        )
        self.assertEqual(row["subtype_status"], "implied")
        self.assertEqual(row["subtype_model_score"], "")
        self.assertEqual(row["subtype_reporting_cutoff"], "")

    def test_implied_partial_summary_reports_deepest_scored_ancestor_confidence(self) -> None:
        payload = taxonomy_payload()
        payload["levels"]["subtype"] = [
            "myeloid_subtype_1",
            "myeloid_subtype_2",
            "lymphoid_subtype",
        ]
        payload["subtype_by_type"]["myeloid_type"] = [
            "myeloid_subtype_1",
            "myeloid_subtype_2",
        ]
        taxonomy = Taxonomy.from_dict(payload)
        logits = _logits(taxonomy)
        _select(logits, "hematolymphoid_tumor_presence", 1)
        lineage_score = 0.87654
        logits["lineage"].zero_()
        logits["lineage"][0, 0] = math.log(lineage_score / (1.0 - lineage_score))
        result = results_from_logits(
            ["sample-1"],
            logits,
            _thresholds(),
            taxonomy,
            _release(),
            runtime=_runtime(),
            input_metadata=[_input()],
            observed_cpg_counts=[1500],
            minimum_observed_cpgs=1500,
        )[0]

        self.assertEqual(result["status"], "partially_resolved")
        self.assertEqual(result["path"][-1]["status"], "implied")
        self.assertEqual(
            result["result_summary"],
            "Type: myeloid_type (87.7% confidence). Subtype unresolved. "
            "Leading candidate: myeloid_subtype_1 (50.0% confidence; threshold 80.0%).",
        )
        row = _result_csv_row(result)
        self.assertEqual(row["resolved_basis"], "implied_by_hierarchy")
        self.assertEqual(row["type_model_score"], "")
        self.assertEqual(row["type_reporting_cutoff"], "")

    def test_old_result_shape_and_invalid_differential_order_fail_closed(self) -> None:
        result = _result(unresolved_level="subtype")
        old = copy.deepcopy(result)
        old.pop("observed_cpg_count")
        with self.assertRaisesRegex(DxContractError, "fields are invalid"):
            validate_result(old)

        reversed_differential = copy.deepcopy(result)
        reversed_differential["decision"]["differential"].reverse()
        with self.assertRaisesRegex(DxContractError, "ranked deterministically"):
            validate_result(reversed_differential)

        canonical_summary = result["result_summary"]
        tampered_summaries = {
            "wrong candidate": canonical_summary.replace("m1t1s1", "m1t1s2"),
            "wrong candidate score": canonical_summary.replace("59.9% confidence", "59.8% confidence"),
            "wrong threshold": canonical_summary.replace("threshold 80.0%", "threshold 79.9%"),
            "legacy prose": "Type: m1t1 (100.0% confidence). Subtype unresolved. See differential.",
            "altered ordering": (
                "Leading candidate: m1t1s1 (59.9% confidence; threshold 80.0%). "
                "Type: m1t1 (100.0% confidence). Subtype unresolved."
            ),
        }
        for description, summary in tampered_summaries.items():
            with self.subTest(description=description):
                tampered_summary = copy.deepcopy(result)
                tampered_summary["result_summary"] = summary
                with self.assertRaisesRegex(DxContractError, "summary is invalid"):
                    validate_result(tampered_summary)

        schema_v1 = copy.deepcopy(result)
        schema_v1["schema_version"] = 1
        with self.assertRaisesRegex(DxContractError, "schema version is invalid"):
            validate_result(schema_v1)

    def test_runtime_input_and_sample_id_contracts_fail_closed(self) -> None:
        result = _result()
        invalid_mutations = {
            "runtime hash": lambda value: value["runtime"].__setitem__("contract_sha256", "bad"),
            "runtime device": lambda value: value["runtime"].__setitem__("device", "cuda"),
            "input format": lambda value: value["input"].__setitem__("format", "csv"),
            "mvalue clipping": lambda value: value.__setitem__(
                "input",
                {"format": "array", "value_mode": "mvalue", "clipped_value_count": 1},
            ),
            "excess clipping": lambda value: value["input"].__setitem__(
                "clipped_value_count", value["observed_cpg_count"] + 1
            ),
        }
        for name, mutate in invalid_mutations.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(result)
                mutate(candidate)
                with self.assertRaises(DxContractError):
                    validate_result(candidate)

        bedmethyl = _result()
        bedmethyl["input"] = _input(input_format="bedmethyl")
        validate_result(bedmethyl)
        for name, mutate in {
            "missing mode": lambda value: value["input"].pop("modification_mode"),
            "invalid mode": lambda value: value["input"].__setitem__(
                "modification_mode", "unknown"
            ),
        }.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(bedmethyl)
                mutate(candidate)
                with self.assertRaises(DxContractError):
                    validate_result(candidate)

        array_with_mode = _result()
        array_with_mode["input"]["modification_mode"] = "5mc_plus_5hmc"
        with self.assertRaises(DxContractError):
            validate_result(array_with_mode)

        for sample_id in ("", " sample", "sample ", "sample\nname", "=formula", "+sum", "-1", "@cmd"):
            with self.subTest(sample_id=sample_id):
                candidate = copy.deepcopy(result)
                candidate["sample_id"] = sample_id
                with self.assertRaisesRegex(DxContractError, "sample ID"):
                    validate_result(candidate)

        safe = copy.deepcopy(result)
        safe["sample_id"] = "Tumor α 01"
        validate_result(safe)

    def test_retired_status_names_fail_closed(self) -> None:
        for retired_status in ("classified", "tumor_not_detected"):
            result = _result()
            result["status"] = retired_status
            with self.subTest(status=retired_status), self.assertRaisesRegex(
                DxContractError, "status is invalid"
            ):
                validate_result(result)


if __name__ == "__main__":
    unittest.main()
