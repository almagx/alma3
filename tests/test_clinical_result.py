from __future__ import annotations

import copy
import unittest

import torch

from alma3.clinical_result import RESULT_LEVELS, results_from_logits, validate_result
from alma3.dx import DX_TARGETS, DxContractError, Taxonomy
from alma3.infer import _result_csv_row
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
        _thresholds(),
        taxonomy,
        _release(),
        observed_cpg_counts=[2000],
        minimum_observed_cpgs=1500,
    )[0]


class ClinicalResultContractTests(unittest.TestCase):
    def test_every_unresolved_level_reports_two_ranked_valid_candidates(self) -> None:
        taxonomy = _taxonomy()
        for level_index, level in enumerate(RESULT_LEVELS):
            with self.subTest(level=level):
                result = _result(unresolved_level=level, taxonomy=taxonomy)
                self.assertEqual(result["status"], "no_call" if level == "presence" else "unresolved")
                self.assertEqual(result["decision"]["level"], level)
                self.assertEqual(len(result["path"]), level_index)
                candidates = result["decision"]["candidates"]
                self.assertEqual(len(candidates), 2)
                self.assertGreater(candidates[0]["score"], candidates[1]["score"])
                self.assertLess(candidates[0]["score"], result["decision"]["threshold"])
                if level == "presence":
                    self.assertEqual([candidate["label"] for candidate in candidates], ["absent", "present"])
                elif level == "family":
                    self.assertEqual([candidate["label"] for candidate in candidates], ["m1", "m2"])
                elif level == "type":
                    self.assertEqual([candidate["label"] for candidate in candidates], ["m1t1", "m1t2"])
                elif level == "subtype":
                    self.assertEqual([candidate["label"] for candidate in candidates], ["m1t1s1", "m1t1s2"])

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
            observed_cpg_counts=[1500],
            minimum_observed_cpgs=1500,
        )[0]
        self.assertEqual(
            [candidate["index"] for candidate in result["decision"]["candidates"]],
            [0, 1],
        )

    def test_csv_is_clinician_centered_for_each_status(self) -> None:
        classified = _result()
        tumor_absent = _result(tumor_absent=True)
        unresolved = _result(unresolved_level="subtype")
        no_call = _result(unresolved_level="presence")

        classified_row = _result_csv_row(classified)
        self.assertEqual(classified_row["result_summary"], "Classified at subtype: m1t1s1.")
        self.assertEqual(classified_row["resolved_basis"], "scored")
        self.assertEqual(classified_row["candidate_1_label"], "")

        absent_row = _result_csv_row(tumor_absent)
        self.assertEqual(absent_row["result_summary"], "No hematolymphoid tumor signal detected.")
        self.assertEqual(absent_row["resolved_label"], "absent")

        unresolved_row = _result_csv_row(unresolved)
        self.assertEqual(
            unresolved_row["result_summary"],
            "Resolved through type: m1t1; subtype unresolved.",
        )
        self.assertEqual(unresolved_row["candidate_1_label"], "m1t1s1")
        self.assertEqual(unresolved_row["candidate_2_label"], "m1t1s2")
        self.assertEqual(unresolved_row["observed_cpg_count"], 2000)
        self.assertEqual(unresolved_row["minimum_observed_cpgs"], 1500)
        self.assertEqual(unresolved_row["model_version"], "3.0.0")

        no_call_row = _result_csv_row(no_call)
        self.assertEqual(no_call_row["result_summary"], "No call: hematolymphoid tumor presence unresolved.")
        self.assertEqual(no_call_row["resolved_label"], "")
        self.assertEqual(no_call_row["candidate_1_label"], "absent")
        self.assertEqual(no_call_row["candidate_2_label"], "present")

    def test_implied_terminal_label_is_explicit_in_csv(self) -> None:
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
            observed_cpg_counts=[1500],
            minimum_observed_cpgs=1500,
        )[0]
        self.assertEqual(result["status"], "classified")
        self.assertEqual(_result_csv_row(result)["resolved_basis"], "implied_by_hierarchy")

    def test_old_result_shape_and_invalid_candidate_order_fail_closed(self) -> None:
        result = _result(unresolved_level="subtype")
        old = copy.deepcopy(result)
        old.pop("observed_cpg_count")
        with self.assertRaisesRegex(DxContractError, "fields are invalid"):
            validate_result(old)

        reversed_candidates = copy.deepcopy(result)
        reversed_candidates["decision"]["candidates"].reverse()
        with self.assertRaisesRegex(DxContractError, "ranked deterministically"):
            validate_result(reversed_candidates)


if __name__ == "__main__":
    unittest.main()
