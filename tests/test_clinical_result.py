"""Release-critical decoder, reporting and input-boundary tests."""

import copy
import itertools
import json
import unittest

import jsonschema
import torch

from alma3.clinical_result import result_schema_path, results_from_probabilities, scaled_subtype_score, validate_result
from alma3.dx import DX_TARGETS, Taxonomy
from alma3.final_ensemble import FinalEnsemble
from alma3.infer import _result_csv_row
from alma3.terminal import SUBTYPE_CUTOFF, selected_path_scores


def taxonomy():
    # 107 subtype leaves plus one broad terminal, with both singleton and competing branches.
    return Taxonomy.from_dict(
        {
            "kind": "alma3_taxonomy",
            "levels": dict(
                zip(
                    DX_TARGETS,
                    [
                        ["absent", "present"],
                        ["a", "b"],
                        ["fa", "fb"],
                        ["ta", "tb", "broad"],
                        [f"s{i}" for i in range(107)],
                    ],
                )
            ),
            "family_by_lineage": {"a": ["fa"], "b": ["fb"]},
            "type_by_family": {"fa": ["ta"], "fb": ["tb", "broad"]},
            "subtype_by_type": {"ta": ["s0"], "tb": [f"s{i}" for i in range(1, 107)], "broad": []},
        }
    )


class ReportingTests(unittest.TestCase):
    def setUp(self):
        raw = taxonomy()
        self.tax = raw
        self.release = dict(
            version="3.1.0",
            **{
                k: "a" * 64
                for k in (
                    "manifest_sha256",
                    "ensemble_sha256",
                    "taxonomy_sha256",
                    "cpg_manifest_sha256",
                    "thresholds_sha256",
                )
            },
        )
        self.runtime = {"package_version": "3.1.0", "contract_sha256": "b" * 64, "device": "cpu"}
        self.cuts = {"thresholds": dict(zip(DX_TARGETS, [0.0, 0.0, 0.0, 0.0, SUBTYPE_CUTOFF]))}

    def result(self, p, cuts=None):
        return results_from_probabilities(
            ["specimen"],
            p[None],
            cuts or self.cuts,
            self.tax,
            self.release,
            runtime=self.runtime,
            input_metadata=[{"format": "array", "value_mode": "beta", "clipped_value_count": 0}],
            observed_cpg_counts=[5000],
            minimum_observed_cpgs=5000,
        )[0]

    def test_type_first_differs_from_terminal_map_and_keeps_path(self):
        p = torch.zeros(109, dtype=torch.float64)
        p[0] = 0.2
        p[1] = 0.3
        p[2:4] = 0.25
        selected, _path, _, _ = selected_path_scores(p[None], self.tax)
        self.assertEqual(int(selected), 2)
        r = self.result(p)
        self.assertEqual(r["prediction"]["terminal_index"], 2)
        self.assertEqual(r["accepted"]["classification"], "tb")
        self.assertEqual(r["path"][-1]["level"], "type")
        self.assertNotIn("level", r["decision"])

    def test_subtype_singleton_checks_joint_probability_at_boundary(self):
        for delta in (-1e-12, 0, 1e-12):
            p = torch.zeros(109, dtype=torch.float64)
            p[1] = SUBTYPE_CUTOFF + delta
            p[0] = 0.22
            p[108] = 0.23
            p[2] = 1 - p[0] - p[108] - p[1]
            r = self.result(p)
            self.assertEqual(r["prediction"]["terminal_index"], 1)
            self.assertEqual(r["prediction"]["subtype_cutoff_passed"], delta >= 0)
            self.assertEqual(r["status"], "fully_resolved" if delta >= 0 else "partially_resolved")
            before = copy.deepcopy(r)
            row = _result_csv_row(r)
            self.assertEqual(r, before)
            self.assertEqual(row["subtype_scaled_cutoff"], 85)
            self.assertEqual(f"{row['subtype_scaled_score']:.1f}", "85.0")
            self.assertEqual(row["result_status"], r["status"])
            self.assertEqual(row["subtype_scaled_score"] >= 85, delta >= 0)
            if delta < 0:
                self.assertEqual(len(r["decision"]["differential"]), 1)

    def test_display_scale_is_monotonic_and_preserves_probability_endpoints(self):
        self.assertEqual([scaled_subtype_score(p) for p in (0, SUBTYPE_CUTOFF, 1)], [0, 85, 100])
        self.assertAlmostEqual(scaled_subtype_score(.1), 25.788837)
        self.assertAlmostEqual(scaled_subtype_score(.489), 88.56652788466175, places=12)
        scores = [scaled_subtype_score(i / 1000) for i in range(1001)]
        self.assertTrue(all(a < b for a, b in itertools.pairwise(scores)))
        for invalid in (True, None, "0.5", -.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                scaled_subtype_score(invalid)

    def test_added_upstream_gates_are_rejected(self):
        p = torch.zeros(109, dtype=torch.float64)
        p[0] = 0.1
        p[1] = 0.8
        p[2] = 0.1
        cuts = copy.deepcopy(self.cuts)
        cuts["thresholds"][DX_TARGETS[0]] = 0.95
        with self.assertRaisesRegex(ValueError, "only the frozen subtype cutoff"):
            self.result(p, cuts)

    def test_changed_path_and_eligibility_are_rejected(self):
        p = torch.zeros(109, dtype=torch.float64)
        p[0] = 0.1
        p[1] = 0.8
        p[2] = 0.1
        for key, value in [("path", [0, -1, -1, -1, -1]), ("subtype_cutoff_passed", False)]:
            r = self.result(p)
            r["prediction"][key] = value
            with self.assertRaises(ValueError):
                validate_result(r)

    def test_json_schema_and_csv_cover_issued_unresolved_and_negative(self):
        schema = json.loads(result_schema_path().read_text())
        for masses, status in [
            ([0.1, 0.8, 0.1], "fully_resolved"),
            ([0.2, 0.3, 0.25, 0.25], "partially_resolved"),
            ([0.8, 0.1, 0.1], "heme_tumor_not_detected"),
        ]:
            p = torch.zeros(109, dtype=torch.float64)
            p[: len(masses)] = torch.tensor(masses, dtype=p.dtype)
            r = self.result(p)
            self.assertEqual(r["status"], status)
            jsonschema.validate(r, schema)
            self.assertTrue({"kind", "schema_version", "minimum_observed_cpgs"}.isdisjoint(r))
            self.assertNotIn("manifest_sha256", r["release"])
            self.assertEqual(r["runtime"], {"device": "cpu"})
            self.assertNotIn("level", r["accepted"])
            if r["decision"]:
                self.assertTrue({"level", "reporting_cutoff"}.isdisjoint(r["decision"]))
            self.assertEqual(len(_result_csv_row(r)), 29)
            self.assertNotIn("confidence", r["result_summary"])
            for node in r["path"]:
                self.assertNotIn("reporting_cutoff", node)
                if node["level"] == "subtype":
                    self.assertNotIn("model_score", node)
            if r["decision"]:
                for entry in r["decision"]["differential"]:
                    self.assertEqual(set(entry), {"index", "classification"})
            before = copy.deepcopy(r)
            row = _result_csv_row(r)
            self.assertEqual(row["result_status"], status)
            self.assertEqual(r, before)
            if status == "heme_tumor_not_detected":
                self.assertEqual(row["subtype_scaled_score"], "")
                self.assertEqual(row["subtype_scaled_cutoff"], "")
                self.assertEqual(row["result_summary"], r["result_summary"])
            else:
                self.assertNotIn("scaled subtype score", row["result_summary"])
                self.assertNotIn("/100", row["result_summary"])
                self.assertEqual(row["result_summary"], r["result_summary"])
                self.assertEqual(row["subtype_scaled_cutoff"], 85)

    def test_nonfinite_probabilities_and_misleading_support_are_rejected(self):
        p = torch.zeros(109, dtype=torch.float64)
        p[0] = 0.1
        p[1] = 0.9
        r = self.result(p)
        r["observed_cpg_count"] = 4999
        with self.assertRaisesRegex(ValueError, "support"):
            validate_result(r)
        p[2] = float("nan")
        with self.assertRaises(ValueError):
            self.result(p)

    def test_input_floor_before_members_are_evaluated(self):
        class Dummy(FinalEnsemble):
            def _member(self, *args, **kwargs):
                raise RuntimeError("member reached")

        m = object.__new__(Dummy)
        m.batch_size = 64
        m.device = torch.device("cpu")
        m.manifest = {"members": [{"key": "test"}]}
        beta = torch.zeros((1, 5000))
        mask = torch.ones_like(beta, dtype=torch.bool)
        mask[0, -1] = False
        args = (beta, mask, beta, torch.zeros_like(beta, dtype=torch.int64), beta)
        with self.assertRaisesRegex(ValueError, "5,000"):
            m.predict(*args)
        mask[0, -1] = True
        with self.assertRaisesRegex(RuntimeError, "member reached"):
            m.predict(*args)


if __name__ == "__main__":
    unittest.main()
