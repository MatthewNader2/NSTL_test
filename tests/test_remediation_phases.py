# tests/test_remediation_phases.py
import unittest
import os
import sys
import json
import tempfile

# Ensure src/ directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from unification import (
    UnificationGate,
    ExecutionContext,
    UnresolvedPlaceholderError,
    assert_placeholders_resolved,
    enforce_lineage_integrity,
    resolve_node_slots,
    ParameterExtractor,
)
from router import log_coverage_gap
from synthesis import SynthesisEngine


class TestRemediationPhases(unittest.TestCase):

    def test_phase1_unbound_placeholder_raises_error(self):
        """Phase 1: Pre-flight placeholder gate raises UnresolvedPlaceholderError on unbound placeholders."""
        template = "df = pd.read_csv({unknown_slot})"
        bindings = {"input_var": "data.csv"}  # {unknown_slot} is missing!
        with self.assertRaises(UnresolvedPlaceholderError):
            assert_placeholders_resolved(template, bindings=bindings)

    def test_phase1_bound_placeholder_passes(self):
        """Phase 1: Valid bindings pass placeholder gate."""
        template = "df = pd.read_csv({input_filename})"
        bindings = {"input_filename": "data.csv"}
        try:
            assert_placeholders_resolved(template, bindings=bindings)
        except UnresolvedPlaceholderError:
            self.fail("assert_placeholders_resolved raised UnresolvedPlaceholderError unexpectedly!")

    def test_phase1_declared_dependency_imports_only(self):
        """Phase 1: Imports are constructed strictly from declared dependencies, not scanning unbound variable names."""
        code_text = "output_var = cv2.cvtColor(input_var, cv2.COLOR_BGR2GRAY)"
        
        class MockNode:
            dependencies = ["cv2"]
            
        context = ExecutionContext()
        res = UnificationGate.resolve_imports(code_text, context=context, chain_nodes=[MockNode()])
        self.assertIn("import cv2", res)
        # Ensure it does NOT guess imports for random variable names
        self.assertNotIn("import input_var", res)

    def test_phase2_domain_calibrated_thresholds(self):
        """Phase 2: Uniform similarity threshold from config.py."""
        from config import SIMILARITY_THRESHOLD
        self.assertEqual(SIMILARITY_THRESHOLD, 0.25)

    def test_phase2_coverage_gap_telemetry_logging(self):
        """Phase 2: Below-threshold query creates entry in logs/coverage_gaps.log."""
        log_coverage_gap("synthesize custom segment tree", "algorithms", 0.12, "NODE_0")
        log_file = os.path.join("logs", "coverage_gaps.log")
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertTrue(len(lines) > 0)
            last_entry = json.loads(lines[-1])
            self.assertEqual(last_entry["prompt"], "synthesize custom segment tree")
            self.assertEqual(last_entry["domain_guess"], "algorithms")

    def test_phase3_lineage_reanchor_safety_net(self):
        """Phase 3: Verify DAG lineage tracker safety net re-anchors stale terminal sink calls."""
        raw_code = (
            "df_clean = df.dropna()\n"
            "df_sorted = df_clean.sort_values(by='age')\n"
            "df_clean.to_csv('output.csv')\n"
        )
        corrected = enforce_lineage_integrity(raw_code)
        self.assertIn("df_sorted.to_csv", corrected)

    def test_phase4_synthesis_memo_cache_hit(self):
        """Phase 4: Synthesis memo-cache reuses cached synthesized micro-cell without invoking LLM."""
        engine = SynthesisEngine()
        
        class DummyFetcher:
            def fetch(self, concept):
                raise RuntimeError("LLM/LiveDoc fetcher should NOT be invoked on cache hit!")

        cache_path = os.path.join("trees", "micro", "synthesized_nodes.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cells = data.get("cells", [])
                if cells:
                    target_cell = cells[0]
                    kws = target_cell.get("keywords", ["test"])[0]
                    c_in = target_cell.get("inputs", {}).get("type_name", "str")
                    c_out = target_cell.get("outputs", {}).get("type_name", "DataFrame")
                    
                    res = engine.synthesize_micro_cell(kws, c_in, c_out, fetcher=DummyFetcher())
                    self.assertIsNotNone(res)
                    self.assertEqual(res.get("cell_id"), target_cell.get("cell_id"))

    def test_phase7_pandas_slot_resolution(self):
        """Phase 7: Schema-driven slot resolution deterministically binds Pandas template slots."""
        prompt = "Read 'data.csv', sort by the 'age' column in descending order, and save to 'cleaned.csv'"
        extracted = ParameterExtractor.extract_parameters(prompt)
        template = "df = pd.read_csv({filename})\ndf_sorted = df.sort_values(by={by_column}, ascending={ascending})\ndf_sorted.to_csv({output_filename}, index=False)"
        
        slots = resolve_node_slots(template, extracted)
        self.assertEqual(slots.get("filename"), "'data.csv'")
        self.assertEqual(slots.get("by_column"), "'age'")
        self.assertEqual(slots.get("ascending"), "False")
        self.assertEqual(slots.get("output_filename"), "'cleaned.csv'")

    def test_phase7_opencv_slot_resolution(self):
        """Phase 7: Schema-driven slot resolution deterministically binds OpenCV template slots."""
        prompt = "Read 'input.jpg' using opencv, convert to grayscale, and save to 'output.jpg'"
        extracted = ParameterExtractor.extract_parameters(prompt)
        template = "img = cv2.imread({image_path})\ngray = cv2.cvtColor(img, {code})\ncv2.imwrite({output_path}, gray)"
        
        slots = resolve_node_slots(template, extracted)
        self.assertEqual(slots.get("image_path"), "'input.jpg'")
        self.assertEqual(slots.get("code"), "cv2.COLOR_BGR2GRAY")
        self.assertEqual(slots.get("output_path"), "'output.jpg'")


if __name__ == "__main__":
    unittest.main()
