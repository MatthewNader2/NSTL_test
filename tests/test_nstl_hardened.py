# tests/test_nstl_hardened.py
"""
Verification and regression test suite for hardened NSTL engine components.
Tests:
- Environment-validated import resolution without placeholder hallucinations
- SSA Lineage Tracker auto-re-anchoring
- Parameter Extractor classification
- AlgebraicSignature qualifiers and subtyping
- GEVR Sandbox execution verification
"""

import os
import sys
import unittest

# Ensure src/ is on Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unification import UnificationGate, enforce_lineage_integrity, ParameterExtractor, ExecutionContext
from lattice import AlgebraicSignature, canonical_type_name, is_subtype
from gevr_sandbox import GEVRSandbox


class TestNSTLHardened(unittest.TestCase):

    def test_import_resolution_no_placeholders(self):
        code = "df = pd.read_csv(input_var)\nres = X.transform(df)"
        ctx = ExecutionContext()
        ctx.declare_variable("input_var", AlgebraicSignature(type_name="str", state="source"))
        ctx.declare_variable("X", AlgebraicSignature(type_name="ndarray", state="raw"))
        
        resolved = UnificationGate.resolve_imports(code, ctx)
        self.assertIn("import pandas as pd", resolved)
        self.assertNotIn("import input_var", resolved)
        self.assertNotIn("import X", resolved)
        self.assertNotIn("import pd", resolved)

    def test_lineage_auto_reanchor(self):
        buggy_code = (
            "import pandas as pd\n"
            "df = pd.read_csv('data.csv')\n"
            "df_clean = df.dropna()\n"
            "df_sorted = df_clean.sort_values('age')\n"
            "df_clean.to_csv('out.csv')"
        )
        fixed = enforce_lineage_integrity(buggy_code)
        self.assertIn("df_sorted.to_csv", fixed)

    def test_parameter_disambiguation(self):
        prompt = "Read 'data.csv', sort by the 'age' column, and save to 'cleaned.csv'"
        params = ParameterExtractor.extract_parameters(prompt)
        self.assertIn("data.csv", params["input_files"])
        self.assertIn("cleaned.csv", params["output_files"])
        self.assertIn("age", params["columns"])

    def test_algebraic_signature_subtyping(self):
        sig_df = AlgebraicSignature.from_string("DataFrame", "cleaned")
        sig_any = AlgebraicSignature.from_string("any", "any")
        sig_sparse = AlgebraicSignature.from_string("SparseMatrix", "raw")
        sig_ndarray = AlgebraicSignature.from_string("ndarray", "raw")

        self.assertTrue(sig_df.matches(sig_any))
        self.assertTrue(sig_sparse.matches(sig_ndarray))
        self.assertEqual(canonical_type_name("pd.DataFrame"), "DataFrame")
        self.assertTrue(is_subtype("SparseMatrix", "ndarray"))

    def test_gevr_sandbox_execution(self):
        sandbox = GEVRSandbox(timeout_seconds=3)
        valid_code = "a = 5\nb = 7\nassert a + b == 12\nprint('Success')"
        success, stdout, stderr = sandbox.execute_and_verify(valid_code)
        self.assertTrue(success)
        self.assertIn("Success", stdout)
        self.assertEqual(stderr, "")

    def test_gevr_sandbox_heuristic_repair(self):
        sandbox = GEVRSandbox(timeout_seconds=3)
        code_with_missing_import = "df = pd.DataFrame({'a': [1, 2]})\nassert len(df) == 2"
        success, repaired_code, stdout = sandbox.repair_cycle(code_with_missing_import, max_attempts=2)
        self.assertTrue(success)
        self.assertIn("import pandas as pd", repaired_code)


if __name__ == "__main__":
    unittest.main()
