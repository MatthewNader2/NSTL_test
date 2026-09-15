"""
tests/test_phase4_macros.py - Phase 4 Macro Abstraction & Dynamic Harvesting Tests
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from lattice import LatticeOrchestrator, MacroCell, Cell
from macro_harvester import MacroHarvester
from unification import UnificationGate, ExecutionContext, Success
from preflight import PreflightLinter


class TestPhase4Macros(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orch = LatticeOrchestrator("trees")
        cls.gate = UnificationGate(cls.orch)

    def test_harvest_and_unify_macro(self):
        """Test harvesting a composite macro and verifying its structural expansion."""
        cell_ids = ["PD_READ_CSV", "PD_DROPNA"]
        macro = MacroHarvester.harvest_macro(
            cell_ids,
            self.orch,
            macro_id="MACRO_LOAD_AND_CLEAN",
            docstring="Load CSV and drop missing values."
        )

        self.assertIsInstance(macro, MacroCell)
        self.assertEqual(macro.cell_id, "MACRO_LOAD_AND_CLEAN")
        self.assertEqual(macro.sub_cells, cell_ids)
        self.assertEqual(macro.internal_topology, {"PD_READ_CSV": ["PD_DROPNA"]})
        self.assertEqual(macro.stage, 1)  # Preserves stage 1 of PD_READ_CSV
        self.assertIn("filepath", macro.inputs)
        self.assertIn("MACRO_LOAD_AND_CLEAN", self.orch.loaded_cells)

        # Pipeline: [MACRO_LOAD_AND_CLEAN, PD_TO_CSV]
        to_csv_cell = self.orch.loaded_cells["PD_TO_CSV"]
        ctx = ExecutionContext(prompt="Load data.csv, clean missing rows, and export to clean.csv")

        # Unify
        res = self.gate.unify_pipeline([macro, to_csv_cell], ctx)
        self.assertTrue(isinstance(res, Success), f"Macro unification failed: {res}")
        pipeline_bindings = res.value

        # Check expansion: must expand into 3 constituent micro-cells
        self.assertEqual(len(pipeline_bindings), 3)
        expanded_ids = [c.cell_id for c, _ in pipeline_bindings]
        self.assertEqual(expanded_ids, ["PD_READ_CSV", "PD_DROPNA", "PD_TO_CSV"])

        # Pre-flight lint check
        lint_res = PreflightLinter.lint(pipeline_bindings)
        self.assertTrue(lint_res.is_valid, f"Preflight linter failed on macro pipeline: {lint_res.violations}")

        # Code emission
        code = self.gate.emit_code([macro, to_csv_cell], ctx)
        self.assertTrue(len(code) > 0)
        # AST syntax check
        compiled = compile(code, "<macro_pipeline>", "exec")
        self.assertIsNotNone(compiled)

    def test_harvest_incompatible_cells_rejected(self):
        """Test that harvesting an incompatible or ill-typed sequence raises ValueError."""
        incompatible_candidates = [cid for cid in self.orch.loaded_cells if "CONFUSION" in cid or "ACCURACY" in cid]
        if incompatible_candidates:
            with self.assertRaises(ValueError):
                MacroHarvester.harvest_macro(["PD_READ_CSV", incompatible_candidates[0]], self.orch)

    def test_nested_macro_expansion(self):
        """Test that nested macros expand recursively down to primitive micro-cells."""
        # 1. Harvest base macro: PD_READ_CSV -> PD_DROPNA
        base_macro = MacroHarvester.harvest_macro(
            ["PD_READ_CSV", "PD_DROPNA"],
            self.orch,
            macro_id="MACRO_BASE_CLEAN"
        )
        # 2. Create another macro whose sub_cells include the first macro and PD_TO_CSV
        to_csv = self.orch.loaded_cells["PD_TO_CSV"]
        outer_macro = MacroCell(
            cell_id="MACRO_FULL_ETL",
            stage=1,
            inputs=base_macro.inputs,
            outputs=to_csv.outputs,
            sub_cells=["MACRO_BASE_CLEAN", "PD_TO_CSV"],
            dependencies=base_macro.dependencies + to_csv.dependencies
        )
        self.orch.loaded_cells["MACRO_FULL_ETL"] = outer_macro

        ctx = ExecutionContext(prompt="Run full ETL on data.csv to out.csv")
        res = self.gate.unify_pipeline([outer_macro], ctx)
        self.assertTrue(isinstance(res, Success), f"Nested macro expansion failed: {res}")

        expanded_ids = [c.cell_id for c, _ in res.value]
        self.assertEqual(expanded_ids, ["PD_READ_CSV", "PD_DROPNA", "PD_TO_CSV"])


if __name__ == "__main__":
    unittest.main()
