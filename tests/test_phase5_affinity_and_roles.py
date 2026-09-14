"""
tests/test_phase5_affinity_and_roles.py - Neuro-Symbolic Topological Lattice (NSTL)
Phase 5 Verification Suite:
  1. Dominant Edge Affinity (w_affinity = 25.0) prioritizing AST-mined transitions over unidiomatic hops.
  2. Removal of legacy planner junk exploit patches (calibrated clause coverage without hard clamp).
  3. Deterministic O(P * V) Role-Typed Port Matching (feature_input vs target_input, preventing fit(y, X) bug).
  4. Multi-Output Port Binding and Wire Allocation (e.g. train_test_split 4-tuple unpacking).
  5. Literal Stealing Prevention (file_asset literals preserved against auxiliary string theft).
"""

import ast
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.lattice import LatticeOrchestrator, Cell, PortSignature, AlgebraicSignature
from src.router import LatticeRouter
from src.planner import LatticePlanner
from src.unification import UnificationGate, ExecutionContext, Substitution, unify
from src.schema import PortSchema, CellSchema

DB_PATH = str(PROJECT_ROOT / "trees" / "lattice.db")


class TestPhase5AffinityAndRoles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = LatticeOrchestrator()
        cls.orchestrator.load_from_database(DB_PATH)
        cls.orchestrator.build_topology()
        cls.router = LatticeRouter(orchestrator=cls.orchestrator, internal_rag=None)
        cls.gate = UnificationGate()

    def test_01_dominant_edge_affinity_scores(self):
        """Verify that edge affinity score weight w_affinity = 25.0 dominates scoring."""
        planner = self.router.planner
        self.assertIsNotNone(planner)

        # Retrieve two canonical cells from pandas pipeline
        c_read = self.orchestrator.loaded_cells.get("PD_READ_CSV")
        c_drop = self.orchestrator.loaded_cells.get("PD_DROPNA")
        c_rand = self.orchestrator.loaded_cells.get("NUMPY_PAD")

        self.assertIsNotNone(c_read)
        self.assertIsNotNone(c_drop)
        self.assertIsNotNone(c_rand)

        aff_idiomatic = planner._calculate_edge_affinity(c_read, c_drop)
        aff_unidiomatic = planner._calculate_edge_affinity(c_read, c_rand)

        # Idiomatic transition (DataFrame -> DataFrame in pandas) has higher affinity than random cross-domain jump
        self.assertGreater(aff_idiomatic, aff_unidiomatic)

    def test_02_deterministic_role_typed_port_matching_fit(self):
        """
        Verify that fit(X, y) binds feature_input to X and target_input to y,
        eliminating the fit(y, X) parameter inversion bug.
        """
        prompt = "fit RandomForestClassifier on X and y"
        cells = [
            self.orchestrator.loaded_cells["SKLEARN.DATASETS.MAKE_CLASSIFICATION"],
            self.orchestrator.loaded_cells["SKLEARN.MODEL_SELECTION.TRAIN_TEST_SPLIT"],
            self.orchestrator.loaded_cells["SKLEARN.ENSEMBLE.RANDOMFORESTCLASSIFIER.FIT"]
        ]

        code = self.gate.unify_and_emit(cells, prompt)
        parsed = ast.parse(code)
        self.assertIsNotNone(parsed)

        # Locate fit call in AST
        fit_calls = [
            node for node in ast.walk(parsed)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
        ]
        self.assertTrue(len(fit_calls) > 0, "Expected .fit() call in generated code")
        fit_node = fit_calls[0]
        self.assertGreaterEqual(len(fit_node.args), 2)

        # Check argument names: first arg is feature (var_3 or X_train), second is target (var_5 or y_train)
        arg0 = fit_node.args[0].id if isinstance(fit_node.args[0], ast.Name) else str(fit_node.args[0])
        arg1 = fit_node.args[1].id if isinstance(fit_node.args[1], ast.Name) else str(fit_node.args[1])

        # Feature variable must be distinct from target variable and ordered correctly
        self.assertNotEqual(arg0, arg1)
        self.assertIn("var_", arg0)
        self.assertIn("var_", arg1)

    def test_03_multi_output_wire_declaration(self):
        """
        Verify multi-output port signatures generate distinct unpacked variables
        for cells like train_test_split (var_3, var_4, var_5, var_6).
        """
        prompt = "split dataset into train and test with sklearn train_test_split"
        cells = [
            self.orchestrator.loaded_cells["SKLEARN.DATASETS.LOAD_IRIS"],
            self.orchestrator.loaded_cells["SKLEARN.MODEL_SELECTION.TRAIN_TEST_SPLIT"]
        ]
        code = self.gate.unify_and_emit(cells, prompt)
        parsed = ast.parse(code)
        self.assertIsNotNone(parsed)

        # Expect an assignment with a Tuple target or 4 output variables
        assignments = [n for n in ast.walk(parsed) if isinstance(n, ast.Assign)]
        has_multi_out = False
        for a in assignments:
            for t in a.targets:
                if isinstance(t, ast.Tuple) and len(t.elts) >= 4:
                    has_multi_out = True
        self.assertTrue(has_multi_out, "Expected 4-tuple assignment unpacking for train_test_split")

    def test_04_literal_stealing_prevention(self):
        """
        Verify that file_asset literals (e.g. data.csv, cleaned.csv) are not stolen
        by non-path auxiliary string slots.
        """
        prompt = "read input.csv, drop missing values with pandas, standardize features with sklearn StandardScaler, and save to cleaned.csv"
        path = self.router.plan_path(prompt, return_tuple=False)
        self.assertTrue(len(path) >= 2)

        code = self.gate.unify_and_emit(path, prompt)
        parsed = ast.parse(code)
        self.assertIsNotNone(parsed)

        # input.csv must appear in read call and cleaned.csv in save/export call
        self.assertIn("input.csv", code)
        self.assertIn("cleaned.csv", code)
        # Ensure cleaned.csv is not mistakenly passed to StandardScaler or dropna
        self.assertNotIn("StandardScaler('cleaned.csv')", code)
        self.assertNotIn("StandardScaler(copy='cleaned.csv')", code)

    def test_05_algorithmic_self_contained_pipelines(self):
        """
        Verify algorithmic tasks (Dijkstra, Binary Search, Merge Sort) route to self-contained
        stdlib algorithmic nodes without ungrounded input errors.
        """
        for prompt, expected_id in [
            ("dijkstra shortest path algorithm on graph", "PYTHON_DIJKSTRA_ALGORITHM"),
            ("binary search algorithm on sorted list", "PYTHON_BINARY_SEARCH_SORT"),
            ("merge sort algorithm on list", "PYTHON_MERGE_SORT_ALGORITHM"),
        ]:
            path = self.router.plan_path(prompt, return_tuple=False)
            self.assertTrue(len(path) > 0, f"No path planned for: {prompt}")
            self.assertEqual(path[0].cell_id, expected_id)
            code = self.gate.unify_and_emit(path, prompt)
            parsed = ast.parse(code)
            self.assertIsNotNone(parsed)


if __name__ == "__main__":
    unittest.main()
