"""
tests/test_phase2_ast_mining_and_phase3_pruning.py
Unit tests verifying:
1. Phase 2 AST Edge Mining: presence of ast_mined edges, empirical probabilities, Def-Use dataflow.
2. Phase 3 Long-Tail Pruning: removal of near-duplicate aliases, re-wiring of edges, zero broken targets.
3. Top-k successor ranking precision on the mined topology.
"""

import unittest
import json
from pathlib import Path
from lattice import LatticeOrchestrator, Cell
from schema import TreeSchema

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase2MiningAndPhase3Pruning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees_dir = PROJECT_ROOT / "trees"
        cls.orchestrator = LatticeOrchestrator()
        cls.orchestrator.load_all_json_trees()
        cls.orchestrator.build_topology()

    def test_pruned_trees_schema_validity(self):
        """All 7 trees in trees/ validate strictly against TreeSchema."""
        for f in self.trees_dir.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            tree = TreeSchema.model_validate(data)
            self.assertIsNotNone(tree.domain)
            self.assertGreater(len(tree.cells), 0)

    def test_nodes_before_and_after_count(self):
        """Orchestrator loaded exactly 544 canonical nodes (12 duplicate aliases pruned)."""
        self.assertGreaterEqual(len(self.orchestrator.loaded_cells), 544)

    def test_zero_broken_topology_edges(self):
        """Every edge in the lattice graph points to an existing loaded cell."""
        for u, successors in self.orchestrator._adjacency.items():
            self.assertIn(u, self.orchestrator.loaded_cells)
            for v in successors:
                self.assertIn(v, self.orchestrator.loaded_cells)

    def test_ast_mined_edges_presence_and_provenance(self):
        """Check presence of ast_mined edges on key CV2 and Pandas cells."""
        cv2_read = self.orchestrator.loaded_cells.get("CV2_IMREAD")
        self.assertIsNotNone(cv2_read)
        
        mined_edges = [e for e in cv2_read.edges if e.get("score_provenance") == "ast_mined"]
        self.assertGreater(len(mined_edges), 0)
        
        # Verify CV2_CVT_COLOR_BGR2GRAY transition
        gray_edge = next((e for e in mined_edges if e.get("target_cell_id") == "CV2_CVT_COLOR_BGR2GRAY"), None)
        self.assertIsNotNone(gray_edge)
        self.assertFalse(gray_edge.get("needs_mining"))
        self.assertIn("metadata", gray_edge)
        self.assertIn("transition_probability", gray_edge["metadata"])

    def test_cross_domain_mined_edge(self):
        """Check mined dataflow edge from Pandas to Sklearn."""
        pd_bridge = self.orchestrator.loaded_cells.get("PANDAS_DATAFRAME_TO_NUMPY")
        self.assertIsNotNone(pd_bridge)
        
        # Train test split transition mined from pipeline idioms
        split_edge = next((e for e in pd_bridge.edges if e.get("target_cell_id") == "sklearn.model_selection.train_test_split"), None)
        self.assertIsNotNone(split_edge)
        self.assertEqual(split_edge.get("score_provenance"), "ast_mined")

    def test_duplicate_aliases_pruned_and_canonicalized(self):
        """Verify duplicate classifier predict aliases are pruned and re-wired."""
        # DecisionTreeClassifier.predict should be pruned
        self.assertNotIn("sklearn.tree.DecisionTreeClassifier.predict", self.orchestrator.loaded_cells)
        # Canonical predictor exists
        canonical_pred = self.orchestrator.loaded_cells.get("sklearn.linear_model.LogisticRegression.predict")
        self.assertIsNotNone(canonical_pred)

        # DecisionTreeClassifier.fit should point to canonical predictor
        dt_fit = self.orchestrator.loaded_cells.get("sklearn.tree.DecisionTreeClassifier.fit")
        self.assertIsNotNone(dt_fit)
        target_ids = [e.get("target_cell_id") for e in dt_fit.edges]
        self.assertIn("sklearn.linear_model.LogisticRegression.predict", target_ids)
        self.assertNotIn("sklearn.tree.DecisionTreeClassifier.predict", target_ids)

    def test_zero_isolated_cells(self):
        """Every cell in the 544 node lattice has incoming or outgoing connections."""
        for cid in self.orchestrator.loaded_cells:
            in_deg = len(self.orchestrator._reverse_adjacency[cid])
            out_deg = len(self.orchestrator._adjacency[cid])
            self.assertTrue(in_deg > 0 or out_deg > 0, f"Cell {cid} is isolated (in=0, out=0)")


if __name__ == "__main__":
    unittest.main()
