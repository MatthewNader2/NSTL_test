"""
tests/test_phase3_graph_model.py - Neuro-Symbolic Topological Lattice (NSTL)
Comprehensive test suite for Phase 3:
- T3.1: 4-term edge score model
- T3.2: Dynamic cross-tree bridge promotion
- T3.3: Static reachability and disconnected node auditor
"""

import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from lattice import LatticeOrchestrator, Cell
from planner import LatticePlanner
from lattice_auditor import LatticeAuditor, LatticeAuditReport


class TestPhase3GraphModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trees_dir = os.path.join(ROOT_DIR, "trees")
        cls.orchestrator = LatticeOrchestrator(trees_directory=trees_dir)
        cls.planner = LatticePlanner(orchestrator=cls.orchestrator)
        cls.auditor = LatticeAuditor(orchestrator=cls.orchestrator)

    def test_01_formal_4_term_edge_score_model(self):
        """Verify compute_edge_score implements the 4-term edge scoring function (T3.1)."""
        cells = list(self.orchestrator.loaded_cells.values())
        self.assertGreater(len(cells), 5)

        # Pick two related cells (e.g. read_csv -> dropna or imread -> cvtColor)
        src_cell = self.orchestrator.loaded_cells.get("PD_READ_CSV")
        dst_cell = self.orchestrator.loaded_cells.get("PD_DROPNA")
        self.assertIsNotNone(src_cell)
        self.assertIsNotNone(dst_cell)

        rel_map = {dst_cell.cell_id: 0.85}
        score = self.planner.compute_edge_score(
            src_cell=src_cell,
            dst_cell=dst_cell,
            relevance_map=rel_map,
            weights=(0.35, 0.30, 0.25, 0.10)
        )

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Edge score between unrelated cells should be lower
        unrelated_cell = self.orchestrator.loaded_cells.get("CV2_IMREAD")
        if unrelated_cell:
            unrelated_score = self.planner.compute_edge_score(
                src_cell=src_cell,
                dst_cell=unrelated_cell,
                relevance_map={unrelated_cell.cell_id: 0.05}
            )
            self.assertGreater(score, unrelated_score)

    def test_02_dynamic_cross_tree_bridge_promotion(self):
        """Verify dynamic cross-tree bridge promotion at runtime (T3.2)."""
        src_id = "PD_READ_CSV"
        dst_id = "CV2_IMWRITE"  # Synthetic cross-domain bridge for test verification

        initial_promoted_count = len(self.orchestrator.get_promoted_bridges())

        # Promote dynamic cross-tree edge
        success = self.orchestrator.register_dynamic_edge(
            src_cell_id=src_id,
            dst_cell_id=dst_id,
            affinity_score=0.92,
            provenance="test_runtime_synthesis"
        )
        self.assertTrue(success)

        # Verify updated adjacency
        self.assertIn(dst_id, self.orchestrator._adjacency[src_id])
        self.assertIn(src_id, self.orchestrator._reverse_adjacency[dst_id])

        # Verify tracking
        promoted = self.orchestrator.get_promoted_bridges()
        self.assertEqual(len(promoted), initial_promoted_count + 1)
        last_promoted = promoted[-1]
        self.assertEqual(last_promoted["source"], src_id)
        self.assertEqual(last_promoted["target"], dst_id)
        self.assertEqual(last_promoted["affinity_score"], 0.92)
        self.assertTrue(last_promoted["is_cross_tree"])

    def test_03_static_reachability_auditor(self):
        """Verify static topology auditor detects entry nodes, reachable nodes, and metrics (T3.3)."""
        report = self.auditor.audit()

        self.assertIsInstance(report, LatticeAuditReport)
        self.assertGreater(report.total_cells, 0)
        self.assertGreater(len(report.entry_nodes), 0, "Lattice must have at least one entry node")
        self.assertGreater(len(report.terminal_nodes), 0, "Lattice must have at least one terminal node")
        self.assertGreater(len(report.reachable_nodes), 0, "Lattice must have reachable nodes")

        print(f"\n[Lattice Audit Summary]: {report.summary}")
        self.assertTrue(report.is_healthy(max_unreachable_ratio=0.50))


if __name__ == "__main__":
    unittest.main()
