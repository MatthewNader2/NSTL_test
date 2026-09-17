"""
tests/test_macro_synapses.py - Neuro-Symbolic Topological Lattice (NSTL)

Tests the Synaptic Macro Node Pipeline:
1. Macro internal topology dynamic edge registration (macro synapses).
2. Synaptic micro-cell score boosting when prompt is relevant to the macro goal.
3. Synaptic edge affinity reinforcement along constituent micro-paths.
4. Clean A/B toggleability (--macros / --no-macros).
5. End-to-end code synthesis for macro-reinforced pipelines.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lattice import LatticeOrchestrator, MacroCell, MicroCell
from router import LatticeRouter
from planner import LatticePlanner
from unification import UnificationGate, ExecutionContext, Success


class TestMacroSynapses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees_dir = str(ROOT_DIR / "trees")
        cls.orch = LatticeOrchestrator(trees_directory=cls.trees_dir)

    def test_01_macro_topology_dynamic_edge_registration(self):
        """Verify orchestrator registers macro internal topology edges as macro synapses."""
        self.assertIn("MACRO_CV2_CONTOUR_ANNOTATION", self.orch.loaded_cells)
        macro = self.orch.loaded_cells["MACRO_CV2_CONTOUR_ANNOTATION"]
        self.assertIsInstance(macro, MacroCell)
        self.assertTrue(len(macro.sub_cells) >= 2)

        # Check that edges inside internal_topology are recorded in dynamic_edges
        edge_key = ("CV2_IMREAD", "CV2_CVT_COLOR_BGR2GRAY")
        self.assertIn(edge_key, self.orch.dynamic_edges)
        edge_info = self.orch.dynamic_edges[edge_key]
        self.assertEqual(edge_info.get("score_provenance"), "macro_synapse")
        self.assertGreaterEqual(edge_info.get("affinity_score", 0.0), 0.8)

        # Check that orchestrator adjacency includes the micro-cells
        adj = self.orch._adjacency.get("CV2_IMREAD", [])
        self.assertIn("CV2_CVT_COLOR_BGR2GRAY", adj)

    def test_02_macro_goal_tunnel_promotion_and_isolation(self):
        """Verify that relevant macro goals are promoted in the tunnel when enabled, and cleanly isolated when disabled."""
        prompt = "Load image, find contours with otsu thresholding, draw them and save the annotated image"

        # A. Macros ENABLED
        router_on = LatticeRouter(self.orch, macros_enabled=True)
        tunnel_on, rel_on = router_on.route(prompt)
        tunnel_ids_on = {c.cell_id for c in tunnel_on}

        # B. Macros DISABLED
        router_off = LatticeRouter(self.orch, macros_enabled=False)
        tunnel_off, rel_off = router_off.route(prompt)
        tunnel_ids_off = {c.cell_id for c in tunnel_off}

        # When macros disabled: MacroCell must NOT appear in tunnel (clean A/B benchmark baseline)
        self.assertNotIn("MACRO_CV2_CONTOUR_ANNOTATION", tunnel_ids_off)
        for c in tunnel_off:
            self.assertNotIsInstance(c, MacroCell)

        # When macros enabled: macro must appear and be promoted decisively
        self.assertIn("MACRO_CV2_CONTOUR_ANNOTATION", tunnel_ids_on)
        macro_score = rel_on.get("MACRO_CV2_CONTOUR_ANNOTATION", 0.0)
        self.assertGreater(macro_score, 1.0, "Macro goal did not receive promotion score")

        # Macro must outrank constituent micro-cells that are in the tunnel
        macro = self.orch.loaded_cells["MACRO_CV2_CONTOUR_ANNOTATION"]
        for sid in macro.sub_cells:
            if sid in rel_on:
                self.assertGreater(macro_score, rel_on[sid], f"Macro {macro.cell_id} did not outrank constituent {sid}")

    def test_03_synaptic_edge_affinity_reinforcement(self):
        """Verify edge affinity between consecutive micro-cells is reinforced when macro is active."""
        prompt = "Load image, find contours with otsu thresholding, draw them and save the annotated image"

        router_on = LatticeRouter(self.orch, macros_enabled=True)
        _, rel_on = router_on.route(prompt)
        planner_on = LatticePlanner(self.orch, macros_enabled=True)
        planner_on.current_relevance_map = rel_on

        router_off = LatticeRouter(self.orch, macros_enabled=False)
        _, rel_off = router_off.route(prompt)
        planner_off = LatticePlanner(self.orch, macros_enabled=False)
        planner_off.current_relevance_map = rel_off

        u = self.orch.loaded_cells["CV2_CVT_COLOR_BGR2GRAY"]
        v = self.orch.loaded_cells["CV2_THRESHOLD_OTSU"]

        aff_on = planner_on._calculate_edge_affinity(u, v)
        aff_off = planner_off._calculate_edge_affinity(u, v)

        self.assertGreaterEqual(aff_on, 0.85, f"Expected reinforced affinity >= 0.85, got {aff_on}")
        self.assertGreater(aff_on, aff_off, f"Synaptic affinity {aff_on} not strictly greater than baseline {aff_off}")

    def test_04_end_to_end_contour_macro_synthesis(self):
        """Verify end-to-end plan and code synthesis with active contour macro."""
        prompt = "Load image input.jpg, find contours with otsu thresholding, draw them onto the image and save to output.jpg"
        router = LatticeRouter(self.orch, macros_enabled=True)
        path, _ = router.plan_path(prompt)

        self.assertTrue(len(path) >= 1)
        gate = UnificationGate(self.orch)
        ctx = ExecutionContext(prompt=prompt)
        res = gate.unify_pipeline(path, ctx)

        self.assertIsInstance(res, Success, f"Unification failed: {res}")
        code = gate.emit_code(path, ctx)

        self.assertIn("cv2.imread", code)
        self.assertIn("cv2.cvtColor", code)
        self.assertIn("cv2.threshold", code)
        self.assertIn("cv2.findContours", code)
        self.assertIn("cv2.drawContours", code)
        self.assertIn("cv2.imwrite", code)

    def test_05_router_macro_toggle_property_sync(self):
        """Verify router.macros_enabled setter synchronizes planner.macros_enabled."""
        router = LatticeRouter(self.orch, macros_enabled=True)
        self.assertTrue(router.macros_enabled)
        self.assertTrue(router.planner.macros_enabled)

        router.macros_enabled = False
        self.assertFalse(router.macros_enabled)
        self.assertFalse(router.planner.macros_enabled)


if __name__ == "__main__":
    unittest.main()
