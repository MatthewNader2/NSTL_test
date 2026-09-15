"""
tests/test_route_methods.py - Neuro-Symbolic Topological Lattice (NSTL)
Comprehensive unit test and benchmark suite for RouteMethods (M0 - M6).
"""

import os
import sys
import unittest
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from lattice import LatticeOrchestrator
from internal_rag import LocalRAG
from router import LatticeRouter
from unification import UnificationGate, ExecutionContext
from route_methods import (
    get_route_method,
    ROUTE_METHOD_REGISTRY,
    RouteMethod,
    M0TrellisRouteMethod,
    M1ClauseAnchorRouteMethod,
    M2EndpointAnchorRouteMethod,
    M3GreedyFreezeRouteMethod,
    M4LLMStepwiseRouteMethod,
    M5LLMOneShotRouteMethod,
    M6HybridAnchorsRouteMethod,
)


class TestRouteMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inference import ModelManager
        mm = ModelManager.get_instance()
        if mm.profile is None:
            mm.initialize_profile("A")
        trees_dir = os.path.join(ROOT_DIR, "trees")
        cls.orchestrator = LatticeOrchestrator(trees_directory=trees_dir)
        cls.rag = LocalRAG(trees_dir=trees_dir, orchestrator=cls.orchestrator)
        cls.router = LatticeRouter(orchestrator=cls.orchestrator, rag_engine=cls.rag)
        cls.gate = UnificationGate()

    def test_01_factory_instantiation(self):
        """Verify all route methods M0-M6 instantiate properly via get_route_method."""
        methods = ["m0", "m1", "m2", "m3", "m4", "m5", "m6"]
        for m_name in methods:
            m = get_route_method(m_name, orchestrator=self.orchestrator)
            self.assertIsInstance(m, RouteMethod)
            self.assertTrue(hasattr(m, "plan"))

        with self.assertRaises(ValueError):
            get_route_method("non_existent_method")

    def test_02_tabular_pipeline_all_methods(self):
        """Verify M0, M1, M2, M3, M4, M5, M6 plan valid tabular pipelines."""
        prompt = "load input.csv and drop missing values then save to output.csv"
        for m_name in ["m0", "m1", "m2", "m3", "m4", "m5", "m6"]:
            path, _ = self.router.plan_path(prompt, route_method=m_name)
            self.assertGreater(len(path), 0, f"Route method {m_name} returned empty path for tabular prompt!")
            first_cell = path[0]
            last_cell = path[-1]
            self.assertIn("READ", first_cell.cell_id.upper(), f"{m_name} should start with a read morphism, got {first_cell.cell_id}")
            self.assertIn("CSV", last_cell.cell_id.upper(), f"{m_name} should conclude with a csv sink, got {last_cell.cell_id}")

            # Verify code generation and binding
            ctx = ExecutionContext(prompt=prompt)
            code = self.gate.emit_code(path, ctx)
            self.assertIn("read_csv", code)
            self.assertIn("to_csv", code)

    def test_03_vision_pipeline_all_methods(self):
        """Verify M0, M1, M2, M3, M4, M5, M6 plan valid vision pipelines."""
        prompt = "read image input.jpg and convert to grayscale then save to output.jpg"
        for m_name in ["m0", "m1", "m2", "m3", "m4", "m5", "m6"]:
            path, _ = self.router.plan_path(prompt, route_method=m_name)
            self.assertGreater(len(path), 0, f"Route method {m_name} returned empty path for vision prompt!")
            first_cell = path[0]
            last_cell = path[-1]
            self.assertIn("IMREAD", first_cell.cell_id.upper(), f"{m_name} should start with IMREAD, got {first_cell.cell_id}")
            self.assertIn("IMWRITE", last_cell.cell_id.upper(), f"{m_name} should conclude with IMWRITE, got {last_cell.cell_id}")

            ctx = ExecutionContext(prompt=prompt)
            code = self.gate.emit_code(path, ctx)
            self.assertIn("imread", code)
            self.assertIn("imwrite", code)

    def test_04_m6_hybrid_lint_clean(self):
        """Verify M6 flagship hybrid produces lint-clean pipeline."""
        prompt = "load input.csv and drop missing values then save to output.csv"
        path, _ = self.router.plan_path(prompt, route_method="m6")
        ctx = ExecutionContext(prompt=prompt)
        res_pipeline = self.gate.unify_pipeline(path, ctx)
        self.assertFalse(res_pipeline.is_bottom())
        bindings = res_pipeline.value
        code = self.gate.emit_code(path, ctx)
        from preflight import PreflightLinter
        # Check that preflight lint passes
        res = PreflightLinter.lint(bindings, prompt=prompt, code_str=code)
        self.assertTrue(res.is_valid, f"M6 pipeline failed preflight lint: {res.violations}")


if __name__ == "__main__":
    unittest.main()
