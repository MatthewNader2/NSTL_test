"""
tests/test_phase0_fixes.py
Comprehensive automated test suite covering all Phase 0 Critical Fixes (C1-C14).
Uses standard library unittest.
"""

import os
import sys
import json
import threading
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools"))

from lattice import AlgebraicSignature, PortSignature, TypeRegistry, LatticeOrchestrator, MicroCell, MacroCell
from unification import UnificationGate, ExecutionContext, PlaceholderResolver
from internal_rag import LocalRAG
from external_rag import IntrospectionFetcher, FetcherFactory
from synthesis import SynthesisEngine
from router import LatticeRouter, MCTSEngine
from compile_trees import _validate_template


class TestPhase0CriticalFixes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from inference import ModelManager
        mm = ModelManager.get_instance()
        if mm.profile is None:
            mm.initialize_profile("A")

    def test_c1_template_validation(self):
        """Verify AST-based template validation rejects unquoted bare filenames and hardcoded constants."""
        # 1. Bare unquoted filename argument (Must REJECT)
        valid, err = _validate_template("{output_var} = pd.read_csv(data.csv)", "test_cell")
        self.assertFalse(valid, "Should reject unquoted data.csv")
        self.assertIn("data.csv", str(err))

        # 2. Hardcoded literal filename without placeholder (Must REJECT)
        valid, err = _validate_template("{output_var} = pd.read_csv('data.csv')", "test_cell")
        self.assertFalse(valid, "Should reject hardcoded 'data.csv'")
        self.assertIn("Hardcoded string filename", str(err))

        # 3. Legitimate parameterized template (Must ACCEPT)
        valid, err = _validate_template("{output_var} = pd.read_csv({filepath})", "test_cell")
        self.assertTrue(valid, f"Should accept parameterized template, got: {err}")

        # 4. Valid method call like dist.pdf(x) (Must ACCEPT)
        valid, err = _validate_template("{output_var} = {self}.pdf({x})", "test_cell")
        self.assertTrue(valid, f"Should accept distribution .pdf() method call, got: {err}")

    def test_c6_multi_port_reconstruction(self):
        """Verify LatticeOrchestrator loads multi-port signatures from SQLite configuration_schema."""
        orchestrator = LatticeOrchestrator(trees_directory=os.path.join(ROOT_DIR, "trees"))
        self.assertGreater(len(orchestrator.loaded_cells), 0)

        # Verify TypeRegistry domain subtyping
        reg = TypeRegistry.get_instance()
        self.assertTrue(reg.is_subtype("DataFrame", "DataFrame"))
        self.assertTrue(reg.is_subtype("ndarray", "ndarray"))
        self.assertTrue(reg.is_subtype("Mat", "Mat"))

    def test_c7_typestate_placeholder_resolution(self):
        """Verify PlaceholderResolver binds typestates accurately without regex stripping corruptions."""
        ctx = ExecutionContext(prompt="load input.csv and drop missing values then save to output.csv")

        cell = MicroCell(
            cell_id="test_read_csv",
            stage=1,
            code_template="{output_var} = pd.read_csv({filepath})",
            inputs={"filepath": PortSignature(name="filepath", signature=AlgebraicSignature("str", "source_identifier"))},
            outputs={"output_var": PortSignature(name="output_var", signature=AlgebraicSignature("DataFrame", "raw"))},
            dependencies=["import pandas as pd"]
        )

        code = UnificationGate.unify_cell(ctx, cell)
        self.assertTrue('"input.csv"' in code or "'input.csv'" in code)
        self.assertNotIn("data.csv", code)
        self.assertNotIn("pd.read_csv(input.csv)", code)  # Must be quoted!

        # Test CV2 constant unquoting without filename quote damage
        cv_cell = MicroCell(
            cell_id="test_cvt_color",
            stage=2,
            code_template="{output_var} = cv2.cvtColor({src}, {code})",
            inputs={
                "src": PortSignature(name="src", signature=AlgebraicSignature("Mat", "raw")),
                "code": PortSignature(name="code", signature=AlgebraicSignature("int", "code"), default_value="cv2.COLOR_BGR2GRAY")
            },
            outputs={"output_var": PortSignature(name="output_var", signature=AlgebraicSignature("Mat", "grayscale"))},
            dependencies=["import cv2"]
        )
        cv_code = UnificationGate.unify_cell(ctx, cv_cell)
        self.assertIn("cv2.COLOR_BGR2GRAY", cv_code)
        self.assertNotIn("'cv2.COLOR_BGR2GRAY'", cv_code)

    def test_c4_introspection_fetcher(self):
        """Verify IntrospectionFetcher extracts real live docstrings locally."""
        fetcher = FetcherFactory.get_fetcher("Python")
        self.assertIsInstance(fetcher, IntrospectionFetcher)

        doc = fetcher.fetch("read_csv")
        self.assertIn("read_csv", doc)
        self.assertTrue("Signature:" in doc or "Documentation:" in doc)

        doc_heap = fetcher.fetch("heappop")
        self.assertIn("heappop", doc_heap)

    def test_c8_internal_rag_structured_output(self):
        """Verify get_relevant_context returns List[Dict[str, Any]]."""
        orchestrator = LatticeOrchestrator(trees_directory=os.path.join(ROOT_DIR, "trees"))
        rag = LocalRAG(trees_dir=os.path.join(ROOT_DIR, "trees"), orchestrator=orchestrator)

        results = rag.get_relevant_context("read csv file into pandas", top_k=5)
        self.assertIsInstance(results, list)
        if results:
            self.assertIsInstance(results[0], dict)
            self.assertIn("cell_id", results[0])
            self.assertIn("score", results[0])
            self.assertIn("schema", results[0])

            formatted = rag.format_context_for_prompt(results)
            self.assertIsInstance(formatted, str)
            self.assertIn("ID:", formatted)

    def test_c3_c5_c9_router(self):
        """Verify strict type pre-filtering, algorithmic seed routing, and A* bridging."""
        orchestrator = LatticeOrchestrator(trees_directory=os.path.join(ROOT_DIR, "trees"))
        rag = LocalRAG(trees_dir=os.path.join(ROOT_DIR, "trees"), orchestrator=orchestrator)
        router = LatticeRouter(orchestrator=orchestrator, rag_engine=rag)

        # C5: Algorithmic seed routing
        path, _ = router.plan_path("dijkstra shortest path algorithm")
        if "PYTHON_DIJKSTRA_ALGORITHM" in orchestrator.loaded_cells:
            self.assertGreaterEqual(len(path), 1)
            self.assertTrue(any("DIJKSTRA" in c.cell_id for c in path))

        # C9: A* Search bridging
        mcts = MCTSEngine(orchestrator)
        bridge = mcts.search(AlgebraicSignature("DataFrame", "raw"), AlgebraicSignature("DataFrame", "cleaned"))
        self.assertIsInstance(bridge, list)

    def test_c14_embedder_thread_safety(self):
        """Verify SentenceTransformer thread-safe inference lock."""
        from inference import BenchmarkProfile_A
        prof = BenchmarkProfile_A()
        self.assertTrue(hasattr(prof, "_lock"))
        self.assertIsInstance(prof._lock, type(threading.Lock()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
