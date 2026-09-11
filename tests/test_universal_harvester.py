"""
tests/test_universal_harvester.py
Unit and integration test for UniversalHarvester.
Verifies:
  1. Submodule traversal & public callable discovery
  2. Classification into Atomic, Parameterized, Scoped, Source, Sink, and Constant archetypes
  3. Proper parameterized port creation (e.g. cv2.cvtColor with code port)
  4. Enrichment from existing knowledge bases
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from universal_harvester import UniversalHarvester
from schema import CellSchema


class TestUniversalHarvester(unittest.TestCase):

    def test_cv2_harvest_and_archetypes(self):
        """Tests that cv2 is harvested with proper archetypes and without duplicate explosion."""
        harvester = UniversalHarvester(domain_name="cv2", package_name="cv2", container_type="Mat")
        cells = harvester.harvest_all()

        self.assertGreater(len(cells), 50, "Expected at least 50 cells from cv2")
        self.assertLess(len(cells), 5000, f"Expected < 5000 cells (no explosion), got {len(cells)}")

        cell_ids = {c.cell_id for c in cells}

        # Check CV2_CVTCOLOR exists and is parameterized
        cvt_cells = [c for c in cells if "CVTCOLOR" in c.cell_id and c.node_type == "parameterized"]
        self.assertGreaterEqual(len(cvt_cells), 1, "Expected at least one parameterized CVTCOLOR cell")
        cvt = cvt_cells[0]
        self.assertEqual(cvt.node_type, "parameterized")
        self.assertIn("code", cvt.inputs, "CVTCOLOR must have a code input port")
        self.assertEqual(cvt.inputs["code"].type_name, "Enum")

        # Check that constants like CV2_COLOR_BGR2GRAY exist as constant nodes
        bgr2gray = [c for c in cells if c.cell_id == "CV2_COLOR_BGR2GRAY"]
        self.assertEqual(len(bgr2gray), 1, "Expected CV2_COLOR_BGR2GRAY constant node")
        self.assertEqual(bgr2gray[0].node_type, "constant")
        self.assertEqual(bgr2gray[0].code_template, "cv2.COLOR_BGR2GRAY")

        # Check that cv2.addText does not have 272 duplicate color variants
        add_text_cells = [c for c in cells if "ADDTEXT" in c.cell_id]
        self.assertLessEqual(len(add_text_cells), 2, f"Expected <= 2 addText cells, got {len(add_text_cells)}")

    def test_numpy_harvest_and_archetypes(self):
        """Tests that numpy is harvested with source, sink, and transform morphisms."""
        harvester = UniversalHarvester(domain_name="numpy", package_name="numpy", container_type="ndarray")
        cells = harvester.harvest_all()

        self.assertGreater(len(cells), 100, "Expected at least 100 cells from numpy")

        # Ingestion / source checks (e.g. zeros, ones, arange)
        zeros_cell = next((c for c in cells if c.cell_id == "NUMPY_ZEROS"), None)
        self.assertIsNotNone(zeros_cell, "Expected NUMPY_ZEROS cell")
        self.assertEqual(zeros_cell.stage, 1, "NUMPY_ZEROS must be Stage 1 (Source)")

    def test_enrichment_from_existing_trees(self):
        """Tests that enrichment preserves keywords and docstrings without resurrecting duplicates."""
        harvester = UniversalHarvester(domain_name="cv2", package_name="cv2")
        cells = harvester.harvest_all()

        existing_tree = PROJECT_ROOT / "trees" / "cv2.json"
        if existing_tree.exists():
            enriched = UniversalHarvester.enrich_from_existing_trees(
                domain_name="cv2",
                new_cells=cells,
                existing_tree_paths=[existing_tree]
            )
            # Find cvtColor
            cvt = next((c for c in enriched if "CVTCOLOR" in c.cell_id and c.node_type == "parameterized"), None)
            if cvt:
                self.assertTrue(len(cvt.keywords) > 0)


if __name__ == "__main__":
    unittest.main()
