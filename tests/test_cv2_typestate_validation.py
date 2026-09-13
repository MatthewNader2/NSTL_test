# tests/test_cv2_typestate_validation.py
import unittest
import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lattice import LatticeOrchestrator, TypeRegistry, PortSignature, AlgebraicSignature
from unification import unify, TypestateTerm, TOP


class TestCV2TypestateValidation(unittest.TestCase):
    """Validation test suite using the normalized CV2 tree (new trees/cv2_v1.1.0_normalized.json)."""

    @classmethod
    def setUpClass(cls):
        cls.cv2_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "new trees", "cv2_v1.1.0_normalized.json")
        )
        assert os.path.exists(cls.cv2_path), f"CV2 tree file not found at {cls.cv2_path}"

        cls.orchestrator = LatticeOrchestrator()
        cls.orchestrator.load_tree_file(cls.cv2_path)
        cls.registry = TypeRegistry.get_instance()

    def test_tree_loaded_and_typestates_registered(self):
        """Verify load_tree_file registers parent_state hierarchy from typestates block."""
        self.assertGreater(len(self.orchestrator.loaded_cells), 0)

        # Check registered hierarchy from cv2_v1.1.0_normalized.json
        # binary -> parent_state: gray
        # edge_map -> parent_state: binary
        self.assertEqual(self.registry.get_state_parent("binary"), "gray")
        self.assertEqual(self.registry.get_state_parent("edge_map"), "binary")

    def test_cell_slots_populated_from_tree(self):
        """Verify preconditions, postconditions, effects, and edges slots are populated."""
        canny = self.orchestrator.loaded_cells.get("CV2_CANNY")
        self.assertIsNotNone(canny)

        # Slots must exist on Cell instance
        self.assertTrue(hasattr(canny, "preconditions"))
        self.assertTrue(hasattr(canny, "postconditions"))
        self.assertTrue(hasattr(canny, "effects"))
        self.assertTrue(hasattr(canny, "edges"))

        self.assertIsInstance(canny.preconditions, list)
        self.assertIsInstance(canny.postconditions, list)
        self.assertIsInstance(canny.effects, list)
        self.assertIsInstance(canny.edges, list)

        # Check that edges exist if declared in cell JSON
        self.assertGreaterEqual(len(canny.edges), 1)

    def test_port_signature_accepted_states_and_parent_state(self):
        """Verify PortSignature and AlgebraicSignature contain accepted_states and parent_state."""
        imwrite = self.orchestrator.loaded_cells.get("CV2_IMWRITE")
        self.assertIsNotNone(imwrite)

        image_port = imwrite.inputs["image"]
        self.assertIsInstance(image_port, PortSignature)
        self.assertEqual(image_port.signature.type_name, "ndarray")
        self.assertEqual(image_port.signature.state, "color_bgr")

        # accepted_states populated on both PortSignature and AlgebraicSignature
        self.assertIn("gray", image_port.accepted_states)
        self.assertIn("binary", image_port.accepted_states)
        self.assertIn("edge_map", image_port.accepted_states)

        self.assertIn("gray", image_port.signature.accepted_states)
        self.assertIn("binary", image_port.signature.accepted_states)
        self.assertIn("edge_map", image_port.signature.accepted_states)

        # Check CV2_CANNY input src accepts blurred
        canny = self.orchestrator.loaded_cells.get("CV2_CANNY")
        src_port = canny.inputs["src"]
        self.assertIn("blurred", src_port.accepted_states)

    def test_unifies_with_via_accepted_states(self):
        """Consumer with accepted_states unifies with producer producing any accepted state."""
        imwrite = self.orchestrator.loaded_cells.get("CV2_IMWRITE")
        image_consumer_port = imwrite.inputs["image"]
        q = image_consumer_port.signature.qualifiers

        # Exact match (color_bgr)
        producer_bgr = AlgebraicSignature("ndarray", "color_bgr", qualifiers=q)
        self.assertTrue(producer_bgr.unifies_with(image_consumer_port))

        # Accepted state: gray
        producer_gray = AlgebraicSignature("ndarray", "gray", qualifiers=q)
        self.assertTrue(producer_gray.unifies_with(image_consumer_port))

        # Accepted state: binary
        producer_binary = AlgebraicSignature("ndarray", "binary", qualifiers=q)
        self.assertTrue(producer_binary.unifies_with(image_consumer_port))

        # Accepted state: edge_map
        producer_edge = AlgebraicSignature("ndarray", "edge_map", qualifiers=q)
        self.assertTrue(producer_edge.unifies_with(image_consumer_port))

        # Also test with actual loaded cell outputs
        imread = self.orchestrator.loaded_cells.get("CV2_IMREAD")
        canny = self.orchestrator.loaded_cells.get("CV2_CANNY")
        self.assertTrue(imread.outputs["output_var"].signature.unifies_with(image_consumer_port))
        self.assertTrue(canny.outputs["output_var"].signature.unifies_with(image_consumer_port))

    def test_unifies_with_via_parent_state_walk_single_hop(self):
        """Producer with child state unifies with consumer expecting parent state (single-hop)."""
        erode = self.orchestrator.loaded_cells.get("CV2_ERODE")
        canny = self.orchestrator.loaded_cells.get("CV2_CANNY")
        self.assertIsNotNone(erode)
        self.assertIsNotNone(canny)

        # CV2_ERODE requires binary input
        erode_src = erode.inputs["src"]
        self.assertEqual(erode_src.signature.state, "binary")

        # CV2_CANNY outputs edge_map (parent_state -> binary)
        canny_out = canny.outputs["output_var"]
        self.assertEqual(canny_out.signature.state, "edge_map")

        # Single hop: edge_map -> parent_state is binary -> should unify
        self.assertTrue(canny_out.signature.unifies_with(erode_src))

    def test_unifies_with_via_parent_state_walk_multi_hop(self):
        """Producer with descendant state unifies with consumer expecting ancestor (multi-hop)."""
        canny = self.orchestrator.loaded_cells.get("CV2_CANNY")
        canny_out = canny.outputs["output_var"]  # edge_map

        # Consumer expecting gray (edge_map -> binary -> gray)
        gray_consumer = AlgebraicSignature("ndarray", "gray")
        self.assertTrue(canny_out.signature.unifies_with(gray_consumer))

    def test_unifies_with_fail_closed_on_unrelated_typestates(self):
        """Incompatible states without ancestor or accepted relation must be rejected."""
        erode = self.orchestrator.loaded_cells.get("CV2_ERODE")
        erode_src = erode.inputs["src"]  # binary

        # BGR image cannot directly satisfy binary consumer without conversion
        bgr_producer = AlgebraicSignature("ndarray", "color_bgr")
        self.assertFalse(bgr_producer.unifies_with(erode_src))

        # File path cannot satisfy ndarray consumer
        path_producer = AlgebraicSignature("str", "file_path")
        self.assertFalse(path_producer.unifies_with(erode_src))

    def test_robinson_unification_with_cv2_cells(self):
        """unify() function correctly handles accepted_states and parent_state walking."""
        canny = self.orchestrator.loaded_cells.get("CV2_CANNY")
        erode = self.orchestrator.loaded_cells.get("CV2_ERODE")
        imwrite = self.orchestrator.loaded_cells.get("CV2_IMWRITE")

        # 1. CANNY output (edge_map) unifies with ERODE input (binary)
        sigma1 = unify(canny.outputs["output_var"], erode.inputs["src"])
        self.assertIsNotNone(sigma1)

        # 2. CANNY output (edge_map) unifies with IMWRITE input (color_bgr with accepted_states)
        sigma2 = unify(canny.outputs["output_var"], imwrite.inputs["image"])
        self.assertIsNotNone(sigma2)

        # 3. Incompatible states produce None (bottom)
        imread = self.orchestrator.loaded_cells.get("CV2_IMREAD")
        self.assertIsNotNone(imread)
        sigma3 = unify(imread.outputs["output_var"], erode.inputs["src"])
        self.assertIsNone(sigma3)


if __name__ == "__main__":
    unittest.main()
