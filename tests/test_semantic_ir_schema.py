# tests/test_semantic_ir_schema.py
import unittest
import json
import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from schema import (
    ConditionPredicate,
    EdgeSchema,
    TypestateDefinition,
    TypestateVocabularySchema,
    PortSchema,
    CellSchema,
    TreeSchema,
)
from lattice import MicroCell, MacroCell, Cell, LatticeOrchestrator


class TestSemanticIRSchema(unittest.TestCase):
    """Test suite for Phase 1 Semantic IR schema extension."""

    def test_condition_predicate_string_parsing(self):
        """ConditionPredicate parses expressions like 'channels == 1', 'dtype == binary'."""
        cp1 = ConditionPredicate.from_any("channels == 1")
        self.assertEqual(cp1.property, "channels")
        self.assertEqual(cp1.operator, "==")
        self.assertEqual(cp1.value, 1)

        cp2 = ConditionPredicate.from_any("dtype == 'binary'")
        self.assertEqual(cp2.property, "dtype")
        self.assertEqual(cp2.operator, "==")
        self.assertEqual(cp2.value, "binary")

        cp3 = ConditionPredicate.from_any("is_fitted == True")
        self.assertEqual(cp3.property, "is_fitted")
        self.assertEqual(cp3.operator, "==")
        self.assertIs(cp3.value, True)

        cp4 = ConditionPredicate.from_any("ndim >= 2")
        self.assertEqual(cp4.property, "ndim")
        self.assertEqual(cp4.operator, ">=")
        self.assertEqual(cp4.value, 2)

    def test_condition_predicate_dict_parsing(self):
        """ConditionPredicate accepts structured dicts or single key-value mappings."""
        cp_dict = ConditionPredicate.from_any({
            "target": "input_image",
            "property": "channels",
            "operator": "==",
            "value": 3,
            "description": "Requires RGB image"
        })
        self.assertEqual(cp_dict.target, "input_image")
        self.assertEqual(cp_dict.property, "channels")
        self.assertEqual(cp_dict.value, 3)

        cp_kv = ConditionPredicate.from_any({"channels": 1})
        self.assertEqual(cp_kv.property, "channels")
        self.assertEqual(cp_kv.value, 1)

    def test_edge_schema_creation_and_normalization(self):
        """EdgeSchema captures target_cell_id, affinity_score, and bridging_precondition."""
        edge = EdgeSchema(
            target_cell_id="CV2_CANNY",
            affinity_score=0.85,
            bridging_precondition="channels == 1",
            metadata={"domain_bridge": "vision_filtering"}
        )
        self.assertEqual(edge.target_cell_id, "CV2_CANNY")
        self.assertEqual(edge.affinity_score, 0.85)
        self.assertIsInstance(edge.bridging_precondition, ConditionPredicate)
        self.assertEqual(edge.bridging_precondition.property, "channels")
        self.assertEqual(edge.bridging_precondition.value, 1)
        self.assertEqual(edge.metadata["domain_bridge"], "vision_filtering")

    def test_typestate_vocabulary_schema_dynamic_domains(self):
        """Domain-level typestate vocabulary supports arbitrary domains without hardcoding."""
        # cv2 typestate example: color/gray/binary/edge-map
        cv2_vocab = TypestateVocabularySchema(
            domain="cv2",
            states=[
                TypestateDefinition(name="color", properties={"channels": 3}),
                TypestateDefinition(name="gray", properties={"channels": 1}),
                TypestateDefinition(name="binary", properties={"channels": 1, "dtype": "bool"}),
                TypestateDefinition(name="edge-map", properties={"channels": 1})
            ],
            transitions=[
                {"from": "color", "to": "gray", "via": "cv2.cvtColor"},
                {"from": "gray", "to": "binary", "via": "cv2.threshold"},
                {"from": "gray", "to": "edge-map", "via": "cv2.Canny"}
            ]
        )
        self.assertEqual(cv2_vocab.domain, "cv2")
        self.assertEqual(len(cv2_vocab.states), 4)
        self.assertEqual(cv2_vocab.states[0].properties["channels"], 3)

        # pandas typestate example: raw/deduped/indexed/numeric-only
        pandas_vocab = TypestateVocabularySchema(
            domain="pandas",
            states=["raw", "deduped", "indexed", "numeric-only"]
        )
        self.assertEqual(pandas_vocab.domain, "pandas")
        self.assertEqual(pandas_vocab.states, ["raw", "deduped", "indexed", "numeric-only"])

        # sklearn typestate example: unfit/fit/transformed
        sklearn_vocab = TypestateVocabularySchema(
            domain="sklearn",
            states=["unfit", "fit", "transformed"],
            initial_state="unfit"
        )
        self.assertEqual(sklearn_vocab.domain, "sklearn")
        self.assertEqual(sklearn_vocab.initial_state, "unfit")

        # custom domain (demonstrating no hardcoding constraint)
        custom_vocab = TypestateVocabularySchema(
            domain="audio_dsp",
            states=["waveform", "spectrogram", "mel_features"],
            initial_state="waveform"
        )
        self.assertEqual(custom_vocab.domain, "audio_dsp")

    def test_cell_schema_backward_compatibility(self):
        """Existing cells without preconditions, effects, or edges parse with default empty lists."""
        cell = CellSchema(
            cell_id="TEST_CELL",
            stage=2,
            code_template="{output_var} = {input_var}.copy()"
        )
        self.assertEqual(cell.preconditions, [])
        self.assertEqual(cell.postconditions, [])
        self.assertEqual(cell.effects, [])
        self.assertEqual(cell.edges, [])

    def test_cell_schema_preconditions_and_effects_normalization(self):
        """CellSchema normalizes strings, dicts, and ConditionPredicates into preconditions and effects."""
        cell = CellSchema(
            cell_id="CV2_CANNY_CELL",
            stage=2,
            code_template="{output_var} = cv2.Canny({input_var}, 100, 200)",
            preconditions=["channels == 1", {"dtype": "uint8"}],
            effects=["state == 'edge-map'", {"channels": 1}],
            edges=[
                {
                    "target_cell_id": "CV2_FIND_CONTOURS",
                    "affinity_score": 0.9,
                    "bridging_precondition": "dtype == 'uint8'"
                }
            ]
        )
        self.assertEqual(len(cell.preconditions), 2)
        self.assertEqual(cell.preconditions[0].property, "channels")
        self.assertEqual(cell.preconditions[0].value, 1)
        self.assertEqual(cell.preconditions[1].property, "dtype")
        self.assertEqual(cell.preconditions[1].value, "uint8")

        # Check effects and postconditions synchronization
        self.assertEqual(len(cell.effects), 2)
        self.assertEqual(len(cell.postconditions), 2)
        self.assertEqual(cell.effects[0].property, "state")
        self.assertEqual(cell.effects[0].value, "edge-map")

        # Check edges
        self.assertEqual(len(cell.edges), 1)
        self.assertEqual(cell.edges[0].target_cell_id, "CV2_FIND_CONTOURS")
        self.assertEqual(cell.edges[0].affinity_score, 0.9)
        self.assertEqual(cell.edges[0].bridging_precondition.property, "dtype")
        self.assertEqual(cell.edges[0].bridging_precondition.value, "uint8")

    def test_tree_schema_with_typestates_and_cells(self):
        """TreeSchema parses full trees containing typestates and extended cells."""
        tree_dict = {
            "domain": "cv2",
            "version": "1.1.0",
            "typestates": {
                "domain": "cv2",
                "states": ["color", "gray", "binary", "edge-map"],
                "initial_state": "color"
            },
            "cells": [
                {
                    "cell_id": "CV2_CVT_COLOR_GRAY",
                    "stage": 2,
                    "code_template": "{output_var} = cv2.cvtColor({image}, cv2.COLOR_BGR2GRAY)",
                    "preconditions": ["channels == 3"],
                    "effects": ["channels == 1", "state == 'gray'"],
                    "edges": [
                        {
                            "target_cell_id": "CV2_CANNY",
                            "affinity_score": 0.95,
                            "bridging_precondition": "channels == 1"
                        }
                    ]
                }
            ]
        }
        tree = TreeSchema(**tree_dict)
        self.assertEqual(tree.domain, "cv2")
        self.assertIsNotNone(tree.typestates)
        self.assertIsInstance(tree.typestates, TypestateVocabularySchema)
        self.assertEqual(tree.typestates.states, ["color", "gray", "binary", "edge-map"])
        self.assertEqual(len(tree.cells), 1)
        self.assertEqual(tree.cells[0].cell_id, "CV2_CVT_COLOR_GRAY")
        self.assertEqual(tree.cells[0].preconditions[0].property, "channels")

    def test_existing_fixture_backward_compatibility(self):
        """Existing fixture phase1_micro_lattice.json parses cleanly through CellSchema."""
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "phase1_micro_lattice.json")
        if os.path.exists(fixture_path):
            with open(fixture_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cells = [CellSchema(**c) for c in data]
            self.assertGreater(len(cells), 0)
            for c in cells:
                self.assertIsInstance(c.preconditions, list)
                self.assertIsInstance(c.effects, list)
                self.assertIsInstance(c.edges, list)

    def test_lattice_cell_slots_and_attributes(self):
        """Cell in lattice.py accepts and stores preconditions, effects, edges in __slots__."""
        cell = MicroCell(
            cell_id="TEST_LATTICE_CELL",
            stage=2,
            preconditions=["channels == 1"],
            effects=["state == 'edge-map'"],
            edges=[{"target_cell_id": "TARGET_1", "affinity_score": 0.8}]
        )
        self.assertEqual(len(cell.preconditions), 1)
        self.assertEqual(len(cell.effects), 1)
        self.assertEqual(len(cell.postconditions), 1)
        self.assertEqual(len(cell.edges), 1)
        self.assertEqual(cell.edges[0]["target_cell_id"], "TARGET_1")


if __name__ == "__main__":
    unittest.main()
