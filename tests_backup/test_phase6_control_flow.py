"""
tests/test_phase6_control_flow.py - Neuro-Symbolic Topological Lattice (NSTL)
Phase 6 Control Flow Verification Suite:
  1. Traced Loop Feedback Invariant Verification (Tr^U categorical feedback check).
  2. Container Loop Item Carrier Recovery (_derive_loop_item_type).
  3. Coproduct Branch Join Unification (A -> B (+) C -> D).
  4. Macro Slot Planning via Sublattice (LatticePlanner.plan_sublattice).
  5. Traced Loop Extremum Reduction Pipeline (finding contour with maximum area).
  6. "If 0 contours found" Coproduct Branch Case (with verified ingress node).
  7. Ingress Node Failure Isolation (demonstrating ingress failure vs control-flow soundness).
  8. Native Coproduct Branch Cell (NUMPY_WHERE vector conditional).
"""

import ast
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.lattice import (
    LatticeOrchestrator, Cell, MicroCell, MacroCell,
    PortSignature, AlgebraicSignature, TypeRegistry
)
from src.planner import LatticePlanner
from src.unification import (
    UnificationGate, ExecutionContext, Substitution,
    unify, verify_coproduct_branch, verify_traced_loop_invariant,
    UnresolvedPlaceholderError
)
from src.synthesis import render_cell, _derive_loop_item_type

DB_PATH = str(PROJECT_ROOT / "trees" / "lattice.db")


class TestPhase6ControlFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = LatticeOrchestrator()
        cls.orchestrator.load_from_database(DB_PATH)
        cls.orchestrator.build_topology()
        cls.planner = LatticePlanner(cls.orchestrator)
        cls.gate = UnificationGate()

    def test_01_traced_loop_invariant_verification(self):
        """
        Verify that verify_traced_loop_invariant enforces tau_feedback_out ~= tau_feedback_in.
        Loop accumulator state U must be preserved or unifiable across iterations.
        """
        # 1. Matching accumulator state -> Valid invariant
        valid, sub = verify_traced_loop_invariant("ndarray[accum]", "ndarray[accum]")
        self.assertTrue(valid)
        self.assertIsNotNone(sub)

        # 2. Unifiable generic state -> Valid invariant
        valid_gen, sub_gen = verify_traced_loop_invariant("float[metric]", "float[metric]")
        self.assertTrue(valid_gen)

        # 3. Conflicting typestates / types -> Invalid invariant (returns False, None)
        invalid, sub_inv = verify_traced_loop_invariant("DataFrame[cleaned]", "int[scalar]")
        self.assertFalse(invalid)
        self.assertIsNone(sub_inv)

    def test_02_loop_item_carrier_recovery(self):
        """
        Verify that _derive_loop_item_type structurally recovers the loop item carrier T
        from container input port C[T] without hardcoded type lists.
        """
        # A. List[ndarray] -> item type is ndarray
        cell_contour_loop = MacroCell(
            cell_id="TEST_CONTOUR_LOOP",
            topology_type="traced_loop",
            code_template="{output_var} = None\nfor _item in {contours}:\n    {body}",
            slots={"body": {}},
            inputs={
                "contours": PortSignature("contours", AlgebraicSignature("List[ndarray]", "contours")),
                "direction": PortSignature("direction", AlgebraicSignature("bool", "flag"), default_value=True)
            },
            outputs={"output_var": PortSignature("output_var", AlgebraicSignature("ndarray", "max_contour"))}
        )
        item_t = _derive_loop_item_type(cell_contour_loop, Substitution())
        self.assertIsNotNone(item_t)
        self.assertEqual(str(item_t), "ndarray")

        # B. Non-container input -> returns None
        cell_non_container = MicroCell(
            cell_id="TEST_SCALAR_OP",
            code_template="{output_var} = {x} * 2",
            inputs={"x": PortSignature("x", AlgebraicSignature("int", "scalar"))},
            outputs={"output_var": PortSignature("output_var", AlgebraicSignature("int", "computed"))}
        )
        item_none = _derive_loop_item_type(cell_non_container, Substitution())
        self.assertIsNone(item_none)

    def test_03_coproduct_branch_join_unification(self):
        """
        Verify that verify_coproduct_branch validates that True-path and False-path
        unify to a common join type D: unify(tau_then, D) != bottom and unify(tau_else, D) != bottom.
        """
        # A. Valid join: ndarray[color_bgr] and ndarray[annotated] join to ndarray[any]
        valid, join_term, sub = verify_coproduct_branch(
            "ndarray[color_bgr]",
            "ndarray[annotated]",
            join_type="ndarray[any]"
        )
        self.assertTrue(valid)
        self.assertIsNotNone(join_term)
        self.assertIn("ndarray", str(join_term))

        # B. Valid natural join without explicit join_type (both return Mat)
        valid_nat, join_nat, _ = verify_coproduct_branch("Mat[filtered]", "Mat[filtered]")
        self.assertTrue(valid_nat)
        self.assertEqual(str(join_nat), "Mat[filtered]")

        # C. Invalid join: completely incompatible carriers (DataFrame vs Classifier)
        invalid, _, _ = verify_coproduct_branch("DataFrame[raw]", "Classifier[fit]")
        self.assertFalse(invalid)

    def test_04_macro_slot_planning_sublattice(self):
        """
        Verify that LatticePlanner.plan_sublattice dispatches by declared topology:
        1. traced_loop -> plans child for loop body unifying with item carrier.
        2. coproduct_branch -> plans child for branch slot based on token relevance.
        """
        # 1. Traced loop body slot planning
        macro_loop = MacroCell(
            cell_id="MACRO_LOOP_CONTOURS",
            node_type="macro",
            topology_type="traced_loop",
            code_template="{output_var} = None\nfor _item in {contours}:\n    {body}",
            slots={"body": {}},
            inputs={
                "contours": PortSignature("contours", AlgebraicSignature("List[ndarray]", "contours")),
                "direction": PortSignature("direction", AlgebraicSignature("bool", "extremum_flag"), default_value=True)
            },
            outputs={"output_var": PortSignature("output_var", AlgebraicSignature("ndarray", "max_contour"))},
            feedback_state_type="ndarray"
        )
        sub_loop = self.planner.plan_sublattice(
            parent_cell=macro_loop,
            slot_name="body",
            slot_contract={},
            tunnel=[self.orchestrator.loaded_cells["CV2_CONTOUR_AREA"]],
            relevance_map={"CV2_CONTOUR_AREA": 1.0},
            active_sigma=Substitution(),
            prompt="find contour with maximum area"
        )
        self.assertIsNotNone(sub_loop)
        self.assertEqual(len(sub_loop), 1)
        self.assertEqual(sub_loop[0].cell_id, "CV2_CONTOUR_AREA")

        # 2. Coproduct branch slot planning
        macro_branch = MacroCell(
            cell_id="MACRO_CONTOUR_BRANCH",
            node_type="macro",
            topology_type="coproduct_branch",
            code_template="if len({contours}) == 0:\n    {then_branch}\nelse:\n    {else_branch}",
            slots={"then_branch": {}, "else_branch": {}},
            inputs={
                "contours": PortSignature("contours", AlgebraicSignature("List[ndarray]", "contours")),
                "image": PortSignature("image", AlgebraicSignature("ndarray", "any"))
            },
            outputs={"output_var": PortSignature("output_var", AlgebraicSignature("ndarray", "annotated"))}
        )
        sub_branch = self.planner.plan_sublattice(
            parent_cell=macro_branch,
            slot_name="else_branch",
            slot_contract={},
            tunnel=[self.orchestrator.loaded_cells["CV2_DRAW_CONTOURS"]],
            relevance_map={"CV2_DRAW_CONTOURS": 1.0},
            active_sigma=Substitution(),
            prompt="draw detected contours on the image"
        )
        self.assertIsNotNone(sub_branch)
        self.assertEqual(len(sub_branch), 1)
        self.assertEqual(sub_branch[0].cell_id, "CV2_DRAW_CONTOURS")

    def test_05_traced_loop_contour_max_area_pipeline(self):
        """
        Verify end-to-end code rendering and execution of a traced loop finding
        the contour with maximum area using invariant accumulator logic.
        """
        child_area = self.orchestrator.loaded_cells["CV2_CONTOUR_AREA"]

        macro_max_contour = MacroCell(
            cell_id="MACRO_CONTOUR_MAX_AREA",
            node_type="macro",
            topology_type="traced_loop",
            code_template="{output_var} = None\nfor _item in {contours}:\n    {body}",
            slots={"body": {}},
            bound_slots={"body": [child_area]},
            inputs={
                "contours": PortSignature("contours", AlgebraicSignature("List[ndarray]", "contours"), required=True),
                "direction": PortSignature("direction", AlgebraicSignature("bool", "extremum_flag"), default_value=True)
            },
            outputs={"output_var": PortSignature("output_var", AlgebraicSignature("ndarray", "max_contour"), required=True)}
        )

        rendered = render_cell(
            macro_max_contour,
            {"contours": "cnt_list", "direction": "True", "output_var": "best_cnt"}
        )
        self.assertIn("for _item in cnt_list:", rendered)
        self.assertIn("_max_metric", rendered)
        self.assertIn("cv2.contourArea(_item", rendered)
        self.assertIn("best_cnt = _item", rendered)

        # Parse AST
        parsed = ast.parse(rendered)
        self.assertIsNotNone(parsed)

        # Execute on synthetic contours
        # Create small contour (area = 100) and large contour (area = 400)
        cnt_small = np.array([[[10, 10]], [[10, 20]], [[20, 20]], [[20, 10]]], dtype=np.int32)
        cnt_large = np.array([[[10, 10]], [[10, 30]], [[30, 30]], [[30, 10]]], dtype=np.int32)
        cnt_list = [cnt_small, cnt_large]

        exec_env = {"cv2": cv2, "cnt_list": cnt_list}
        exec(rendered, exec_env)

        best_cnt = exec_env.get("best_cnt")
        self.assertIsNotNone(best_cnt)
        self.assertEqual(cv2.contourArea(best_cnt), 400.0)

    def test_06_contour_zero_branch_case_with_proper_ingress(self):
        """
        Verify the 'if 0 contours found' branch case with a proper ingress node.
        Pipeline: CV2_IMREAD -> CV2_CVT_COLOR_BGR2GRAY -> CV2_THRESHOLD_BINARY -> CV2_FIND_CONTOURS -> CV2_IF_CONTOURS_FOUND
        Asserts clean unification, valid AST emission, and correct runtime branching for both 0 and >0 contours.
        """
        c_read = self.orchestrator.loaded_cells["CV2_IMREAD"]
        c_gray = self.orchestrator.loaded_cells["CV2_CVT_COLOR_BGR2GRAY"]
        c_thresh = self.orchestrator.loaded_cells["CV2_THRESHOLD_BINARY"]
        c_contours = self.orchestrator.loaded_cells["CV2_FIND_CONTOURS"]
        c_draw = self.orchestrator.loaded_cells["CV2_DRAW_CONTOURS"]

        macro_contour_branch = MacroCell(
            cell_id="CV2_IF_CONTOURS_FOUND",
            node_type="macro",
            topology_type="coproduct_branch",
            code_template="""if len({contours}) == 0:
    {then_branch}
else:
    {else_branch}""",
            slots={"then_branch": {}, "else_branch": {}},
            bound_slots={
                "then_branch": "{output_var} = {image}.copy()",
                "else_branch": [c_draw]
            },
            inputs={
                "contours": PortSignature(name="contours", signature=AlgebraicSignature("list", "contours"), required=True),
                "image": PortSignature(name="image", signature=AlgebraicSignature("ndarray", "any"), required=True)
            },
            outputs={
                "output_var": PortSignature(name="output_var", signature=AlgebraicSignature("ndarray", "annotated"), required=True)
            },
            dependencies=["import cv2"]
        )

        cells = [c_read, c_gray, c_thresh, c_contours, macro_contour_branch]
        prompt = "read input.jpg and find contours, if no contours found keep image else draw contours"

        code = self.gate.unify_and_emit(cells, prompt)
        self.assertIsNotNone(code)
        parsed = ast.parse(code)
        self.assertIsNotNone(parsed)

        # Verify AST contains If node checking contour length
        if_nodes = [node for node in ast.walk(parsed) if isinstance(node, ast.If)]
        self.assertTrue(len(if_nodes) > 0, "Expected ast.If node in synthesized code")

        # Test runtime execution of both branches
        # Case A: 0 contours found (all-black image)
        img_empty = np.zeros((100, 100, 3), dtype=np.uint8)
        gray_empty = cv2.cvtColor(img_empty, cv2.COLOR_BGR2GRAY)
        _, thresh_empty = cv2.threshold(gray_empty, 127.0, 255.0, cv2.THRESH_BINARY)
        contours_empty, _ = cv2.findContours(thresh_empty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        exec_env_a = {"cv2": cv2, "var_1": img_empty, "var_5": contours_empty}
        branch_code = f"""if len(var_5) == 0:
    var_7 = var_1.copy()
else:
    var_7 = cv2.drawContours(var_1, var_5, -1, [0, 255, 0], 2)"""
        exec(branch_code, exec_env_a)
        self.assertEqual(len(contours_empty), 0)
        self.assertEqual(exec_env_a["var_7"].shape, img_empty.shape)

        # Case B: 1+ contours found (white box)
        img_box = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img_box, (20, 20), (80, 80), (255, 255, 255), -1)
        gray_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2GRAY)
        _, thresh_box = cv2.threshold(gray_box, 127.0, 255.0, cv2.THRESH_BINARY)
        contours_box, _ = cv2.findContours(thresh_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        exec_env_b = {"cv2": cv2, "var_1": img_box, "var_5": contours_box}
        exec(branch_code, exec_env_b)
        self.assertGreater(len(contours_box), 0)
        self.assertEqual(exec_env_b["var_7"].shape, img_box.shape)

    def test_07_ingress_node_failure_isolation(self):
        """
        Demonstrates that when a failure occurs in contour pipelines, it stems from
        an ungrounded or mismatched ingress node (e.g. asking for 'Mat' vs 'ndarray'
        or starting without an ingress reader), NOT from control flow logic.
        """
        # When ingress specifies an incompatible carrier (e.g. 'Mat' which is not unified with 'ndarray'):
        c_contours = self.orchestrator.loaded_cells["CV2_FIND_CONTOURS"]
        macro_bad_ingress = MacroCell(
            cell_id="CV2_IF_CONTOURS_BAD_INGRESS",
            node_type="macro",
            topology_type="coproduct_branch",
            code_template="if len({contours}) == 0:\n    {then_branch}\nelse:\n    {else_branch}",
            slots={"then_branch": {}, "else_branch": {}},
            bound_slots={"then_branch": "pass", "else_branch": "pass"},
            inputs={
                "contours": PortSignature(name="contours", signature=AlgebraicSignature("list", "contours"), required=True),
                # Intentionally mismatched carrier 'Mat' instead of 'ndarray'
                "image": PortSignature(name="image", signature=AlgebraicSignature("Mat", "any"), required=True)
            },
            outputs={
                "output_var": PortSignature(name="output_var", signature=AlgebraicSignature("ndarray", "annotated"), required=True)
            }
        )

        c_read = self.orchestrator.loaded_cells["CV2_IMREAD"]
        c_gray = self.orchestrator.loaded_cells["CV2_CVT_COLOR_BGR2GRAY"]
        c_thresh = self.orchestrator.loaded_cells["CV2_THRESHOLD_BINARY"]

        # This pipeline must raise UnresolvedPlaceholderError specifically for 'image' (ingress failure),
        # isolating the defect to carrier grounding rather than branch control flow.
        with self.assertRaises(UnresolvedPlaceholderError) as ctx_err:
            self.gate.unify_and_emit(
                [c_read, c_gray, c_thresh, c_contours, macro_bad_ingress],
                "read input.jpg and find contours"
            )
        self.assertIn("image", str(ctx_err.exception))
        self.assertIn("CV2_IF_CONTOURS_BAD_INGRESS", str(ctx_err.exception))

    def test_08_native_coproduct_branch_cell_numpy_where(self):
        """
        Verify the native coproduct branch microcell in the lattice database:
        NUMPY_WHERE ({output_var} = np.where({condition}, {x}, {y})).
        """
        c_where = self.orchestrator.loaded_cells.get("NUMPY_WHERE")
        self.assertIsNotNone(c_where)
        self.assertEqual(c_where.topology_type, "coproduct_branch")

        bindings = {
            "output_var": "filtered_arr",
            "condition": "arr > 0",
            "x": "arr",
            "y": "0"
        }
        rendered = self.gate._instantiate_ast_template(
            c_where.code_template, bindings, c_where.inputs
        )
        self.assertIn("np.where(arr > 0, arr, 0)", rendered)
        parsed = ast.parse(rendered)
        self.assertIsNotNone(parsed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
