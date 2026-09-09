"""
benchmarks/test_universal_control_flow.py - Neuro-Symbolic Topological Lattice (NSTL)
Universal Control Flow & Categorical Computation Topologies Benchmark.

Tests the 4 universal topologies of Traced Symmetric Monoidal Categories:
  1. Sequential Composition (∘): Flat linear pipelines (A -> B)
  2. Monoidal Product & Port Sharing (⊗, Δ): Multi-argument calls & variable reuse
  3. Coproduct Branching (⊕): Conditionals and branching (if/else)
  4. Traced Feedback (Tr^U): Loops and aggregations with invariant accumulator state U

Plus End-to-End Synthesis and Execution in GEVRSandbox:
  - Test A: Vision + Loop Reduction (imread -> findContours -> loop min contourArea -> drawContours)
  - Test B: Tabular + Filter Comprehension (read_csv -> query('value > 100') -> to_csv)
"""

import ast
import os
import shutil
import tempfile
import pytest
import numpy as np
import cv2
import pandas as pd

from src.lattice import LatticeOrchestrator, Cell, PortSignature, AlgebraicSignature, TypeRegistry
from src.unification import (
    unify,
    Substitution,
    GenericTypeTerm,
    TypeVariable,
    AtomicType,
    substitute_generics,
    verify_traced_loop_invariant,
    verify_coproduct_branch,
    UnificationGate,
    ExecutionContext,
)
from src.router import LatticeRouter
from src.gevr_sandbox import GEVRSandbox


# =====================================================================
# 1. Unit Tests for Category Theory Topologies
# =====================================================================

class TestUniversalCategoryTopologies:
    """Mathematical verification of Traced Symmetric Monoidal Categories."""

    def test_sequential_composition(self):
        """Sequential Composition (∘): Hom(A, B) x Hom(B, C) -> Hom(A, C)."""
        sig_a = AlgebraicSignature(type_name="MatLike", state="raw")
        sig_b = AlgebraicSignature(type_name="MatLike", state="grayscale")
        sig_c = AlgebraicSignature(type_name="Sequence[MatLike]", state="contours")

        sub1 = unify(sig_a, sig_a)
        assert sub1 is not None

        sub2 = unify(sig_b, sig_b, sub1)
        assert sub2 is not None

        sub3 = unify(sig_c, sig_c, sub2)
        assert sub3 is not None

        # Bottom on invalid types
        assert unify("MatLike", "DataFrame") is None

    def test_polymorphic_generic_container_unification(self):
        """Unification with Generic Type Variables and Containers: Sequence[T], List[T]."""
        sub = Substitution()
        t_seq = GenericTypeTerm("Sequence", (TypeVariable("T"),))
        concrete_seq = GenericTypeTerm("Sequence", (AtomicType("MatLike"),))

        res_sub = unify(concrete_seq, t_seq, sub)
        assert res_sub is not None
        assert "T" in res_sub.mappings
        assert str(res_sub.mappings["T"]) == "MatLike"

        # Substitute generics in child port
        child_port = PortSignature(
            name="contour",
            type_name="T",
            signature=AlgebraicSignature(type_name="T", state="any"),
            required=True
        )
        concrete_port = substitute_generics(child_port, res_sub)
        assert concrete_port.type_name == "MatLike"

    def test_monoidal_product_and_port_sharing(self):
        """Monoidal Product & Port Sharing (⊗, Δ): Multi-argument call reusing in-scope variables."""
        gate = UnificationGate()
        ctx = ExecutionContext()

        # Wire producer var_1 (MatLike image)
        sig_img = PortSignature(
            name="image",
            type_name="MatLike",
            signature=AlgebraicSignature(type_name="MatLike", state="raw"),
            required=True
        )
        ctx.declare_variable("var_1", sig_img, "var_1")

        # Wire producer var_3 (MatLike contour)
        sig_cnt = PortSignature(
            name="contour",
            type_name="MatLike",
            signature=AlgebraicSignature(type_name="MatLike", state="contour"),
            required=True
        )
        ctx.declare_variable("var_3", sig_cnt, "var_3")

        # Consumer cell requiring both image and contours
        consumer_cell = Cell(
            cell_id="TEST_DRAW",
            stage=2,
            inputs={
                "image": PortSignature(
                    name="image",
                    type_name="MatLike",
                    signature=AlgebraicSignature(type_name="MatLike", state="any"),
                    required=True
                ),
                "contours": PortSignature(
                    name="contours",
                    type_name="Sequence[MatLike] | MatLike",
                    signature=AlgebraicSignature(type_name="Sequence[MatLike] | MatLike", state="any"),
                    required=True
                ),
                "color": PortSignature(
                    name="color",
                    type_name="Scalar",
                    signature=AlgebraicSignature(type_name="Scalar", state="color"),
                    required=False,
                    default_value="(0, 255, 0)"
                )
            },
            outputs={
                "output_data": PortSignature(
                    name="output_data",
                    type_name="MatLike",
                    signature=AlgebraicSignature(type_name="MatLike", state="annotated"),
                    required=True
                )
            },
            slots={},
            code_template="{output_var} = draw({image}, {contours}, {color})",
            dependencies=[],
            keywords=["draw"],
            semantic_tags=["draw"],
            docstring="Draws contours on image",
            domain_name="test",
            node_type="transform",
            node_role="transform",
            topology_type="monoidal_product"
        )

        res = gate.unify_pipeline([consumer_cell], ctx)
        assert not res.is_bottom()
        pipeline_bindings = res.value
        assert len(pipeline_bindings) == 1
        _, bindings = pipeline_bindings[0]
        # Port sharing: both required ports are satisfied without collision from in-scope variables
        assert {bindings["image"], bindings["contours"]} == {"var_1", "var_3"}

    def test_coproduct_branching(self):
        """Coproduct Branching (⊕): verify_coproduct_branch ensures joins unify with target carrier."""
        then_type = "MatLike"
        else_type = "MatLike"
        join_carrier = "MatLike"
        valid, join_type, sub = verify_coproduct_branch(then_type, else_type, join_carrier)
        assert valid is True
        assert join_type is not None
        assert sub is not None

        # Incompatible branch output types
        valid_bad, _, _ = verify_coproduct_branch("DataFrame", "int", "DataFrame")
        assert valid_bad is False

    def test_traced_feedback_loop_invariant(self):
        """Traced Feedback (Tr^U): verify_traced_loop_invariant ensures invariant preservation."""
        # Loop accumulator state U = MatLike
        u_in = "MatLike"
        u_out = "MatLike"
        valid, sub = verify_traced_loop_invariant(u_in, u_out)
        assert valid is True
        assert sub is not None

        # Incompatible feedback loop state
        valid_bad, _ = verify_traced_loop_invariant("MatLike", "DataFrame")
        assert valid_bad is False


# =====================================================================
# 2. End-to-End Control-Flow Benchmarks
# =====================================================================

class TestUniversalControlFlowEndToEnd:
    """End-to-End synthesis, AST validation, and GEVRSandbox execution."""

    @pytest.fixture
    def nstl_system(self):
        orch = LatticeOrchestrator()
        orch.load_from_database()
        router = LatticeRouter(orch)
        gate = UnificationGate()
        sandbox = GEVRSandbox()
        return orch, router, gate, sandbox

    def test_e2e_vision_loop_reduction(self, nstl_system, tmp_path):
        """
        Test A: Vision + Loop Reduction
        Prompt: 'Load image, find contours, loop over them to find the contour with the minimum area, and draw it.'
        Topology: Sequential + Traced Feedback Loop (cv2.contourArea) + Monoidal Port Sharing (drawContours).
        """
        orch, router, gate, sandbox = nstl_system

        # 1. Create a synthetic test image with two distinct contours:
        # One small rectangle (area 400), one large rectangle (area 2500)
        work_dir = str(tmp_path)
        img_path = os.path.join(work_dir, "input.png")
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (10, 10), (30, 30), (255, 255, 255), -1)   # Small: 20x20 = 400
        cv2.rectangle(test_img, (60, 60), (110, 110), (255, 255, 255), -1) # Large: 50x50 = 2500
        cv2.imwrite(img_path, test_img)

        prompt = "Load image, find contours, loop over them to find the contour with the minimum area, and draw it."

        # 2. Plan path through semantic tunnel
        path = router.plan_path(prompt, return_tuple=False)
        cell_ids = [c.cell_id for c in path]

        # Verify topological structure
        assert "CV2_IMREAD" in cell_ids
        assert "CV2_FINDCONTOURS" in cell_ids
        assert "PYTHON_FOR_REDUCE_ACCUMULATOR" in cell_ids
        assert "CV2_DRAWCONTOURS" in cell_ids

        # Verify sub-lattice slot binding inside loop
        loop_cell = next(c for c in path if c.cell_id == "PYTHON_FOR_REDUCE_ACCUMULATOR")
        assert "loop_body" in loop_cell.bound_slots
        sub_slot_ids = [c.cell_id for c in loop_cell.bound_slots["loop_body"]]
        assert "CV2_CONTOURAREA" in sub_slot_ids

        # 3. Emit block-structured code
        ctx = ExecutionContext(prompt=prompt)
        code = gate.emit_code(path, context=ctx)
        assert code and len(code) > 0

        # 4. AST Structure Verification
        tree = ast.parse(code)
        loop_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
        assert len(loop_nodes) >= 1, "Code must contain an indented for-loop AST node"
        assert "cv2.findContours" in code
        assert "cv2.contourArea" in code
        assert "cv2.drawContours" in code

        # 5. GEVRSandbox Execution
        res = sandbox.execute(code, cwd=work_dir)
        assert res["success"] is True, f"Execution failed in sandbox: {res.get('error')}"

    def test_e2e_tabular_filter_comprehension(self, nstl_system, tmp_path):
        """
        Test B: Tabular + Filter Comprehension
        Prompt: 'Read CSV data, filter rows where column value > 100, and save results.'
        Topology: Sequential + Predicate Grounding + Sink Terminal Export.
        """
        orch, router, gate, sandbox = nstl_system

        work_dir = str(tmp_path)
        csv_in_path = os.path.join(work_dir, "input.csv")
        csv_out_path = os.path.join(work_dir, "output.csv")

        # 1. Create synthetic dataset with known values
        raw_df = pd.DataFrame({"value": [45, 120, 85, 250, 100, 310]})
        raw_df.to_csv(csv_in_path, index=False)

        prompt = "Read CSV data, filter rows where column value > 100, and save results."

        # 2. Plan path through semantic tunnel
        path = router.plan_path(prompt, return_tuple=False)
        cell_ids = [c.cell_id for c in path]

        # Verify topological structure
        assert "PANDAS_READ_CSV" in cell_ids
        assert "PANDAS_DATAFRAME_QUERY" in cell_ids
        assert "PANDAS_DATAFRAME_TO_CSV" in cell_ids

        # 3. Emit code
        ctx = ExecutionContext(prompt=prompt)
        code = gate.emit_code(path, context=ctx)
        assert code and len(code) > 0

        # 4. AST Structure Verification
        assert "pandas.read_csv" in code
        assert ".query(" in code or "value > 100" in code
        assert ".to_csv(" in code

        # 5. GEVRSandbox Execution with egress artifact verification
        res = sandbox.execute(code, egress_paths=[csv_out_path], cwd=work_dir)
        assert res["success"] is True, f"Execution failed in sandbox: {res.get('error')}"

        # 6. Non-vacuity & Functional Correctness Verification
        assert os.path.exists(csv_out_path), "Filtered CSV output artifact must exist"
        result_df = pd.read_csv(csv_out_path)
        assert len(result_df) == 3, f"Expected 3 rows with value > 100, found {len(result_df)}"
        assert sorted(result_df["value"].tolist()) == [120, 250, 310]
