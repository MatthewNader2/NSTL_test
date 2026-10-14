"""
tests/test_unified_architecture.py
Verifies that all remaining items from the Architecture Audit (Issues 1-5 & Proposals 4.A-4.C)
are solved and operational.
"""

import numpy as np
import pytest

from src.lattice import MicroCell, PortSignature, AlgebraicSignature, TypeRegistry
from src.unification import VerificationContract
from src.dag_plan import DAGPlan, OpenPortFrontier, SemanticBlackboard, TypedHoleCell
from src.algebraic_effects import (
    PropertyBundle,
    infer_cell_algebraic_effects,
    build_universal_verification_contract,
)
from src.experiments import (
    EarlyExitLatentProbe,
    SparseConceptCoding,
    AbsorbingMarkovRouter,
    SinkhornPortMatcher,
)


def test_issue_1_and_4b_universal_algebraic_effects_domain_agnostic():
    """Issue 1 & Proposal 4.B: Verify universal linear mutation and partition coherence laws."""
    # 1. Custom domain splitter (e.g. Graph partitioner in NetworkX - not sklearn!)
    splitter = MicroCell(
        cell_id="GRAPH_SPLIT",
        stage=2,
        inputs={"g": PortSignature("g", AlgebraicSignature("Graph", "raw"))},
        outputs={
            "g_train": PortSignature("g_train", AlgebraicSignature("Graph", "split_train"), port_role="feature_input"),
            "g_test": PortSignature("g_test", AlgebraicSignature("Graph", "split_test"), port_role="feature_input"),
        },
    )
    # 2. Custom in-place mutator (e.g. Graph node attribute annotator)
    annotator = MicroCell(
        cell_id="GRAPH_ANNOTATE",
        stage=2,
        mutation_type="in_place",
        inputs={"g": PortSignature("g", AlgebraicSignature("Graph", "any"), port_role="receiver")},
        outputs={"g_out": PortSignature("g_out", AlgebraicSignature("Graph", "annotated"))},
    )
    # 3. Custom sink
    exporter = MicroCell(
        cell_id="GRAPH_EXPORT",
        stage=3,
        inputs={
            "g": PortSignature("g", AlgebraicSignature("Graph", "any")),
            "path": PortSignature("path", AlgebraicSignature("filepath", "dest_identifier")),
        },
        outputs={},
    )

    eff_split = infer_cell_algebraic_effects(splitter)
    assert eff_split["out:g_train"].partition_index == 0
    assert eff_split["out:g_test"].partition_index == 1

    eff_ann = infer_cell_algebraic_effects(annotator)
    assert eff_ann["out:g_out"].mutates_port == "g"

    pipeline_bindings = [
        (splitter, {"g": "var_0", "g_train": "var_1", "g_test": "var_2"}),
        (annotator, {"g": "var_1", "g_out": "var_3", "output_var": "var_3"}),
        (exporter, {"g": "var_3", "path": "'out.gml'"}),
    ]
    contract = build_universal_verification_contract(pipeline_bindings, VerificationContract)
    egress_checks = [c for c in contract.terminal_checks if c["type"] == "algebraic_linear_egress"]
    assert len(egress_checks) == 1
    assert egress_checks[0]["expected_var"] == "var_3"
    assert egress_checks[0]["actual_var"] == "var_3"


def test_issue_2_dag_plan_is_first_class_list_subclass():
    """Issue 2: DAGPlan subclasses list so all list operations work while preserving port edges."""
    c1 = MicroCell(
        cell_id="INGRESS_A",
        stage=1,
        inputs={},
        outputs={"out_a": PortSignature("out_a", AlgebraicSignature("ndarray", "raw"))},
    )
    c2 = MicroCell(
        cell_id="TRANSFORM_B",
        stage=2,
        inputs={"in_b": PortSignature("in_b", AlgebraicSignature("ndarray", "raw"))},
        outputs={"out_b": PortSignature("out_b", AlgebraicSignature("ndarray", "clean"))},
    )
    dag = OpenPortFrontier.build_dag_plan([c1, c2])
    assert isinstance(dag, list)
    assert isinstance(dag, DAGPlan)
    assert len(dag) == 2
    assert dag[0].cell_id == "INGRESS_A"
    assert len(dag.detailed_edges) == 1
    assert dag.detailed_edges[0].src_port == "out_a"
    assert dag.detailed_edges[0].dst_port == "in_b"
    assert c2.planned_port_wires["in_b"][1] == "out_a"


def test_issue_3_semantic_blackboard_locks_neural_authority():
    """Issue 3: SemanticBlackboard locks neural extractions against heuristic overwrites."""
    bb = SemanticBlackboard()
    bb.lock_from_intent({
        "target_col": "revenue",
        "columns": ["age", "tenure"],
        "slots": {"PD_READ": {"filepath": "sales.csv"}},
    })
    assert bb.locked_by_neural is True
    assert bb.target_sink == "revenue"
    assert bb.feature_columns == ["age", "tenure"]
    assert bb.cell_slot_locks["PD_READ"]["filepath"] == "sales.csv"


def test_issue_4_heyting_property_bundle_accumulates_stacked_transforms():
    """Issue 4: PropertyBundle accumulates stacked states and predicates without overwriting."""
    b0 = PropertyBundle(carrier_type="DataFrame", active_states=frozenset({"raw"}))
    b1 = b0.transition(new_state="cleaned", added_predicates=[("has_nans", "==", False)])
    b2 = b1.transition(new_state="normalized", added_predicates=[("is_normalized", "==", True)])

    # Both 'cleaned' and 'normalized' states and predicates remain active simultaneously!
    assert b2.entails("cleaned", [("has_nans", "==", False)])
    assert b2.entails("normalized", [("is_normalized", "==", True)])


def test_issue_5_typed_hole_cell_instantiates_open_world_bridge():
    """Issue 5: TypedHoleCell bridges arbitrary type gaps mid-search."""
    hole = TypedHoleCell(
        hole_id="HOLE_CUSTOM_SCORE",
        task_description="Compute z-score normalized array",
        in_sig=AlgebraicSignature("ndarray", "raw"),
        out_sig=AlgebraicSignature("ndarray", "normalized"),
    )
    assert hole.is_typed_hole is True
    assert hole.primary_input.type_name == "ndarray"
    assert hole.primary_output.state == "normalized"


def test_proposal_4c_sinkhorn_optimal_transport_and_sparse_coding():
    """Proposal 4.C & Experiments: Verify Sinkhorn-Knopp OT port matcher and Sparse Concept Coding."""
    port_embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    tok_embs = np.array([[0.95, 0.05, 0.0], [0.05, 0.95, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    matched = SinkhornPortMatcher.match_tokens_to_ports(
        token_embeddings=tok_embs,
        port_embeddings=port_embs,
        tokens=["age", "50", "noise"],
        port_names=["column", "threshold"],
    )
    assert matched["column"] == "age"
    assert matched["threshold"] == "50"

    # Sparse Autoencoder OMP decomposition
    D = np.eye(4, dtype=np.float32)
    compound_prompt = (0.7 * D[0] + 0.7 * D[2]).astype(np.float32)
    sparse_acts = SparseConceptCoding.decompose_prompt_embedding(
        compound_prompt, D, ["CELL_A", "CELL_B", "CELL_C", "CELL_D"], top_k=2
    )
    assert set(sparse_acts.keys()) == {"CELL_A", "CELL_C"}
