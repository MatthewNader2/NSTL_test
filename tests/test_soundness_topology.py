"""
tests/test_soundness_topology.py
PR 1 Regression Test: Full Dual-Layer Type System Soundness and Edge Formation.

Asserts:
  1. No directed edge (u -> v) in LatticeOrchestrator._adjacency fails unify_pipeline's wire check
     (is_edge_valid).
  2. color_bgr -> CV2_CANNY is rejected (never in _adjacency).
  3. gray -> CV2_GAUSSIAN_BLUR is accepted (positive control).
  4. gray -> CV2_CANNY is accepted (positive control).
  5. 1-to-1 dataflow precondition bridge candidate qualification with config ports.
"""

from src.lattice import LatticeOrchestrator
from src.unification import is_edge_valid


def test_soundness_topology_no_invalid_adjacency_edges():
    orch = LatticeOrchestrator()

    # 1. Negative control / Named case: No cell producing color_bgr can connect to CV2_CANNY
    bgr_to_canny = []
    for u_id, neighbors in orch._adjacency.items():
        u = orch.loaded_cells[u_id]
        if u.primary_output and str(getattr(u.primary_output, "state", "")).lower() == "color_bgr":
            if "CV2_CANNY" in neighbors:
                bgr_to_canny.append(u_id)
    assert not bgr_to_canny, f"Soundness violation: color_bgr connected to CV2_CANNY: {bgr_to_canny}"

    # 2. Positive control: CV2_CVT_COLOR_BGR2GRAY (gray) connects to CV2_GAUSSIAN_BLUR
    assert "CV2_GAUSSIAN_BLUR" in orch._adjacency.get("CV2_CVT_COLOR_BGR2GRAY", []), (
        "Positive control failed: CV2_CVT_COLOR_BGR2GRAY -> CV2_GAUSSIAN_BLUR missing from _adjacency"
    )

    # 3. Positive control: CV2_CVT_COLOR_BGR2GRAY (gray) connects to CV2_CANNY
    assert "CV2_CANNY" in orch._adjacency.get("CV2_CVT_COLOR_BGR2GRAY", []), (
        "Positive control failed: CV2_CVT_COLOR_BGR2GRAY -> CV2_CANNY missing from _adjacency"
    )

    # 4. Universal check: Every single edge in _adjacency must pass is_edge_valid
    invalid_edges = []
    for u_id, neighbors in orch._adjacency.items():
        u = orch.loaded_cells[u_id]
        u_out = u.primary_output
        for v_id in neighbors:
            v = orch.loaded_cells[v_id]
            v_in = v.primary_input
            if not is_edge_valid(u_out, v_in, producer_cell=u, consumer_cell=v):
                invalid_edges.append((u_id, v_id))

    assert not invalid_edges, f"Found {len(invalid_edges)} invalid edges in _adjacency: {invalid_edges[:10]}"


def test_bridge_candidate_qualification_with_config_ports():
    orch = LatticeOrchestrator()
    bgr2gray = orch.loaded_cells.get("CV2_CVT_COLOR_BGR2GRAY")
    assert bgr2gray is not None
    assert orch._is_valid_bridge_candidate(bgr2gray), "CV2_CVT_COLOR_BGR2GRAY must qualify as a bridge candidate"


if __name__ == "__main__":
    test_soundness_topology_no_invalid_adjacency_edges()
    test_bridge_candidate_qualification_with_config_ports()
    print("All PR 1 regression tests passed successfully!")
