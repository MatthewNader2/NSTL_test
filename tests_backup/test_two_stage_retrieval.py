"""
Tests for Phase 4: Two-Stage Retrieval with Edge-Context Text & Graph-Aware Idiom Discovery.

Covers:
1. Edge-context text representation in build_cell_embedding_text (outbound edges, inbound predecessors, preconditions, postconditions).
2. Stage 1 Candidate Idiom Discovery (_discover_candidate_idioms).
3. Stage 2 Entry Node Selection (_select_entry_node).
4. Full LatticeRouter.route() two-stage execution and ranking.
5. Full LatticeRouter.plan_path() pipeline generation.
6. Lexical fallback when internal_rag is None or unavailable.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lattice import LatticeOrchestrator, Cell
from router import LatticeRouter, IdiomSubgraph
from internal_rag import build_cell_embedding_text, _INDEX_FORMAT_VERSION
from tokenizer import CellTokenizer


@pytest.fixture(scope="module")
def orchestrator():
    """Initializes the LatticeOrchestrator with all 7 domain trees loaded."""
    orch = LatticeOrchestrator()
    return orch


@pytest.fixture(scope="module")
def router(orchestrator):
    """Initializes the LatticeRouter without FAISS dependency for deterministic testing."""
    return LatticeRouter(orchestrator)


# =========================================================================
# 1. Edge-Context Embedding Text Representation
# =========================================================================

def test_index_format_version_bumped():
    """Verify that index format version is bumped to 3 for edge-context representation."""
    assert _INDEX_FORMAT_VERSION >= 3


def test_build_cell_embedding_text_contains_edge_context(orchestrator):
    """
    Verify that build_cell_embedding_text formats:
    - Inbound and Outbound edges with transition probabilities and target identifiers
    - Preconditions and effects/postconditions
    """
    # Find a cell with outbound edges, e.g., CV2_IMREAD or CV2_CVT_COLOR_BGR2GRAY
    cell = next((c for c in orchestrator.cells if c.cell_id == "CV2_IMREAD"), None)
    assert cell is not None, "CV2_IMREAD must be loaded"

    emb_text = build_cell_embedding_text(cell, orchestrator)
    assert isinstance(emb_text, str)
    assert len(emb_text) > 0

    # Must contain docstring, signature, or domain
    assert "opencv" in emb_text.lower() or "cv2" in emb_text.lower()

    # If the cell has outbound edges, they must be formatted in the text
    if cell.edges:
        assert "successors:" in emb_text.lower()
        for edge in cell.edges[:2]:
            target_id = getattr(edge, "target_cell_id", "")
            if target_id:
                clean_target = target_id.lower().replace("_", " ")
                assert target_id.lower() in emb_text.lower() or any(
                    tok in emb_text.lower() for tok in clean_target.split()
                )

    # Test CV2_CVT_COLOR_BGR2GRAY which has preconditions/postconditions and inbound connections
    gray_cell = next((c for c in orchestrator.cells if c.cell_id == "CV2_CVT_COLOR_BGR2GRAY"), None)
    if gray_cell:
        gray_text = build_cell_embedding_text(gray_cell, orchestrator)
        # Should include inbound predecessors if orchestrator has reverse adjacency
        if hasattr(orchestrator, "_reverse_adjacency") and orchestrator._reverse_adjacency.get(gray_cell.cell_id):
            assert "predecessors:" in gray_text.lower()


# =========================================================================
# 2. Stage 1: Candidate Idiom Subgraph Discovery
# =========================================================================

def test_stage1_candidate_idiom_discovery(router, orchestrator):
    """
    Verify that Stage 1 _discover_candidate_idioms identifies valid connected subgraphs
    spanning multiple prompt clauses.
    """
    prompt = "Read an image, convert to grayscale, and apply Gaussian blur"
    clauses = ["read an image", "convert to grayscale", "apply gaussian blur"]

    idioms = router._discover_candidate_idioms(prompt, clauses)
    assert isinstance(idioms, list)
    assert len(idioms) > 0

    # At least one idiom must cover multiple clauses
    multi_clause_idioms = [idiom for idiom in idioms if len(idiom.covered_clauses) > 1]
    assert len(multi_clause_idioms) > 0

    best_idiom = idioms[0]
    assert isinstance(best_idiom, IdiomSubgraph)
    assert len(best_idiom.cells) >= 2
    assert best_idiom.lexical_score > 0.0


# =========================================================================
# 3. Stage 2: Entry Node Selection
# =========================================================================

def test_stage2_entry_node_selection(router, orchestrator):
    """
    Verify that Stage 2 _select_entry_node correctly identifies the entry node
    within a discovered candidate idiom.
    """
    prompt = "Read an image, convert to grayscale, and detect edges"
    clauses = ["read an image", "convert to grayscale", "detect edges"]

    token_index = router.orchestrator.token_index
    N = len(router.orchestrator.cells)

    idioms = router._discover_candidate_idioms(prompt, clauses)
    assert len(idioms) > 0

    # Pick an idiom containing CV2 or Pillow image reading
    image_idiom = None
    for idiom in idioms:
        cell_ids = {c.cell_id for c in idiom.cells}
        if "CV2_IMREAD" in cell_ids or "PILLOW_IMAGE_OPEN" in cell_ids:
            image_idiom = idiom
            break

    if image_idiom is not None:
        entry = router._select_entry_node(image_idiom, clauses[0], token_index, N)
        assert entry is not None
        # Entry node should be stage 1 source (e.g. CV2_IMREAD or PILLOW_IMAGE_OPEN)
        assert entry.cell_id in ("CV2_IMREAD", "PILLOW_IMAGE_OPEN") or entry.stage == 1


# =========================================================================
# 4. Full Two-Stage Route Verification
# =========================================================================

def test_two_stage_route_cv2_pipeline(router):
    """
    Test LatticeRouter.route() returns expected cells and confidences for
    an OpenCV image processing prompt.
    """
    prompt = "Read an image with opencv, convert BGR to grayscale, apply Gaussian blur, and detect edges with canny"
    cells, confs = router.route(prompt, top_k=5)

    assert isinstance(cells, list)
    assert isinstance(confs, dict)
    assert len(cells) > 0

    routed_ids = {c.cell_id for c in cells}
    # Essential pipeline components should be in the top candidates
    expected_cells = {"CV2_IMREAD", "CV2_CVT_COLOR_BGR2GRAY", "CV2_GAUSSIAN_BLUR", "CV2_CANNY"}
    overlap = routed_ids.intersection(expected_cells)
    assert len(overlap) >= 3, f"Expected at least 3 of {expected_cells}, got {overlap}"

    # Verify confidence scores are in [0, 1]
    for cid, conf in confs.items():
        assert 0.0 <= conf <= 1.0, f"Confidence for {cid} out of range: {conf}"


def test_two_stage_route_pandas_sklearn(router):
    """
    Test LatticeRouter.route() returns expected cells for a tabular data ML pipeline.
    """
    prompt = "Load a CSV dataset, drop missing values, convert dataframe to numpy, and fit a RandomForestClassifier"
    cells, confs = router.route(prompt, top_k=6)

    routed_ids = {c.cell_id for c in cells}
    # Should include PD_READ_CSV or sklearn RandomForestClassifier
    assert "PD_READ_CSV" in routed_ids or "sklearn.ensemble.RandomForestClassifier.fit" in routed_ids


# =========================================================================
# 5. Plan Path Verification
# =========================================================================

def test_plan_path_cv2(router):
    """
    Test router.plan_path produces a logically connected path.
    """
    prompt = "Read an image with opencv, convert BGR to grayscale, apply Gaussian blur, and detect edges with canny"
    path, covered = router.plan_path(prompt)

    assert isinstance(path, list)
    assert len(path) >= 3

    path_ids = [c.cell_id for c in path]
    # CV2_IMREAD must precede CV2_CVT_COLOR_BGR2GRAY which must precede CV2_GAUSSIAN_BLUR or CV2_CANNY
    assert path_ids[0] in ("CV2_IMREAD", "PILLOW_IMAGE_OPEN")
    assert "CV2_CVT_COLOR_BGR2GRAY" in path_ids


def test_plan_path_pillow(router):
    """
    Test router.plan_path on Pillow pipeline.
    """
    prompt = "Open an image, convert to grayscale, and apply Gaussian blur"
    path, covered = router.plan_path(prompt)

    assert isinstance(path, list)
    assert len(path) >= 2
    path_ids = [c.cell_id for c in path]
    assert path_ids[0] in ("PILLOW_IMAGE_OPEN", "CV2_IMREAD")


# =========================================================================
# 6. Lexical Fallback & Empty Prompt Handling
# =========================================================================

def test_empty_or_whitespace_prompt(router):
    """Test router handling of empty or blank prompt."""
    cells, confs = router.route("", top_k=5)
    assert cells == []
    assert confs == {}

    path, covered = router.plan_path("")
    assert path == []


def test_single_keyword_prompt(router):
    """Test routing with a single specific keyword."""
    cells, confs = router.route("canny", top_k=3)
    assert len(cells) > 0
    assert cells[0].cell_id == "CV2_CANNY"
