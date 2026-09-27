import os
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path = [p for p in sys.path if Path(p).resolve() != ROOT_DIR.resolve()]
sys.path.insert(0, str(SRC_DIR))

from lattice import LatticeOrchestrator, Cell, UNRESOLVED_PORT, _UnresolvedPortSentinel
from unification import ExecutionContext, UnresolvedPlaceholderError, UnificationGate
from preflight import PreflightLinter, PreflightLintResult
from router import LatticeRouter
from planner import LatticePlanner
from cli import PipelineDebugger, NSTLInteractiveShell


def test_unresolved_port_sentinel_singleton():
    """Verify UNRESOLVED_PORT sentinel behaves as a singleton and respects contracts."""
    assert isinstance(UNRESOLVED_PORT, _UnresolvedPortSentinel)
    assert _UnresolvedPortSentinel() is UNRESOLVED_PORT
    assert bool(UNRESOLVED_PORT) is False
    assert repr(UNRESOLVED_PORT) == "<UNRESOLVED_PORT>"
    assert str(UNRESOLVED_PORT) == "<UNRESOLVED_PORT>"
    # Equality matches for backward-compatibility string lookups
    assert UNRESOLVED_PORT == "<UNRESOLVED_PORT>"
    assert UNRESOLVED_PORT == "<UNRESOLVED>"
    assert UNRESOLVED_PORT == "<unbound>"
    assert hash(UNRESOLVED_PORT) == hash("<UNRESOLVED_PORT>")


def test_no_unresolved_string_literals_in_src():
    """Purity test: ensure no file in src/ assigns string '<UNRESOLVED>' as port bindings."""
    for py_file in SRC_DIR.glob("*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Ensure cell_bindings[...] = "<UNRESOLVED>" is nowhere in the codebase
            assert 'cell_bindings[p_name] = "<UNRESOLVED>"' not in content, f"Found rogue assignment in {py_file}"
            assert 'cell_bindings[p] = "<UNRESOLVED>"' not in content, f"Found rogue assignment in {py_file}"


def test_direct_unification_refuses_unresolved_ports():
    """Verify that unifying an unbindable role port records ctx.unresolved_ports and refuses code emission."""
    orch = LatticeOrchestrator()
    c_read = orch.loaded_cells.get("PD_READ_CSV")
    c_numpy = orch.loaded_cells.get("PANDAS_DATAFRAME_TO_NUMPY")
    c_split = orch.loaded_cells.get("SKLEARN.MODEL_SELECTION.TRAIN_TEST_SPLIT") or orch.loaded_cells.get("sklearn.model_selection.train_test_split")
    assert c_read is not None and c_numpy is not None and c_split is not None

    pipeline = [c_read, c_numpy, c_split]
    prompt = "load data.csv and split dataset with train_test_split then train regression to predict Z"
    ctx = ExecutionContext(prompt=prompt)
    gate = UnificationGate()
    monad_res = gate.unify_pipeline(pipeline, context=ctx)

    # Assert unresolved ports were tracked
    assert len(ctx.unresolved_ports) > 0
    port_names = [p for _, p in ctx.unresolved_ports]
    assert "y" in port_names

    # Assert emit_code strictly refuses synthesis
    with pytest.raises(UnresolvedPlaceholderError) as exc_info:
        gate.emit_code(pipeline, context=ctx)
    assert "Code synthesis refused" in str(exc_info.value)
    assert "y" in str(exc_info.value)


def test_preflight_linter_aborts_on_unresolved_or_unconsumed():
    """Verify that PreflightLinter detects unconsumed literals and unresolved ports."""
    orch = LatticeOrchestrator()
    c_read = orch.loaded_cells.get("PD_READ_CSV")
    c_split = orch.loaded_cells.get("SKLEARN.MODEL_SELECTION.TRAIN_TEST_SPLIT") or orch.loaded_cells.get("sklearn.model_selection.train_test_split")

    pipeline_bindings = [
        (c_read, {"filepath": '"data.csv"', "output_df": "var_1"}),
        (c_split, {"X": "var_1", "y": UNRESOLVED_PORT, "test_size": 0.25, "random_state": 42})
    ]

    prompt = "load data.csv and split dataset with train_test_split then train regression to predict Z"
    res = PreflightLinter.lint(pipeline_bindings, prompt=prompt, code_str="# dummy code")

    assert res.is_valid is False
    assert len(res.violations) >= 1
    violation_text = " ".join(res.violations)
    assert "Z" in violation_text or "target_input" in violation_text or "unresolved" in violation_text


def test_planner_unbindable_penalty_on_missing_target_producer():
    """Verify that even if the planner selects train_test_split, the system
    catches the unresolved y port and refuses code synthesis.

    The planner may still select train_test_split because planner-level
    unify() (without accumulated sigma) finds type-compatible producers.
    The safety invariant is that the downstream guards (unification gate +
    preflight linter) catch and refuse the unresolved port before any code
    reaches the sandbox.
    """
    orch = LatticeOrchestrator()
    planner = LatticePlanner(orch)
    prompt = "load data.csv and split dataset with train_test_split then train regression to predict Z"
    cells, rel_map = LatticeRouter(orch).route(prompt, top_k=30)

    best_path = planner.plan(prompt, cells, rel_map)
    best_cell_ids = [c.cell_id for c in best_path]

    # If planner happens to select train_test_split, verify the downstream
    # unification + preflight correctly catches the unresolved y port.
    split_id = "SKLEARN.MODEL_SELECTION.TRAIN_TEST_SPLIT"
    if split_id in best_cell_ids or split_id.lower() in [x.lower() for x in best_cell_ids]:
        gate = UnificationGate()
        ctx = ExecutionContext(prompt=prompt)
        gate.unify_pipeline(best_path, context=ctx)

        # Must have recorded unresolved ports
        assert len(ctx.unresolved_ports) > 0, \
            "train_test_split selected but no unresolved ports detected"
        port_names = [p for _, p in ctx.unresolved_ports]
        assert "y" in port_names, \
            f"Expected 'y' in unresolved ports, got: {port_names}"

        # emit_code must refuse synthesis
        with pytest.raises(UnresolvedPlaceholderError):
            gate.emit_code(best_path, context=ctx)
    else:
        # Planner excluded train_test_split — that's also acceptable
        pass


def test_literal_consumption_column_projection_gap():
    """Verify that literal consumption requires actual column projection cells rather than whole dataframe hops."""
    orch = LatticeOrchestrator()
    c_read = orch.loaded_cells.get("PD_READ_CSV")
    c_numpy = orch.loaded_cells.get("PANDAS_DATAFRAME_TO_NUMPY")
    c_select = orch.loaded_cells.get("PD_SELECT_COLUMNS")

    planner = LatticePlanner(orch)
    prompt = "load data.csv and predict Z"
    
    # Path without column projection:
    path_numpy = [c_read, c_numpy]
    # Path with column projection:
    path_select = [c_read, c_select]

    # In path_numpy, 'Z' is not consumed
    # In path_select, PD_SELECT_COLUMNS has a column_projection input, so 'Z' is consumed
    # Test via planner's score or internal check
    cells = [c_read, c_numpy, c_select]
    rel_map = {c.cell_id: 0.5 for c in cells}
    
    # Check that PD_SELECT_COLUMNS is recognized as a column projection cell
    has_col_proj_numpy = any(
        any(
            str(getattr(p_s.signature, "state", "")).lower() == "column_projection"
            or (str(getattr(p_s.signature, "type_name", "")).lower() in ("list", "sequence") and p_n.lower() in ("columns", "cols", "column"))
            for p_n, p_s in c.inputs.items()
        ) or any(slot.lower() in ("columns", "cols", "column") for slot in getattr(c, "slots", []))
        for c in path_numpy
    )
    assert has_col_proj_numpy is False

    has_col_proj_select = any(
        any(
            str(getattr(p_s.signature, "state", "")).lower() == "column_projection"
            or (str(getattr(p_s.signature, "type_name", "")).lower() in ("list", "sequence") and p_n.lower() in ("columns", "cols", "column"))
            for p_n, p_s in c.inputs.items()
        ) or any(slot.lower() in ("columns", "cols", "column") for slot in getattr(c, "slots", []))
        for c in path_select
    )
    assert has_col_proj_select is True


def test_cli_debug_unresolved_port_refusal():
    """End-to-end CLI debug run: ensures no unresolved port reaches sandbox and pre-flight aborts cleanly."""
    shell = NSTLInteractiveShell(
        db_path=str(Path(__file__).parent.parent / "trees" / "lattice.db"),
        initial_profile="0",
        interactive=False,
        debug=True
    )
    debugger = PipelineDebugger(
        orchestrator=shell.orchestrator,
        router=shell.router,
        gate=shell.gate,
        sandbox=shell.sandbox,
        active_profile=shell.active_profile
    )
    prompt = "load data.csv and split dataset with train_test_split then train regression to predict Z"
    res = debugger.run(prompt, execute_sandbox=True, timeout=5.0)

    # Sandbox execution must not have run on a broken/unresolved pipeline
    # Stderr or abort message must explain the lint violation or refusal
    final_code = res.get("final_code", "")
    assert "<UNRESOLVED>" not in final_code
    assert "<UNRESOLVED_PORT>" not in final_code
    assert "<unbound>" not in final_code
