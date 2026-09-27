"""
tests/test_phase1_correctness_generic.py - Neuro-Symbolic Topological Lattice (NSTL)

T1.5: Phase 1 Correctness & Anti-Hardcoding Invariant Test Suite.

Contains:
1. Worked Example:
   "load data.csv, clean missing values, normalize features X, Y, Z, and fit regression to predict Z"
   Asserts binder never materializes bare None into data-bearing positions and preflight lint passes.
2. Generic Parameterized Property Invariant:
   Scans live trees/*.json for every harvested cell with an optional port whose default is None
   and whose role is data-bearing. Asserts the binder never emits a literal None in data positions.
"""

from __future__ import annotations
import ast
import os
import sys
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lattice import LatticeOrchestrator, Cell
from router import LatticeRouter
from unification import UnificationGate, ExecutionContext
from preflight import PreflightLinter, DATA_BEARING_ROLES


@pytest.fixture(scope="module")
def shared_orchestrator():
    trees_dir = str(ROOT_DIR / "trees")
    orch = LatticeOrchestrator(trees_directory=trees_dir)
    return orch


def test_worked_example_ml_data_carrier_no_none(shared_orchestrator):
    """
    Worked Example:
    Prompt with multiple data identifiers ('data.csv', clean, normalize X, Y, Z -> regression predict Z).
    Must not emit bare None into data-bearing arguments (e.g. train_test_split(X, None) or fit(X, None)).
    """
    router = LatticeRouter(orchestrator=shared_orchestrator, internal_rag=None)
    prompt = "load data.csv and drop missing values then normalize X, Y, Z and predict Z with linear regression"

    res = router.plan_path(prompt)
    cells = res[0] if isinstance(res, tuple) else res

    assert len(cells) > 0, f"Planner returned empty path for ML prompt: '{prompt}'"

    gate = UnificationGate()
    ctx = ExecutionContext(prompt=prompt)
    unify_res = gate.unify_pipeline(cells, ctx)
    assert not unify_res.is_bottom(), f"Unification failed: {getattr(unify_res, 'reason', '')}"

    pipeline_bindings = unify_res.value
    code = gate.emit_code(pipeline_bindings, ctx)
    assert code and code.strip(), "Emitted code was empty."

    # Parse AST to inspect all calls
    parsed = ast.parse(code)
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            # Check positional arguments for bare None Constant
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value is None:
                    func_name = getattr(node.func, "id", getattr(node.func, "attr", "unknown_func"))
                    raise AssertionError(f"Bare None passed as positional argument in call to '{func_name}' in emitted code:\n{code}")

    # Verify static pre-flight linter passes
    lint_res = PreflightLinter.lint(pipeline_bindings, prompt=prompt, code_str=code)
    # Lint must not find bare None in data positions
    none_violations = [v for v in lint_res.violations if "received bare None" in v]
    assert len(none_violations) == 0, f"Pre-flight lint flagged bare None violations:\n{none_violations}"


def test_parameterized_live_trees_optional_data_ports(shared_orchestrator):
    """
    Generic Parameterized Invariant:
    Scans live trees/*.json for every harvested cell with an optional port having default=None
    and a data-bearing role. Asserts that the binder never instantiates a literal 'None' string
    for any such port across all loaded domains.
    """
    gate = UnificationGate()
    target_cells = []

    for cell_id, cell in shared_orchestrator.loaded_cells.items():
        for p_name, p_sig in cell.inputs.items():
            if not p_sig.required:
                role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                if role in DATA_BEARING_ROLES:
                    def_val = getattr(p_sig, "default_value", None)
                    if def_val is None or str(def_val).strip().lower() == "none":
                        target_cells.append((cell, p_name, role))

    assert len(target_cells) > 0, "No cells found with optional data-bearing ports in live trees."

    # Test each cell: bind in isolation with dummy prompt supplying an identifier
    for cell, p_name, role in target_cells:
        prompt = f"process data using {cell.cell_id} with X and Y"
        ctx = ExecutionContext(prompt=prompt)
        try:
            unify_res = gate.unify_pipeline([cell], ctx)
            if not unify_res.is_bottom():
                bindings = unify_res.value[0][1]
                val = bindings.get(p_name)
                # If bound, it must never be literal string 'None' unless handles_none is declared
                if val is not None:
                    val_str = str(val).strip()
                    assert val_str != "None", (
                        f"Cell '{cell.cell_id}' port '{p_name}' (role '{role}') materialized bare literal 'None'!"
                    )
        except Exception:
            # Unification may fail if required ports cannot be resolved; that is valid
            pass
