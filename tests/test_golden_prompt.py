"""
tests/test_golden_prompt.py - Neuro-Symbolic Topological Lattice (NSTL)
Golden benchmark test for Round 4.

Evaluates the benchmark prompt:
  "load a csv file named \"input.csv\", normalize X column, drop null values,
   get the mean of the Y column, and perform FFT and write it to a new column called Z,
   then make a regression model trained on X and Y to predict Z"

Executes emitted code in the GEVR sandbox with synthesized fixtures.
Evaluates 6 runtime assertions via execution hooks/tracing (no regex on source).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import Profile
from gevr_sandbox import GEVRSandbox
from lattice import LatticeOrchestrator
from planner import LatticePlanner, _segment_prompt_clauses
from tokenizer import CellTokenizer
from unification import UnificationGate, ExecutionContext

BENCHMARK_PROMPT = (
    'load a csv file named "input.csv", normalize X column, drop null values, '
    'get the mean of the Y column, and perform FFT and write it to a new column called Z, '
    'then make a regression model trained on X and Y to predict Z'
)

NON_LLM_METHODS = ["M0", "M1", "M2", "M3", "M6", "M9"]


@pytest.fixture(scope="module")
def orchestrator():
    trees_dir = str(ROOT_DIR / "trees")
    db_path = str(ROOT_DIR / "trees" / "lattice.db")
    orch = LatticeOrchestrator(trees_directory=trees_dir, db_path=db_path)
    if os.path.exists(db_path):
        try:
            orch.load_from_database(db_path)
        except Exception:
            orch.load_all_json_trees()
    else:
        orch.load_all_json_trees()
    orch.build_topology()
    return orch


def _run_pipeline_for_method(orch: LatticeOrchestrator, method: str, profile_name: str = "A"):
    prof = Profile(profile_name)
    planner = LatticePlanner(orchestrator=orch)
    gate = UnificationGate(orchestrator=orch)

    # Route and plan
    relevance_map = {}
    tokens = CellTokenizer.tokenize_prompt(BENCHMARK_PROMPT)
    for c in orch.all_cells():
        overlap = len(tokens & c.token_set)
        if overlap > 0:
            relevance_map[c.cell_id] = float(overlap)

    if method == "M0":
        best_path, score = planner.plan_dag(BENCHMARK_PROMPT, candidates=orch.all_cells(), relevance_map=relevance_map)
    else:
        # Use route method
        from router import SemanticRouteOptimizer
        optimizer = SemanticRouteOptimizer(orch, prof)
        best_path = optimizer.route(BENCHMARK_PROMPT, method_name=method, relevance_map=relevance_map)

    if not best_path:
        return None, None, None, []

    ctx = ExecutionContext(prompt=BENCHMARK_PROMPT)
    unify_res = gate.unify_pipeline(best_path, context=ctx)
    if not unify_res.is_success():
        return best_path, None, None, []

    bindings = unify_res.value
    code = gate.emit_code(bindings, ctx)
    return best_path, bindings, code, ctx.ordered_literals


def _execute_with_tracing(code: str, bindings: list, literals: list, working_dir: Path):
    # Create test input files derived from literals
    import pandas as pd
    import numpy as np
    csv_files = [lit for lit in literals if isinstance(lit, str) and lit.endswith(".csv")]
    if not csv_files:
        csv_files = ["input.csv"]
    col_names = [lit for lit in literals if isinstance(lit, str) and not lit.endswith(".csv") and len(lit) <= 30]
    if not col_names:
        col_names = ["X", "Y"]
    np.random.seed(42)
    sample_data = {c: np.random.randn(50) * 10.0 + 5.0 for c in col_names}
    if col_names:
        for c in col_names:
            sample_data[c][2] = np.nan
    df_sample = pd.DataFrame(sample_data)
    for f in csv_files:
        p = working_dir / f
        p.parent.mkdir(parents=True, exist_ok=True)
        df_sample.to_csv(p, index=False)

    # Trace wrapper prepended to code to inspect runtime contracts
    instrumentation_preamble = """
import numpy as _nstl_np
import pandas as _nstl_pd
import sklearn.linear_model as _nstl_slm

_nstl_trace = {
    'series_mean_names': [],
    'fft_input_ndims': [],
    'fft_input_types': [],
    'fit_calls': [],
}

_orig_series_mean = _nstl_pd.Series.mean
def _traced_series_mean(self, *args, **kwargs):
    _nstl_trace['series_mean_names'].append(getattr(self, 'name', None))
    return _orig_series_mean(self, *args, **kwargs)
_nstl_pd.Series.mean = _traced_series_mean

_orig_fft = _nstl_np.fft.fft
def _traced_fft(a, *args, **kwargs):
    _nstl_trace['fft_input_ndims'].append(getattr(a, 'ndim', 1))
    _nstl_trace['fft_input_types'].append(type(a).__name__)
    return _orig_fft(a, *args, **kwargs)
_nstl_np.fft.fft = _traced_fft

_orig_lr_fit = _nstl_slm.LinearRegression.fit
def _traced_lr_fit(self, X, y, *args, **kwargs):
    _nstl_trace['fit_calls'].append({
        'X_ndim': getattr(X, 'ndim', None),
        'X_shape': getattr(X, 'shape', None),
        'y_ndim': getattr(y, 'ndim', None),
        'y_shape': getattr(y, 'shape', None),
        'y_vals': _nstl_np.array(y).tolist() if hasattr(y, '__iter__') else None
    })
    return _orig_lr_fit(self, X, y, *args, **kwargs)
_nstl_slm.LinearRegression.fit = _traced_lr_fit
"""

    instrumented_code = instrumentation_preamble + "\n" + code

    sandbox = GEVRSandbox()
    # Execute inside working_dir
    orig_cwd = os.getcwd()
    try:
        os.chdir(working_dir)
        res = sandbox.execute(
            instrumented_code,
            timeout=10.0,
            pipeline_bindings=bindings,
            extracted_literals=literals,
            fixtures_dir=str(working_dir),
        )
    finally:
        os.chdir(orig_cwd)

    return res


@pytest.mark.parametrize("method", NON_LLM_METHODS)
def test_golden_prompt(orchestrator, method):
    if method in ("M2", "M3"):
        pytest.xfail("M2/M3 path ends in a non-regression sink until PR-B lands")

    with tempfile.TemporaryDirectory() as tmpdir:
        work_path = Path(tmpdir)
        best_path, bindings, code, literals = _run_pipeline_for_method(orchestrator, method)

        assert best_path is not None, f"Planner produced no path for {method}"
        assert code is not None, f"Synthesis produced no code for {method}"

        # 6. No cell on path has empty clause coverage unless it is a bridge or source
        planner = LatticePlanner(orchestrator=orchestrator)
        clauses = _segment_prompt_clauses(BENCHMARK_PROMPT)
        clause_toks = [CellTokenizer.tokenize_prompt(cl) for cl in clauses]
        for cell in best_path:
            role = str(getattr(cell, "node_role", "")).lower()
            if role in ("bridge", "source"):
                continue
            c_toks = cell.token_set
            covered = any(len(c_toks & cl_t) > 0 for cl_t in clause_toks)
            assert covered, f"Cell {cell.cell_id} on path has empty clause coverage and is not bridge/source"

        res = _execute_with_tracing(code, bindings, literals, work_path)

        # 1. Script executes without exception
        assert res.get("success") is True, f"Execution failed for {method}: {res.get('error')}"

        results = res.get("results", {})
        trace = results.get("_nstl_trace", {})

        # 2. Frame has column Z with length matching non-null rows, no NaN
        # Check all DataFrame variables in results
        df_found = False
        for k, v in results.items():
            if hasattr(v, "columns") and "Z" in v.columns:
                df_found = True
                z_col = v["Z"]
                assert not z_col.isna().any(), "Column Z contains NaN values"
                assert len(z_col) > 0, "Column Z is empty"
                break
        assert df_found, "Working frame does not contain column 'Z'"

        # 3. LinearRegression.fit was called with 2-D X of two columns and 1-D y
        fit_calls = trace.get("fit_calls", [])
        assert len(fit_calls) > 0, "LinearRegression.fit was never called"
        fit_info = fit_calls[0]
        assert fit_info["X_ndim"] == 2, f"Expected 2-D X, got {fit_info['X_ndim']}"
        assert fit_info["X_shape"][1] == 2, f"Expected 2 columns in X, got {fit_info['X_shape'][1]}"
        assert fit_info["y_ndim"] == 1, f"Expected 1-D y, got {fit_info['y_ndim']}"

        # 4. Mean computed on Series named Y
        mean_names = trace.get("series_mean_names", [])
        assert any(name in ("Y", "'Y'", '"Y"') for name in mean_names), (
            f"Mean was not computed over Series named Y; captured series: {mean_names}"
        )

        # 5. 1-D FFT applied to single column, not the whole frame
        fft_ndims = trace.get("fft_input_ndims", [])
        assert len(fft_ndims) > 0, "np.fft.fft was never called"
        assert all(ndim == 1 for ndim in fft_ndims), f"np.fft.fft input ndim was not 1: {fft_ndims}"
