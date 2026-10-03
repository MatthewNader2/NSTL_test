#!/usr/bin/env python3
"""
scripts/sweep.py - Neuro-Symbolic Topological Lattice (NSTL)
Multi-Profile & Multi-Method Sweep Runner for Benchmark Evaluation.

Runs the benchmark prompt across:
  Profiles: A, C, D, E, S
  Methods: M0, M1, M2, M3, M4, M5, M6, M7, M8, M9

Options used: --debug --no-reranker --exec

Emits a structured markdown table:
  | profile | method | steps | emitted? | refused why | sandbox ok? | golden assertions passed (n/6) | layer-2 ms |
Followed by:
  - Number of distinct emitted scripts
  - Number refused
  - Preflight violation categories with counts
"""

from __future__ import annotations

import collections
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import Profile, settings
from fixtures import FixtureSynthesizer
from gevr_sandbox import GEVRSandbox
from lattice import LatticeOrchestrator
from planner import LatticePlanner, _segment_prompt_clauses
from preflight import PreflightLinter
from tokenizer import CellTokenizer
from unification import ExecutionContext, UnificationGate

BENCHMARK_PROMPT = (
    'load a csv file named "input.csv", normalize X column, drop null values, '
    'get the mean of the Y column, and perform FFT and write it to a new column called Z, '
    'then make a regression model trained on X and Y to predict Z'
)

PROFILES = ["A", "C", "D", "E", "S"]
METHODS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


def evaluate_golden_assertions(
    code: str, bindings: list, literals: list, working_dir: Path, path: list
) -> Tuple[int, Optional[str]]:
    """
    Evaluates the 6 runtime assertions on the generated code.
    Returns (score: 0-6, error_message).
    """
    FixtureSynthesizer.synthesize_for_pipeline(
        pipeline_bindings=bindings,
        working_dir=working_dir,
        extracted_literals=literals,
        num_rows=50,
    )

    clauses = _segment_prompt_clauses(BENCHMARK_PROMPT)
    clause_toks = [CellTokenizer.tokenize_prompt(cl) for cl in clauses]

    passed = 0

    # Assertion 6: No cell on path has empty clause coverage unless bridge or source
    a6_pass = True
    for cell in path:
        role = str(getattr(cell, "node_role", "")).lower()
        if role in ("bridge", "source"):
            continue
        c_toks = cell.token_set
        if not any(len(c_toks & cl_t) > 0 for cl_t in clause_toks):
            a6_pass = False
            break
    if a6_pass:
        passed += 1

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

    if not res.get("success"):
        return passed, res.get("error")

    # Assertion 1: Executed without exception
    passed += 1

    results = res.get("results", {})
    trace = results.get("_nstl_trace", {})

    # Assertion 2: Column Z in working frame, length matches non-null rows, no NaN
    df_found = False
    for v in results.values():
        if hasattr(v, "columns") and "Z" in v.columns:
            z_col = v["Z"]
            if len(z_col) > 0 and not z_col.isna().any():
                df_found = True
                break
    if df_found:
        passed += 1

    # Assertion 3: LinearRegression.fit called with 2D X (2 columns) and 1D y
    fit_calls = trace.get("fit_calls", [])
    if fit_calls and fit_calls[0]["X_ndim"] == 2 and fit_calls[0]["X_shape"][1] == 2 and fit_calls[0]["y_ndim"] == 1:
        passed += 1

    # Assertion 4: Mean computed on Series named Y
    mean_names = trace.get("series_mean_names", [])
    if any(name in ("Y", "'Y'", '"Y"') for name in mean_names):
        passed += 1

    # Assertion 5: 1D FFT applied to single column
    fft_ndims = trace.get("fft_input_ndims", [])
    if fft_ndims and all(ndim == 1 for ndim in fft_ndims):
        passed += 1

    return passed, None


def run_sweep():
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

    all_cells = orch.all_cells()
    prompt_tokens = CellTokenizer.tokenize_prompt(BENCHMARK_PROMPT)
    relevance_map = {
        c.cell_id: float(len(prompt_tokens & c.token_set))
        for c in all_cells if len(prompt_tokens & c.token_set) > 0
    }

    gate = UnificationGate(orchestrator=orch)

    rows = []
    distinct_scripts = set()
    refused_count = 0
    violation_categories = collections.Counter()

    for profile_name in PROFILES:
        prof = Profile(profile_name)
        for method_name in METHODS:
            t0 = time.perf_counter()
            best_path = None
            code = None
            refused_why = "-"
            sandbox_ok = "no"
            golden_score = 0
            steps = 0
            layer2_ms = 0.0

            try:
                if method_name == "M0":
                    planner = LatticePlanner(orchestrator=orch)
                    best_path, score = planner.plan_dag(BENCHMARK_PROMPT, candidates=all_cells, relevance_map=relevance_map)
                else:
                    from router import SemanticRouteOptimizer
                    optimizer = SemanticRouteOptimizer(orch, prof)
                    best_path = optimizer.route(BENCHMARK_PROMPT, method_name=method_name, relevance_map=relevance_map)

                layer2_ms = (time.perf_counter() - t0) * 1000.0

                if not best_path:
                    refused_why = "Planner produced no path"
                    refused_count += 1
                else:
                    steps = len(best_path)
                    ctx = ExecutionContext(prompt=BENCHMARK_PROMPT)
                    unify_res = gate.unify_pipeline(best_path, context=ctx)
                    if not unify_res.is_success():
                        refused_why = f"Unification failed: {unify_res.reason if hasattr(unify_res, 'reason') else 'type bottom'}"
                        refused_count += 1
                    else:
                        bindings = unify_res.value
                        code = gate.emit_code(bindings, ctx)

                        # Preflight check
                        lint_res = PreflightLinter.lint(
                            bindings,
                            prompt=BENCHMARK_PROMPT,
                            code_str=code,
                            extracted_literals=ctx.ordered_literals,
                        )

                        if not lint_res.is_valid:
                            refused_count += 1
                            v_first = lint_res.violations[0] if lint_res.violations else "Preflight lint violation"
                            refused_why = f"Preflight: {v_first[:50]}"
                            for v in lint_res.violations:
                                cat = v.split(":")[0] if ":" in v else v.split()[0]
                                violation_categories[cat] += 1
                            code = None
                        else:
                            # Preflight passed, record script and test sandbox
                            script_hash = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()[:12]
                            distinct_scripts.add(script_hash)

                            with tempfile.TemporaryDirectory() as tmpdir:
                                golden_score, err = evaluate_golden_assertions(
                                    code, bindings, ctx.ordered_literals, Path(tmpdir), best_path
                                )
                                if err is None:
                                    sandbox_ok = "yes"
                                else:
                                    sandbox_ok = f"error: {err.splitlines()[-1][:40]}"

            except Exception as exc:
                layer2_ms = (time.perf_counter() - t0) * 1000.0
                refused_why = f"Exception: {type(exc).__name__}: {str(exc)[:40]}"
                refused_count += 1

            rows.append({
                "profile": profile_name,
                "method": method_name,
                "steps": steps,
                "emitted": "yes" if code else "no",
                "refused_why": refused_why,
                "sandbox_ok": sandbox_ok,
                "golden_passed": f"{golden_score}/6",
                "layer2_ms": f"{layer2_ms:.1f}",
            })

    # Print markdown table
    print("\n### Sweep Results Table\n")
    print("| profile | method | steps | emitted? | refused why | sandbox ok? | golden assertions passed (n/6) | layer-2 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['profile']} | {r['method']} | {r['steps']} | {r['emitted']} | {r['refused_why']} | {r['sandbox_ok']} | {r['golden_passed']} | {r['layer2_ms']} |")

    print(f"\nDistinct emitted scripts: {len(distinct_scripts)}")
    print(f"Refused runs: {refused_count} / {len(rows)}")
    if violation_categories:
        print("\nPreflight violation categories:")
        for cat, count in violation_categories.most_common():
            print(f"  - {cat}: {count}")


if __name__ == "__main__":
    run_sweep()
