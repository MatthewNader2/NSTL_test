"""
tests/test_evaluation_matrix.py - NSTL Evaluation Matrix & Thesis Validation (Phase 5).
Evaluates RouteMethods (M0, M1, M2, M3, M6) across cross-domain benchmark tasks.
Logs latency, path length, monadic unification, preflight linting, AST validity,
and pass attribution to evaluation_results.json.
"""

from __future__ import annotations
import ast
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from lattice import LatticeOrchestrator
from router import LatticeRouter
from unification import UnificationGate, ExecutionContext, Success
from preflight import PreflightLinter
from route_methods import ROUTE_METHOD_REGISTRY

DB_PATH = str(ROOT_DIR / "trees" / "lattice.db")
EVAL_RESULTS_PATH = str(ROOT_DIR / "evaluation_results.json")

EVAL_TASKS: List[Tuple[str, str, str]] = [
    ("TAB_01", "Tabular", "load input.csv and drop missing values then save to output.csv"),
    ("TAB_02", "Tabular", "read data.csv and sort by age ascending then save to cleaned.csv"),
    ("VIS_01", "Vision", "read image input.jpg and convert to grayscale then save to output.jpg"),
    ("VIS_02", "Vision", "read image photo.jpg and apply gaussian blur then save to output.jpg"),
    ("CR_01", "Cross-Domain", "read input.csv, standardize features with sklearn StandardScaler, plot histogram of feature_1 with matplotlib, and save figure to plot.png"),
    ("CR_09", "Cross-Domain", "load input.csv, drop missing values, group by region and sum revenue, and save to report.csv"),
    ("ALG_01", "Algorithmic", "dijkstra shortest path algorithm on graph"),
]

TEST_METHODS = ["M0", "M1", "M2", "M3", "M6"]


class TestEvaluationMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orch = LatticeOrchestrator("trees")
        if os.path.exists(DB_PATH):
            try:
                cls.orch.load_from_database(DB_PATH)
            except Exception:
                pass
        cls.orch.build_topology()
        cls.router = LatticeRouter(orchestrator=cls.orch, internal_rag=None)
        cls.gate = UnificationGate(cls.orch)

    def test_run_evaluation_matrix(self):
        """
        Runs full RouteMethods (M0-M6) evaluation matrix across domain tasks.
        Measures routing latency, path validity, linting, and AST compilation.
        """
        matrix_records: List[Dict[str, Any]] = []
        method_stats: Dict[str, Dict[str, Any]] = {
            m: {"latencies": [], "passed": 0, "total": 0, "path_lens": []}
            for m in TEST_METHODS
        }

        print("\n" + "=" * 80)
        print(" NSTL PHASE 5 EVALUATION MATRIX: ROUTE METHODS BENCHMARK")
        print("=" * 80)

        for method in TEST_METHODS:
            for task_id, domain, prompt in EVAL_TASKS:
                method_stats[method]["total"] += 1
                ctx = ExecutionContext(prompt=prompt)

                t0 = time.perf_counter()
                try:
                    cells = self.router.plan_path(prompt, return_tuple=False, route_method=method)
                except Exception as e:
                    cells = []
                dt_ms = (time.perf_counter() - t0) * 1000.0

                if not cells:
                    matrix_records.append({
                        "method": method,
                        "task_id": task_id,
                        "domain": domain,
                        "prompt": prompt,
                        "passed": False,
                        "path": [],
                        "path_len": 0,
                        "latency_ms": round(dt_ms, 3),
                        "unification": False,
                        "preflight_lint": False,
                        "ast_valid": False,
                        "pass_attribution": "none",
                        "error": "Routing produced empty path"
                    })
                    continue

                path_ids = [c.cell_id for c in cells]
                method_stats[method]["path_lens"].append(len(cells))

                # Unification check
                try:
                    res = self.gate.unify_pipeline(cells, ctx)
                    unif_success = isinstance(res, Success)
                    pipeline_bindings = res.value if unif_success else []
                except Exception as e:
                    unif_success = False
                    pipeline_bindings = []

                # Preflight lint check
                lint_valid = False
                if unif_success and pipeline_bindings:
                    lint_res = PreflightLinter.lint(pipeline_bindings)
                    lint_valid = lint_res.is_valid

                # Code emission & AST validity check
                ast_valid = False
                code = ""
                if unif_success:
                    try:
                        code = self.gate.emit_code(cells, ctx)
                        ast.parse(code)
                        ast_valid = True
                    except Exception:
                        ast_valid = False

                passed = unif_success and ast_valid
                if passed:
                    method_stats[method]["passed"] += 1
                    method_stats[method]["latencies"].append(dt_ms)

                matrix_records.append({
                    "method": method,
                    "task_id": task_id,
                    "domain": domain,
                    "prompt": prompt,
                    "passed": passed,
                    "path": path_ids,
                    "path_len": len(cells),
                    "latency_ms": round(dt_ms, 3),
                    "unification": unif_success,
                    "preflight_lint": lint_valid,
                    "ast_valid": ast_valid,
                    "pass_attribution": "path" if passed else "none",
                    "code_snippet": code[:120] if code else ""
                })

        # Save to evaluation_results.json
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_evaluations": len(matrix_records),
            "methods_evaluated": TEST_METHODS,
            "tasks_evaluated": [t[0] for t in EVAL_TASKS],
            "method_summary": {},
            "records": matrix_records
        }

        print("\n| Method | Description | Pass Rate | Mean Latency (ms) | p50 (ms) | Mean Path Length |")
        print("| :--- | :--- | :---: | :---: | :---: | :---: |")

        for m in TEST_METHODS:
            tot = method_stats[m]["total"]
            pas = method_stats[m]["passed"]
            rate = (pas / tot * 100.0) if tot > 0 else 0.0
            lats = method_stats[m]["latencies"] or [0.0]
            lens = method_stats[m]["path_lens"] or [0.0]
            mean_lat = statistics.mean(lats)
            p50_lat = statistics.median(lats)
            mean_len = statistics.mean(lens)

            method_cls = ROUTE_METHOD_REGISTRY.get(m)
            desc = getattr(method_cls, "name", m) if method_cls else m

            summary["method_summary"][m] = {
                "description": desc,
                "total": tot,
                "passed": pas,
                "pass_rate_pct": round(rate, 2),
                "mean_latency_ms": round(mean_lat, 2),
                "p50_latency_ms": round(p50_lat, 2),
                "mean_path_len": round(mean_len, 2)
            }
            print(f"| **{m}** | {desc} | {rate:.1f}% ({pas}/{tot}) | {mean_lat:.2f} ms | {p50_lat:.2f} ms | {mean_len:.1f} |")

        print("=" * 80 + "\n")

        with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Evaluation matrix written to {EVAL_RESULTS_PATH}")

        # Assert baseline requirements: M0, M1, M6 should pass their core tasks
        self.assertGreaterEqual(method_stats["M0"]["passed"], 4)
        self.assertGreaterEqual(method_stats["M1"]["passed"], 4)
        self.assertGreaterEqual(method_stats["M6"]["passed"], 4)


if __name__ == "__main__":
    unittest.main()
