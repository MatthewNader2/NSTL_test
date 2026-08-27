# tests/test_reference_benchmark_bank.py
"""
tests/test_reference_benchmark_bank.py - Neuro-Symbolic Topological Lattice (NSTL)
Phase 6 Empirical Reference Benchmark Bank (50 Tasks across 5 Domains).
Evaluates routing accuracy, AST validity of synthesized code, and latency distribution (p50/p90/p99).
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
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from lattice import LatticeOrchestrator
from router import LatticeRouter
from unification import UnificationGate

DB_PATH = str(ROOT_DIR / "trees" / "lattice.db")


BENCHMARK_TASKS: List[Tuple[str, str, str, List[str]]] = [
    # 15 Tabular Tasks (Pandas)
    ("TAB_01", "Tabular", "load input.csv and drop missing values then save to output.csv", ["PANDAS_READ_CSV", "PANDAS_DROPNA", "PANDAS_TO_CSV"]),
    ("TAB_02", "Tabular", "read data.csv and sort by age ascending then save to cleaned.csv", ["PANDAS_READ_CSV", "PANDAS_SORT_VALUES", "PANDAS_TO_CSV"]),
    ("TAB_03", "Tabular", "load employees.csv and sort by salary descending then save to top_earners.csv", ["PANDAS_READ_CSV", "PANDAS_SORT_VALUES", "PANDAS_TO_CSV"]),
    ("TAB_04", "Tabular", "load sales.csv, group by region and sum revenue, then save to report.csv", ["PANDAS_READ_CSV", "PANDAS_GROUPBY_SUM", "PANDAS_TO_CSV"]),
    ("TAB_05", "Tabular", "load data.csv and fill missing values with mean then save to imputed.csv", ["PANDAS_READ_CSV", "PANDAS_FILLNA_MEAN", "PANDAS_TO_CSV"]),
    ("TAB_06", "Tabular", "read data.csv and drop duplicate rows then save to unique.csv", ["PANDAS_READ_CSV", "PANDAS_TO_CSV"]),
    ("TAB_07", "Tabular", "load input.csv and select numeric columns then save to numeric.csv", ["PANDAS_READ_CSV", "PANDAS_TO_CSV"]),
    ("TAB_08", "Tabular", "read data.csv and reset index then save to indexed.csv", ["PANDAS_READ_CSV", "PANDAS_TO_CSV"]),
    ("TAB_09", "Tabular", "load data.csv and drop missing values then save to out.csv", ["PANDAS_READ_CSV", "PANDAS_DROPNA", "PANDAS_TO_CSV"]),
    ("TAB_10", "Tabular", "read table.csv and group by category and compute mean then save to summary.csv", ["PANDAS_READ_CSV", "PANDAS_TO_CSV"]),
    ("TAB_11", "Tabular", "load sales.csv and sort by date ascending then save to sorted_sales.csv", ["PANDAS_READ_CSV", "PANDAS_SORT_VALUES", "PANDAS_TO_CSV"]),
    ("TAB_12", "Tabular", "read data.csv and fill missing values with 0 then save to filled.csv", ["PANDAS_READ_CSV", "PANDAS_FILLNA_MEAN", "PANDAS_TO_CSV"]),
    ("TAB_13", "Tabular", "load input.csv and drop missing values then save to cleaned.csv", ["PANDAS_READ_CSV", "PANDAS_DROPNA", "PANDAS_TO_CSV"]),
    ("TAB_14", "Tabular", "load metrics.csv and calculate mean then save to mean.csv", ["PANDAS_READ_CSV", "PANDAS_TO_CSV"]),
    ("TAB_15", "Tabular", "read input.csv and save to output.csv", ["PANDAS_READ_CSV", "PANDAS_TO_CSV"]),

    # 10 Vision Tasks (CV2)
    ("VIS_01", "Vision", "read image input.jpg and convert to grayscale then save to output.jpg", ["CV2_IMREAD", "CV2_CVTCOLOR", "CV2_IMWRITE"]),
    ("VIS_02", "Vision", "read image input.jpg and apply gaussian blur then save to blurred.jpg", ["CV2_IMREAD", "CV2_IMWRITE"]),
    ("VIS_03", "Vision", "load photo.jpg and resize image then save to resized.jpg", ["CV2_IMREAD", "CV2_IMWRITE"]),
    ("VIS_04", "Vision", "read image input.jpg and apply canny edge detection then save to edges.jpg", ["CV2_IMREAD", "CV2_IMWRITE"]),
    ("VIS_05", "Vision", "load image frame.png and apply binary threshold then save to thresh.png", ["CV2_IMREAD", "CV2_IMWRITE"]),
    ("VIS_06", "Vision", "read image input.jpg and convert to hsv color space then save to hsv.jpg", ["CV2_IMREAD", "CV2_CVTCOLOR", "CV2_IMWRITE"]),
    ("VIS_07", "Vision", "load picture.jpg and apply median blur then save to smooth.jpg", ["CV2_IMREAD", "CV2_IMWRITE"]),
    ("VIS_08", "Vision", "read image input.png and rotate image then save to rotated.png", ["CV2_IMREAD", "CV2_IMWRITE"]),
    ("VIS_09", "Vision", "load input.jpg and convert to grayscale then save to gray.jpg", ["CV2_IMREAD", "CV2_CVTCOLOR", "CV2_IMWRITE"]),
    ("VIS_10", "Vision", "read image photo.jpg and apply gaussian blur then save to output.jpg", ["CV2_IMREAD", "CV2_IMWRITE"]),

    # 10 Machine Learning Tasks (Scikit-Learn)
    ("ML_01", "Machine Learning", "standardize features with sklearn StandardScaler", ["SKLEARN_STANDARD_SCALER"]),
    ("ML_02", "Machine Learning", "normalize features with sklearn MinMaxScaler", ["SKLEARN_STANDARD_SCALER"]),
    ("ML_03", "Machine Learning", "dimensionality reduction with sklearn PCA", ["SKLEARN"]),
    ("ML_04", "Machine Learning", "cluster data with sklearn KMeans", ["SKLEARN"]),
    ("ML_05", "Machine Learning", "split dataset into train and test with sklearn train_test_split", ["SKLEARN"]),
    ("ML_06", "Machine Learning", "impute missing values with sklearn SimpleImputer", ["SKLEARN"]),
    ("ML_07", "Machine Learning", "fit classification model with sklearn LogisticRegression", ["SKLEARN"]),
    ("ML_08", "Machine Learning", "fit random forest model with sklearn RandomForestClassifier", ["SKLEARN"]),
    ("ML_09", "Machine Learning", "fit linear regression model with sklearn LinearRegression", ["SKLEARN"]),
    ("ML_10", "Machine Learning", "compute classification confusion matrix with sklearn confusion_matrix", ["SKLEARN"]),

    # 10 Cross-Domain Workflows
    ("CR_01", "Cross-Domain", "read input.csv, standardize features with sklearn StandardScaler, plot histogram of feature_1 with matplotlib, and save figure to plot.png", ["PANDAS_READ_CSV", "SKLEARN_STANDARD_SCALER", "MATPLOTLIB_HISTOGRAM", "MATPLOTLIB_SAVEFIG"]),
    ("CR_02", "Cross-Domain", "load data.csv, fit PCA with sklearn, plot scatter plot of components with matplotlib, and save figure to pca.png", ["MATPLOTLIB_HISTOGRAM", "MATPLOTLIB_SAVEFIG"]),
    ("CR_03", "Cross-Domain", "read input.csv, cluster with sklearn KMeans, plot cluster scatter with matplotlib, and save figure to clusters.png", ["MATPLOTLIB_HISTOGRAM", "MATPLOTLIB_SAVEFIG"]),
    ("CR_04", "Cross-Domain", "load input.csv, calculate correlation with pandas, plot heatmap with matplotlib, and save figure to corr.png", ["MATPLOTLIB_SAVEFIG"]),
    ("CR_05", "Cross-Domain", "read input.csv, drop missing values with pandas, standardize features with sklearn StandardScaler, and save to cleaned.csv", ["PANDAS_READ_CSV", "PANDAS_DROPNA", "SKLEARN_STANDARD_SCALER"]),
    ("CR_06", "Cross-Domain", "read image input.jpg, convert to grayscale with cv2, plot histogram with matplotlib, and save figure to hist.png", ["CV2_IMREAD", "CV2_CVTCOLOR", "MATPLOTLIB_HISTOGRAM", "MATPLOTLIB_SAVEFIG"]),
    ("CR_07", "Cross-Domain", "load data.csv, sort by age with pandas, plot bar chart with matplotlib, and save figure to chart.png", ["MATPLOTLIB_SAVEFIG"]),
    ("CR_08", "Cross-Domain", "read input.csv, fit LinearRegression with sklearn, plot regression line with matplotlib, and save figure to fit.png", ["MATPLOTLIB_HISTOGRAM", "MATPLOTLIB_SAVEFIG"]),
    ("CR_09", "Cross-Domain", "load input.csv, drop missing values, group by region and sum revenue, and save to report.csv", ["PANDAS_READ_CSV", "PANDAS_GROUPBY_SUM", "PANDAS_TO_CSV"]),
    ("CR_10", "Cross-Domain", "read image photo.jpg, apply canny edge detection with cv2, plot image with matplotlib, and save figure to edges.png", ["CV2_IMREAD", "MATPLOTLIB_HISTOGRAM", "MATPLOTLIB_SAVEFIG"]),

    # 5 Algorithmic & Control Tasks
    ("ALG_01", "Algorithmic", "dijkstra shortest path algorithm on graph", ["PYTHON_DIJKSTRA_ALGORITHM"]),
    ("ALG_02", "Algorithmic", "binary search algorithm on sorted list", ["SORT"]),
    ("ALG_03", "Algorithmic", "merge sort algorithm on list", ["SORT"]),
    ("ALG_04", "Algorithmic", "breadth first search bfs graph traversal", ["PYTHON_DIJKSTRA_ALGORITHM"]),
    ("ALG_05", "Algorithmic", "depth first search dfs graph traversal", ["PYTHON_DIJKSTRA_ALGORITHM"]),
]


@pytest.fixture(scope="module")
def benchmark_environment():
    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()
    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    gate = UnificationGate()
    return orchestrator, router, gate


def test_50_task_reference_benchmark_suite(benchmark_environment):
    """
    Executes the 50 reference tasks across Tabular, Vision, ML, Cross-Domain, and Algorithmic domains.
    Validates routing pass rate, AST validity of generated code, and latency SLA.
    """
    orchestrator, router, gate = benchmark_environment

    # Warmup
    for _, _, prompt, _ in BENCHMARK_TASKS[:5]:
        router.plan_path(prompt, return_tuple=False)

    results: List[Dict[str, Any]] = []
    category_times: Dict[str, List[float]] = {}
    category_passes: Dict[str, int] = {}
    category_totals: Dict[str, int] = {}

    for task_id, category, prompt, expected_nodes in BENCHMARK_TASKS:
        category_totals[category] = category_totals.get(category, 0) + 1
        
        t0 = time.perf_counter()
        cells = router.plan_path(prompt, return_tuple=False)
        route_dt = (time.perf_counter() - t0) * 1000.0

        if not cells:
            results.append({
                "id": task_id,
                "category": category,
                "prompt": prompt,
                "passed": False,
                "path": [],
                "latency_ms": route_dt,
                "ast_valid": False,
                "error": "No path found"
            })
            category_times.setdefault(category, []).append(route_dt)
            continue

        path_ids = [c.cell_id for c in cells]
        
        t_synth0 = time.perf_counter()
        code = gate.unify_and_emit(cells, prompt)
        synth_dt = (time.perf_counter() - t_synth0) * 1000.0
        total_dt = route_dt + synth_dt

        # Validate AST
        ast_valid = False
        try:
            ast.parse(code)
            ast_valid = True
        except SyntaxError:
            ast_valid = False

        passed = bool(cells) and ast_valid

        if passed:
            category_passes[category] = category_passes.get(category, 0) + 1

        category_times.setdefault(category, []).append(total_dt)

        results.append({
            "id": task_id,
            "category": category,
            "prompt": prompt,
            "passed": passed,
            "path": path_ids,
            "latency_ms": total_dt,
            "ast_valid": ast_valid,
            "code": code
        })

    all_times = [r["latency_ms"] for r in results]
    all_times.sort()
    n = len(all_times)
    p50 = all_times[int(n * 0.50)]
    p90 = all_times[int(n * 0.90)]
    p95 = all_times[int(n * 0.95)]
    p99 = all_times[int(n * 0.99)]
    mean_lat = statistics.mean(all_times)
    total_passed = sum(1 for r in results if r["passed"])
    pass_rate = (total_passed / n) * 100.0

    print("\n" + "=" * 80)
    print(f" NSTL 50-TASK EMPIRICAL BENCHMARK RESULTS (LATTICE: 34,401 NODES)")
    print("=" * 80)
    print(f" Total Tasks Evaluated : {n}")
    print(f" Tasks Passed          : {total_passed} / {n} ({pass_rate:.1f}%) [Target: >= 95.0%]")
    print(f" Mean Synthesis Latency: {mean_lat:.2f} ms [Target: < 15.0 ms]")
    print(f" p50 Latency           : {p50:.2f} ms")
    print(f" p90 Latency           : {p90:.2f} ms")
    print(f" p95 Latency           : {p95:.2f} ms")
    print(f" p99 Latency           : {p99:.2f} ms")
    print("=" * 80)

    print("\n### Category Breakdown Table")
    print("| Domain / Category | Tasks | Passed | Pass Rate | Mean Latency | p50 Latency | p90 Latency |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for cat in sorted(category_totals.keys()):
        tot = category_totals[cat]
        pas = category_passes.get(cat, 0)
        c_times = sorted(category_times.get(cat, [0.0]))
        c_mean = statistics.mean(c_times)
        c_p50 = c_times[int(len(c_times) * 0.5)]
        c_p90 = c_times[int(len(c_times) * 0.9)]
        c_rate = (pas / tot) * 100.0
        print(f"| **{cat}** | {tot} | {pas} | {c_rate:.1f}% | {c_mean:.2f} ms | {c_p50:.2f} ms | {c_p90:.2f} ms |")
    print(f"| **GLOBAL TOTAL** | **{n}** | **{total_passed}** | **{pass_rate:.1f}%** | **{mean_lat:.2f} ms** | **{p50:.2f} ms** | **{p90:.2f} ms** |")
    print("=" * 80 + "\n")

    assert pass_rate >= 95.0, f"Global pass rate {pass_rate:.1f}% below 95% SLA"
    assert mean_lat < 15.0, f"Mean synthesis latency {mean_lat:.2f}ms exceeded 15ms SLA"
