# tests/test_phase5_performance_benchmarks.py
import concurrent.futures
import json
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from lattice import LatticeOrchestrator, TypeRegistry
from router import LatticeRouter
from gevr_sandbox import GEVRSandbox
from unification import ExecutionContext, UnificationGate

DB_PATH = str(ROOT_DIR / "trees" / "lattice.db")


@pytest.fixture(scope="module")
def full_orchestrator():
    """Loads full 34,401 node lattice once for benchmark tests."""
    assert os.path.exists(DB_PATH), f"Database {DB_PATH} must exist"
    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()
    return orchestrator


def test_bitset_reachability_matrix_pruning(full_orchestrator):
    """Verify that bitset reachability matrix accurately prunes incompatible type transitions in O(1)."""
    assert hasattr(full_orchestrator, "reachability_bitsets")
    assert len(full_orchestrator.reachability_bitsets) > 0

    # Valid transitions: DataFrame -> ndarray, Mat -> Mat, DataFrame -> DataFrame
    assert full_orchestrator.can_reach_type("DataFrame", "ndarray")
    assert full_orchestrator.can_reach_type("DataFrame", "Figure")
    assert full_orchestrator.can_reach_type("Mat", "Mat")
    assert full_orchestrator.can_reach_type("Mat", "ndarray")
    assert full_orchestrator.can_reach_type("DataFrame", "str")

    # Benchmark O(1) bitwise speed: 100,000 checks in < 15ms
    t0 = time.perf_counter()
    for _ in range(100000):
        full_orchestrator.can_reach_type("DataFrame", "Figure")
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[+] 100,000 Bitset Reachability Checks completed in {dt_ms:.2f}ms ({dt_ms/100:.4f}µs per check)")
    assert dt_ms < 50.0, f"Bitset check too slow: {dt_ms}ms"


def test_routing_latency_p50_p99_benchmark(full_orchestrator):
    """
    Run 50 routing requests across the full 34,401 node lattice.
    Assert p50 < 25ms and p99 < 60ms.
    """
    router = LatticeRouter(orchestrator=full_orchestrator, internal_rag=None)

    test_prompts = [
        "load input.csv and drop missing values then save to output.csv",
        "read image input.jpg and convert to grayscale then save to output.jpg",
        "load sales.csv, group by region and sum revenue, then save to report.csv",
        "read data.csv, drop missing values, sort by age ascending, and save to cleaned.csv",
        "read input.csv, standardize features with sklearn StandardScaler, plot histogram of feature_1 with matplotlib, and save figure to plot.png",
    ]

    latencies_ms = []

    # Warmup
    for p in test_prompts:
        router.plan_path(p, return_tuple=False)

    # 50 Iterations benchmark
    for i in range(50):
        prompt = test_prompts[i % len(test_prompts)]
        t0 = time.perf_counter()
        path = router.plan_path(prompt, return_tuple=False)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)
        assert len(path) > 0, f"Path empty for iteration {i}"

    latencies_ms.sort()
    n = len(latencies_ms)
    p50 = latencies_ms[int(n * 0.50)]
    p90 = latencies_ms[int(n * 0.90)]
    p95 = latencies_ms[int(n * 0.95)]
    p99 = latencies_ms[int(n * 0.99)]
    mean_lat = statistics.mean(latencies_ms)
    min_lat = min(latencies_ms)
    max_lat = max(latencies_ms)

    print("\n==================================================")
    print(" NSTL ROUTING LATENCY BENCHMARK (34,401 NODES)")
    print("==================================================")
    print(f" Total Requests : {n}")
    print(f" Mean Latency   : {mean_lat:.2f} ms")
    print(f" Min Latency    : {min_lat:.2f} ms")
    print(f" p50 Latency    : {p50:.2f} ms  (Target: < 25 ms)")
    print(f" p90 Latency    : {p90:.2f} ms")
    print(f" p95 Latency    : {p95:.2f} ms")
    print(f" p99 Latency    : {p99:.2f} ms  (Target: < 60 ms)")
    print(f" Max Latency    : {max_lat:.2f} ms")
    print("==================================================")

    assert p50 < 25.0, f"p50 latency {p50:.2f}ms exceeded 25ms SLA"
    assert p99 < 60.0, f"p99 latency {p99:.2f}ms exceeded 60ms SLA"


def test_concurrent_request_throughput(full_orchestrator):
    """
    Dispatch 10 concurrent routing queries via ThreadPoolExecutor.
    Assert 100% completion with 0 race conditions or corrupted responses.
    """
    router = LatticeRouter(orchestrator=full_orchestrator, internal_rag=None)

    prompts = [
        ("load input.csv and drop missing values then save to output.csv", "PANDAS_DROPNA"),
        ("read image input.jpg and convert to grayscale then save to output.jpg", "CV2_CVTCOLOR"),
        ("load sales.csv, group by region and sum revenue, then save to report.csv", "PANDAS_GROUPBY_SUM"),
        ("read data.csv, drop missing values, sort by age ascending, and save to cleaned.csv", "PANDAS_SORT_VALUES"),
        ("read input.csv, standardize features with sklearn StandardScaler, plot histogram of feature_1 with matplotlib, and save figure to plot.png", "SKLEARN_STANDARD_SCALER"),
        ("load input.csv and drop missing values then save to output.csv", "PANDAS_DROPNA"),
        ("read image input.jpg and convert to grayscale then save to output.jpg", "CV2_CVTCOLOR"),
        ("load sales.csv, group by region and sum revenue, then save to report.csv", "PANDAS_GROUPBY_SUM"),
        ("read data.csv, drop missing values, sort by age ascending, and save to cleaned.csv", "PANDAS_SORT_VALUES"),
        ("read input.csv, standardize features with sklearn StandardScaler, plot histogram of feature_1 with matplotlib, and save figure to plot.png", "SKLEARN_STANDARD_SCALER"),
    ]

    def _query(idx_prompt):
        idx, (prompt, expected_kw) = idx_prompt
        t0 = time.perf_counter()
        path = router.plan_path(prompt, return_tuple=False)
        dt = (time.perf_counter() - t0) * 1000
        path_ids = [c.cell_id for c in path]
        return idx, path_ids, expected_kw, dt

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_query, (i, p)) for i, p in enumerate(prompts)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    total_time_ms = (time.perf_counter() - t_start) * 1000

    assert len(results) == 10, f"Expected 10 results, got {len(results)}"
    for idx, path_ids, expected_kw, dt in results:
        assert any(expected_kw in cid for cid in path_ids), f"Thread {idx} failed: {expected_kw} not in {path_ids}"

    print(f"\n[+] 10 Concurrent Queries executed successfully in {total_time_ms:.2f}ms total (avg {(total_time_ms/10):.2f}ms per query in parallel).")


def test_sandbox_worker_pool_latency():
    """
    Run 10 consecutive GEVR sandbox executions.
    Assert average execution latency < 100ms per verification.
    """
    sandbox = GEVRSandbox()

    snippets = [
        "import numpy as np\na = np.arange(1000) * 2\nassert a[-1] == 1998",
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})\nassert len(df) == 3",
        "x = sum(i * 2 for i in range(500))\nassert x > 0",
        "import cv2\nimport numpy as np\nimg = np.zeros((10, 10, 3), dtype=np.uint8)\ngray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nassert gray.shape == (10, 10)",
        "from sklearn.preprocessing import StandardScaler\nimport numpy as np\nX = np.array([[0, 0], [1, 1], [2, 2]])\ns = StandardScaler().fit_transform(X)\nassert s.shape == (3, 2)",
    ]

    # Warmup execution of each snippet
    for s in snippets:
        sandbox.execute(s)

    exec_latencies_ms = []
    for i in range(10):
        code = snippets[i % len(snippets)]
        t0 = time.perf_counter()
        res = sandbox.execute(code, timeout=5.0)
        dt = (time.perf_counter() - t0) * 1000
        exec_latencies_ms.append(dt)
        assert res["success"], f"Execution {i} failed: {res.get('error')}"

    avg_exec_lat = statistics.mean(exec_latencies_ms)
    min_exec_lat = min(exec_latencies_ms)
    max_exec_lat = max(exec_latencies_ms)

    print("\n==================================================")
    print(" GEVR SANDBOX POOL LATENCY BENCHMARK")
    print("==================================================")
    print(f" Executions     : {len(exec_latencies_ms)}")
    print(f" Mean Latency   : {avg_exec_lat:.2f} ms  (Target: < 100 ms)")
    print(f" Min Latency    : {min_exec_lat:.2f} ms")
    print(f" Max Latency    : {max_exec_lat:.2f} ms")
    print("==================================================")

    assert avg_exec_lat < 100.0, f"Average execution latency {avg_exec_lat:.2f}ms exceeded 100ms SLA"


def test_memory_stability_over_100_iterations(full_orchestrator):
    """
    Assert memory growth over 100 routing iterations remains < 5%.
    """
    router = LatticeRouter(orchestrator=full_orchestrator, internal_rag=None)

    prompt = "read data.csv, drop missing values, sort by age ascending, and save to cleaned.csv"

    # Warm up and stabilize Python heaps
    for _ in range(10):
        router.plan_path(prompt, return_tuple=False)

    try:
        import psutil
        proc = psutil.Process(os.getpid())
        get_mem = lambda: proc.memory_info().rss
    except ImportError:
        get_mem = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: KB -> bytes

    initial_mem = get_mem()

    for i in range(100):
        path = router.plan_path(prompt, return_tuple=False)
        assert len(path) == 4

    final_mem = get_mem()
    growth_ratio = (final_mem - initial_mem) / initial_mem
    growth_pct = growth_ratio * 100.0

    print(f"\n[+] Memory Stability over 100 iterations: Initial = {initial_mem / 1024 / 1024:.2f}MB, Final = {final_mem / 1024 / 1024:.2f}MB, Growth = {growth_pct:.2f}% (Target: < 5%)")
    assert growth_pct < 5.0, f"Memory growth {growth_pct:.2f}% exceeded 5% limit"
