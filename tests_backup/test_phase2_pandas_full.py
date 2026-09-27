# tests/test_phase2_pandas_full.py
import os
import sys
import time
import tempfile
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
TOOLS_DIR = ROOT_DIR / "tools"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lattice import LatticeOrchestrator, PortSignature
from router import LatticeRouter
from unification import UnificationGate, ExecutionContext
from gevr_sandbox import GEVRSandbox
from compile_trees import compile_database

DB_PATH = str(ROOT_DIR / "tests/fixtures/pandas_full.db")


def ensure_pandas_full_db():
    if not os.path.exists(DB_PATH):
        print("[Phase 2] Compiling tests/fixtures/pandas_full.db...")
        compile_database(output_db=DB_PATH, domain_filter=["pandas", "macro"])


def test_task1_groupby_and_aggregate():
    ensure_pandas_full_db()

    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()

    assert len(orchestrator.loaded_cells) > 1000, f"Expected full pandas tree, got {len(orchestrator.loaded_cells)}"
    assert len(orchestrator._cells_by_input) > 0, "Poset reverse index _cells_by_input is empty!"
    assert len(orchestrator._cells_by_output) > 0, "Poset reverse index _cells_by_output is empty!"

    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    sandbox = GEVRSandbox()

    with tempfile.TemporaryDirectory() as tmpdir:
        sales_csv = os.path.join(tmpdir, "sales.csv")
        report_csv = os.path.join(tmpdir, "report.csv")
        df_sales = pd.DataFrame({
            "region": ["North", "South", "North", "East", "South"],
            "revenue": [100.0, 200.0, 150.0, 300.0, 50.0],
            "salesperson": ["Alice", "Bob", "Charlie", "David", "Eve"]
        })
        df_sales.to_csv(sales_csv, index=False)

        prompt = f"Load {sales_csv}, group by region and sum revenue, then save to {report_csv}"

        t0 = time.perf_counter()
        path, _ = router.plan_path(prompt, return_tuple=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        cell_ids = [c.cell_id for c in path]
        print(f"\n[Phase 2 - Task 1] Path: {cell_ids} in {latency_ms:.2f}ms")

        # 1. Zero cross-domain leakage assertion
        for c in path:
            assert (c.domain_name or "").lower() in ("pandas", "generic", "python_core"), f"Cross-domain leak: {c.cell_id} in {c.domain_name}"
            assert not any(p in c.cell_id.lower() for p in ("cv2", "opencv", "numpy", "scipy", "sklearn")), f"Unrelated domain node in path: {c.cell_id}"

        # 2. Latency assertion (< 50ms)
        assert latency_ms < 50.0, f"Latency exceeded 50ms: {latency_ms:.2f}ms"

        # 3. Path structure assertions
        assert any("READ" in cid for cid in cell_ids), "Missing read node in path!"
        assert any("GROUPBY" in cid or "SUM" in cid for cid in cell_ids), "Missing groupby/sum node in path!"
        assert any("TO_CSV" in cid or "WRITE" in cid or "SAVE" in cid for cid in cell_ids), "Missing sink node in path!"

        # 4. Code Generation & Sandbox Execution
        ctx = ExecutionContext(prompt=prompt)
        lines = [UnificationGate.unify_cell(ctx, c) for c in path]
        code = UnificationGate.resolve_imports("\n".join(lines), ctx)
        print(f"[Phase 2 - Task 1] Generated Code:\n{code}\n")

        result = sandbox.execute(code, timeout=5)
        assert result["success"], f"GEVR execution failed: {result.get("error")}"
        assert os.path.exists(report_csv), "report.csv was not created!"

        df_out = pd.read_csv(report_csv)
        assert len(df_out) == 3, f"Expected 3 regions, got {len(df_out)}"
        assert set(df_out["region"]) == {"North", "South", "East"}
        assert df_out.loc[df_out["region"] == "North", "revenue"].values[0] == 250.0
        assert df_out.loc[df_out["region"] == "East", "revenue"].values[0] == 300.0
        assert df_out.loc[df_out["region"] == "South", "revenue"].values[0] == 250.0
        print("✅ Task 1 (Group By & Aggregate) PASSED")


def test_task2_missing_value_imputation():
    ensure_pandas_full_db()

    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()

    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    sandbox = GEVRSandbox()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_csv = os.path.join(tmpdir, "data.csv")
        imputed_csv = os.path.join(tmpdir, "imputed.csv")
        df_data = pd.DataFrame({
            "val1": [10.0, None, 30.0, 40.0],
            "val2": [None, 2.0, 4.0, None]
        })
        df_data.to_csv(data_csv, index=False)

        prompt = f"Load {data_csv}, fill missing values with mean, and save to {imputed_csv}"

        t0 = time.perf_counter()
        path, _ = router.plan_path(prompt, return_tuple=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        cell_ids = [c.cell_id for c in path]
        print(f"\n[Phase 2 - Task 2] Path: {cell_ids} in {latency_ms:.2f}ms")

        # 1. Zero cross-domain leakage assertion
        for c in path:
            assert (c.domain_name or "").lower() in ("pandas", "generic", "python_core"), f"Cross-domain leak: {c.cell_id} in {c.domain_name}"
            assert not any(p in c.cell_id.lower() for p in ("cv2", "opencv", "numpy", "scipy", "sklearn")), f"Unrelated domain node in path: {c.cell_id}"

        # 2. Latency assertion (< 50ms)
        assert latency_ms < 50.0, f"Latency exceeded 50ms: {latency_ms:.2f}ms"

        # 3. Path structure assertions
        assert any("READ" in cid for cid in cell_ids), "Missing read node in path!"
        assert any("FILLNA" in cid or "IMPUTE" in cid or "MEAN" in cid for cid in cell_ids), "Missing fillna/imputation node in path!"
        assert any("TO_CSV" in cid or "WRITE" in cid or "SAVE" in cid for cid in cell_ids), "Missing sink node in path!"

        # 4. Code Generation & Sandbox Execution
        ctx = ExecutionContext(prompt=prompt)
        lines = [UnificationGate.unify_cell(ctx, c) for c in path]
        code = UnificationGate.resolve_imports("\n".join(lines), ctx)
        print(f"[Phase 2 - Task 2] Generated Code:\n{code}\n")

        result = sandbox.execute(code, timeout=5)
        assert result["success"], f"GEVR execution failed: {result.get("error")}"
        assert os.path.exists(imputed_csv), "imputed.csv was not created!"

        df_out = pd.read_csv(imputed_csv)
        assert df_out.isnull().sum().sum() == 0, f"Null values remain in imputed data!\n{df_out}"
        assert len(df_out) == 4, f"Expected 4 rows, got {len(df_out)}"
        print("✅ Task 2 (Missing Value Imputation & Mean Filling) PASSED")


def test_task3_multiport_sorting_descending():
    ensure_pandas_full_db()

    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()

    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    sandbox = GEVRSandbox()

    with tempfile.TemporaryDirectory() as tmpdir:
        emp_csv = os.path.join(tmpdir, "employees.csv")
        top_csv = os.path.join(tmpdir, "top_earners.csv")
        df_emp = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "salary": [70000, 120000, 95000, 50000]
        })
        df_emp.to_csv(emp_csv, index=False)

        prompt = f"Read {emp_csv}, sort by salary descending, and write to {top_csv}"

        t0 = time.perf_counter()
        path, _ = router.plan_path(prompt, return_tuple=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        cell_ids = [c.cell_id for c in path]
        print(f"\n[Phase 2 - Task 3] Path: {cell_ids} in {latency_ms:.2f}ms")

        # 1. Zero cross-domain leakage assertion
        for c in path:
            assert (c.domain_name or "").lower() in ("pandas", "generic", "python_core"), f"Cross-domain leak: {c.cell_id} in {c.domain_name}"
            assert not any(p in c.cell_id.lower() for p in ("cv2", "opencv", "numpy", "scipy", "sklearn")), f"Unrelated domain node in path: {c.cell_id}"

        # 2. Latency assertion (< 50ms)
        assert latency_ms < 50.0, f"Latency exceeded 50ms: {latency_ms:.2f}ms"

        # 3. Path structure assertions
        assert any("READ" in cid for cid in cell_ids), "Missing read node in path!"
        assert any("SORT" in cid for cid in cell_ids), "Missing sort node in path!"
        assert any("TO_CSV" in cid or "WRITE" in cid or "SAVE" in cid for cid in cell_ids), "Missing sink node in path!"

        # 4. Code Generation & Sandbox Execution
        ctx = ExecutionContext(prompt=prompt)
        lines = [UnificationGate.unify_cell(ctx, c) for c in path]
        code = UnificationGate.resolve_imports("\n".join(lines), ctx)
        print(f"[Phase 2 - Task 3] Generated Code:\n{code}\n")

        # Multi-port parameter binding assertions
        assert ("'salary'" in code) or ('"salary"' in code), "Failed to bind by='salary' parameter!"
        assert "ascending=False" in code, "Failed to bind ascending=False parameter!"

        result = sandbox.execute(code, timeout=5)
        assert result["success"], f"GEVR execution failed: {result.get("error")}"
        assert os.path.exists(top_csv), "top_earners.csv was not created!"

        df_out = pd.read_csv(top_csv)
        assert list(df_out["salary"]) == [120000, 95000, 70000, 50000], f"Incorrect sort order: {list(df_out['salary'])}"
        print("✅ Task 3 (Multi-Port Sorting & Projection) PASSED")


if __name__ == "__main__":
    test_task1_groupby_and_aggregate()
    test_task2_missing_value_imputation()
    test_task3_multiport_sorting_descending()
    print("\n🎉 ALL 3 PHASE 2 TASKS PASSED AND VERIFIED!")
