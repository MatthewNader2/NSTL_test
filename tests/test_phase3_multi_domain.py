# tests/test_phase3_multi_domain.py
import os
import sys
import time
import tempfile
import cv2
import numpy as np
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

from lattice import LatticeOrchestrator, MicroCell, PortSignature, AlgebraicSignature
from router import LatticeRouter
from unification import UnificationGate, ExecutionContext
from gevr_sandbox import GEVRSandbox
from compile_trees import compile_database

DB_PATH = str(ROOT_DIR / "tests/fixtures/all_trees_full.db")


def ensure_all_trees_db():
    if not os.path.exists(DB_PATH):
        print("[Phase 3] Compiling tests/fixtures/all_trees_full.db across all domains...")
        compile_database(output_db=DB_PATH)


def test_task1_pure_tabular_domain_isolation():
    """Task 1: Pure Tabular Domain Isolation with zero CV2/Vision/Sklearn leakage."""
    ensure_all_trees_db()

    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()

    assert len(orchestrator.loaded_cells) > 30000, f"Expected 30k+ cells, got {len(orchestrator.loaded_cells)}"

    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    sandbox = GEVRSandbox()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_csv = os.path.join(tmpdir, "data.csv")
        cleaned_csv = os.path.join(tmpdir, "cleaned.csv")
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [30.0, None, 22.0, 45.0]
        })
        df.to_csv(data_csv, index=False)

        prompt = f"Read {data_csv}, drop missing values, sort by age ascending, and save to {cleaned_csv}"

        t0 = time.perf_counter()
        path, _ = router.plan_path(prompt, return_tuple=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        cell_ids = [c.cell_id for c in path]
        print(f"\n[Phase 3 - Task 1] Path: {cell_ids} in {latency_ms:.2f}ms")

        # 1. Zero Cross-Domain Leakage
        for c in path:
            domain = (c.domain_name or "").lower()
            assert domain in ("pandas", "generic", "python_core"), f"Cross-domain leak: {c.cell_id} in {domain}"
            assert not any(p in c.cell_id.lower() for p in ("cv2", "opencv", "sklearn", "scipy")), (
                f"Unrelated domain node in tabular path: {c.cell_id}"
            )

        # 2. Latency assertion (< 100ms)
        assert latency_ms < 100.0, f"Latency exceeded 100ms: {latency_ms:.2f}ms"

        # 3. Path structure assertions
        assert "PANDAS_READ_CSV" in cell_ids, "Missing PANDAS_READ_CSV!"
        assert "PANDAS_DROPNA" in cell_ids, "Missing PANDAS_DROPNA!"
        assert "PANDAS_SORT_VALUES" in cell_ids, "Missing PANDAS_SORT_VALUES!"
        assert "PANDAS_TO_CSV" in cell_ids, "Missing PANDAS_TO_CSV!"

        # 4. Code Generation & Sandbox Execution
        ctx = ExecutionContext(prompt=prompt)
        lines = [UnificationGate.unify_cell(ctx, c) for c in path]
        code = UnificationGate.resolve_imports("\n".join(lines), ctx)
        print(f"[Phase 3 - Task 1] Generated Code:\n{code}\n")

        result = sandbox.execute(code, timeout=5)
        assert result["success"], f"GEVR execution failed: {result.get('error')}"
        assert os.path.exists(cleaned_csv), "cleaned.csv was not created!"

        df_out = pd.read_csv(cleaned_csv)
        assert len(df_out) == 3, f"Expected 3 rows after dropna, got {len(df_out)}"
        assert df_out["age"].isna().sum() == 0, "Missing values were not dropped!"
        assert list(df_out["age"]) == [22.0, 30.0, 45.0], f"Ages not sorted ascending: {list(df_out['age'])}"
        print("✅ Task 1 (Pure Tabular Domain Isolation) PASSED")


def test_task2_pure_vision_domain_isolation():
    """Task 2: Pure Vision Domain Isolation with zero Tabular/Pandas/Sklearn leakage."""
    ensure_all_trees_db()

    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()

    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    sandbox = GEVRSandbox()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_jpg = os.path.join(tmpdir, "input.jpg")
        gray_jpg = os.path.join(tmpdir, "gray.jpg")

        # Create test RGB image
        img_rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        img_rgb[:, :] = (200, 100, 50)
        cv2.imwrite(input_jpg, img_rgb)

        prompt = f"Read image {input_jpg}, convert to grayscale, and save to {gray_jpg}"

        t0 = time.perf_counter()
        path, _ = router.plan_path(prompt, return_tuple=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        cell_ids = [c.cell_id for c in path]
        print(f"\n[Phase 3 - Task 2] Path: {cell_ids} in {latency_ms:.2f}ms")

        # 1. Zero Cross-Domain Leakage
        for c in path:
            domain = (c.domain_name or "").lower()
            assert domain in ("cv2", "opencv", "generic", "python_core"), f"Cross-domain leak: {c.cell_id} in {domain}"
            assert not any(p in c.cell_id.lower() for p in ("pandas", "sklearn", "scipy", "matplotlib")), (
                f"Unrelated domain node in vision path: {c.cell_id}"
            )

        # 2. Latency assertion (< 100ms)
        assert latency_ms < 100.0, f"Latency exceeded 100ms: {latency_ms:.2f}ms"

        # 3. Path structure assertions
        assert cell_ids == ["CV2_IMREAD", "CV2_CVTCOLOR", "CV2_IMWRITE"], f"Unexpected path: {cell_ids}"

        # 4. Code Generation & Sandbox Execution
        ctx = ExecutionContext(prompt=prompt)
        lines = [UnificationGate.unify_cell(ctx, c) for c in path]
        code = UnificationGate.resolve_imports("\n".join(lines), ctx)
        print(f"[Phase 3 - Task 2] Generated Code:\n{code}\n")

        result = sandbox.execute(code, timeout=5)
        assert result["success"], f"GEVR execution failed: {result.get('error')}"
        assert os.path.exists(gray_jpg), "gray.jpg was not created!"

        img_out = cv2.imread(gray_jpg, cv2.IMREAD_UNCHANGED)
        assert img_out is not None, "Failed to read output image!"
        assert len(img_out.shape) == 2, f"Expected 1-channel grayscale image, got shape {img_out.shape}"
        assert img_out.shape == (120, 160), f"Image shape mismatch: {img_out.shape}"
        print("✅ Task 2 (Pure Vision Domain Isolation) PASSED")


def test_task3_mixed_cross_domain_pipeline():
    """Task 3: Mixed Cross-Domain Pipeline (DataFrame -> ndarray -> Figure -> File)."""
    ensure_all_trees_db()

    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()

    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    sandbox = GEVRSandbox()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "input.csv")
        plot_png = os.path.join(tmpdir, "plot.png")

        df_in = pd.DataFrame({
            "feature_1": [1.0, 2.5, 3.0, 4.5, 5.0, 6.0, 7.5, 8.0],
            "feature_2": [10.0, 20.0, 15.0, 40.0, 35.0, 60.0, 55.0, 80.0]
        })
        df_in.to_csv(input_csv, index=False)

        prompt = f"Read {input_csv}, standardize features with sklearn StandardScaler, plot histogram of feature_1 with matplotlib, and save figure to {plot_png}"

        t0 = time.perf_counter()
        path, _ = router.plan_path(prompt, return_tuple=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        cell_ids = [c.cell_id for c in path]
        print(f"\n[Phase 3 - Task 3] Path: {cell_ids} in {latency_ms:.2f}ms")

        # 1. Path structure assertions
        expected_path = [
            "PANDAS_READ_CSV",
            "SKLEARN_STANDARD_SCALER",
            "MATPLOTLIB_HISTOGRAM",
            "MATPLOTLIB_SAVEFIG"
        ]
        assert cell_ids == expected_path, f"Expected {expected_path}, got {cell_ids}"

        # 2. Latency assertion (< 100ms)
        assert latency_ms < 100.0, f"Latency exceeded 100ms: {latency_ms:.2f}ms"

        # 3. Typestate Continuity Check
        for i in range(len(path) - 1):
            curr_out = path[i].primary_output
            next_in = path[i+1].primary_input
            assert curr_out.unifies_with(next_in) or path[i+1].can_accept(curr_out), (
                f"Typestate mismatch between {path[i].cell_id} ({curr_out}) and {path[i+1].cell_id} ({next_in})"
            )

        # 4. Code Generation & Sandbox Execution
        ctx = ExecutionContext(prompt=prompt)
        lines = [UnificationGate.unify_cell(ctx, c) for c in path]
        code = UnificationGate.resolve_imports("\n".join(lines), ctx)
        print(f"[Phase 3 - Task 3] Generated Code:\n{code}\n")

        result = sandbox.execute(code, timeout=5)
        assert result["success"], f"GEVR execution failed: {result.get('error')}"
        assert os.path.exists(plot_png), "plot.png was not created!"
        assert os.path.getsize(plot_png) > 0, "plot.png is empty!"
        print("✅ Task 3 (Mixed Cross-Domain Pipeline) PASSED")


def test_task4_adversarial_distractor_inoculation():
    """Task 4: Adversarial Distractor Inoculation (Typestate gating rejects invalid distractors)."""
    ensure_all_trees_db()

    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()

    # Inoculate malicious distractor with high keyword overlap but incompatible typestate
    evil_cell = MicroCell(
        cell_id="EVIL_READ_CSV",
        stage=1,
        inputs={"x": PortSignature(name="x", signature=AlgebraicSignature(type_name="InvalidType", state="source_identifier"))},
        outputs={"output_data": PortSignature(name="output_data", signature=AlgebraicSignature(type_name="DataFrame", state="raw"))},
        code_template="{output_var} = pd.read_csv({x})",
        domain_name="pandas",
        keywords={"read", "csv", "load", "data", "table", "dataset"}
    )
    orchestrator.register_cell(evil_cell)

    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    sandbox = GEVRSandbox()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_csv = os.path.join(tmpdir, "data.csv")
        out_csv = os.path.join(tmpdir, "out.csv")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(data_csv, index=False)

        prompt = f"Load {data_csv} and save to {out_csv}"

        t0 = time.perf_counter()
        path, _ = router.plan_path(prompt, return_tuple=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        cell_ids = [c.cell_id for c in path]
        print(f"\n[Phase 3 - Task 4] Path: {cell_ids} in {latency_ms:.2f}ms")

        # 1. Distractor Inoculation Assertion: EVIL_READ_CSV must NOT be selected
        assert "EVIL_READ_CSV" not in cell_ids, "Adversarial distractor EVIL_READ_CSV was erroneously selected!"
        assert "PANDAS_READ_CSV" in cell_ids, "Valid loader PANDAS_READ_CSV was not selected!"
        assert "PANDAS_TO_CSV" in cell_ids, "Valid sink PANDAS_TO_CSV was not selected!"

        # 2. Latency assertion (< 100ms)
        assert latency_ms < 100.0, f"Latency exceeded 100ms: {latency_ms:.2f}ms"

        # 3. Code Generation & Sandbox Execution
        ctx = ExecutionContext(prompt=prompt)
        lines = [UnificationGate.unify_cell(ctx, c) for c in path]
        code = UnificationGate.resolve_imports("\n".join(lines), ctx)
        print(f"[Phase 3 - Task 4] Generated Code:\n{code}\n")

        result = sandbox.execute(code, timeout=5)
        assert result["success"], f"GEVR execution failed: {result.get('error')}"
        assert os.path.exists(out_csv), "out.csv was not created!"

        df_res = pd.read_csv(out_csv)
        assert df_res.equals(df), "Saved dataframe does not match original data!"
        print("✅ Task 4 (Adversarial Distractor Inoculation) PASSED")


def test_out_of_order_prompt_planning():
    """Task 5: Out-of-Order Natural Language Prompt Planning and Typestate Synthesis."""
    ensure_all_trees_db()

    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()

    router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
    sandbox = GEVRSandbox()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_csv = os.path.join(tmpdir, "data.csv")
        cleaned_csv = os.path.join(tmpdir, "cleaned.csv")
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [30.0, None, 22.0, 45.0]
        })
        df.to_csv(data_csv, index=False)

        # Intentionally out-of-order: destination stated first, source middle, transformations last
        prompt = f"Save to {cleaned_csv} the dataset from {data_csv} after dropping missing values and sorting by age ascending"

        t0 = time.perf_counter()
        path, _ = router.plan_path(prompt, return_tuple=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        cell_ids = [c.cell_id for c in path]
        print(f"\n[Phase 3 - Out of Order] Path: {cell_ids} in {latency_ms:.2f}ms")

        # 1. Path structure assertion: despite out-of-order phrasing, data flow must be correct
        expected_path = [
            "PANDAS_READ_CSV",
            "PANDAS_DROPNA",
            "PANDAS_SORT_VALUES",
            "PANDAS_TO_CSV"
        ]
        assert cell_ids == expected_path, f"Expected {expected_path}, got {cell_ids}"

        # 2. Latency assertion (< 100ms)
        assert latency_ms < 100.0, f"Latency exceeded 100ms: {latency_ms:.2f}ms"

        # 3. Code Generation & Sandbox Execution
        ctx = ExecutionContext(prompt=prompt)
        lines = [UnificationGate.unify_cell(ctx, c) for c in path]
        code = UnificationGate.resolve_imports("\n".join(lines), ctx)
        print(f"[Phase 3 - Out of Order] Generated Code:\n{code}\n")

        result = sandbox.execute(code, timeout=5)
        assert result["success"], f"GEVR execution failed: {result.get('error')}"
        assert os.path.exists(cleaned_csv), "cleaned.csv was not created!"

        df_res = pd.read_csv(cleaned_csv)
        assert len(df_res) == 3, f"Expected 3 rows after dropna, got {len(df_res)}"
        assert list(df_res["name"]) == ["Charlie", "Alice", "David"], f"Incorrect sort order: {list(df_res['name'])}"
        print("✅ Task 5 (Out-of-Order Prompt Planning) PASSED")

