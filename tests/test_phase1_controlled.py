# tests/test_phase1_controlled.py
import os
import sys
import json
import sqlite3
import tempfile
import time
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.lattice import LatticeOrchestrator, Cell, PortSignature
from src.router import LatticeRouter
from src.unification import DynamicPlaceholderResolver, UnificationGate
from src.gevr_sandbox import GEVRSandbox

def setup_micro_lattice_db(db_path: str, json_fixture_path: str):
    with open(json_fixture_path, "r") as f:
        cells_data = json.load(f)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cells (
            cell_id TEXT PRIMARY KEY,
            stage INTEGER,
            input_type TEXT,
            input_state TEXT,
            output_type TEXT,
            output_state TEXT,
            code_template TEXT,
            configuration_schema TEXT,
            dependencies TEXT
        )
    """)
    for c in cells_data:
        in_p = next(iter(c["inputs"].values()))
        out_p = next(iter(c["outputs"].values()))
        cur.execute("""
            INSERT OR REPLACE INTO cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            c["cell_id"],
            c["stage"],
            in_p["type_name"],
            in_p.get("state", "default"),
            out_p["type_name"],
            out_p.get("state", "default"),
            c["code_template"],
            json.dumps({"inputs": c["inputs"], "outputs": c["outputs"]}),
            json.dumps(["pandas"])
        ))
    conn.commit()
    conn.close()

def test_phase1_controlled_micro_lattice():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "micro_lattice.db")
        fixture_path = os.path.join(str(ROOT_DIR), "tests/fixtures/phase1_micro_lattice.json")
        
        setup_micro_lattice_db(db_path, fixture_path)
        
        # Load orchestrator
        orchestrator = LatticeOrchestrator()
        orchestrator.load_from_database(db_path)
        orchestrator.build_topology()
        
        # Create input CSV
        input_csv = os.path.join(tmpdir, "data.csv")
        output_csv = os.path.join(tmpdir, "cleaned.csv")
        df_in = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [35, None, 25, 40]
        })
        df_in.to_csv(input_csv, index=False)
        
        prompt = "Read data.csv, drop missing values, sort by age ascending, save to cleaned.csv"
        
        # Execution Context Mock
        class Context:
            source_files = [input_csv]
            dest_files = [output_csv]
            columns = ["age"]
            flags = {"ascending": True}
            prompt_lower = prompt.lower()
            scope_variables = {}
            parameters = {}
        
        ctx = Context()
        
        t0 = time.perf_counter()
        
        # Target Path Sequence
        expected_path = ["PANDAS_READ_CSV", "PANDAS_DROPNA", "PANDAS_SORT_VALUES", "PANDAS_TO_CSV"]
        
        # Verify Path Selection
        router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)
        start_sig = PortSignature("str", "source_identifier")
        goal_sig = PortSignature("str", "filepath_written")
        
        selected_cells = router.plan_path(
            prompt=prompt,
            start_sig=start_sig,
            goal_sig=goal_sig
        )
        
        selected_ids = [c.cell_id for c in selected_cells]
        lat_ms = (time.perf_counter() - t0) * 1000
        
        print(f"\n[Phase 1] Selected path: {selected_ids} in {lat_ms:.2f}ms")
        
        # Assertions
        assert selected_ids == expected_path, f"Path mismatch! Got: {selected_ids}, Expected: {expected_path}"
        
        # Negative Assertions: Distractors must NEVER be selected
        distractors = ["DISTRACTOR_BROKEN_READ", "DISTRACTOR_BROKEN_SORT", "DISTRACTOR_BROKEN_WRITE"]
        for d in distractors:
            assert d not in selected_ids, f"Distractor {d} was improperly selected!"
        
        # Code Generation & Monadic Unification
        gate = UnificationGate()
        resolver = DynamicPlaceholderResolver()
        
        generated_lines = ["import pandas as pd"]
        var_counter = 0
        
        for cell in selected_cells:
            var_counter += 1
            out_var = f"var_{var_counter}"
            code_line = cell.code_template
            
            for p_name, p_sig in cell.inputs.items():
                val = resolver.resolve_port(p_name, p_sig, cell.stage, ctx, out_var)
                code_line = code_line.replace(f"{{{p_name}}}", val)
            
            code_line = code_line.replace("{output_var}", out_var)
            resolver.assert_placeholders_resolved(code_line)
            
            # Register in scope
            out_sig = next(iter(cell.outputs.values()))
            ctx.scope_variables[out_var] = out_sig
            generated_lines.append(code_line)
            
        full_code = "\n".join(generated_lines)
        print(f"\n[Phase 1] Generated Code:\n{full_code}\n")
        
        # Sandbox functional verification
        sandbox = GEVRSandbox()
        result = sandbox.execute(full_code, timeout=5)
        
        assert result["success"], f"GEVR execution failed: {result.get("error")}"
        assert os.path.exists(output_csv), "Output CSV was not created!"
        
        df_out = pd.read_csv(output_csv)
        assert len(df_out) == 3, f"Expected 3 rows (1 dropped), got {len(df_out)}"
        assert list(df_out["age"]) == [25.0, 35.0, 40.0], f"Incorrect sort order: {list(df_out["age"])}"
        
        print("✅ PHASE 1 CONTROLLED TEST PASSED (100% Correctness, 0 Distractors Selected)")

if __name__ == "__main__":
    test_phase1_controlled_micro_lattice()
