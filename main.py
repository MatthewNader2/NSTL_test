# main.py — NSTL Engine (FastAPI + PyWebview Integration)
import json
import logging
import os
import socket
import sys
import threading
import time
import warnings
from dataclasses import asdict

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # Dynamically locate the Python DLL bundled by PyInstaller and set PYTHONNET_PYDLL for pythonnet/clr.
    # We must target the version-specific DLL (e.g. python313.dll) and avoid python3.dll (stable ABI shim).
    dll_name = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    dll_path = os.path.join(sys._MEIPASS, dll_name)
    if os.path.exists(dll_path):
        os.environ["PYTHONNET_PYDLL"] = dll_path
    else:
        # Fallback to search if name is slightly different, but explicitly skip python3.dll
        for file in os.listdir(sys._MEIPASS):
            if file.lower().startswith("python") and file.lower().endswith(".dll") and file.lower() != "python3.dll":
                os.environ["PYTHONNET_PYDLL"] = os.path.join(sys._MEIPASS, file)
                break

import uvicorn
import webview
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =====================================================================
#  CRITICAL THREAD & ENVIRONMENT FIXES
# =====================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

TREES_DIR = get_resource_path("trees")
FRONTEND_DIR = get_resource_path("frontend_dist")

# Import our new modular architecture
from router import HardwareProfiler, MCTSEngine
from lattice import LatticeOrchestrator, AlgebraicSignature
from unification import ExecutionContext, UnificationGate
from planner import ZeroShotPlanner
from synthesis import SynthesisEngine
from external_rag import FetcherFactory
from internal_rag import LocalRAG
from inference import ModelManager

global_orchestrator = None
global_rag_engine = None
engine_device = "cpu"

# BUG 12 FIX: Lock to protect concurrent mutations of the shared orchestrator
# (inject_transient_macro modifies loaded_cells and rebuilds topology).
_orchestrator_lock = threading.Lock()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    prompt: str

@app.get("/api/status")
def get_status():
    if global_orchestrator is None:
        return {"status": "uninitialized", "device": engine_device, "cells_loaded": 0}
    return {"status": "ready", "device": engine_device, "cells_loaded": len(global_orchestrator.loaded_cells)}

@app.get("/api/health")
def health():
    if global_orchestrator is None:
        return {"status": "ok", "cells_loaded": 0}
    return {"status": "ok", "cells_loaded": len(global_orchestrator.loaded_cells)}

@app.get("/api/cells")
def get_cells():
    if global_orchestrator is None:
        return {"cells": [], "count": 0}
    cells = []
    for cell in global_orchestrator.get_all_available_cells():
        in_dict  = asdict(cell.inputs) if hasattr(cell, 'inputs') else {}
        out_dict = asdict(cell.outputs) if hasattr(cell, 'outputs') else {}
        cells.append({
            "cell_id": cell.cell_id,
            "stage": cell.stage if hasattr(cell, 'stage') else 0,
            "type": cell.type,
            "keywords": list(cell.keywords) if hasattr(cell, 'keywords') else [],
            "inputs": in_dict,
            "outputs": out_dict,
            "code_template": getattr(cell, 'code_template', "# No implementation")
        })
    return {"cells": cells, "count": len(cells)}

@app.post("/api/run")
def run_prompt(req: RunRequest):
    if global_orchestrator is None:
        return {
            "logs": [{"msg": "Engine is currently loading neural model, please wait...", "type": "warn"}],
            "path": [],
            "virtual_edges": [],
            "code": "# Engine is loading."
        }

    log_buffer = []
    log_buffer.append({"msg": "Extracting parameters and context...", "type": "info"})
    
    # We initialize context
    context = ExecutionContext()
    context.extract_prompt_parameters(req.prompt)
    
    # Using 'str' and 'source_identifier' for dummy bootstrap
    context.declare_variable(
        name="input_source", signature=AlgebraicSignature(type_name="str", state="source_identifier")
    )

    # 1. ZeroShotPlanner
    log_buffer.append({"msg": "Running ZeroShotPlanner pass...", "type": "info"})
    planner = ZeroShotPlanner(global_orchestrator, rag_engine=global_rag_engine)
    try:
        macro_graph = planner.run_planning_pass(req.prompt)
    except Exception as e:
        log_buffer.append({"msg": f"Planner failed: {e}", "type": "error"})
        return {"logs": log_buffer, "path": [], "virtual_edges": [], "code": f"# Planner Error: {e}"}

    # 2. Iterate Macro Graph and Gap Bridging
    final_micro_path = []
    virtual_edges = set()
    current_signature = AlgebraicSignature(type_name="str", state="source_identifier")
    
    if isinstance(macro_graph, dict):
        cells_list = macro_graph.get('cells', [macro_graph])
        macro_cell = cells_list[0] if cells_list else {}
    elif isinstance(macro_graph, list) and len(macro_graph) > 0:
        macro_cell = macro_graph[0]
    else:
        macro_cell = {}
        
    sub_cells_ids = macro_cell.get('sub_cells', []) if isinstance(macro_cell, dict) else []
    log_buffer.append({"msg": f"MCTS routing for {len(sub_cells_ids)} macro steps...", "type": "info"})
    
    for i, step_id in enumerate(sub_cells_ids):
        target_cell = global_orchestrator.loaded_cells.get(step_id)

        if target_cell is None:
            expected_inputs = current_signature.type_name
            expected_outputs = "any"
            for next_id in sub_cells_ids[i+1:]:
                next_cell = global_orchestrator.loaded_cells.get(next_id)
                if next_cell:
                    expected_outputs = next_cell.inputs.type_name
                    break

            log_buffer.append({"msg": f"Planner flagged MISSING_NODE ({step_id}) for {expected_inputs}->{expected_outputs}.", "type": "warn"})
            
            # 1. Composition Confidence
            mcts = MCTSEngine(global_orchestrator)
            comp_path = mcts.search(expected_inputs, expected_outputs, iterations=200)
            comp_confidence = 1.0 / (len(comp_path) + 1) if comp_path else 0.0

            # 2. Synthesis Confidence
            synth_micro_json = None
            synth_confidence = 0.0
            
            can_synth = ModelManager.get_instance().can_synthesize()
            if can_synth:
                synth = SynthesisEngine()
                fetcher = FetcherFactory.get_fetcher(global_orchestrator.active_domain)
                try:
                    gap_concept = f"{step_id}: convert {expected_inputs} to {expected_outputs}"
                    synth_micro_json = synth.synthesize_micro_cell(gap_concept, expected_inputs, expected_outputs, fetcher)
                    if UnificationGate.validate_synthesis(synth_micro_json, expected_inputs, expected_outputs, trees_dir=TREES_DIR):
                        synth_confidence = 0.85
                except Exception as e:
                    log_buffer.append({"msg": f"Synthesis failed: {e}", "type": "error"})

            # Decision Matrix
            if comp_confidence > synth_confidence and comp_confidence > 0:
                log_buffer.append({"msg": f"Composition confidence ({comp_confidence:.2f}) > Synthesis. Bridging using existing nodes.", "type": "info"})
                for n in comp_path:
                    final_micro_path.append(n)
                    virtual_edges.add(n.cell_id)
                    current_signature = n.outputs
                continue
            elif synth_confidence > 0:
                with _orchestrator_lock:
                    target_cell = global_orchestrator.inject_transient_macro(synth_micro_json)
                virtual_edges.add(target_cell.cell_id)
                log_buffer.append({"msg": f"Synthesis chosen (conf {synth_confidence:.2f}) for {expected_inputs}->{expected_outputs}", "type": "info"})
            else:
                log_buffer.append({"msg": "SAFETY ABORT: Cannot bridge or synthesize missing node.", "type": "error"})
                target_cell = None
                break

        if target_cell is None:
            continue

        if not current_signature.matches(target_cell.inputs):
            log_buffer.append({
                "msg": (
                    f"Bridging {current_signature.type_name}[{current_signature.state}] "
                    f"-> {target_cell.inputs.type_name}[{target_cell.inputs.state}] before {target_cell.cell_id}"
                ),
                "type": "info",
            })
            bridge_path = []
            if current_signature.type_name != target_cell.inputs.type_name:
                mcts = MCTSEngine(global_orchestrator)
                bridge_path = mcts.search(current_signature.type_name, target_cell.inputs.type_name, iterations=500)

            for bridge_node in bridge_path:
                final_micro_path.append(bridge_node)
                virtual_edges.add(bridge_node.cell_id)
                current_signature = bridge_node.outputs

            if not current_signature.matches(target_cell.inputs):
                log_buffer.append({
                    "msg": (
                        f"No exact typestate bridge found; applying {target_cell.cell_id} "
                        f"with latest compatible runtime value."
                    ),
                    "type": "warn",
                })

        final_micro_path.append(target_cell)
        current_signature = target_cell.outputs

    # 3. Code Generation
    compiled_blocks = []
    explicit_filename = context.extracted_parameters.get("explicit_filename")
    if explicit_filename:
        compiled_blocks.append(f"input_source = {explicit_filename!r}")
    for cell in final_micro_path:
        code_block = UnificationGate.unify(context, cell)
        if code_block:
            compiled_blocks.append(code_block)
            
    # Feedback Loop Check
    final_code = "\n".join(compiled_blocks)
    if ModelManager.get_instance().can_synthesize():
        log_buffer.append({"msg": "Running final generated code through feedback check...", "type": "info"})
        final_code = ModelManager.get_instance().feedback_check(final_code)
    
    # We rebuild compiled_blocks for the API response
    compiled_blocks = [final_code]

    # Format the path for the React frontend
    path_formatted = []
    for c in final_micro_path:
        # BUG 5 FIX: AlgebraicSignature has no .to_dict() method.
        # Use dataclasses.asdict() instead, which works correctly on @dataclass instances.
        in_dict  = asdict(c.inputs)  if hasattr(c, 'inputs')  else {}
        out_dict = asdict(c.outputs) if hasattr(c, 'outputs') else {}
        
        path_formatted.append({
            "cell_id": c.cell_id,
            "stage": c.stage if hasattr(c, 'stage') else 0,
            "type": c.type if hasattr(c, 'type') else 'micro',
            "keywords": list(c.keywords) if hasattr(c, 'keywords') else [],
            "inputs": in_dict,
            "outputs": out_dict,
        })
            
    return {
        "logs": log_buffer,
        "path": path_formatted,
        "virtual_edges": list(virtual_edges),
        "code": "\n".join(compiled_blocks) if compiled_blocks else "# No code generated.",
    }


class InitRequest(BaseModel):
    profile: str = "B"
    embedder_model: str = "jina-embeddings-v5-text-nano"
    llm_model: str = "qwen2.5-coder-1.5b-instruct"
    embedder_device: str = "auto"
    llm_device: str = "auto"
    trees_storage: str = "ram"

@app.post("/api/initialize")
def initialize_engine(req: InitRequest = InitRequest()):
    global global_orchestrator, global_rag_engine, engine_device
    try:
        HardwareProfiler.set_config(req.embedder_device, req.llm_device, req.trees_storage)

        # Initialize selected profile
        ModelManager.get_instance().initialize_profile(req.profile, req.embedder_model, req.llm_model)

        engine_device = HardwareProfiler.get_optimal_device()
        global_orchestrator = LatticeOrchestrator(trees_directory=TREES_DIR)
        
        global_rag_engine = LocalRAG(trees_dir=TREES_DIR)
        # Include device in response so the frontend can update its hardware indicator
        # immediately after initialization without waiting for the next /api/status poll.
        return {"status": "ready", "device": engine_device}
    except Exception as e:
        print(f"Error during initialization: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/benchmark/toggle")
def toggle_benchmark():
    """Enable or disable latency benchmarking. Off by default.
    The frontend calls this when the user clicks the Benchmark button in the GUI.
    """
    mm = ModelManager.get_instance()
    mm.benchmarking_enabled = not mm.benchmarking_enabled
    state = "enabled" if mm.benchmarking_enabled else "disabled"
    print(f"[BENCHMARK] Benchmarking {state} by user request.")
    return {"benchmarking": mm.benchmarking_enabled}

@app.get("/api/benchmark/status")
def get_benchmark_status():
    mm = ModelManager.get_instance()
    return {"benchmarking": getattr(mm, 'benchmarking_enabled', False)}

if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    def index():
        return {"msg": "Frontend not found, serving API only"}

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=58102, log_level="error")

def _wait_for_server(host: str, port: int, timeout: float = 10.0):
    """BUG 18 FIX: Poll until the uvicorn server is actually accepting connections
    before opening the WebView, to avoid loading a connection-refused page."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # BUG 18 FIX: Wait for the server to be ready before opening WebView.
    if not _wait_for_server("127.0.0.1", 58102):
        print("[WARNING] Server did not start in time; opening WebView anyway.")

    try:
        webview.create_window(
            "NSTL Engine",
            "http://127.0.0.1:58102",
            width=1400,
            height=900,
            min_size=(1000, 700),
            background_color="#0d1117",
        )
        webview.start(debug=True)
    except Exception as e:
        print(f"Failed to start pywebview: {e}")
        # Keep process alive for headless
        while True:
            time.sleep(1)
