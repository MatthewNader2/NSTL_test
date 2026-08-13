# main.py — NSTL Engine (FastAPI + PyWebview Integration)
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import asdict

if sys.platform.startswith("win"):
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

# Initialize centralized logging
from log_config import setup_logging, get_logger
setup_logging()
logger = get_logger("main")

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_resource_path(relative_path):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass_path = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(meipass_path):
            return meipass_path

    src_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(src_dir, relative_path)
    if os.path.exists(src_path):
        return src_path

    root_dir = os.path.abspath(os.path.join(src_dir, ".."))
    root_path = os.path.join(root_dir, relative_path)
    if os.path.exists(root_path):
        return root_path

    return root_path

TREES_DIR = get_resource_path("trees")
FRONTEND_DIR = get_resource_path("frontend_dist")

# Import our new modular architecture
from router import HardwareProfiler, MCTSEngine, SynthesisContext
from lattice import LatticeOrchestrator, AlgebraicSignature
from unification import ExecutionContext, UnificationGate
from planner import ZeroShotPlanner
from synthesis import SynthesisEngine
from external_rag import FetcherFactory
from internal_rag import LocalRAG
from inference import ModelManager
from config import API_HOST, API_PORT, CORS_ORIGINS

global_orchestrator = None
global_rag_engine = None
engine_device = "cpu"
# State machine: None=uninitialized, False=loading, True=ready, str=error message
_engine_ready = None

# BUG 12 FIX: Lock to protect concurrent mutations of the shared orchestrator
# (inject_transient_macro modifies loaded_cells and rebuilds topology).
_orchestrator_lock = threading.Lock()
_engine_state_lock = threading.Lock()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    prompt: str

@app.get("/api/status")
def get_status():
    if _engine_ready is None:
        return {"status": "uninitialized", "device": engine_device, "cells_loaded": 0}
    if _engine_ready is False:
        cells = len(global_orchestrator.loaded_cells) if global_orchestrator else 0
        return {"status": "loading", "device": engine_device, "cells_loaded": cells}
    if isinstance(_engine_ready, str):   # error message stored as string
        return {"status": "error", "message": _engine_ready, "device": engine_device, "cells_loaded": 0}
    if global_orchestrator is None:
        return {"status": "uninitialized", "device": engine_device, "cells_loaded": 0}
    return {"status": "ready", "device": engine_device, "cells_loaded": len(global_orchestrator.loaded_cells)}

@app.get("/api/health")
def health():
    if global_orchestrator is None:
        return {"status": "ok", "cells_loaded": 0}
    return {"status": "ok", "cells_loaded": len(global_orchestrator.loaded_cells)}

@app.get("/api/models")
def get_available_models():
    """Scan the local models directories and return only what is actually installed."""
    models_base = get_resource_path("models")
    embedders_dir = os.path.join(models_base, "embeddings")
    llms_dir = os.path.join(models_base, "llms")

    embedders = []
    if os.path.isdir(embedders_dir):
        for name in os.listdir(embedders_dir):
            if os.path.isdir(os.path.join(embedders_dir, name)):
                embedders.append(name)

    llms = []
    if os.path.isdir(llms_dir):
        for name in os.listdir(llms_dir):
            model_dir = os.path.join(llms_dir, name)
            if os.path.isdir(model_dir):
                gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
                if gguf_files:
                    llms.append(name)

    return {"embedders": embedders, "llms": llms}

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
    if _engine_ready is not True or global_orchestrator is None or global_rag_engine is None:
        return {
            "logs": [{"msg": "Engine is not ready; wait for /api/status to report ready.", "type": "warn"}],
            "path": [],
            "virtual_edges": [],
            "code": "# Engine is loading."
        }

    log_buffer = []
    prompt_preview = req.prompt[:120] + ("..." if len(req.prompt) > 120 else "")
    log_buffer.append({"msg": f"[PROMPT] {prompt_preview}", "type": "system"})
    log_buffer.append({"msg": "Extracting parameters and context...", "type": "info"})
    
    # We initialize context
    context = ExecutionContext()
    context.extract_prompt_parameters(req.prompt)
    
    extracted = context.extracted_parameters
    if extracted:
        for k, v in extracted.items():
            if v:
                log_buffer.append({"msg": f"  param[{k}] = {repr(v)}", "type": "debug"})
    
    explicit_filename = extracted.get("explicit_filename") if extracted else None
    
    if explicit_filename:
        # Using 'str' and 'source_identifier' for dummy bootstrap
        context.declare_variable(
            name="input_source", signature=AlgebraicSignature(type_name="str", state="source_identifier")
        )
        current_signature = AlgebraicSignature(type_name="str", state="source_identifier")
    else:
        # Start from any/any if there is no explicit input source
        current_signature = AlgebraicSignature(type_name="any", state="any")

    # Build MCTSEngine ONCE per request and reuse for all gap/bridge calls
    shared_mcts = MCTSEngine(global_orchestrator)

    final_micro_path = []
    virtual_edges = set()

    can_synth = ModelManager.get_instance().can_synthesize()

    if not can_synth:
        # Profile A: Embedding-only Lattice Routing
        log_buffer.append({"msg": "Phase 1: LatticeRouter (Embedding only) — finding path...", "type": "info"})
        from router import LatticeRouter
        router = LatticeRouter(global_orchestrator, global_rag_engine)
        try:
            # Revert BUG 3 FIX: Use actual current_signature to enforce typestate correctness from the start
            # But pass 'any' for state so that we don't enforce strict state matching on the bootstrap input
            final_micro_path, virtual_edges_list = router.plan_path(req.prompt, current_signature.type_name, "any")
            virtual_edges = set(virtual_edges_list)
        except Exception as e:
            print(f"[FATAL ROUTER ERROR] Exception type: {type(e)}, repr: {repr(e)}")
            import traceback
            traceback.print_exc()
            log_buffer.append({"msg": f"LatticeRouter failed: {repr(e)}", "type": "error"})
            return {"logs": log_buffer, "path": [], "virtual_edges": [], "code": f"# Routing Error: {e}"}
    else:
        # 1. ZeroShotPlanner
        if ModelManager.get_instance().has_translator_pass():
            log_buffer.append({"msg": "Phase 0: LLM Pre-Translator — translating natural language prompt...", "type": "info"})
        log_buffer.append({"msg": "Phase 1: ZeroShotPlanner — building macro execution graph...", "type": "info"})
        planner = ZeroShotPlanner(global_orchestrator, rag_engine=global_rag_engine)
        try:
            macro_graph = planner.run_planning_pass(req.prompt)
        except Exception as e:
            log_buffer.append({"msg": f"Planner failed: {e}", "type": "error"})
            return {"logs": log_buffer, "path": [], "virtual_edges": [], "code": f"# Planner Error: {e}"}

        # 2. Iterate Macro Graph and Gap Bridging
    
        if isinstance(macro_graph, dict):
            cells_list = macro_graph.get('cells', [macro_graph])
            macro_cell = cells_list[0] if cells_list else {}
        elif isinstance(macro_graph, list) and len(macro_graph) > 0:
            macro_cell = macro_graph[0]
        else:
            macro_cell = {}
            
        sub_cells_ids = macro_cell.get('sub_cells', []) if isinstance(macro_cell, dict) else []
        log_buffer.append({"msg": f"Phase 2: MCTS routing — {len(sub_cells_ids)} macro steps to resolve...", "type": "info"})
        if sub_cells_ids:
            log_buffer.append({"msg": f"  Plan: {' → '.join(sub_cells_ids)}", "type": "debug"})
        
        for i, step_id in enumerate(sub_cells_ids):
            target_cell = global_orchestrator.loaded_cells.get(step_id)

            # Auto-correct: if the planner hallucinated a cell ID that doesn't exist,
            # try fuzzy matching against all known cells before falling back to gap-bridging.
            # (Excludes SYNTH_* steps which are intended for direct LLM synthesis).
            if target_cell is None and not step_id.startswith("SYNTH_"):
                corrected_id = planner._find_closest_existing_cell(step_id, req.prompt)
                if corrected_id:
                    target_cell = global_orchestrator.loaded_cells.get(corrected_id)
                    if target_cell:
                        log_buffer.append({"msg": f"[AUTO-CORRECT] '{step_id}' → '{corrected_id}'", "type": "info"})
                        step_id = corrected_id

            if target_cell is None:
                expected_inputs = current_signature.type_name
                expected_outputs = "any"
                for next_id in sub_cells_ids[i+1:]:
                    next_cell = global_orchestrator.loaded_cells.get(next_id)
                    if next_cell:
                        expected_outputs = next_cell.inputs.type_name
                        break

                log_buffer.append({"msg": f"Planner flagged MISSING_NODE ({step_id}) for {expected_inputs}->{expected_outputs}.", "type": "warn"})

                # 1. Composition Confidence (reuse shared_mcts — no re-construction overhead)
                comp_path = shared_mcts.search(expected_inputs, expected_outputs, iterations=200)
                comp_confidence = 1.0 / (len(comp_path) + 1) if comp_path else 0.0

                # 2. Synthesis Confidence
                synth_micro_json = None
                synth_confidence = 0.0
                
                # Profile B/C are already here, so we just check if synthesis is explicitly supported by the active model
                if ModelManager.get_instance().can_synthesize():
                    synth = SynthesisEngine()
                    fetcher = FetcherFactory.get_fetcher(global_orchestrator.active_domain)
                    try:
                        gap_concept = f"{step_id}: convert {expected_inputs} to {expected_outputs}"
                        synth_ctx = SynthesisContext(
                            gap_concept=gap_concept,
                            input_type=expected_inputs,
                            output_type=expected_outputs,
                            domain=global_orchestrator.active_domain or '',
                            input_file_hint=context.extracted_parameters.get("input_filename", ""),
                            prompt_hint=req.prompt[:150],
                        )
                        synth_micro_json = synth.synthesize_micro_cell(
                            gap_concept, expected_inputs, expected_outputs, fetcher,
                            context_hint=synth_ctx.to_context_hint(),
                        )
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
                        if global_rag_engine:
                            global_rag_engine.add_dynamic_cell(synth_micro_json)
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
                    # Reuse shared_mcts — avoids another O(N) construction
                    bridge_path = shared_mcts.search(current_signature.type_name, target_cell.inputs.type_name, iterations=500)

                for bridge_node in bridge_path:
                    final_micro_path.append(bridge_node)
                    virtual_edges.add(bridge_node.cell_id)
                    current_signature = bridge_node.outputs

                if not current_signature.matches(target_cell.inputs):
                    log_buffer.append({
                        "msg": f"No valid typestate bridge found before {target_cell.cell_id}; aborting the route.",
                        "type": "error",
                    })
                    break

            final_micro_path.append(target_cell)
            current_signature = target_cell.outputs

    # 3. Code Generation

    compiled_blocks = []
    explicit_filename = context.extracted_parameters.get("explicit_filename")
    if explicit_filename:
        compiled_blocks.append(f"input_source = {explicit_filename!r}")
    else:
        compiled_blocks.append(f"input_source = None")
        
    log_buffer.append({"msg": f"Phase 3: Code Generation — unifying {len(final_micro_path)} micro cells...", "type": "info"})
    for cell in final_micro_path:
        code_block = UnificationGate.unify(context, cell)
        if code_block:
            compiled_blocks.append(code_block)
            
    # Feedback Loop Check
    final_code = "\n".join(compiled_blocks)
    
    # AST Dynamic Import Resolver
    final_code = UnificationGate.resolve_imports(final_code, context)
    
    if ModelManager.get_instance().can_feedback_check():
        log_buffer.append({"msg": "Phase 4: LLM feedback check running on generated code...", "type": "info"})
        reviewed_code = ModelManager.get_instance().feedback_check(final_code)
        try:
            compile(reviewed_code, "<generated-code-feedback>", "exec")
        except SyntaxError as exc:
            log_buffer.append({"msg": f"Feedback result rejected as invalid Python: {exc.msg}", "type": "warn"})
        else:
            final_code = UnificationGate.resolve_imports(reviewed_code, context)
    
    # We rebuild compiled_blocks for the API response
    compiled_blocks = [final_code]

    # Log summary
    code_lines = final_code.count("\n") + 1 if final_code.strip() else 0
    log_buffer.append({"msg": f"[DONE] Generated {code_lines} lines across {len(final_micro_path)} cells. Virtual edges: {len(virtual_edges)}.", "type": "system"})
    if final_code.strip():
        preview = final_code.strip()[:200].replace("\n", " ↵ ") + ("..." if len(final_code.strip()) > 200 else "")
        log_buffer.append({"msg": f"[CODE PREVIEW] {preview}", "type": "debug"})

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
    embedder_model: str = ""
    llm_model: str = ""
    embedder_device: str = "auto"
    llm_device: str = "auto"
    trees_storage: str = "ram"

@app.post("/api/initialize")
def initialize_engine(req: InitRequest = InitRequest()):
    """Non-blocking initialization — spawns a daemon thread for the heavy model/FAISS
    load and returns {status: initializing} immediately. The frontend's existing
    /api/status polling handles the wait with zero extra client-side changes."""
    global global_orchestrator, global_rag_engine, engine_device, _engine_ready

    with _engine_state_lock:
        if _engine_ready is False:
            return {"status": "initializing", "device": engine_device}
        _engine_ready = False

    def _do_init():
        global global_orchestrator, global_rag_engine, engine_device, _engine_ready
        try:
            print(f"[*] Starting engine initialization (Profile={req.profile}, Embedder={req.embedder_model or 'auto'}, LLM={req.llm_model or 'auto'}, Device={req.embedder_device})...")
            logger.info(f"[*] Starting engine initialization (Profile={req.profile})")
            HardwareProfiler.set_config(req.embedder_device, req.llm_device, req.trees_storage)
            ModelManager.get_instance().initialize_profile(req.profile, req.embedder_model, req.llm_model)
            engine_device = HardwareProfiler.get_optimal_device()
            print(f"[*] Loading Lattice Orchestrator from {TREES_DIR}...")
            new_orchestrator = LatticeOrchestrator(trees_directory=TREES_DIR)
            print(f"[*] Loading Local RAG Engine & FAISS index...")
            new_rag_engine = LocalRAG(trees_dir=TREES_DIR, orchestrator=new_orchestrator)
            with _engine_state_lock:
                global_orchestrator = new_orchestrator
                global_rag_engine = new_rag_engine
                _engine_ready = True
            print(f"  [+] Engine fully ready on {engine_device.upper()}")
            logger.info(f"Engine fully ready on {engine_device.upper()}")
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            with _engine_state_lock:
                _engine_ready = str(e) if str(e) else "Unknown Error (see console)"
            print(f"[ERROR] Engine initialization failed: {e}\n{tb_str}")
            logger.error(f"Engine initialization failed: {e}\n{tb_str}")

    t = threading.Thread(target=_do_init, daemon=True)
    t.start()
    return {"status": "initializing", "device": engine_device}

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

def ensure_port_available(host: str, port: int):
    """Fail safely when another process owns the requested listener."""
    try:
        with socket.create_connection((host, port), timeout=0.2):
            raise RuntimeError(f"Port {host}:{port} is already in use; choose --port instead of terminating its owner.")
    except ConnectionRefusedError:
        return
    except socket.timeout:
        return

def run_server(host: str = API_HOST, port: int = API_PORT):
    ensure_port_available(host, port)
    uvicorn.run(app, host=host, port=port, log_level="error")

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

def _execute_cli_prompt(prompt_text: str):
    """Helper to run a prompt and print formatted logs, path, and code to terminal."""
    print(f"\n[QUERY] {prompt_text}")
    print("-" * 50)
    result = run_prompt(RunRequest(prompt=prompt_text))

    logs = result.get("logs", [])
    if logs:
        print("[LOGS]")
        for log in logs:
            msg_type = log.get("type", "info").upper()
            msg_text = log.get("msg", "")
            print(f"  [{msg_type}] {msg_text}")

    path = result.get("path", [])
    if path:
        path_str = " → ".join(c.get("cell_id", "?") for c in path)
        print(f"\n[PATH] {path_str}")

    code = result.get("code", "# No code generated")
    print("\n" + "=" * 50)
    print("GENERATED CODE:")
    print("=" * 50)
    print(code)
    print("=" * 50 + "\n")


def _handle_cli_command(cmd_str: str, current_config: dict):
    """Processes interactive slash-commands (/profile, /models, /status, /cells, /help)."""
    parts = cmd_str.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("/help", "/h"):
        print("\n  Available Commands:")
        print("    /profile <A|B|C|D>    Switch active inference profile")
        print("    /models               List installed local embedders & LLMs")
        print("    /model <emb> <llm>    Hot-swap embedder & LLM models")
        print("    /status               Show engine status and active device")
        print("    /cells                List loaded semantic cells")
        print("    /help                 Show this help message\n")

    elif cmd == "/status":
        prof_name = ModelManager.get_instance().current_profile_name
        active_prof = ModelManager.get_instance().active_profile
        emb_name = getattr(active_prof, 'embedder_name', 'N/A')
        llm_name = getattr(active_prof, 'llm_name', 'N/A')
        cell_cnt = len(global_orchestrator.loaded_cells) if global_orchestrator else 0
        print(f"\n  [STATUS] Device: {engine_device.upper()} | Ready: {_engine_ready}")
        print(f"    Profile:  {prof_name}")
        print(f"    Embedder: {emb_name}")
        print(f"    LLM:      {llm_name}")
        print(f"    Cells:    {cell_cnt}\n")

    elif cmd == "/models":
        m = get_available_models()
        print("\n  [INSTALLED MODELS]")
        print(f"    Embedders: {', '.join(m['embedders']) if m['embedders'] else 'None found'}")
        print(f"    LLMs:      {', '.join(m['llms']) if m['llms'] else 'None found'}\n")

    elif cmd == "/cells":
        if not global_orchestrator:
            print("  [CELLS] Engine not initialized.")
            return
        print(f"\n  [CELLS] {len(global_orchestrator.loaded_cells)} loaded cells:")
        for cell_id, cell in global_orchestrator.loaded_cells.items():
            cell_type = getattr(cell, 'type', 'micro')
            print(f"    - {cell_id} ({cell_type})")
        print()

    elif cmd == "/profile":
        if not args:
            print("  Usage: /profile <A|B|C|D>")
            return
        new_prof = args[0].upper()
        if new_prof not in ("A", "B", "C", "D"):
            print(f"  Invalid profile '{new_prof}'. Choose from A, B, C, D.")
            return
        print(f"  [*] Hot-swapping profile to {new_prof}...")
        current_config["profile"] = new_prof
        initialize_engine(InitRequest(
            profile=new_prof,
            embedder_model=current_config["embedder"],
            llm_model=current_config["llm"]
        ))
        while _engine_ready is not True:
            if isinstance(_engine_ready, str):
                print(f"  [ERROR] Hot-swap failed: {_engine_ready}")
                return
            time.sleep(0.1)
        print(f"  [+] Profile updated to {new_prof}!")

    elif cmd in ("/model", "/modelswap"):
        if len(args) < 2:
            print("  Usage: /model <embedder_name> <llm_name>")
            return
        new_emb, new_llm = args[0], args[1]
        print(f"  [*] Hot-swapping models to EMB={new_emb}, LLM={new_llm}...")
        current_config["embedder"] = new_emb
        current_config["llm"] = new_llm
        initialize_engine(InitRequest(
            profile=current_config["profile"],
            embedder_model=new_emb,
            llm_model=new_llm
        ))
        while _engine_ready is not True:
            if isinstance(_engine_ready, str):
                print(f"  [ERROR] Hot-swap failed: {_engine_ready}")
                return
            time.sleep(0.1)
        print(f"  [+] Models updated!")

    else:
        print(f"  Unknown command '{cmd}'. Type /help for available commands.")


def run_interactive_cli(default_profile: str, default_embedder: str, default_llm: str, initial_prompt: str = None):
    """Interactive REPL mode: keeps model weights allocated in memory across multiple prompts."""
    config = {
        "profile": default_profile,
        "embedder": default_embedder,
        "llm": default_llm,
    }

    print("\n" + "=" * 64)
    print(" ⬡ NSTL CYBER-LATTICE INTERACTIVE CLI REPL")
    print("=" * 64)
    print(f" [*] Bootstrapping Engine (Profile={config['profile']}, Embedder={config['embedder'] or 'auto'}, LLM={config['llm'] or 'auto'})...")

    initialize_engine(InitRequest(
        profile=config["profile"],
        embedder_model=config["embedder"],
        llm_model=config["llm"]
    ))

    while _engine_ready is not True:
        if isinstance(_engine_ready, str):
            print(f" [!] Initialization failed: {_engine_ready}")
            return
        time.sleep(0.1)

    cells_cnt = len(global_orchestrator.loaded_cells) if global_orchestrator else 0
    print(f" [+] Engine Ready! Hardware: {engine_device.upper()} | Loaded Cells: {cells_cnt}")
    print("-" * 64)
    print(" Commands:")
    print("   /profile <A|B|C|D>    Switch inference profile")
    print("   /models               List installed local embedders & LLMs")
    print("   /model <emb> <llm>    Hot-swap embedder and LLM models")
    print("   /status               Show engine status & active hardware")
    print("   /cells                List loaded semantic cell IDs")
    print("   /help                 Show help menu")
    print("   exit / quit           Exit REPL")
    print("=" * 64 + "\n")

    if initial_prompt:
        _execute_cli_prompt(initial_prompt)

    # Enable line-editing & command history in terminal on Unix/Linux if available
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    while True:
        try:
            user_input = input("nstl> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", ":q"):
                print("Exiting NSTL CLI. Goodbye!")
                break

            if user_input.startswith("/"):
                _handle_cli_command(user_input, config)
                continue

            _execute_cli_prompt(user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting NSTL CLI. Goodbye!")
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NSTL Engine")
    parser.add_argument("--profile", type=str, default="C", help="Inference profile (A, B, C, D)")
    parser.add_argument("--embedder", type=str, default="", help="Embedder model name")
    parser.add_argument("--llm", type=str, default="", help="LLM model name")
    parser.add_argument("--prompt", type=str, default=None, help="Direct prompt to execute via CLI")
    parser.add_argument("--cli", "-i", "--interactive", action="store_true", help="Start interactive CLI REPL mode (model weights stay allocated)")
    parser.add_argument("--port", type=int, default=API_PORT, help="Port for API server")
    args = parser.parse_args()

    port = args.port

    # 1. Interactive REPL Mode (--cli or -i)
    if args.cli:
        run_interactive_cli(
            default_profile=args.profile,
            default_embedder=args.embedder,
            default_llm=args.llm,
            initial_prompt=args.prompt
        )
        sys.exit(0)

    # 2. Single-prompt direct CLI mode (--prompt without --cli)
    if args.prompt:
        print(f"[*] Single-Prompt CLI Mode (Profile={args.profile}, Embedder={args.embedder or 'auto'}, LLM={args.llm or 'auto'})")
        initialize_engine(InitRequest(
            profile=args.profile,
            embedder_model=args.embedder,
            llm_model=args.llm
        ))
        while _engine_ready is not True:
            if isinstance(_engine_ready, str):
                print(f"[ERROR] Initialization failed: {_engine_ready}")
                sys.exit(1)
            time.sleep(0.1)

        _execute_cli_prompt(args.prompt)
        sys.exit(0)

    # 3. Server Mode (GUI + REST API)
    ensure_port_available(API_HOST, port)
    server_thread = threading.Thread(target=run_server, args=(API_HOST, port), daemon=True)
    server_thread.start()

    # Auto-initialize engine in server mode so API is ready out-of-the-box
    if _engine_ready is None:
        initialize_engine(InitRequest(
            profile=args.profile,
            embedder_model=args.embedder,
            llm_model=args.llm
        ))

    if os.environ.get("TEST_HEADLESS") == "1":
        print("[HEADLESS] TEST_HEADLESS is set. Running in headless mode — press Ctrl+C to stop.")
        import signal
        shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: shutdown_event.set())
        signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
        shutdown_event.wait()
        print("[HEADLESS] Shutdown signal received. Exiting.")
    else:
        # BUG 18 FIX: Wait for the server to be ready before opening WebView.
        if not _wait_for_server(API_HOST, port):
            print(f"[WARNING] Server did not start on {API_HOST}:{port} in time; opening WebView anyway.")

        try:
            webview.create_window(
                "NSTL Engine",
                f"http://{API_HOST}:{port}",
                width=1400,
                height=900,
                min_size=(1000, 700),
                background_color="#0d1117",
            )
            webview.start(debug=True)
        except Exception as e:
            print(f"Failed to start pywebview: {e}")
            print("[HEADLESS] Running in headless fallback mode — press Ctrl+C to stop.")
            import signal
            shutdown_event = threading.Event()
            signal.signal(signal.SIGINT, lambda *_: shutdown_event.set())
            signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
            shutdown_event.wait()
            print("[HEADLESS] Shutdown signal received. Exiting.")
