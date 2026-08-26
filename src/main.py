"""
src/main.py - Neuro-Symbolic Topological Lattice (NSTL) Engine
Main FastAPI Server, REST API Endpoints, and 4-Phase Synthesis Pipeline.
"""

from __future__ import annotations
import os
import sys
import socket
import subprocess
import threading
import time
import warnings
from dataclasses import asdict
from typing import List, Dict, Set, Any, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize centralized logging
from log_config import setup_logging, get_logger
setup_logging()
logger = get_logger("main")

from config import API_HOST, API_PORT, CORS_ORIGINS
from lattice import LatticeOrchestrator, AlgebraicSignature, Cell, MicroCell
from unification import ExecutionContext, UnificationGate
from router import HardwareProfiler, LatticeRouter, MCTSEngine
from planner import ZeroShotPlanner
from synthesis import SynthesisEngine
from gevr_sandbox import GEVRSandbox
from external_rag import FetcherFactory
from internal_rag import LocalRAG
from inference import ModelManager


def get_resource_path(relative_path: str) -> str:
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
    return root_path if os.path.exists(root_path) else relative_path


TREES_DIR = get_resource_path("trees")
FRONTEND_DIR = get_resource_path("frontend_dist")

# Global engine state
global_orchestrator: Optional[LatticeOrchestrator] = None
global_rag_engine: Optional[LocalRAG] = None
engine_device: str = "cpu"
_engine_ready: Optional[bool | str] = None
_engine_state_lock = threading.Lock()
_current_init_thread: Optional[threading.Thread] = None

app = FastAPI(title="NSTL Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    prompt: str


class InitRequest(BaseModel):
    profile: str = "C"
    embedder_model: str = ""
    llm_model: str = ""
    embedder_device: str = "auto"
    llm_device: str = "auto"
    trees_storage: str = "ram"


@app.get("/api/status")
def get_status():
    if _engine_ready is None:
        return {"status": "uninitialized", "device": engine_device, "cells_loaded": 0}
    if _engine_ready is False:
        cells = len(global_orchestrator.loaded_cells) if global_orchestrator else 0
        return {"status": "loading", "device": engine_device, "cells_loaded": cells}
    if isinstance(_engine_ready, str):
        return {"status": "error", "message": _engine_ready, "device": engine_device, "cells_loaded": 0}
    return {"status": "ready", "device": engine_device, "cells_loaded": len(global_orchestrator.loaded_cells)}


@app.get("/api/health")
def health():
    return {"status": "ok", "cells_loaded": len(global_orchestrator.loaded_cells) if global_orchestrator else 0}


@app.get("/api/cells")
def get_cells():
    if global_orchestrator is None:
        return {"cells": [], "count": 0}
    cells = []
    for cell in global_orchestrator.loaded_cells.values():
        cells.append({
            "cell_id": cell.cell_id,
            "stage": cell.stage,
            "type": cell.cell_type,
            "keywords": list(cell.keywords),
            "primary_input": asdict(cell.primary_input),
            "primary_output": asdict(cell.primary_output),
            "code_template": cell.code_template
        })
    return {"cells": cells, "count": len(cells)}


@app.post("/api/run")
def run_prompt(req: RunRequest):
    """
    Main 4-Phase Program Synthesis Pipeline:
    Phase 1: Semantic Intent Routing / Planning
    Phase 2: Typestate Resolution & Dynamic Gap Bridging
    Phase 3: Type-Monadic Unification & Code Generation
    Phase 4: Sandboxed Verification & Feedback Repair
    """
    if _engine_ready is not True or global_orchestrator is None or global_rag_engine is None:
        return {
            "logs": [{"msg": "Engine is loading. Please wait for initialization.", "type": "warn"}],
            "path": [],
            "virtual_edges": [],
            "code": "# Engine is not ready."
        }

    log_buffer = []
    log_buffer.append({"msg": f"[PROMPT] {req.prompt[:100]}...", "type": "system"})

    # ─────────────────────────────────────────────────────────────
    # Phase 1: Planning / Routing
    # ─────────────────────────────────────────────────────────────
    model_mgr = ModelManager.get_instance()
    resolved_path: List[Cell] = []
    virtual_edges: Set[str] = set()

    if not model_mgr.can_synthesize():
        # Profile A: Pure Embedding-Guided Beam Search
        log_buffer.append({"msg": "Phase 1: Embedding-guided Beam Search pathfinding...", "type": "info"})
        router = LatticeRouter(global_orchestrator, global_rag_engine)
        resolved_path, virtual_edges = router.plan_path(req.prompt)
    else:
        # Profiles B/C/D: Structured Macro Planning
        log_buffer.append({"msg": "Phase 1: ZeroShotPlanner decomposing prompt into topological DAG...", "type": "info"})
        planner = ZeroShotPlanner(global_orchestrator, global_rag_engine)
        plan_dict = planner.run_planning_pass(req.prompt)
        sub_cells = plan_dict.get("cells", [{}])[0].get("sub_cells", [])

        # ─────────────────────────────────────────────────────────
        # Phase 2: Typestate Resolution & Gap Bridging
        # ─────────────────────────────────────────────────────────
        log_buffer.append({"msg": f"Phase 2: Resolving {len(sub_cells)} plan steps across the lattice...", "type": "info"})
        mcts = MCTSEngine(global_orchestrator)
        synthesis_engine = SynthesisEngine(trees_dir=TREES_DIR)
        fetcher = FetcherFactory.get_fetcher(global_orchestrator.active_domain)

        current_sig = AlgebraicSignature("str", "source_identifier")

        for step_id in sub_cells:
            target_cell = global_orchestrator.loaded_cells.get(step_id)

            if target_cell is None or step_id.startswith("SYNTH_"):
                # Bridge gap: Attempt MCTS or Live Doc Synthesis
                log_buffer.append({"msg": f"Gap detected for step '{step_id}'. Bridging...", "type": "warn"})
                expected_out = AlgebraicSignature("any", "any")

                # Try MCTS first
                bridge = mcts.search(current_sig, expected_out)
                if bridge:
                    for b in bridge:
                        resolved_path.append(b)
                        virtual_edges.add(b.cell_id)
                        current_sig = b.primary_output
                    continue

                # Fallback to LLM Synthesis
                if model_mgr.can_synthesize():
                    try:
                        synth_dict = synthesis_engine.synthesize_micro_cell(
                            gap_concept=step_id.replace("SYNTH_", "").replace("_", " "),
                            expected_input=current_sig.type_name,
                            expected_output="any",
                            fetcher=fetcher
                        )
                        if UnificationGate.validate_synthesis(synth_dict, current_sig, expected_out, TREES_DIR):
                            # Instantiate and inject
                            in_sig = AlgebraicSignature(synth_dict["inputs"]["type_name"], synth_dict["inputs"]["state"])
                            out_sig = AlgebraicSignature(synth_dict["outputs"]["type_name"], synth_dict["outputs"]["state"])
                            new_cell = MicroCell(
                                cell_id=synth_dict["cell_id"],
                                stage=synth_dict.get("stage", 2),
                                keywords=set(synth_dict.get("keywords", [])),
                                inputs={"input_data": in_sig},
                                outputs={"output_data": out_sig},
                                dependencies=synth_dict.get("dependencies", []),
                                code_template=synth_dict.get("code_template", "")
                            )
                            global_orchestrator.inject_cell(new_cell)
                            target_cell = new_cell
                            virtual_edges.add(target_cell.cell_id)
                    except Exception as e:
                        log_buffer.append({"msg": f"Synthesis failed for {step_id}: {e}", "type": "error"})

            if target_cell:
                # Type continuity check
                if not current_sig.unifies_with(target_cell.primary_input):
                    bridge = mcts.search(current_sig, target_cell.primary_input)
                    for b in bridge:
                        resolved_path.append(b)
                        virtual_edges.add(b.cell_id)
                        current_sig = b.primary_output

                resolved_path.append(target_cell)
                current_sig = target_cell.primary_output

    if not resolved_path:
        log_buffer.append({"msg": "Failed to construct a valid execution route.", "type": "error"})
        return {"logs": log_buffer, "path": [], "virtual_edges": [], "code": "# Path generation failed."}

    # ─────────────────────────────────────────────────────────────
    # Phase 3: Monadic Code Unification
    # ─────────────────────────────────────────────────────────────
    log_buffer.append({"msg": f"Phase 3: Monadic unification across {len(resolved_path)} cells...", "type": "info"})
    context = ExecutionContext(prompt=req.prompt)
    code_blocks: List[str] = []

    for cell in resolved_path:
        try:
            block = UnificationGate.unify_cell(context, cell)
            if block:
                code_blocks.append(block)
        except Exception as e:
            log_buffer.append({"msg": f"Unification error on {cell.cell_id}: {e}", "type": "error"})

    raw_synthesized_code = "\n".join(code_blocks)
    full_code = UnificationGate.resolve_imports(raw_synthesized_code, context)

    # ─────────────────────────────────────────────────────────────
    # Phase 4: Sandboxed Verification & Feedback Repair
    # ─────────────────────────────────────────────────────────────
    log_buffer.append({"msg": "Phase 4: Running sandboxed execution check...", "type": "info"})
    sandbox = GEVRSandbox(timeout_seconds=5.0)

    feedback_func = model_mgr.feedback_check if model_mgr.can_feedback_check() else None
    verified, final_code, exec_msg = sandbox.repair_cycle(full_code, llm_repair_func=feedback_func)

    if verified:
        log_buffer.append({"msg": "[GEVR SUCCESS] Generated code verified successfully.", "type": "info"})
    else:
        log_buffer.append({"msg": f"[GEVR WARNING] Execution check encountered an issue: {exec_msg[:100]}", "type": "warn"})

    path_formatted = [
        {
            "cell_id": c.cell_id,
            "stage": c.stage,
            "type": c.cell_type,
            "keywords": list(c.keywords),
            "primary_input": asdict(c.primary_input),
            "primary_output": asdict(c.primary_output),
        }
        for c in resolved_path
    ]

    return {
        "logs": log_buffer,
        "path": path_formatted,
        "virtual_edges": list(virtual_edges),
        "code": final_code
    }


@app.post("/api/initialize")
def initialize_engine(req: InitRequest = InitRequest()):
    global global_orchestrator, global_rag_engine, engine_device, _engine_ready, _current_init_thread

    with _engine_state_lock:
        if _current_init_thread is not None and _current_init_thread.is_alive():
            _current_init_thread.join(timeout=30)
        _engine_ready = False

    def _do_init():
        global global_orchestrator, global_rag_engine, engine_device, _engine_ready
        try:
            logger.info(f"Initializing NSTL Engine (Profile={req.profile})...")
            HardwareProfiler.set_config(req.embedder_device, req.llm_device, req.trees_storage)
            ModelManager.get_instance().initialize_profile(req.profile, req.embedder_model, req.llm_model)
            engine_device = HardwareProfiler.get_optimal_device()

            new_orchestrator = LatticeOrchestrator(trees_directory=TREES_DIR)
            new_rag_engine = LocalRAG(trees_dir=TREES_DIR, orchestrator=new_orchestrator)

            with _engine_state_lock:
                global_orchestrator = new_orchestrator
                global_rag_engine = new_rag_engine
                _engine_ready = True

            logger.info(f"[READY] Engine fully initialized on {engine_device.upper()}")
        except Exception as e:
            with _engine_state_lock:
                _engine_ready = str(e)
            logger.error(f"[INIT ERROR] {e}")

    t = threading.Thread(target=_do_init, daemon=True)
    _current_init_thread = t
    t.start()
    return {"status": "initializing", "device": engine_device}


if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    def index():
        return {"msg": "Frontend not found, serving API only."}


def run_server(host: str = API_HOST, port: int = API_PORT):
    uvicorn.run(app, host=host, port=port, log_level="error")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NSTL Engine Server")
    parser.add_argument("--profile", type=str, default="C")
    parser.add_argument("--port", type=int, default=API_PORT)
    args = parser.parse_args()

    initialize_engine(InitRequest(profile=args.profile))
    run_server(API_HOST, args.port)
