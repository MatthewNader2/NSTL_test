# main.py — NSTL Engine (FastAPI + PyWebview Integration)
import json
import logging
import math
import os
import platform
import re
import sys
import threading
import time
import warnings
from collections import defaultdict

import faiss
import numpy as np
import torch
import uvicorn
import webview
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

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


# =====================================================================
#  HARDWARE PROFILER
# =====================================================================
class HardwareProfiler:
    @staticmethod
    def get_optimal_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


# =====================================================================
#  UNIFICATION LAYER
# =====================================================================
class ExecutionContext:
    def __init__(self):
        self.registry = {}
        self.extracted_parameters = {}

    def extract_prompt_parameters(self, user_prompt: str):
        self.extracted_parameters = {}

        # 1. Filename extraction
        quoted_items = re.findall(r'["\']([^"\']+)["\']', user_prompt)
        if quoted_items:
            self.extracted_parameters["explicit_filename"] = quoted_items[0]
        else:
            file_match = re.search(
                r"\b([\w\-_.]+\.(?:csv|json|xlsx|parquet|feather|html))\b",
                user_prompt.lower(),
            )
            if file_match:
                self.extracted_parameters["explicit_filename"] = file_match.group(1)

        # 2. Dynamic Parameter Extraction (e.g., "by score", "column age")
        sort_match = re.search(
            r"\b(?:by|on)\s+(?:the\s+)?(?:score|column\s+)?['\"]?([a-zA-Z0-9_]+)['\"]?",
            user_prompt.lower(),
        )
        if sort_match:
            self.extracted_parameters["dynamic_col"] = sort_match.group(1)

    def declare_variable(self, name: str, var_type: str, state: str) -> str:
        base_name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
        sanitized_name = base_name
        counter = 2
        while sanitized_name in self.registry:
            sanitized_name = f"{base_name}_v{counter}"
            counter += 1
        self.registry[sanitized_name] = {"type": var_type, "state": state}
        return sanitized_name

    def find_compatible_variable(self, expected_type: str, expected_state: str) -> str:
        expected_type_clean = expected_type.lower().strip()
        expected_state_clean = (
            expected_state.lower().replace("_", "").replace("-", "").strip()
        )
        for var_name, tracking_info in reversed(list(self.registry.items())):
            current_type = tracking_info["type"].lower().strip()
            current_state = (
                tracking_info["state"].lower().replace("_", "").replace("-", "").strip()
            )
            if current_type == expected_type_clean and (
                expected_state_clean in current_state
                or current_state in expected_state_clean
            ):
                return var_name
        return None


class UnificationGate:
    @staticmethod
    def unify(context: ExecutionContext, target_cell, log_buffer: list = None) -> str:
        matching_input_var = context.find_compatible_variable(
            expected_type=target_cell.inputs["input_type"],
            expected_state=target_cell.inputs["expected_state"],
        )
        if not matching_input_var:
            matching_input_var = (
                list(context.registry.keys())[-1]
                if context.registry
                else "input_source"
            )

        raw_output_name = target_cell.outputs["resulting_state"].lower().strip()
        output_var_name = context.declare_variable(
            name=raw_output_name,
            var_type=target_cell.outputs["output_type"],
            state=target_cell.outputs["resulting_state"],
        )

        compiled_snippet = target_cell.code_template
        compiled_snippet = compiled_snippet.replace("{input_var}", matching_input_var)
        compiled_snippet = compiled_snippet.replace("{output_var}", output_var_name)

        # Apply extracted parameters dynamically
        if "explicit_filename" in context.extracted_parameters:
            user_assigned_name = context.extracted_parameters["explicit_filename"]
            compiled_snippet = re.sub(
                r"['\"]export\.(?:csv|json|html|feather|parquet)['\"]",
                f"'{user_assigned_name}'",
                compiled_snippet,
            )
            compiled_snippet = compiled_snippet.replace(
                "export.csv", user_assigned_name
            )

        if "dynamic_col" in context.extracted_parameters:
            user_assigned_col = context.extracted_parameters["dynamic_col"]
            compiled_snippet = re.sub(
                r"by=['\"]\w+['\"]", f"by='{user_assigned_col}'", compiled_snippet
            )
            compiled_snippet = re.sub(
                r"\['\w+'\]", f"['{user_assigned_col}']", compiled_snippet
            )

        return compiled_snippet


# =====================================================================
#  LATTICE DATA STRUCTURES
# =====================================================================
class MicroCell:
    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        code_template: str,
        inputs: dict,
        outputs: dict,
        cell_type: str = "micro",
        intent_expansion: list = None,
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = set(keywords)
        self.type = cell_type
        self.intent_expansion = intent_expansion or []
        self.code_template = code_template
        self.inputs = inputs or {}
        self.outputs = outputs or {}


class LatticeOrchestrator:
    def __init__(self, trees_directory="trees"):
        self.trees_directory = trees_directory
        self.loaded_cells = {}
        self.topology = {}
        self.discover_and_load_trees()
        self.build_topology()

    def discover_and_load_trees(self):
        if not os.path.exists(self.trees_directory):
            return
        for file_name in os.listdir(self.trees_directory):
            if file_name.endswith(".json"):
                try:
                    with open(
                        os.path.join(self.trees_directory, file_name),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        tree_data = json.load(f)
                    for raw_cell in tree_data.get("cells", []):
                        cell = MicroCell(
                            cell_id=raw_cell.get("cell_id", "UNKNOWN"),
                            stage=raw_cell.get("stage", 0),
                            keywords=raw_cell.get("keywords", []),
                            code_template=raw_cell.get("code_template", ""),
                            inputs=raw_cell.get("inputs", {}),
                            outputs=raw_cell.get("outputs", {}),
                            cell_type=raw_cell.get("type", "micro"),
                            intent_expansion=raw_cell.get("intent_expansion", []),
                        )
                        self.loaded_cells[cell.cell_id] = cell
                except Exception:
                    pass

    def build_topology(self):
        for cell_id in self.loaded_cells:
            self.topology[cell_id] = []
        for cell_a in self.loaded_cells.values():
            if cell_a.type == "macro":
                continue
            for cell_b in self.loaded_cells.values():
                if cell_b.type == "macro":
                    continue
                if cell_a.outputs.get("output_type") == cell_b.inputs.get(
                    "input_type"
                ) and cell_a.outputs.get("resulting_state") == cell_b.inputs.get(
                    "expected_state"
                ):
                    self.topology[cell_a.cell_id].append(cell_b.cell_id)

    def get_all_available_cells(self) -> list:
        return list(self.loaded_cells.values())

    def get_neighbors(self, cell_id: str) -> list:
        return [self.loaded_cells[nid] for nid in self.topology.get(cell_id, [])]

    def find_type_bridge(self, from_type: str, to_type: str):
        for cell in self.loaded_cells.values():
            if (
                cell.type == "micro"
                and cell.inputs.get("input_type") == from_type
                and cell.outputs.get("output_type") == to_type
            ):
                return cell
        return None


# =====================================================================
#  NEURAL ROUTER (2026 SOTA: Task-Driven Asymmetric Retrieval)
# =====================================================================
class LatticeRouter:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        try:
            self.device = HardwareProfiler.get_optimal_device()
        except NameError:
            self.device = "cpu"

        # 🚀 Check for local cached model first, otherwise download
        model_path = get_resource_path("model_cache")
        if os.path.exists(model_path):
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                trust_remote_code=True,
            )
        else:
            self.model = SentenceTransformer(
                "jinaai/jina-embeddings-v5-text-nano",
                device=self.device,
                trust_remote_code=True,
            )

        self.cells_list = []
        self.index = None
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        cells = self.orchestrator.get_all_available_cells()
        self.cells_list = cells
        if not cells:
            return

        # Clean documentation format with Jina's expected Document prefix
        texts = [
            f"Document: Function {cell.cell_id.replace('_', ' ').lower()}. Tags: {' '.join(cell.keywords).lower()}."
            for cell in cells
        ]

        encode_kwargs = {
            "batch_size": 256,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }

        # 🚀 FIX: Jina v5 uses the unified "retrieval" task adapter
        if "jina" in self.model.config.name_or_path.lower():
            encode_kwargs["task"] = "retrieval"

        embeddings = self.model.encode(texts, **encode_kwargs).astype(np.float32)
        self.index = faiss.IndexHNSWFlat(
            embeddings.shape[1], 32, faiss.METRIC_INNER_PRODUCT
        )
        self.index.add(embeddings)

    def plan_path(
        self,
        user_intent: str,
        initial_type: str,
        initial_state: str,
        log_buffer: list = None,
    ) -> tuple:
        if log_buffer is None:
            log_buffer = []

        def log(msg, log_type="info"):
            log_buffer.append({"msg": msg, "type": log_type})
            print(f"[{log_type.upper()}] {msg}")

        raw_goals = re.split(r",|\band\b|\bthen\b", user_intent)
        goals = [g.strip() for g in raw_goals if g.strip()]
        log(f"🧠 INITIAL PARSED GOALS: {goals}", "system")

        final_path = []
        virtual_edges = set()
        expanded_macro_ids = set()
        MAX_GOAL_QUEUE = 60
        current_type, current_state = initial_type, initial_state
        current_node = None

        MIN_CONFIDENCE = 0.55
        MACRO_THRESHOLD = 0.70
        step = 0

        while goals:
            if len(goals) > MAX_GOAL_QUEUE:
                log(f"⚠️ Goal queue exceeded {MAX_GOAL_QUEUE}. TRUNCATING.", "warn")
                goals = goals[:MAX_GOAL_QUEUE]

            goal = goals.pop(0)
            log(f"--------------------------------------------------", "system")
            log(f"🎯 EVALUATING GOAL: '{goal}'", "system")

            goal_text_clean = goal.lower()

            encode_kwargs = {"convert_to_numpy": True, "normalize_embeddings": True}

            # 🚀 FIX: Pass the unified 'retrieval' task and prefix the user goal with 'Query: '
            if "jina" in self.model.config.name_or_path.lower():
                encode_kwargs["task"] = "retrieval"
                goal_text_clean = f"Query: {goal_text_clean}"
            elif "snowflake" in self.model.config.name_or_path.lower():
                goal_text_clean = f"query: {goal_text_clean}"

            goal_embedding = self.model.encode(
                [goal_text_clean], **encode_kwargs
            ).astype(np.float32)

            global_micro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "micro"
                and c.cell_id != (current_node.cell_id if current_node else "")
            ]
            best_global_micro, global_micro_score = self._score_and_select_best(
                global_micro_candidates, goal_text_clean, goal_embedding
            )
            log(
                f"   [Search] Top Micro Match: {best_global_micro.cell_id if best_global_micro else 'None'} (Score: {global_micro_score:.2f})",
                "debug",
            )

            macro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "macro"
            ]
            best_macro, macro_score = self._score_and_select_best(
                macro_candidates, goal_text_clean, goal_embedding
            )
            log(
                f"   [Search] Top Macro Match: {best_macro.cell_id if best_macro else 'None'} (Score: {macro_score:.2f})",
                "debug",
            )

            if (
                best_macro
                and macro_score > MACRO_THRESHOLD
                and macro_score > global_micro_score
                and best_macro.cell_id not in expanded_macro_ids
            ):
                expanded_macro_ids.add(best_macro.cell_id)
                log(
                    f"💥 MACRO EXPLOSION! '{goal}' triggered '{best_macro.cell_id}'.",
                    "warn",
                )
                goals = best_macro.intent_expansion + goals
                continue

            if step == 0:
                strict_entry_candidates = [
                    c
                    for c in self.orchestrator.get_all_available_cells()
                    if c.type == "micro"
                    and c.inputs.get("input_type") == current_type
                    and c.inputs.get("expected_state") == current_state
                ]
                best_node, best_score = self._score_and_select_best(
                    strict_entry_candidates, goal_text_clean, goal_embedding
                )

                if best_score < MIN_CONFIDENCE:
                    global_entry_candidates = [
                        c
                        for c in self.orchestrator.get_all_available_cells()
                        if c.type == "micro"
                    ]
                    best_global_node, best_global_score = self._score_and_select_best(
                        global_entry_candidates, goal_text_clean, goal_embedding
                    )

                    if best_global_score >= MIN_CONFIDENCE:
                        target_type = best_global_node.inputs.get("input_type")
                        bridge_node = self.orchestrator.find_type_bridge(
                            current_type, target_type
                        )

                        if bridge_node:
                            log(
                                f"🌉 STEP 0 COERCION: Bridging '{current_type}' -> '{target_type}' via {bridge_node.cell_id}",
                                "tunnel",
                            )
                            final_path.append(bridge_node)
                            virtual_edges.add(bridge_node.cell_id)
                            best_node = best_global_node
                            best_score = best_global_score
                        else:
                            log(
                                f"❌ ROUTER HALT: No entry bridge exists for '{current_type}' -> '{target_type}'.",
                                "warn",
                            )
                            break
                    else:
                        log(
                            f"❌ ROUTER HALT: No valid entry node found for '{goal}'.",
                            "warn",
                        )
                        break

                log(f"✅ Step 0 Selected -> {best_node.cell_id}", "success")
                final_path.append(best_node)
                current_node = best_node
                current_type, current_state = (
                    current_node.outputs.get("output_type"),
                    current_node.outputs.get("resulting_state"),
                )
                step += 1
                continue

            strict_candidates = self.orchestrator.get_neighbors(current_node.cell_id)
            best_local_node, best_local_score = self._score_and_select_best(
                strict_candidates, goal_text_clean, goal_embedding
            )

            if best_local_score > -1.0:
                best_local_score += 0.08

            global_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "micro" and c.cell_id != current_node.cell_id
            ]
            best_global_node, best_global_score = self._score_and_select_best(
                global_candidates, goal_text_clean, goal_embedding
            )

            if best_global_score > MIN_CONFIDENCE and (
                best_local_score < MIN_CONFIDENCE
                or (best_global_score - best_local_score > 0.12)
            ):
                log(
                    f"   [Routing] Semantic gravity pulled Router out of local bounds.",
                    "tunnel",
                )
                target_type = best_global_node.inputs.get("input_type")

                if target_type == current_type:
                    log(
                        f"⚡ VIRTUAL EDGE COMPILED! Tunneling directly to -> {best_global_node.cell_id}",
                        "tunnel",
                    )
                    best_node = best_global_node
                    virtual_edges.add(best_node.cell_id)
                else:
                    bridge_node = self.orchestrator.find_type_bridge(
                        current_type, target_type
                    )
                    if bridge_node:
                        log(
                            f"🌉 COERCION BRIDGE INJECTED -> {bridge_node.cell_id}",
                            "tunnel",
                        )
                        final_path.append(bridge_node)
                        virtual_edges.add(bridge_node.cell_id)
                        best_node = best_global_node
                        virtual_edges.add(best_node.cell_id)
                    else:
                        if best_local_score >= MIN_CONFIDENCE:
                            best_node = best_local_node
                        else:
                            log(f"❌ DEAD END. No path for '{goal}'. Skipping.", "warn")
                            continue
            elif best_local_score >= MIN_CONFIDENCE:
                log(f"✅ Local Edge Followed -> {best_local_node.cell_id}", "success")
                best_node = best_local_node
            else:
                log(f"❌ NO PATH FOUND for '{goal}'. Skipping.", "warn")
                continue

            final_path.append(best_node)
            current_node = best_node
            current_type, current_state = (
                current_node.outputs.get("output_type"),
                current_node.outputs.get("resulting_state"),
            )
            step += 1

        log(f"🏁 Route Generated: {[c.cell_id for c in final_path]}", "system")
        return final_path, virtual_edges

    def _score_and_select_best(
        self, candidates: list, query_text: str, prompt_embedding: np.ndarray
    ) -> tuple:
        if not candidates or self.index is None:
            return None, -1.0

        valid_ids = {c.cell_id for c in candidates}
        k_search = self.index.ntotal
        if k_search == 0:
            return None, -1.0

        distances, indices = self.index.search(prompt_embedding, k=k_search)

        best_cell = None
        best_score = -1.0

        query_words = [
            w[:-1] if w.endswith("s") and len(w) > 3 else w
            for w in re.findall(r"\w+", query_text.replace("query: ", ""))
            if len(w) > 2
        ]

        for dist, idx in zip(distances[0], indices[0]):
            cell = self.cells_list[idx]

            if cell.cell_id in valid_ids:
                base_score = float(dist)

                cell_words = set(cell.cell_id.lower().replace("_", " ").split()).union(
                    {k.lower() for k in cell.keywords}
                )
                lexical_boost = 0.0

                for qw in query_words:
                    if any(qw == cw or qw in cw for cw in cell_words):
                        lexical_boost += 0.20

                final_score = min(base_score + lexical_boost, 0.99)

                if final_score > best_score:
                    best_score = final_score
                    best_cell = cell

        return best_cell, best_score


# =====================================================================
#  FASTAPI BACKEND
# =====================================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    prompt: str


# Global status state to monitor initialization from frontend
global_orchestrator = None
global_router = None
system_status = {
    "status": "starting",
    "message": "Initializing FastAPI server components...",
    "device": "cpu",
    "cells_loaded": 0
}


@app.get("/api/status")
def get_status():
    return system_status


@app.get("/api/health")
def health():
    if global_orchestrator is None:
        return {"status": system_status["status"], "cells_loaded": 0}
    return {"status": "live", "cells_loaded": len(global_orchestrator.loaded_cells)}


@app.get("/api/cells")
def get_cells():
    if global_orchestrator is None:
        return {"cells": []}
    return {
        "cells": [
            {
                "cell_id": c.cell_id,
                "stage": c.stage,
                "type": c.type,
                "keywords": list(c.keywords),
                "inputs": c.inputs,
                "outputs": c.outputs,
                "intent_expansion": c.intent_expansion,
            }
            for c in global_orchestrator.get_all_available_cells()
        ]
    }


@app.post("/api/run")
def run_prompt(req: RunRequest):
    if global_router is None or global_router.index is None:
        return {
            "logs": [{"msg": "Engine is currently loading neural model, please wait...", "type": "warn"}],
            "path": [],
            "virtual_edges": [],
            "code": "# Engine is loading."
        }

    log_buffer = []
    context = ExecutionContext()
    context.extract_prompt_parameters(req.prompt)
    context.declare_variable(
        name="input_source", var_type="str", state="source_identifier"
    )

    execution_path, virtual_edges = global_router.plan_path(
        req.prompt,
        initial_type="str",
        initial_state="source_identifier",
        log_buffer=log_buffer,
    )

    compiled_blocks = []
    for cell in execution_path:
        code_block = UnificationGate.unify(context, cell, log_buffer=log_buffer)
        if code_block:
            compiled_blocks.append(code_block)

    return {
        "logs": log_buffer,
        "path": [
            {
                "cell_id": c.cell_id,
                "stage": c.stage,
                "type": c.type,
                "keywords": list(c.keywords),
                "inputs": c.inputs,
                "outputs": c.outputs,
            }
            for c in execution_path
        ],
        "virtual_edges": list(virtual_edges),
        "code": "\n".join(compiled_blocks)
        if compiled_blocks
        else "# No code generated.",
    }


if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


def initialize_async():
    global global_orchestrator, global_router
    try:
        # Determine acceleration device
        system_status["status"] = "profiling"
        system_status["message"] = "Optimizing hardware acceleration..."
        device = HardwareProfiler.get_optimal_device()
        system_status["device"] = device

        # Load cells
        system_status["status"] = "loading_trees"
        system_status["message"] = "Loading semantic trees from memory..."
        global_orchestrator = LatticeOrchestrator(trees_directory=TREES_DIR)
        system_status["cells_loaded"] = len(global_orchestrator.loaded_cells)
        time.sleep(0.5)  # Brief delay to make the loading UI feel organic

        # Initialize HNSW index and SentenceTransformer
        system_status["status"] = "loading_model"
        system_status["message"] = f"Loading Jina v5 embedding model on {device.upper()}..."
        global_router = LatticeRouter(global_orchestrator)

        system_status["status"] = "ready"
        system_status["message"] = "NSTL Cyber-Lattice Engine online."
    except Exception as e:
        system_status["status"] = "error"
        system_status["message"] = f"Initialization error: {str(e)}"


if __name__ == "__main__":
    # Start uvicorn server in a separate background thread
    threading.Thread(target=run_server, daemon=True).start()
    
    # Run heavy model and tree loading in a background thread
    threading.Thread(target=initialize_async, daemon=True).start()
    
    # Start webview immediately (loads the loading screen instantly)
    try:
        webview.create_window(
            "NSTL Cyber-Lattice Engine",
            "http://127.0.0.1:8000"
            if os.path.exists(FRONTEND_DIR)
            else "http://localhost:5173",
            width=1400,
            height=900,
            text_select=True,
        )
        webview.start(debug=False)
    except Exception:
        while True:
            time.sleep(100)
