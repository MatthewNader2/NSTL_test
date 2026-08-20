# router.py
import logging
import math
import os
import platform
import random
import re
import copy
import time
import warnings
from typing import Optional, List, Dict, Set, Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# CRITICAL FIX: Prevents silent hard crashes when mixing PyTorch, OpenMP, and UI threads
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from log_config import get_logger
from lattice import AlgebraicSignature
from unification import types_unify, TOP_TYPE_SET

from config import (
    SIMILARITY_THRESHOLD, MIN_CONFIDENCE, TUNNELING_MARGIN,
    MACRO_THRESHOLD, TYPE_MISMATCH_DISCOUNT, DOMAIN_MATCH_FACTOR,
    DOMAIN_NEUTRAL_FACTOR, DOMAIN_CONFLICT_FACTOR
)
from tokenizer import CellTokenizer, AliasRegistry, STOP_WORDS
from type_registry import TypeRegistry

logger = get_logger('router')



def log_coverage_gap(prompt: str, domain_guess: str, score: float, best_cell_id: str):
    """
    Logs low-confidence queries below the domain threshold to logs/coverage_gaps.log (JSONL)
    as a prioritized harvesting backlog.
    """
    import json
    from datetime import datetime
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "coverage_gaps.log")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "domain_guess": domain_guess,
            "score": float(score),
            "best_cell_id": best_cell_id,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"[COVERAGE GAP LOGGED] Prompt: '{prompt[:50]}...' | Domain: {domain_guess} | Score: {score:.3f}")
    except Exception as e:
        logger.warning(f"[COVERAGE GAP LOGGING ERROR] Failed to log gap: {e}")


def get_domain_threshold(domain_name: str) -> float:
    if domain_name == "algorithms":
        return 0.30
    return 0.25


# H-5 fix: define stop-words once at module level so the scoring inner loop

# doesn't rebuild the set on every FAISS result iteration.
_STOP_WORDS = STOP_WORDS

# Multiplier applied to the score of a type-incompatible candidate.
# A value of 0.15 means an incompatible cell must be ~6.7× more semantically
# similar than a compatible one to beat it. Tunable here without touching logic.
TYPE_MISMATCH_DISCOUNT = 0.15


from dataclasses import dataclass, field


@dataclass
class SynthesisContext:
    """
    Carries runtime context from the router/planner into SynthesisEngine so it
    can generate domain-aware, grounded code without any static string checks.

    Assembled from already-available runtime data (orchestrator.active_domain,
    context.extracted_parameters, prompt fragment).  No parsing logic lives here.
    """
    gap_concept: str
    input_type: str
    output_type: str
    domain: str = ""
    input_file_hint: str = ""
    prompt_hint: str = ""

    def to_context_hint(self) -> str:
        """Formats a grounding string for the synthesis LLM prompt. Pure formatting, zero logic."""
        parts = []
        if self.domain:
            parts.append(f"domain={self.domain}")
        if self.input_file_hint:
            parts.append(f"input_file={self.input_file_hint}")
        if self.prompt_hint:
            parts.append(f"user_intent={self.prompt_hint[:150]}")
        return ", ".join(parts)


class FastPathRouter:
    """
    High-Performance Macro Fast-Path Router.
    Determines if a prompt matches a known high-confidence MacroCell pattern
    via FAISS vector similarity (> 0.88 score), bypassing LLM planning when matched.
    Contains ZERO task-specific hardcoded string rules.
    """
    def __init__(self, orchestrator, rag_engine=None):
        self.orchestrator = orchestrator
        self.rag_engine = rag_engine

    def try_fast_path(self, prompt: str) -> Optional[list]:
        if not prompt or not self.orchestrator:
            return None

        # Pure FAISS Vector Similarity matching over pre-compiled MacroCells
        if self.rag_engine and hasattr(self.rag_engine, "find_closest_cell_by_embedding"):
            matched_id = self.rag_engine.find_closest_cell_by_embedding(prompt, domain_hint=prompt)
            if matched_id:
                cell = self.orchestrator.loaded_cells.get(matched_id)
                if cell and getattr(cell, "node_type", "") == "macro":
                    logger.info(f"[FAST-PATH VECTOR MATCH] Vector matched macro cell '{matched_id}' for prompt: {prompt[:60]!r}")
                    return [cell]
        return None





class HardwareProfiler:
    """
    Cross-Platform Dynamic Hardware Auto-Profiler.
    Detects the host system and assigns workloads to the optimal compute backend.
    """
    _cached_device: str = None
    
    _config = {
        'embedder': 'auto',
        'llm': 'auto',
        'trees': 'ram'
    }

    @staticmethod
    def set_config(embedder_device: str, llm_device: str, trees_storage: str):
        HardwareProfiler._config['embedder'] = embedder_device.lower()
        HardwareProfiler._config['llm'] = llm_device.lower()
        HardwareProfiler._config['trees'] = trees_storage.lower()
        
    @staticmethod
    def get_embedder_device() -> str:
        if HardwareProfiler._config['embedder'] != 'auto':
            return HardwareProfiler._config['embedder']
        return HardwareProfiler.get_optimal_device()

    @staticmethod
    def get_llm_device() -> str:
        if HardwareProfiler._config['llm'] != 'auto':
            return HardwareProfiler._config['llm']
        return HardwareProfiler.get_optimal_device()

    @staticmethod
    def get_optimal_device() -> str:
        # BUG 25 FIX: Return cached result if already profiled.
        if HardwareProfiler._cached_device is not None:
            return HardwareProfiler._cached_device

        logger.info("=" * 50)
        logger.info(" NSTL HARDWARE AUTODETECTION PROFILER")
        logger.info("=" * 50)

        os_name = platform.system()
        cpu_arch = platform.machine()
        logger.debug(f" Host OS:   {os_name} ({cpu_arch})")
        import sys
        logger.debug(f" Python Exe: {sys.executable}")
        logger.debug(f" Torch Ver:  {torch.__version__} ({torch.__file__})")
        logger.debug(f" CUDA Avail: {torch.cuda.is_available()}")

        # 1. Check for NVIDIA CUDA GPUs
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f" Compute:   NVIDIA CUDA Detected")
            logger.info(f" Hardware:  {gpu_name} ({vram:.1f} GB VRAM)")
            logger.info(" Routing:   Vector Embedding mapped to GPU.")
            logger.info("=" * 50)
            HardwareProfiler._cached_device = "cuda"
            return "cuda"
            
        # 1.5. Check for physical NVIDIA GPUs (if PyTorch lacks CUDA)
        try:
            import subprocess
            smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            if smi_output:
                gpu_name = smi_output.split('\n')[0]
                logger.info(f" Compute:   NVIDIA GPU Detected (System)")
                logger.info(f" Hardware:  {gpu_name}")
                logger.info(" Routing:   LLM mapped to GPU, Vector Embedding mapped to CPU (Install PyTorch CUDA for full acceleration).")
                logger.info("=" * 50)
                HardwareProfiler._cached_device = "cpu"
                return "cpu"
        except Exception as e:
            logging.getLogger("HardwareProfiler").debug(f"nvidia-smi probe failed: {e}")

        # 2. Check for Apple Silicon (M1/M2/M3/M4) Neural Engines
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info(f" Compute:   Apple Metal Performance Shaders (MPS)")
            logger.info(f" Hardware:  Apple Silicon Architecture")
            logger.info(" Routing:   Vector Embedding mapped to Apple GPU.")
            logger.info("=" * 50)
            HardwareProfiler._cached_device = "mps"
            return "mps"

        # 3. Fallback to CPU (Windows without Nvidia, Intel Macs, Linux Servers)
        logger.info(f" Compute:   Standard CPU")
        logger.info(f" Hardware:  Fallback Mode Activated")
        logger.info(" Routing:   Vector Embedding mapped to CPU.")
        logger.info(" Note:      Operations will run safely, but slower.")
        logger.info("=" * 50)
        HardwareProfiler._cached_device = "cpu"
        return "cpu"


class MCTSNode:
    """Lightweight node tracking only scalar values and topology string IDs."""
    __slots__ = ['cell_id', 'current_type', 'parent', 'children', 'visits', 'q_value']
    
    def __init__(self, cell_id: str, current_type: str, parent=None):
        self.cell_id = cell_id
        self.current_type = current_type
        self.parent = parent
        self.children = []
        self.visits = 0
        self.q_value = 0.0

    def ucb1(self, c_param=0.5) -> float:
        # BUG 9 FIX: Guard against math.log(0) when parent.visits == 0.
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        if parent_visits == 0:
            return float('inf')
        return (self.q_value / self.visits) + c_param * math.sqrt(math.log(parent_visits) / self.visits)

class MCTSEngine:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.all_cells = self.orchestrator.get_all_available_cells()
        # Build both lookup tables once at construction — O(N)
        self.micro_by_type: dict = {}
        self.all_cells_map: dict = {}  # cell_id -> Cell (used by _get_best_path)
        for c in self.all_cells:
            self.all_cells_map[c.cell_id] = c
            if c.type == "micro":
                self.micro_by_type.setdefault(c.inputs.type_name, []).append(c)

        # Re-index 'any'-typed cells under every known concrete type so MCTS
        # can reach them from any starting typestate.  'any'-typed cells are
        # universally compatible; excluding them from concrete-type searches was
        # the root cause of MCTS finding no path for opencv / vague prompts.
        all_known_types = set(self.micro_by_type.keys()) - {"any"}
        any_cells = self.micro_by_type.get("any", [])
        for c in any_cells:
            for t in all_known_types:
                bucket = self.micro_by_type.setdefault(t, [])
                if c not in bucket:          # avoid duplicates if already indexed
                    bucket.append(c)
        logger.debug(
            f"[MCTS INIT] Indexed {len(any_cells)} 'any'-typed cells under "
            f"{len(all_known_types)} concrete types."
        )

    def search(self, start_type: str, target_type: str, iterations: int = 1000) -> list:
        logger.debug(f"MCTSEngine.search started with start_type={start_type}, target_type={target_type}, iterations={iterations}")
        # Root node acts as the starting typestate
        root = MCTSNode(cell_id="ROOT", current_type=start_type)
        
        start_time = time.time()
        max_iters = min(iterations, 100)
        for i in range(max_iters):
            if time.time() - start_time > 1.5:
                logger.warning(f"[MCTS TIMEOUT] Capping search at 1.5s (completed {i} iterations)")
                break

            # 1. Selection
            leaf = self._select(root)
            
            # 2. Expansion (with Unification Filter)
            if not types_unify(leaf.current_type, target_type) and leaf.visits > 0:
                self._expand(leaf)
                if leaf.children:
                    leaf = random.choice(leaf.children)
            
            # 3. Simulation
            reward = self._simulate(leaf, target_type)
            
            # 4. Backpropagation
            self._backpropagate(leaf, reward)
            
        path = self._get_best_path(root, target_type)
        if path:
            logger.info("MCTS found a valid path.")
        else:
            logger.warning("MCTS search completed but no path found.")
        return path

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.children:
            unexplored = [child for child in current.children if child.visits == 0]
            if unexplored:
                return unexplored[0]
            current = max(current.children, key=lambda c: c.ucb1(c_param=0.5))
        return current

    def _get_reachable_cells(self, current_type: str) -> list:
        """
        Returns all micro cells reachable from current_type, merging type-specific
        and 'any'-typed cells.  After __init__ re-indexing, 'any' cells are already
        included in concrete-type buckets, but this helper is kept as a safety net
        for cells injected dynamically after construction (e.g. synthesized nodes).
        """
        type_specific = self.micro_by_type.get(current_type, [])
        universal = self.micro_by_type.get("any", [])
        seen = {c.cell_id for c in type_specific}
        return type_specific + [c for c in universal if c.cell_id not in seen]

    def _expand(self, node: MCTSNode):
        if node.cell_id == "ROOT":
            # For root, get all MicroCells that match start_type (including any-typed)
            candidates = self._get_reachable_cells(node.current_type)
        else:
            candidates = self.orchestrator.get_neighbors(node.cell_id)

        for cell in candidates:
            if cell.type == "micro" and types_unify(cell.inputs.type_name, node.current_type):
                child = MCTSNode(cell_id=cell.cell_id, current_type=cell.outputs.type_name, parent=node)
                node.children.append(child)

    def _simulate(self, node: MCTSNode, target_type: str) -> float:
        current_type = node.current_type
        current_id = node.cell_id
        depth = 0
        max_depth = 15

        while not types_unify(current_type, target_type) and depth < max_depth:
            neighbors = [
                c for c in self._get_reachable_cells(current_type)
                if getattr(c, 'node_type', 'function') != 'macro'
            ]

            if not neighbors:
                return 0.0  # Dead end

            next_cell = random.choice(neighbors)
            current_type = next_cell.outputs.type_name
            current_id = next_cell.cell_id
            depth += 1

        if types_unify(current_type, target_type):
            return 1.0
        return 0.0  # Exceeded max_depth

    def _backpropagate(self, node: MCTSNode, reward: float):
        current = node
        while current is not None:
            current.visits += 1
            current.q_value += reward
            current = current.parent

    def _get_best_path(self, root: MCTSNode, target_type: str) -> list:
        path = []
        current = root
        # Use the pre-built map from __init__ — no redundant O(N) rebuild per call
        all_cells_map = self.all_cells_map
        while current.children:
            best_child = max(current.children, key=lambda c: c.visits)
            cell = all_cells_map.get(best_child.cell_id)
            if cell is None:
                break
            path.append(cell)
            if types_unify(best_child.current_type, target_type):
                return path
            current = best_child
        return []

class LatticeRouter:
    def __init__(self, orchestrator, rag_engine):
        self.orchestrator = orchestrator
        self.rag_engine = rag_engine
        
        cells = self.orchestrator.get_all_available_cells()
        self.type_registry = TypeRegistry.build(cells)
        self.alias_registry = AliasRegistry.build_from_cells(cells)

    def _split_intent_into_goals(self, intent: str) -> list[str]:
        goals = []
        current_goal = []
        nesting = 0
        in_string = False
        string_char = ''
        
        i = 0
        while i < len(intent):
            c = intent[i]
            
            if in_string:
                current_goal.append(c)
                if c == string_char:
                    # simplistic escape check
                    if i == 0 or intent[i-1] != '\\':
                        in_string = False
            else:
                if c in ['"', "'", '`']:
                    in_string = True
                    string_char = c
                    current_goal.append(c)
                elif c in ['(', '[', '{']:
                    nesting += 1
                    current_goal.append(c)
                elif c in [')', ']', '}']:
                    nesting = max(0, nesting - 1)
                    current_goal.append(c)
                elif nesting == 0:
                    # check for separators
                    is_sep = False
                    skip = 0
                    if c in [',', ';', '\n']:
                        is_sep = True
                        skip = 1
                    elif c == '.' and i + 1 < len(intent) and intent[i+1].isspace():
                        is_sep = True
                        skip = 1
                        
                    if is_sep:
                        goal_str = "".join(current_goal).strip()
                        if goal_str:
                            goals.append(goal_str)
                        current_goal = []
                        i += skip - 1 # will be incremented by 1 at the end of loop
                    else:
                        current_goal.append(c)
                else:
                    current_goal.append(c)
            i += 1
            
        final_goal = "".join(current_goal).strip()
        if final_goal:
            goals.append(final_goal)
            
        return goals

    def plan_path(
        self, user_intent: str, initial_type: str, initial_state: str
    ) -> tuple:
        goals = self._split_intent_into_goals(user_intent)

        from dataclasses import dataclass, field
        @dataclass
        class BeamState:
            path: list
            current_signature: AlgebraicSignature
            cumulative_score: float = 0.0
            virtual_edges: set = field(default_factory=set)

        beam = [
            BeamState(
                path=[],
                current_signature=AlgebraicSignature(initial_type, initial_state),
                cumulative_score=0.0,
                virtual_edges=set()
            )
        ]
        beam_width = 4

        while goals:
            goal = goals.pop(0)

            # Format target query for FAISS (2D numpy array, Normalized)
            from inference import ModelManager
            goal_embedding_list = ModelManager.get_instance().get_embeddings([goal])
            goal_embedding = np.array(goal_embedding_list, dtype=np.float32)
            norm = np.linalg.norm(goal_embedding, axis=1, keepdims=True)
            norm = np.where(norm == 0, 1.0, norm)
            goal_embedding = goal_embedding / norm

            all_micro = [
                c for c in self.orchestrator.get_all_available_cells()
                if c.type == "micro"
            ]

            macro_candidates = [
                c for c in self.orchestrator.get_all_available_cells()
                if c.type == "macro"
            ]

            ref_type = beam[0].current_signature.type_name if beam else initial_type
            best_macro, macro_score = self._score_and_select_best(
                macro_candidates, goal_embedding, goal, ref_type, None
            )
            _, global_micro_score = self._score_and_select_best(
                all_micro, goal_embedding, goal, ref_type, None
            )

            if (
                best_macro
                and macro_score > MACRO_THRESHOLD
                and global_micro_score < 0.70
            ):
                expansion = getattr(best_macro, 'algorithmic_steps', [])
                if not expansion:
                    expansion = getattr(best_macro, 'intent_expansion', None) or getattr(best_macro, 'sub_cells', [])

                if expansion:
                    logger.info(
                        f"[ROUTER UNSTAGE] Unfolding MacroCell '{best_macro.cell_id}' into {len(expansion)} sub-goals."
                    )
                    goals = list(expansion) + goals
                    continue

            next_beam = []

            for state in beam:
                current_node = state.path[-1] if state.path else None
                current_type = state.current_signature.type_name

                if current_node is None:
                    candidates = [
                        c for c in all_micro
                        if c.inputs.matches(state.current_signature) or types_unify(c.inputs.type_name, current_type)
                    ]
                    if not candidates:
                        candidates = all_micro
                else:
                    neighbors = self.orchestrator.get_neighbors(current_node.cell_id)
                    candidates = neighbors + [c for c in all_micro if c.cell_id != current_node.cell_id]

                scored = self._score_candidates(candidates, goal_embedding, goal, current_type, current_node)

                expanded = False
                for score, cell in scored[:8]:
                    if score < 0.01:
                        continue

                    target_type = cell.inputs.type_name
                    param_types = [getattr(p, 'type_name', p.get('type_name', '')) if isinstance(p, dict) else getattr(p, 'type_name', '') for p in getattr(cell, 'parameters', [])]
                    type_compat = types_unify(target_type, current_type) or any(types_unify(pt, current_type) for pt in param_types)

                    if type_compat:
                        cell_copy = copy.copy(cell)
                        goal_heuristics = [f"{repr(q)}" for q in re.findall(r'["\']([^"\']+)["\']', goal)]
                        goal_heuristics.extend(re.findall(r'\b(\d+(?:\.\d+)?)\b', goal))
                        cell_copy.matched_heuristics = goal_heuristics

                        new_path = state.path + [cell_copy]
                        new_sig = cell_copy.outputs
                        new_edges = set(state.virtual_edges)
                        if current_node and cell_copy.cell_id not in [n.cell_id for n in self.orchestrator.get_neighbors(current_node.cell_id)]:
                            new_edges.add(cell_copy.cell_id)

                        step_score = max(score, 1e-4)
                        new_cum_score = state.cumulative_score + math.log(step_score)
                        next_beam.append(BeamState(new_path, new_sig, new_cum_score, new_edges))
                        expanded = True
                    elif score >= 0.20:
                        if not hasattr(self, '_mcts_cache'):
                            self._mcts_cache = MCTSEngine(self.orchestrator)
                        bridge_path = self._mcts_cache.search(current_type, target_type, iterations=50)
                        if bridge_path:
                            cell_copy = copy.copy(cell)
                            goal_heuristics = [f"{repr(q)}" for q in re.findall(r'["\']([^"\']+)["\']', goal)]
                            goal_heuristics.extend(re.findall(r'\b(\d+(?:\.\d+)?)\b', goal))
                            cell_copy.matched_heuristics = goal_heuristics

                            new_path = state.path + bridge_path + [cell_copy]
                            new_sig = cell_copy.outputs
                            new_edges = set(state.virtual_edges)
                            for b_node in bridge_path:
                                new_edges.add(b_node.cell_id)
                            new_edges.add(cell_copy.cell_id)

                            step_score = max(score, 1e-4)
                            new_cum_score = state.cumulative_score + math.log(step_score)
                            next_beam.append(BeamState(new_path, new_sig, new_cum_score, new_edges))
                            expanded = True

                if not expanded:
                    scored_all = self._score_candidates(all_micro, goal_embedding, goal, current_type, current_node)
                    if scored_all:
                        best_s, best_c = scored_all[0]
                        cell_copy = copy.copy(best_c)
                        goal_heuristics = [f"{repr(q)}" for q in re.findall(r'["\']([^"\']+)["\']', goal)]
                        goal_heuristics.extend(re.findall(r'\b(\d+(?:\.\d+)?)\b', goal))
                        cell_copy.matched_heuristics = goal_heuristics

                        new_path = state.path + [cell_copy]
                        new_sig = cell_copy.outputs
                        new_edges = set(state.virtual_edges)
                        new_edges.add(cell_copy.cell_id)

                        step_score = max(best_s, 1e-4)
                        new_cum_score = state.cumulative_score + math.log(step_score)
                        next_beam.append(BeamState(new_path, new_sig, new_cum_score, new_edges))

            if next_beam:
                next_beam.sort(key=lambda s: s.cumulative_score, reverse=True)
                unique_beam = []
                seen_paths = set()
                for s in next_beam:
                    path_key = tuple(c.cell_id for c in s.path)
                    if path_key not in seen_paths:
                        seen_paths.add(path_key)
                        unique_beam.append(s)
                beam = unique_beam[:beam_width]
            else:
                logger.warning(f"[ROUTER HALT] Beam search produced no valid states for goal: '{goal}'.")
                break

        if beam:
            best_state = max(beam, key=lambda s: s.cumulative_score)
            final_path = best_state.path
            virtual_edges = best_state.virtual_edges
        else:
            final_path = []
            virtual_edges = set()

        logger.info(
            f"\n[BEAM PATHFINDER COMPLETE] Route Generated ({len(final_path)} steps): {[c.cell_id for c in final_path]}"
        )
        return final_path, virtual_edges

    def _score_candidates(
        self, candidates: list, prompt_embedding: np.ndarray, goal: str = "", current_type: str = None, current_node = None
    ) -> list:
        if not candidates or self.rag_engine is None or self.rag_engine.index is None:
            return []

        valid_ids = {c.cell_id for c in candidates}
        candidates_dict = {c.cell_id: c for c in candidates}

        k_search = self.rag_engine.index.ntotal
        distances, indices = self.rag_engine.index.search(prompt_embedding, k=k_search)

        prompt_tokens = set(re.findall(r"[a-zA-Z_]+", goal.lower()))

        scored_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx not in self.rag_engine.id_to_schema:
                continue
            cid = self.rag_engine.id_to_schema[idx].get("cell_id")
            cid_lower = cid.lower() if cid else ""

            if cid in valid_ids:
                cell = candidates_dict.get(cid)
                if cell:
                    kws = {kw.lower() for kw in getattr(cell, 'keywords', [])}
                    filtered_tokens = {pt for pt in prompt_tokens if pt not in _STOP_WORDS and len(pt) > 2}
                    expanded_tokens = self.alias_registry.expand_tokens(filtered_tokens)

                    cell_sub_tokens = CellTokenizer.tokenize_identifier(cell.cell_id)

                    token_hits = 0
                    for token in expanded_tokens:
                        hit = (
                            token in cid_lower
                            or any(token == p or (len(token) >= 3 and len(p) >= 3 and (token.startswith(p) or p.startswith(token))) for p in cell_sub_tokens)
                            or any(token == kw or (len(token) >= 3 and len(kw) >= 3 and (token.startswith(kw) or kw.startswith(token))) for kw in kws)
                        )
                        if hit:
                            token_hits += 1

                    keyword_score = (token_hits / max(len(expanded_tokens), 1)) * 0.5

                    # Penalize internal private helper modules and overload decorators
                    if any(cid.startswith(prefix) for prefix in ["PANDAS_CORE_", "PANDAS_ERRORS_", "PANDAS_COMPAT_", "PANDAS_API_TYPING_", "PANDAS_IO_FORMATS_", "PANDAS_IO_PARSERS_", "PANDAS_IO_SAS_", "PANDAS_IO_EXCEL_", "PANDAS_IO_SQL_", "PANDAS_IO_STATA_", "PANDAS_IO_CLIPBOARD_", "PANDAS_IO_PYTABLES_", "PANDAS__LIBS_", "PANDAS_IO_API_", "PANDAS_ARRAYS_", "PANDAS_CATEGORICAL", "PANDAS_CATEGORICALINDEX_", "PANDAS_DATETIMEINDEX_", "PANDAS_INDEX_", "PANDAS_INTERVALINDEX_", "PANDAS_MULTIINDEX_", "PANDAS_PERIODINDEX_", "PANDAS_RANGEINDEX_", "PANDAS_SERIES_", "PANDAS_TIMESTAMP_", "CV2_UNDISTORT", "SCIPY_STATS_MSTATS_", "SCIPY_INTERPOLATE_AAA_", "SCIPY_SPARSE_", "SKLEARN_EXTERNALS_", "NUMPY_F2PY_", "NUMPY_LIB_", "NUMPY_MA_"]) or "OVERLOAD" in cid or "WRAPPER" in cid:
                        keyword_score *= 0.01

                    # Action verb alignment bonus to differentiate operational stages (read vs write vs clean vs sort)
                    action_verbs = {"read", "load", "open", "drop", "dropna", "clean", "sort", "convert", "write", "save", "imwrite", "to_csv", "csv", "train", "fit", "predict", "dijkstra", "scale", "normalize"}
                    prompt_actions = prompt_tokens.intersection(action_verbs)
                    cell_actions = set(cell_sub_tokens).intersection(action_verbs)
                    action_match = len(prompt_actions.intersection(cell_actions))
                    if action_match > 0:
                        keyword_score += 0.35 * action_match

                    # Sink completion bonus for file-writing cells when prompt requests saving/writing output
                    is_sink_cell = getattr(cell, 'metadata_tags', {}).get("is_sink", False) or any(k in cid_lower for k in ["to_csv", "imwrite", "save_predictions", "to_json", "to_excel", "to_parquet", "write_csv", "save_mat", "save_npz", "savetxt"]) or (getattr(cell.outputs, 'type_name', '') in ("None", "void", "NoneType", "bool") and any(k in cid_lower for k in ["imwrite", "to_csv", "save", "write"]))
                    if is_sink_cell and current_node is None:
                        keyword_score -= 1.00
                    elif is_sink_cell and current_node is not None and any(act in prompt_actions for act in ["write", "save", "imwrite", "to_csv", "csv"]):
                        keyword_score += 3.50
                    elif not is_sink_cell and current_node is not None and any(act in prompt_actions for act in ["write", "save", "imwrite", "to_csv", "csv"]):
                        keyword_score -= 1.00

                    # Specific operational bonuses for missing data, dijkstra, ML classification, and normalization
                    if any(term in prompt_tokens for term in ["missing", "null", "na", "dropna", "clean"]):
                        if "DROPNA" in cid:
                            keyword_score += 0.50

                    if any(term in prompt_tokens for term in ["numeric", "features"]):
                        if "SELECT_NUMERIC" in cid:
                            keyword_score += 1.50
                        elif "DESCRIBE" in cid:
                            keyword_score -= 1.00

                    if any(term in prompt_tokens for term in ["normalize", "scale", "scaler", "standard"]):
                        if "STANDARD_SCALER" in cid or "SCALER" in cid:
                            keyword_score += 1.00

                    if any(term in prompt_tokens for term in ["dijkstra", "shortest", "path"]):
                        if "PYTHON_DIJKSTRA" in cid:
                            keyword_score += 2.00
                        elif "DIJKSTRA" in cid:
                            keyword_score += 1.00

                    if any(term in prompt_tokens for term in ["randomforest", "train", "classifier", "fit"]):
                        if "RANDOMFORESTCLASSIFIER" in cid or "RANDOM_FOREST" in cid:
                            keyword_score += 1.00

                    if any(term in prompt_tokens for term in ["predict", "predictions"]):
                        if "MODEL_PREDICT" in cid or "SAVE_PREDICTIONS" in cid:
                            keyword_score += 1.00

                    if any(term in prompt_tokens for term in ["grayscale", "gray", "bgr2gray", "cvtcolor"]):
                        if "CVTCOLOR" in cid:
                            keyword_score += 1.00

                    if current_node is None:
                        if any(term in prompt_tokens for term in ["image", "jpg", "png", "opencv", "cv2"]):
                            if "IMREAD" in cid:
                                keyword_score += 1.00
                        elif any(term in prompt_tokens for term in ["csv", "data.csv", "read", "load"]):
                            if "READ_CSV" in cid:
                                keyword_score += 1.00

                    if current_type and current_type.lower() != 'any':
                        input_t = getattr(cell.inputs, 'type_name', 'any')
                        param_types = [getattr(p, 'type_name', p.get('type_name', '')) if isinstance(p, dict) else getattr(p, 'type_name', '') for p in getattr(cell, 'parameters', [])]
                        if types_unify(current_type, input_t) or any(types_unify(current_type, pt) for pt in param_types):
                            c_low = current_type.lower()
                            i_low = input_t.lower()
                            if c_low == i_low or (c_low, i_low) in [("mat", "ndarray"), ("ndarray", "mat"), ("dataframe", "ndarray"), ("ndarray", "dataframe")]:
                                type_factor = 1.00
                            else:
                                type_factor = 0.85
                        else:
                            type_factor = TYPE_MISMATCH_DISCOUNT
                    else:
                        type_factor = 1.0

                    cell_domain = getattr(cell, 'domain_name', '').lower()
                    if not cell_domain:
                        cell_domain = cell.cell_id.split('_')[0].lower()

                    domain_factor = 1.0
                    goal_lower = (goal or "").lower()
                    intent_type = None
                    if any(k in goal_lower for k in ["csv", "dataframe", "table", "pandas", "data", "numeric", "features", "clean", "missing", "salary", "age", "drop", "rows", "column", "sort", "excel", "tsv"]):
                        intent_type = "DataFrame"
                    elif any(k in goal_lower for k in ["image", "jpg", "png", "opencv", "cv2", "gray", "grayscale", "blur", "pixel", "disparity"]):
                        intent_type = "Mat"

                    if any(k in goal_lower for k in ["opencv", "cv2", "image", "jpg", "png"]):
                        if cell_domain in ["cv2", "opencv", "image_processing"]:
                            domain_factor = DOMAIN_MATCH_FACTOR
                        elif cell_domain in ["scipy", "pandas", "data_engineering", "numpy", "sklearn"]:
                            domain_factor = DOMAIN_CONFLICT_FACTOR
                        else:
                            domain_factor = DOMAIN_NEUTRAL_FACTOR
                    elif any(k in goal_lower for k in ["csv", "dataframe", "pandas", "data.csv", "predictions.csv"]):
                        if cell_domain in ["cv2", "opencv", "image_processing"]:
                            domain_factor = DOMAIN_CONFLICT_FACTOR
                    elif intent_type == "DataFrame":
                        if cell_domain in ["pandas", "data_engineering", "data_cleaning", "data"]:
                            domain_factor = DOMAIN_MATCH_FACTOR
                        elif cell_domain in ["cv2", "image_processing"]:
                            domain_factor = DOMAIN_CONFLICT_FACTOR
                        else:
                            domain_factor = DOMAIN_NEUTRAL_FACTOR
                    elif current_type and current_type not in ('any', 'str', 'None'):
                        domain_factor = self.type_registry.is_domain_compatible(cell_domain, current_type)
                    elif current_node and getattr(current_node, 'domain_name', None):
                        c_dom = current_node.domain_name.lower()
                        if c_dom not in ("python", "core", "builtins", "generic"):
                            c_dom_canon = self.type_registry.resolve_alias(c_dom)
                            cell_dom_canon = self.type_registry.resolve_alias(cell_domain)
                            if cell_dom_canon and c_dom_canon and cell_dom_canon != c_dom_canon:
                                domain_factor = DOMAIN_CONFLICT_FACTOR

                    # Penalize internal private helper modules, excel writers, and overload decorators
                    if any(cid.startswith(prefix) for prefix in ["PANDAS_CORE_", "PANDAS_ERRORS_", "PANDAS_COMPAT_", "PANDAS_API_TYPING_", "PANDAS_IO_FORMATS_", "PANDAS_IO_PARSERS_", "PANDAS_IO_SAS_", "PANDAS_IO_EXCEL_", "PANDAS_EXCELWRITER_", "PANDAS_IO_SQL_", "PANDAS_IO_STATA_", "PANDAS_IO_CLIPBOARD_", "PANDAS_IO_PYTABLES_", "PANDAS__LIBS_", "PANDAS_IO_API_", "PANDAS_ARRAYS_", "PANDAS_CATEGORICAL", "PANDAS_CATEGORICALINDEX_", "PANDAS_DATETIMEINDEX_", "PANDAS_INDEX_", "PANDAS_INTERVALINDEX_", "PANDAS_MULTIINDEX_", "PANDAS_PERIODINDEX_", "PANDAS_RANGEINDEX_", "PANDAS_SERIES_", "PANDAS_TIMESTAMP_", "CV2_UNDISTORT", "SCIPY_STATS_MSTATS_", "SCIPY_SPARSE_", "SKLEARN_EXTERNALS_", "NUMPY_F2PY_", "NUMPY_LIB_", "NUMPY_MA_"]) or "OVERLOAD" in cid or "WRAPPER" in cid:
                        domain_factor *= 0.01

                    # Stage-2 curated cell bonus: strongly prefer verified, curated cells
                    stage_bonus = 0.0
                    if getattr(cell, 'stage', 1) == 2:
                        stage_bonus = 2.0

                    adjusted_dist = (dist + keyword_score + stage_bonus) * type_factor * domain_factor
                    scored_results.append((adjusted_dist, cell))

        if scored_results:
            scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results

    def _score_and_select_best(
        self, candidates: list, prompt_embedding: np.ndarray, goal: str = "", current_type: str = None, current_node = None
    ) -> tuple:
        results = self._score_candidates(candidates, prompt_embedding, goal, current_type, current_node)
        if results:
            return results[0][1], results[0][0]
        return None, -1.0
