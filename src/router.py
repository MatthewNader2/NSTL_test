# router.py
import logging
import math
import os
import platform
import random
import re
import copy
import warnings

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

logger = get_logger('router')

# H-5 fix: define stop-words once at module level so the scoring inner loop
# doesn't rebuild the set on every FAISS result iteration.
_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'to', 'with', 'any', 'it', 'is',
    'in', 'of', 'for', 'on', 'by', 'function', 'write', 'python', 'code',
    'script', 'create', 'def', 'that', 'returns', 'result',
})


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

    def search(self, start_type: str, target_type: str, iterations: int = 1000) -> list:
        logger.debug(f"MCTSEngine.search started with start_type={start_type}, target_type={target_type}, iterations={iterations}")
        # Root node acts as the starting typestate
        root = MCTSNode(cell_id="ROOT", current_type=start_type)
        
        for _ in range(iterations):
            # 1. Selection
            leaf = self._select(root)
            
            # 2. Expansion (with Unification Filter)
            if leaf.current_type != target_type and leaf.visits > 0:
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

    def _expand(self, node: MCTSNode):
        if node.cell_id == "ROOT":
            # For root, get all MicroCells that match start_type
            candidates = self.micro_by_type.get(node.current_type, [])
        else:
            candidates = self.orchestrator.get_neighbors(node.cell_id)

        for cell in candidates:
            if cell.type == "micro" and cell.inputs.type_name == node.current_type:
                # UnificationGate filtering happens here natively because we verify type_name match
                child = MCTSNode(cell_id=cell.cell_id, current_type=cell.outputs.type_name, parent=node)
                node.children.append(child)

    def _simulate(self, node: MCTSNode, target_type: str) -> float:
        current_type = node.current_type
        current_id = node.cell_id
        depth = 0
        max_depth = 15

        while current_type != target_type and depth < max_depth:
            if current_id == "ROOT":
                neighbors = self.micro_by_type.get(current_type, [])
            else:
                neighbors = [c for c in self.micro_by_type.get(current_type, [])
                             if getattr(c, 'node_type', 'function') == 'function']
            
            if not neighbors:
                return 0.0 # Dead end
            
            next_cell = random.choice(neighbors)
            current_type = next_cell.outputs.type_name
            current_id = next_cell.cell_id
            depth += 1
            
        if current_type == target_type:
            return 1.0
        return 0.0 # Exceeded max_depth

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
            if best_child.current_type == target_type:
                return path
            current = best_child
        return []

class LatticeRouter:
    def __init__(self, orchestrator, rag_engine):
        self.orchestrator = orchestrator
        self.rag_engine = rag_engine

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

        final_path = []
        virtual_edges = set()

        current_type = initial_type
        current_state = initial_state
        current_node = None

        MIN_CONFIDENCE = 0.30
        TUNNELING_MARGIN = 0.15
        MACRO_THRESHOLD = 0.40

        step = 0
        while goals:
            goal = goals.pop(0)

            # Format target query for FAISS (2D numpy array, Normalized)
            from inference import ModelManager
            goal_embedding_list = ModelManager.get_instance().get_embeddings([goal])
            goal_embedding = np.array(goal_embedding_list, dtype=np.float32)
            norm = np.linalg.norm(goal_embedding, axis=1, keepdims=True)
            norm = np.where(norm == 0, 1.0, norm)
            goal_embedding = goal_embedding / norm

            global_micro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                    if c.type == "micro"
                    and (getattr(c, 'node_type', None) in [None, 'function'])
                    and c.cell_id != (current_node.cell_id if current_node else "")
            ]
            best_global_micro, global_micro_score = self._score_and_select_best(
                global_micro_candidates, goal_embedding, goal, current_type
            )

            macro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "macro"
            ]
            best_macro, macro_score = self._score_and_select_best(
                macro_candidates, goal_embedding, goal, current_type
            )

            if (
                best_macro
                and macro_score > MACRO_THRESHOLD
                and global_micro_score < 0.70
            ):
                # BUG 2 FIX: Use algorithmic_steps for purely language-agnostic fractal unfolding
                expansion = getattr(best_macro, 'algorithmic_steps', [])
                if not expansion:
                    # Fallback to intent_expansion or sub_cells if algorithmic_steps is empty
                    expansion = getattr(best_macro, 'intent_expansion', None) or getattr(best_macro, 'sub_cells', [])
                
                expansion_goals = []
                for item in expansion:
                    if isinstance(item, str):
                        expansion_goals.append(item)
                    else:
                        expansion_goals.append(getattr(item, 'cell_id', str(item)))
                logger.info(
                    f"[FRACTAL UNFOLDING] Abstract goal '{goal}' expanded into {len(expansion_goals)} sub-operations."
                )
                goals = expansion_goals + goals
                continue

            if step == 0:
                candidates = [
                    c
                    for c in self.orchestrator.get_all_available_cells()
                    if c.type == "micro"
                    and (getattr(c, 'node_type', None) in [None, 'function'])
                    and c.inputs.matches(AlgebraicSignature(current_type, current_state))
                ]
                    
                best_node, best_score = self._score_and_select_best(
                    candidates, goal_embedding, goal, current_type
                )
                if best_node:
                    logger.debug(f"[ROUTER] best_score={best_score} for node {best_node.cell_id}")
                else:
                    logger.debug(f"[ROUTER] best_score={best_score} (no node)")

                if best_score < MIN_CONFIDENCE:
                    logger.warning(
                        f"[ROUTER HALT] Entry confidence too low ({best_score:.2f}) for goal: '{goal}'."
                    )
                    break

                final_path.append(best_node)
                current_node = best_node
                # BUG 1 FIX: Use dataclass attributes, not .get()
                current_type = current_node.outputs.type_name
                current_state = current_node.outputs.state
                step += 1
                continue

            strict_candidates = self.orchestrator.get_neighbors(current_node.cell_id)
            best_local_node, best_local_score = self._score_and_select_best(
                strict_candidates, goal_embedding, goal, current_type
            )

            # 2. Relaxed Topology (Global) Search
            global_candidates = [
                c for c in self.orchestrator.get_all_available_cells()
                if c.type == "micro" and getattr(c, 'node_type', 'function') == 'function' and c.cell_id != current_node.cell_id
            ]
            best_global_node, best_global_score = self._score_and_select_best(
                global_candidates, goal_embedding, goal, current_type
            )

            # BUG 1 FIX: Initialize best_node to None so the control flow is always safe.
            best_node = None

            if best_global_score > MIN_CONFIDENCE and (
                best_local_score < MIN_CONFIDENCE
                or (best_global_score - best_local_score > TUNNELING_MARGIN)
            ):
                logger.info(
                    f"[ROUTER] Semantic gravity exceeded local bounds! Goal: '{goal}'"
                )
                # BUG 1 FIX: Use .type_name instead of .get("input_type")
                target_type = best_global_node.inputs.type_name

                if target_type == current_type:
                    logger.info(
                        f"  [+] VIRTUAL EDGE COMPILED! Tunneling to -> {best_global_node.cell_id} (Score: {best_global_score:.2f})"
                    )
                    best_node = best_global_node
                    virtual_edges.add(best_node.cell_id)
                else:
                    logger.warning(
                        f"  [!] TYPE MISMATCH: Current '{current_type}' cannot flow into '{target_type}'. Searching for bridge..."
                    )
                    # B-8 fix: reuse the per-router cached MCTSEngine; avoids O(N) rebuild per bridge
                    if not hasattr(self, '_mcts_cache'):
                        self._mcts_cache = MCTSEngine(self.orchestrator)
                    bridge_path = self._mcts_cache.search(current_type, target_type, iterations=1000)
                    
                    if not bridge_path:
                        from inference import ModelManager
                        if not ModelManager.get_instance().can_synthesize():
                            logger.error(f"  [!] COST EVALUATION: C_sub = ∞. Synthesis disabled for current BenchmarkProfile. Path blocked.")
                            break
                        
                        logger.info(f"  [!] COST EVALUATION: C_sub = ∞. C_gen = 1000. C_gen < C_sub. Triggering Synthesis Engine...")
                        from synthesis import SynthesisEngine
                        from external_rag import FetcherFactory
                        
                        synth = SynthesisEngine()
                        # Use DuckDuckGo by default for generic gap bridging if domain isn't known here
                        fetcher = FetcherFactory.get_fetcher("Python")
                        
                        try:
                            # We formulate the gap concept as the coercion between types
                            gap_concept = f"convert {current_type} to {target_type}"
                            micro_json = synth.synthesize_micro_cell(gap_concept, current_type, target_type, fetcher)
                            
                            from unification import UnificationGate
                            if UnificationGate.validate_synthesis(micro_json, current_type, target_type):
                                bridge_node = self.orchestrator.inject_transient_macro(micro_json)
                                # BUG 7 FIX: Rebuild FAISS index so the newly injected cell
                                # is visible to _score_and_select_best in future routing steps.
                                if self.rag_engine:
                                    self.rag_engine.add_dynamic_cell(micro_json)
                                bridge_path = [bridge_node]
                                logger.info(f"  [+] SYNTHESIS COMPLETE: Bridged {current_type} -> {target_type}")
                            else:
                                logger.error("  [-] SYNTHESIS FAILED: Unification Gate Rejected Typestates.")
                        except Exception as e:
                            logger.error(f"  [-] SYNTHESIS FAILED: {e}")

                    if bridge_path:
                        for b_node in bridge_path:
                            logger.info(f"  [+] COERCION BRIDGE FOUND! Injecting -> {b_node.cell_id}")
                            final_path.append(b_node)
                            virtual_edges.add(b_node.cell_id)
                        logger.info(f"  [+] TUNNELING COMPLETED to -> {best_global_node.cell_id} (Score: {best_global_score:.2f})")
                        best_node = best_global_node
                        virtual_edges.add(best_node.cell_id)
                    else:
                        logger.error(f"  [-] FATAL: No coercion bridge exists and Synthesis failed between '{current_type}' and '{target_type}'. Path blocked.")
                        break
            elif best_local_score >= MIN_CONFIDENCE:
                best_node = best_local_node
            else:
                logger.warning(f"[ROUTER HALT] Pathfinding failed for goal: '{goal}'.")
                break

            # Guard: if no branch assigned best_node, stop routing.
            if best_node is None:
                logger.warning(f"[ROUTER HALT] No valid node resolved for goal: '{goal}'.")
                break
                
            # Clone best_node to attach heuristics safely
            best_node = copy.copy(best_node)
            
            # Extract heuristics purely from this goal
            goal_heuristics = []
            all_quoted = re.findall(r'["\']([^"\']+)["\']', goal)
            for q in all_quoted:
                # Naively add all quoted strings
                goal_heuristics.append(f"{repr(q)}")
                
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', goal)
            for n in numbers:
                goal_heuristics.append(n)
                
            best_node.matched_heuristics = goal_heuristics

            final_path.append(best_node)
            current_node = best_node
            # BUG 1 FIX: Use dataclass attributes, not .get()
            current_type = current_node.outputs.type_name
            current_state = current_node.outputs.state
            step += 1

        if goals:
            logger.error(
                f"\n[PATHFINDER FAILED] Route incomplete — {len(goals)} unresolved goals remain. Partial path: {[c.cell_id for c in final_path]}"
            )

        logger.info(
            f"\n[PATHFINDER COMPLETE] Route Generated: {[c.cell_id for c in final_path]}"
        )
        return final_path, virtual_edges

    def _score_and_select_best(
        self, candidates: list, prompt_embedding: np.ndarray, goal: str = "", current_type: str = None
    ) -> tuple:
        """FAISS Vector Database Lookup - O(1) Constraint Filtering inside O(log N) Graph Traverse"""
        if not candidates or self.rag_engine is None or self.rag_engine.index is None:
            return None, -1.0

        valid_ids = {c.cell_id for c in candidates}
        candidates_dict = {c.cell_id: c for c in candidates}

        # Query all nodes across the fractal lattice instantly to ensure keyword boosting can rescue poor embeddings
        k_search = self.rag_engine.index.ntotal
        distances, indices = self.rag_engine.index.search(prompt_embedding, k=k_search)

        # The first valid candidate we hit is mathematically the most semantically aligned
        import re
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
                    id_parts = {p for p in re.split(r"[_\W]+", cell.cell_id.lower()) if p}
                    
                    filtered_tokens = {pt for pt in prompt_tokens if pt not in _STOP_WORDS and len(pt) > 2}

                    # Module alias expansion (e.g. opencv <-> cv2, pandas <-> pd)
                    MODULE_ALIASES = {
                        "opencv": "cv2", "cv2": "opencv",
                        "pandas": "pd", "pd": "pandas",
                        "numpy": "np", "np": "numpy",
                        "matplotlib": "plt", "plt": "matplotlib",
                        "seaborn": "sns", "sns": "seaborn",
                        "scikit": "sklearn", "sklearn": "scikit",
                        "tensorflow": "tf", "tf": "tensorflow"
                    }
                    expanded_tokens = set(filtered_tokens)
                    for pt in filtered_tokens:
                        if pt in MODULE_ALIASES:
                            expanded_tokens.add(MODULE_ALIASES[pt])

                    overlap = 0.0
                    for pt in expanded_tokens:
                        if any(pt in kw or (len(kw) >= 3 and kw in pt) for kw in kws):
                            overlap += 0.3
                        if any(pt in p or (len(p) >= 3 and p in pt) for p in id_parts):
                            overlap += 0.3
                            
                    # Generic domain-agnostic keyword overlap boost
                    intent_boost = 0.0
                    penalty = 0.0
                    if any(w in prompt_tokens for w in ['function', 'def']):
                        intent_boost -= 2.0

                    # Penalize inspector/boolean checker methods over core transformation functions
                    if "haveimage" in cid_lower or "have_image" in cid_lower or cid_lower.startswith("is_") or cid_lower.startswith("has_"):
                        penalty += 0.8

                    # Boost primary action methods matching prompt verb/noun tokens
                    if any(act in cid_lower for act in ["imread", "imwrite", "cvtcolor", "read_csv", "to_csv", "dropna", "sort_values", "predict", "fit", "transform"]):
                        for pt in filtered_tokens:
                            if pt in cid_lower or any(p in cid_lower for p in id_parts):
                                intent_boost += 0.4
                                break

                    # Penalize nodes that require coercion bridges or input type mismatch
                    if current_type and getattr(cell, 'inputs', None) and getattr(cell.inputs, 'type_name', '') != 'any':
                        if getattr(cell.inputs, 'type_name', '') != current_type:
                            penalty += 1.5
                            
                    adjusted_dist = dist + overlap + intent_boost - penalty
                    scored_results.append((adjusted_dist, float(dist), cell))

        if scored_results:
            scored_results.sort(key=lambda x: x[0], reverse=True)
            return scored_results[0][2], scored_results[0][0]

        # Extreme fallback
        if k_search < self.rag_engine.index.ntotal:
            distances, indices = self.rag_engine.index.search(
                prompt_embedding, k=self.rag_engine.index.ntotal
            )
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx not in self.rag_engine.id_to_schema:
                    continue
                cid = self.rag_engine.id_to_schema[idx].get("cell_id")
                if cid in valid_ids:
                    cell = candidates_dict.get(cid)
                    if cell:
                        kws = {kw.lower() for kw in getattr(cell, 'keywords', [])}
                        id_parts = {p for p in re.split(r"[_\W]+", cell.cell_id.lower()) if p}
                        overlap = len(prompt_tokens.intersection(kws)) * 0.2 + len(prompt_tokens.intersection(id_parts)) * 0.1
                        
                        penalty = 0.0
                        if getattr(cell, 'inputs', None) and getattr(cell.inputs, 'type_name', '') == 'any':
                            penalty += 0.3
                        if getattr(cell, 'outputs', None) and getattr(cell.outputs, 'type_name', '') == 'any':
                            penalty += 0.3
                        
                        if current_type and getattr(cell, 'inputs', None) and getattr(cell.inputs, 'type_name', '') != 'any':
                            if getattr(cell.inputs, 'type_name', '') != current_type:
                                penalty += 0.5
                                
                        adjusted_dist = dist + overlap - penalty
                        scored_results.append((adjusted_dist, float(dist), cell))

            if scored_results:
                scored_results.sort(key=lambda x: x[0], reverse=True)
                return scored_results[0][2], scored_results[0][1]

        return None, -1.0
