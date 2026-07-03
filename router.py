# router.py
import logging
import os
import platform
import re
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


class HardwareProfiler:
    """
    Cross-Platform Dynamic Hardware Auto-Profiler.
    Detects the host system and assigns workloads to the optimal compute backend.
    """
    # BUG 25 FIX: Cache result to avoid printing the banner multiple times on startup.
    _cached_device: str = None

    @staticmethod
    def get_optimal_device() -> str:
        # BUG 25 FIX: Return cached result if already profiled.
        if HardwareProfiler._cached_device is not None:
            return HardwareProfiler._cached_device

        print("\n" + "=" * 50)
        print(" NSTL HARDWARE AUTODETECTION PROFILER")
        print("=" * 50)

        os_name = platform.system()
        cpu_arch = platform.machine()
        print(f" Host OS:   {os_name} ({cpu_arch})")
        import sys
        print(f" Python Exe: {sys.executable}")
        print(f" Torch Ver:  {torch.__version__} ({torch.__file__})")
        print(f" CUDA Avail: {torch.cuda.is_available()}")

        # 1. Check for NVIDIA CUDA GPUs
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f" Compute:   NVIDIA CUDA Detected")
            print(f" Hardware:  {gpu_name} ({vram:.1f} GB VRAM)")
            print(" Routing:   Vector Embedding mapped to GPU.")
            print("=" * 50 + "\n")
            HardwareProfiler._cached_device = "cuda"
            return "cuda"
            
        # 1.5. Check for physical NVIDIA GPUs (if PyTorch lacks CUDA)
        try:
            import subprocess
            smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            if smi_output:
                gpu_name = smi_output.split('\n')[0]
                print(f" Compute:   NVIDIA GPU Detected (System)")
                print(f" Hardware:  {gpu_name}")
                print(" Routing:   LLM mapped to GPU, Vector Embedding mapped to CPU (Install PyTorch CUDA for full acceleration).")
                print("=" * 50 + "\n")
                HardwareProfiler._cached_device = "cpu"
                return "cpu"
        except Exception:
            pass

        # 2. Check for Apple Silicon (M1/M2/M3/M4) Neural Engines
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f" Compute:   Apple Metal Performance Shaders (MPS)")
            print(f" Hardware:  Apple Silicon Architecture")
            print(" Routing:   Vector Embedding mapped to Apple GPU.")
            print("=" * 50 + "\n")
            HardwareProfiler._cached_device = "mps"
            return "mps"

        # 3. Fallback to CPU (Windows without Nvidia, Intel Macs, Linux Servers)
        print(f" Compute:   Standard CPU")
        print(f" Hardware:  Fallback Mode Activated")
        print(" Routing:   Vector Embedding mapped to CPU.")
        print(" Note:      Operations will run safely, but slower.")
        print("=" * 50 + "\n")
        HardwareProfiler._cached_device = "cpu"
        return "cpu"

import math
import random

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

    def search(self, start_type: str, target_type: str, iterations: int = 1000) -> list:
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
            
        return self._get_best_path(root, target_type)

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
            candidates = [c for c in self.orchestrator.get_all_available_cells() 
                          if c.type == "micro" and c.inputs.type_name == node.current_type]
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
                neighbors = [c for c in self.orchestrator.get_all_available_cells() 
                             if c.type == "micro" and c.inputs.type_name == current_type]
            else:
                neighbors = [c for c in self.orchestrator.get_neighbors(current_id) 
                             if c.type == "micro" and c.inputs.type_name == current_type]
            
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
        # BUG 6 FIX: Build a full map of ALL cells (including MacroCell sub-cells)
        # to avoid KeyError when best_child refers to a cell not in loaded_cells top-level.
        all_cells_map = {c.cell_id: c for c in self.orchestrator.get_all_available_cells()}
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
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

        # 1. Run the Auto-Profiler to get the absolute best hardware state
        self.device = HardwareProfiler.get_optimal_device()

        # 2. Load the model directly into the optimized silicon
        print(f"[NEURAL CORE] Initializing Transformer on {self.device.upper()}...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        # 3. Initialize High-Speed CPU Memory for the Vector Database
        self.cells_list = []
        self.index = None
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Encodes the topology into a CPU-bound HNSW FAISS Graph for instantaneous O(log N) routing."""
        cells = self.orchestrator.get_all_available_cells()
        self.cells_list = cells

        if not cells:
            print("[NEURAL CORE] No cells found to index.")
            return

        print(
            f"[NEURAL CORE] Pre-computing semantic manifolds for {len(cells)} cells..."
        )
        texts = [
            f"Action: {c.cell_id}. Concept Tags: {' '.join(c.keywords)}." for c in cells
        ]

        # Batch encode & normalize for Cosine Similarity mapping
        embeddings = self.model.encode(
            texts,
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # HNSW (Hierarchical Navigable Small World) Index always stays on CPU for 0.1ms access times
        d = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.add(embeddings)

        print("[NEURAL CORE] FAISS HNSW Vector Database compiled. Routing is ready.")

    def plan_path(
        self, user_intent: str, initial_type: str, initial_state: str
    ) -> tuple:
        raw_goals = re.split(r",|\band\b|\bthen\b", user_intent)
        goals = [g.strip() for g in raw_goals if g.strip()]

        final_path = []
        virtual_edges = set()

        current_type = initial_type
        current_state = initial_state
        current_node = None

        MIN_CONFIDENCE = 0.25
        TUNNELING_MARGIN = 0.15
        MACRO_THRESHOLD = 0.40

        step = 0
        while goals:
            goal = goals.pop(0)

            # Format target query for FAISS (2D numpy array, Normalized)
            goal_embedding = self.model.encode(
                [goal], convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)

            global_micro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "micro"
                and c.cell_id != (current_node.cell_id if current_node else "")
            ]
            best_global_micro, global_micro_score = self._score_and_select_best(
                global_micro_candidates, goal_embedding
            )

            macro_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "macro"
            ]
            best_macro, macro_score = self._score_and_select_best(
                macro_candidates, goal_embedding
            )

            if (
                best_macro
                and macro_score > MACRO_THRESHOLD
                and global_micro_score < 0.70
            ):
                # BUG 2 FIX: MacroCell has no .intent_expansion — use sub_cells as fallback.
                expansion = getattr(best_macro, 'intent_expansion', None) or best_macro.sub_cells
                # sub_cells may contain Cell objects after resolution; extract IDs if needed
                expansion_goals = []
                for item in expansion:
                    if isinstance(item, str):
                        expansion_goals.append(item)
                    else:
                        expansion_goals.append(getattr(item, 'cell_id', str(item)))
                print(
                    f"[FRACTAL UNFOLDING] Abstract goal '{goal}' expanded into {len(expansion_goals)} sub-operations."
                )
                goals = expansion_goals + goals
                continue

            if step == 0:
                candidates = [
                    c
                    for c in self.orchestrator.get_all_available_cells()
                    if c.type == "micro"
                    # BUG 1 FIX: Use .type_name and .state instead of .get() on AlgebraicSignature.
                    and c.inputs.type_name == current_type
                    and c.inputs.state == current_state
                ]
                best_node, best_score = self._score_and_select_best(
                    candidates, goal_embedding
                )

                if best_score < MIN_CONFIDENCE:
                    print(
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
                strict_candidates, goal_embedding
            )

            global_candidates = [
                c
                for c in self.orchestrator.get_all_available_cells()
                if c.type == "micro" and c.cell_id != current_node.cell_id
            ]
            best_global_node, best_global_score = self._score_and_select_best(
                global_candidates, goal_embedding
            )

            # BUG 1 FIX: Initialize best_node to None so the control flow is always safe.
            best_node = None

            if best_global_score > MIN_CONFIDENCE and (
                best_local_score < MIN_CONFIDENCE
                or (best_global_score - best_local_score > TUNNELING_MARGIN)
            ):
                print(
                    f"[ROUTER] Semantic gravity exceeded local bounds! Goal: '{goal}'"
                )
                # BUG 1 FIX: Use .type_name instead of .get("input_type")
                target_type = best_global_node.inputs.type_name

                if target_type == current_type:
                    print(
                        f"  [+] VIRTUAL EDGE COMPILED! Tunneling to -> {best_global_node.cell_id} (Score: {best_global_score:.2f})"
                    )
                    best_node = best_global_node
                    virtual_edges.add(best_node.cell_id)
                else:
                    print(
                        f"  [!] TYPE MISMATCH: Current '{current_type}' cannot flow into '{target_type}'. Searching for bridge..."
                    )
                    mcts = MCTSEngine(self.orchestrator)
                    bridge_path = mcts.search(current_type, target_type, iterations=1000)
                    
                    if not bridge_path:
                        from inference import ModelManager
                        if not ModelManager.get_instance().can_synthesize():
                            print(f"  [!] COST EVALUATION: C_sub = ∞. Synthesis disabled for current BenchmarkProfile.")
                            raise ValueError(f"Topology gap between {current_type} and {target_type} cannot be resolved without synthesis.")
                        
                        print(f"  [!] COST EVALUATION: C_sub = ∞. C_gen = 1000. C_gen < C_sub. Triggering Synthesis Engine...")
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
                                self._precompute_embeddings()
                                bridge_path = [bridge_node]
                                print(f"  [+] SYNTHESIS COMPLETE: Bridged {current_type} -> {target_type}")
                            else:
                                print("  [-] SYNTHESIS FAILED: Unification Gate Rejected Typestates.")
                        except Exception as e:
                            print(f"  [-] SYNTHESIS FAILED: {e}")

                    if bridge_path:
                        for b_node in bridge_path:
                            print(f"  [+] COERCION BRIDGE FOUND! Injecting -> {b_node.cell_id}")
                            final_path.append(b_node)
                            virtual_edges.add(b_node.cell_id)
                        print(f"  [+] TUNNELING COMPLETED to -> {best_global_node.cell_id} (Score: {best_global_score:.2f})")
                        best_node = best_global_node
                        virtual_edges.add(best_node.cell_id)
                    else:
                        print(f"  [-] FATAL: No coercion bridge exists and Synthesis failed between '{current_type}' and '{target_type}'. Path blocked.")
                        break
            elif best_local_score >= MIN_CONFIDENCE:
                best_node = best_local_node
            else:
                print(f"[ROUTER HALT] Pathfinding failed for goal: '{goal}'.")
                break

            # Guard: if no branch assigned best_node, stop routing.
            if best_node is None:
                print(f"[ROUTER HALT] No valid node resolved for goal: '{goal}'.")
                break

            final_path.append(best_node)
            current_node = best_node
            # BUG 1 FIX: Use dataclass attributes, not .get()
            current_type = current_node.outputs.type_name
            current_state = current_node.outputs.state
            step += 1

        print(
            f"\n[PATHFINDER COMPLETE] Route Generated: {[c.cell_id for c in final_path]}"
        )
        return final_path, virtual_edges

    def _score_and_select_best(
        self, candidates: list, prompt_embedding: np.ndarray
    ) -> tuple:
        """FAISS Vector Database Lookup - O(1) Constraint Filtering inside O(log N) Graph Traverse"""
        if not candidates or self.index is None:
            return None, -1.0

        valid_ids = {c.cell_id for c in candidates}

        # Query top 200 closest nodes across the fractal lattice instantly
        k_search = min(200, self.index.ntotal)
        distances, indices = self.index.search(prompt_embedding, k=k_search)

        # The first valid candidate we hit is mathematically the most semantically aligned
        for dist, idx in zip(distances[0], indices[0]):
            cell = self.cells_list[idx]
            if cell.cell_id in valid_ids:
                return cell, float(dist)

        # Extreme fallback
        if k_search < self.index.ntotal:
            distances, indices = self.index.search(
                prompt_embedding, k=self.index.ntotal
            )
            for dist, idx in zip(distances[0], indices[0]):
                cell = self.cells_list[idx]
                if cell.cell_id in valid_ids:
                    return cell, float(dist)

        return None, -1.0
