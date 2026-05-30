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

    @staticmethod
    def get_optimal_device() -> str:
        print("\n" + "=" * 50)
        print(" NSTL HARDWARE AUTODETECTION PROFILER")
        print("=" * 50)

        os_name = platform.system()
        cpu_arch = platform.machine()
        print(f" Host OS:   {os_name} ({cpu_arch})")

        # 1. Check for NVIDIA CUDA GPUs
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f" Compute:   NVIDIA CUDA Detected")
            print(f" Hardware:  {gpu_name} ({vram:.1f} GB VRAM)")
            print(" Routing:   Vector Embedding mapped to GPU.")
            print("=" * 50 + "\n")
            return "cuda"

        # 2. Check for Apple Silicon (M1/M2/M3/M4) Neural Engines
        elif torch.backends.mps.is_available():
            print(f" Compute:   Apple Metal Performance Shaders (MPS)")
            print(f" Hardware:  Apple Silicon Architecture")
            print(" Routing:   Vector Embedding mapped to Apple GPU.")
            print("=" * 50 + "\n")
            return "mps"

        # 3. Fallback to CPU (Windows without Nvidia, Intel Macs, Linux Servers)
        else:
            print(f" Compute:   Standard CPU")
            print(f" Hardware:  Fallback Mode Activated")
            print(" Routing:   Vector Embedding mapped to CPU.")
            print(" Note:      Operations will run safely, but slower.")
            print("=" * 50 + "\n")
            return "cpu"


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
                print(
                    f"[FRACTAL UNFOLDING] Abstract goal '{goal}' expanded into {len(best_macro.intent_expansion)} sub-operations."
                )
                goals = best_macro.intent_expansion + goals
                continue

            if step == 0:
                candidates = [
                    c
                    for c in self.orchestrator.get_all_available_cells()
                    if c.type == "micro"
                    and c.inputs.get("input_type") == current_type
                    and c.inputs.get("expected_state") == current_state
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
                current_type = current_node.outputs.get("output_type")
                current_state = current_node.outputs.get("resulting_state")
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

            if best_global_score > MIN_CONFIDENCE and (
                best_local_score < MIN_CONFIDENCE
                or (best_global_score - best_local_score > TUNNELING_MARGIN)
            ):
                print(
                    f"[ROUTER] Semantic gravity exceeded local bounds! Goal: '{goal}'"
                )
                target_type = best_global_node.inputs.get("input_type")

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
                    bridge_node = self.orchestrator.find_type_bridge(
                        current_type, target_type
                    )
                    if bridge_node:
                        print(
                            f"  [+] COERCION BRIDGE FOUND! Injecting -> {bridge_node.cell_id}"
                        )
                        final_path.append(bridge_node)
                        virtual_edges.add(bridge_node.cell_id)
                        print(
                            f"  [+] TUNNELING COMPLETED to -> {best_global_node.cell_id} (Score: {best_global_score:.2f})"
                        )
                        best_node = best_global_node
                        virtual_edges.add(best_node.cell_id)
                    else:
                        print(
                            f"  [-] FATAL: No coercion bridge exists between '{current_type}' and '{target_type}'. Path blocked."
                        )
                        break
            elif best_local_score >= MIN_CONFIDENCE:
                best_node = best_local_node
            else:
                print(f"[ROUTER HALT] Pathfinding failed for goal: '{goal}'.")
                break

            final_path.append(best_node)
            current_node = best_node
            current_type = current_node.outputs.get("output_type")
            current_state = current_node.outputs.get("resulting_state")
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
