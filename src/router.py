"""
src/router.py - Neuro-Symbolic Topological Lattice (NSTL)
FIXED: Domain-aware scoring, keyword overlap, algorithmic bypass, robust RAG format handling,
       and preserves original return type contract: (List[Cell], Set[str]).
"""

from __future__ import annotations
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any

import numpy as np
import torch
from log_config import get_logger
from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, AlgebraicSignature

logger = get_logger('router')


class HardwareProfiler:
    _cached_device: Optional[str] = None
    _config: Dict[str, str] = {
        'embedder': 'auto',
        'llm': 'auto',
        'trees': 'ram'
    }

    @classmethod
    def set_config(cls, embedder_device: str = "auto", llm_device: str = "auto", trees_storage: str = "ram"):
        cls._config['embedder'] = (embedder_device or 'auto').lower()
        cls._config['llm'] = (llm_device or 'auto').lower()
        cls._config['trees'] = (trees_storage or 'ram').lower()

    @classmethod
    def get_embedder_device(cls) -> str:
        if cls._config.get('embedder', 'auto') != 'auto':
            return cls._config['embedder']
        return cls.get_optimal_device()

    @classmethod
    def get_llm_device(cls) -> str:
        if cls._config.get('llm', 'auto') != 'auto':
            return cls._config['llm']
        return cls.get_optimal_device()

    @classmethod
    def get_optimal_device(cls) -> str:
        if cls._cached_device is not None:
            return cls._cached_device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        cls._cached_device = device
        logger.info(f"[HARDWARE PROFILER] Optimal compute device: {device.upper()}")
        return device


class MCTSNode:
    __slots__ = ['cell_id', 'signature', 'parent', 'children', 'visits', 'q_value']

    def __init__(self, cell_id: str, signature: AlgebraicSignature, parent: Optional[MCTSNode] = None):
        self.cell_id = cell_id
        self.signature = signature
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.q_value: float = 0.0

    def ucb1(self, c_param: float = 0.5) -> float:
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return (self.q_value / self.visits) + c_param * math.sqrt(2 * math.log(parent_visits) / self.visits)


class MCTSEngine:
    """Deterministic A* / Best-First typestate bridge engine over the typed lattice."""

    def __init__(self, orchestrator: LatticeOrchestrator):
        self.orchestrator = orchestrator

    def search(
        self,
        start_sig: AlgebraicSignature,
        goal_sig: AlgebraicSignature,
        max_depth: int = 6,
        iterations: int = 300,
        prompt_keywords: Optional[Set[str]] = None
    ) -> List[Cell]:
        """Returns List[Cell] to bridge between start_sig and goal_sig."""
        if start_sig.unifies_with(goal_sig):
            return []

        import heapq
        # Priority queue entries: (cost, depth, counter, current_sig, [cell_ids])
        counter = 0
        pq = [(0.0, 0, counter, start_sig, [])]
        visited_sigs = set()
        kw_set = prompt_keywords or set()

        while pq:
            cost, depth, _, curr_sig, path = heapq.heappop(pq)

            if curr_sig.unifies_with(goal_sig) and path:
                return [self.orchestrator.loaded_cells[cid] for cid in path if cid in self.orchestrator.loaded_cells]

            if depth >= max_depth:
                continue

            sig_key = (curr_sig.type_name, curr_sig.state)
            if sig_key in visited_sigs:
                continue
            visited_sigs.add(sig_key)

            # Get neighbors from orchestrator using bucket lookup
            with self.orchestrator._lock:
                bucket_items = list(self.orchestrator._cells_by_input.items())

            for (in_type, in_state), target_cells in bucket_items:
                if curr_sig.unifies_with(AlgebraicSignature(in_type, in_state)):
                    for cell in target_cells:
                        if cell.cell_id in path:
                            continue

                        next_sig = cell.primary_output
                        cell_kws = set(k.lower() for k in cell.keywords)
                        overlap = len(kw_set & cell_kws) if kw_set else 0

                        # Priority: prefer fewer hops, higher keyword overlap, avoid internal helper noise
                        step_cost = 1.0 - (0.25 * min(overlap, 3))
                        if any(p in cell.cell_id.lower() for p in ["_group_", "_internal_", "typing_"]):
                            step_cost += 0.5

                        counter += 1
                        heapq.heappush(pq, (cost + step_cost, depth + 1, counter, next_sig, path + [cell.cell_id]))

        return []


class LatticeRouter:
    """
    Semantic router with type-monadic beam search.
    Returns (List[Cell], Set[str]) to match main.py contract.
    """

    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        rag_engine: Any,
        reranker: Optional[Any] = None
    ):
        self.orchestrator = orchestrator
        self.rag = rag_engine
        self.reranker = reranker
        self.mcts = MCTSEngine(orchestrator)
        self._keyword_cache: Dict[str, Set[str]] = {}

    def plan_path(
        self,
        prompt: str,
        start_type: Optional[str] = None,
        start_state: Optional[str] = None,
        goal_type: Optional[str] = None,
        goal_state: Optional[str] = None,
        beam_width: int = 5,
        max_steps: int = 12
    ) -> Tuple[List[Cell], Set[str]]:
        prompt_lower = prompt.lower()
        prompt_keywords = set(re.findall(r"[a-zA-Z_]+", prompt_lower))

        # Check for algorithmic seeds first
        algo_indicators = ["dijkstra", "shortest_path", "bfs", "dfs", "quicksort", "mergesort",
                           "binary_search", "a_star", "astar", "topological_sort"]
        matched_indicators = [ind for ind in algo_indicators if ind in prompt_lower or ind.replace("_", "") in prompt_lower.replace(" ", "")]
        if matched_indicators:
            candidates: List[Tuple[str, int]] = []
            for c in self.orchestrator.loaded_cells.values():
                if c.domain_name in ("algorithms", "python_core", "generic", "macro") or isinstance(c, MacroCell):
                    cid_lower = c.cell_id.lower()
                    cell_kws = {k.lower() for k in c.keywords}
                    algo_overlap = sum(1 for ind in matched_indicators if ind in cid_lower or ind in cell_kws or any(ind in k for k in cell_kws))
                    if algo_overlap > 0:
                        candidates.append((c.cell_id, algo_overlap))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_algo_id = candidates[0][0]
                logger.info(f"[ROUTER] Algorithmic seed found: {best_algo_id}")
                return self._ids_to_cells([best_algo_id]), set()
            logger.info("[ROUTER] Algorithmic task detected without seed; yielding to synthesis.")
            return [], set()

        current_sig = AlgebraicSignature(
            start_type or "str",
            start_state or "source_identifier"
        )
        goal_sig = AlgebraicSignature(
            goal_type or "str",
            goal_state or "filepath_written"
        )

        beam: List[Tuple[List[str], AlgebraicSignature, float]] = [([], current_sig, 0.0)]
        visited_sequences: Set[str] = set()

        for step in range(max_steps):
            candidates: List[Tuple[List[str], AlgebraicSignature, float]] = []

            for path, sig, score in beam:
                if sig.unifies_with(goal_sig) and path:
                    return self._ids_to_cells(path), set()

                seq_key = "->".join(path)
                if seq_key in visited_sequences:
                    continue
                visited_sequences.add(seq_key)

                try:
                    next_nodes = self._score_candidates(sig, prompt, prompt_keywords, top_k=25)
                except Exception as e:
                    logger.error(f"[ROUTER] Scoring failed at step {step}: {e}")
                    next_nodes = []

                if not next_nodes:
                    next_nodes = self._keyword_fallback(sig, prompt_keywords, top_k=10)

                for cid, node_score in next_nodes:
                    if cid in path:
                        continue

                    cell = self.orchestrator.loaded_cells.get(cid)
                    if not cell:
                        continue

                    new_path = path + [cid]
                    new_sig = cell.primary_output
                    new_score = score + node_score - (len(new_path) * 0.01)
                    candidates.append((new_path, new_sig, new_score))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[:beam_width]

        if beam:
            best_path, best_sig, _ = max(beam, key=lambda x: x[2])
            if not best_sig.unifies_with(goal_sig):
                try:
                    bridge = self.mcts.search(best_sig, goal_sig, prompt_keywords=prompt_keywords)
                    if bridge:
                        return self._ids_to_cells(best_path + [c.cell_id for c in bridge]), set()
                except Exception as e:
                    logger.error(f"[ROUTER] MCTS bridging failed: {e}")

            if best_path:
                return self._ids_to_cells(best_path), set()

        return [], set()

    def _ids_to_cells(self, ids: List[str]) -> List[Cell]:
        cells: List[Cell] = []
        for cid in ids:
            cell = self.orchestrator.loaded_cells.get(cid)
            if cell:
                cells.append(cell)
        return cells

    def _score_candidates(
        self,
        current_sig: AlgebraicSignature,
        prompt: str,
        prompt_keywords: Set[str],
        top_k: int = 25
    ) -> List[Tuple[str, float]]:
        """Scores candidate cells with strict type pre-filtering and domain alignment."""
        try:
            raw_candidates = self.rag.get_relevant_context(prompt, top_k=top_k * 2)
        except Exception as e:
            logger.error(f"[ROUTER] RAG query failed: {e}")
            return []

        if not raw_candidates:
            return []

        # Handle list of dicts from RAG directly
        entries: List[Dict[str, Any]] = []
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if isinstance(item, dict):
                    entries.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    entries.append({
                        "cell_id": str(item[0]),
                        "score": float(item[1]) if len(item) > 1 else 0.5
                    })

        scored: List[Tuple[str, float]] = []
        prompt_lower = prompt.lower()

        # Infer active domain dynamically
        domain_hint = None
        if entries:
            top_dom = entries[0].get("domain", "")
            if top_dom and top_dom not in ("generic", "python_core"):
                domain_hint = top_dom.lower()

        if not domain_hint:
            domain_map = {
                "pandas": ["csv", "dataframe", "df", "read_csv", "to_csv", "dropna"],
                "cv2": ["image", "opencv", "grayscale", "cvtcolor", "imread", "imwrite"],
                "numpy": ["array", "ndarray", "reshape", "linspace", "zeros"],
                "sklearn": ["classifier", "regressor", "fit", "predict", "scaler"],
                "scipy": ["sparse", "csgraph", "optimize", "integrate"],
                "matplotlib": ["plot", "figure", "subplot", "savefig"],
            }
            for d, kws in domain_map.items():
                if any(k in prompt_lower for k in kws):
                    domain_hint = d
                    break

        for entry in entries:
            cid = entry.get("cell_id", "")
            if not cid:
                continue

            cell = self.orchestrator.loaded_cells.get(cid)
            if not cell:
                continue

            # STRICT TYPE PRE-FILTERING (C3 FIX):
            # If the candidate cell cannot accept the current signature, reject immediately!
            if not current_sig.unifies_with(cell.primary_input):
                continue

            base_score = float(entry.get("score", 0.5))

            if domain_hint:
                cell_domain = (cell.domain_name or "").lower()
                if domain_hint == cell_domain:
                    base_score *= 1.4
                elif cell_domain not in ("generic", "macro", "python_core"):
                    base_score *= 0.3

            cell_kws = set(k.lower() for k in cell.keywords)
            overlap = len(prompt_keywords & cell_kws)
            base_score += overlap * 0.25

            if any(p in cid.lower() for p in ["_group_", "_core_", "_algos_", "_internal_",
                                               "typing_", "withmetadata", "renderer"]):
                base_score *= 0.3

            scored.append((cid, base_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _keyword_fallback(
        self,
        current_sig: AlgebraicSignature,
        prompt_keywords: Set[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Pure keyword matching fallback when RAG returns nothing or crashes."""
        results: List[Tuple[str, float]] = []
        for cid, cell in self.orchestrator.loaded_cells.items():
            if not current_sig.unifies_with(cell.primary_input):
                continue
            cell_kws = set(k.lower() for k in cell.keywords)
            overlap = len(prompt_keywords & cell_kws)
            if overlap > 0:
                results.append((cid, overlap * 0.5))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class FastPathRouter:
    def __init__(self, orchestrator: LatticeOrchestrator, rag: Any):
        self.orchestrator = orchestrator
        self.rag = rag

    def try_fast_path(self, prompt: str, threshold: float = 0.92) -> Optional[List[str]]:
        result = self.rag.find_closest_cell_by_embedding(prompt)
        if not result or result.get("score", 0.0) < threshold:
            return None

        cid = result.get("cell_id", "")
        cell = self.orchestrator.loaded_cells.get(cid)
        if isinstance(cell, MacroCell):
            return cell.sub_cells
        return None
