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

            # O(1) indexed lookup of downstream successors
            for cell in self.orchestrator.get_successors_for_sig(curr_sig):
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
    Semantic router with type-monadic beam search and planner synchronization.
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
        from planner import ZeroShotPlanner

        prompt_lower = prompt.lower()
        prompt_keywords = set(re.findall(r"[a-zA-Z_]+", prompt_lower))

        # Use ZeroShotPlanner to obtain the topological stage plan / waypoints
        planner = ZeroShotPlanner(self.orchestrator, self.rag)
        plan_dict = planner.run_planning_pass(prompt)
        cells_blocks = plan_dict.get("cells", [])
        sub_cells = cells_blocks[0].get("sub_cells", []) if cells_blocks else []

        if sub_cells:
            resolved_path: List[Cell] = []
            current_sig = AlgebraicSignature(
                start_type or "str",
                start_state or "source_identifier"
            )

            for step_id in sub_cells:
                target_cell = self.orchestrator.loaded_cells.get(step_id)
                if not target_cell:
                    continue

                if current_sig.unifies_with(target_cell.primary_input) or target_cell.can_accept(current_sig):
                    resolved_path.append(target_cell)
                    current_sig = target_cell.primary_output
                else:
                    # Bridge gap between current signature and target cell's required input
                    bridge = self.mcts.search(current_sig, target_cell.primary_input, prompt_keywords=prompt_keywords)
                    if bridge:
                        resolved_path.extend(bridge)
                        current_sig = bridge[-1].primary_output
                    resolved_path.append(target_cell)
                    current_sig = target_cell.primary_output

                # If this cell is a Stage 3 sink (exporter/writer), terminate path immediately
                if target_cell.stage == 3:
                    break

            if resolved_path:
                return resolved_path, set()

        # Pure beam search fallback if planner returned no sub-cells
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
                if (sig.unifies_with(goal_sig) or sig.state in ("filepath_written", "written")) and path:
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

        if not domain_hint and entries:
            domain_counts: Dict[str, float] = {}
            for entry in entries[:5]:
                d = (entry.get("domain") or "").lower()
                if d and d not in ("generic", "python_core", "macro"):
                    domain_counts[d] = domain_counts.get(d, 0.0) + float(entry.get("score", 0.5))
            if domain_counts:
                domain_hint = max(domain_counts, key=domain_counts.get)

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
