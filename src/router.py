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
    """Monte Carlo Tree Search for bridging typestate gaps."""

    def __init__(self, orchestrator: LatticeOrchestrator):
        self.orchestrator = orchestrator

    def search(
        self,
        start_sig: AlgebraicSignature,
        goal_sig: AlgebraicSignature,
        max_depth: int = 5,
        iterations: int = 300,
        prompt_keywords: Optional[Set[str]] = None
    ) -> List[Cell]:
        """Returns List[Cell] to match main.py's append expectations."""
        if start_sig.unifies_with(goal_sig):
            return []

        root = MCTSNode("__root__", start_sig)
        best_path_ids: List[str] = []
        best_score = -1.0

        for _ in range(iterations):
            node = self._select(root)
            if node is None:
                continue
            reward, path_ids = self._rollout(node, goal_sig, max_depth, prompt_keywords)
            self._backpropagate(node, reward)
            if reward > best_score and path_ids:
                best_score = reward
                best_path_ids = path_ids

        cells: List[Cell] = []
        for cid in best_path_ids:
            cell = self.orchestrator.loaded_cells.get(cid)
            if cell:
                cells.append(cell)
        return cells

    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        current = node
        depth = 0
        while current.children and depth < 10:
            if any(c.visits == 0 for c in current.children):
                return next(c for c in current.children if c.visits == 0)
            current = max(current.children, key=lambda c: c.ucb1())
            depth += 1
        return current

    def _rollout(
        self,
        node: MCTSNode,
        goal_sig: AlgebraicSignature,
        max_depth: int,
        prompt_keywords: Optional[Set[str]]
    ) -> Tuple[float, List[str]]:
        current_sig = node.signature
        path: List[str] = []
        visited: Set[str] = set()

        for depth in range(max_depth):
            if current_sig.unifies_with(goal_sig):
                return 1.0, path

            candidates = self._get_compatible_neighbors(current_sig, visited, prompt_keywords)
            if not candidates:
                break

            weights = [score for _, score in candidates]
            total = sum(weights)
            if total == 0:
                break
            r = random.uniform(0, total)
            cumsum = 0.0
            chosen_id = candidates[0][0]
            for cid, w in candidates:
                cumsum += w
                if r <= cumsum:
                    chosen_id = cid
                    break

            cell = self.orchestrator.loaded_cells.get(chosen_id)
            if not cell:
                break

            path.append(chosen_id)
            visited.add(chosen_id)
            current_sig = cell.primary_output

        if current_sig.unifies_with(goal_sig):
            return 1.0, path
        if current_sig.type_name == goal_sig.type_name:
            return 0.5, path
        return 0.0, path

    def _get_compatible_neighbors(
        self,
        sig: AlgebraicSignature,
        visited: Set[str],
        prompt_keywords: Optional[Set[str]]
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        kw_set = prompt_keywords or set()

        for cid, cell in self.orchestrator.loaded_cells.items():
            if cid in visited:
                continue
            if not sig.unifies_with(cell.primary_input):
                continue

            score = 1.0
            if kw_set:
                cell_kws = set(k.lower() for k in cell.keywords)
                overlap = len(kw_set & cell_kws)
                score += overlap * 0.5
            score -= len(cid) * 0.001
            results.append((cid, max(score, 0.1)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:20]

    def _backpropagate(self, node: Optional[MCTSNode], reward: float):
        current = node
        while current is not None:
            current.visits += 1
            current.q_value += reward
            current = current.parent


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

        algo_indicators = ["algorithm", "dijkstra", "bfs", "dfs", "quicksort", "mergesort",
                           "binary search", "a-star", "astar", "topological sort"]
        if any(ind in prompt_lower for ind in algo_indicators):
            logger.info("[ROUTER] Algorithmic task detected; returning empty path for synthesis fallback.")
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
                if sig.unifies_with(goal_sig):
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
        """Robust to any RAG return format: dicts, tuples, or strings."""
        try:
            raw_candidates = self.rag.get_relevant_context(prompt, top_k=top_k * 2)
        except Exception as e:
            logger.error(f"[ROUTER] RAG query failed: {e}")
            return []

        if not raw_candidates:
            return []

        first = raw_candidates[0]
        entries: List[Dict[str, Any]] = []

        if isinstance(first, dict):
            entries = raw_candidates
        elif isinstance(first, (list, tuple)):
            for item in raw_candidates:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    entries.append({
                        "cell_id": str(item[0]),
                        "score": float(item[1]) if len(item) > 1 else 0.5,
                        "text": str(item[2]) if len(item) > 2 else ""
                    })
                else:
                    entries.append({"cell_id": str(item), "score": 0.5, "text": ""})
        elif isinstance(first, str):
            for s in raw_candidates:
                m = re.search(r"ID:\s*([A-Z0-9_]+)", s)
                cid = m.group(1) if m else ""
                sm = re.search(r"score[:\s=]+([0-9.]+)", s, re.IGNORECASE)
                score = float(sm.group(1)) if sm else 0.5
                entries.append({"cell_id": cid, "score": score, "text": s})
        else:
            logger.warning(f"[ROUTER] Unknown RAG format: {type(first)}")
            return []

        scored: List[Tuple[str, float]] = []
        prompt_lower = prompt.lower()

        domain_hint = None
        domain_keywords = {
            "pandas": ["csv", "dataframe", "df", "read_csv", "to_csv", "dropna", "groupby"],
            "cv2": ["image", "opencv", "grayscale", "cvtcolor", "imread", "imwrite", "blur"],
            "numpy": ["array", "ndarray", "reshape", "linspace", "zeros", "ones"],
            "sklearn": ["classifier", "regressor", "fit", "predict", "train_test_split", "scaler"],
            "scipy": ["sparse", "csgraph", "optimize", "integrate", "fft"],
            "matplotlib": ["plot", "figure", "subplot", "savefig"],
        }
        for domain, kws in domain_keywords.items():
            if any(k in prompt_lower for k in kws):
                domain_hint = domain
                break

        for entry in entries:
            cid = entry.get("cell_id", "")
            if not cid:
                continue

            cell = self.orchestrator.loaded_cells.get(cid)
            if not cell:
                continue

            base_score = float(entry.get("score", 0.0))

            type_match = current_sig.unifies_with(cell.primary_input)
            if type_match:
                base_score *= 1.0
            else:
                base_score *= 0.3

            if domain_hint:
                cell_domain = (cell.domain_name or "").lower()
                if domain_hint != cell_domain and cell_domain not in ("generic", "macro", "python_core"):
                    base_score *= 0.4

            cell_kws = set(k.lower() for k in cell.keywords)
            overlap = len(prompt_keywords & cell_kws)
            base_score += overlap * 0.15

            if any(p in cid.lower() for p in ["_group_", "_core_", "_algos_", "_internal_",
                                               "typing_", "withmetadata", "renderer"]):
                base_score *= 0.5

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
