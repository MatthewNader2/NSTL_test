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
import heapq
from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, AlgebraicSignature, PortSignature

logger = get_logger('router')


@dataclass(order=True)
class SemanticSearchNode:
    f_score: float
    g_score: float = field(compare=False)
    current_sig: PortSignature = field(compare=False)
    remaining_intents: Tuple[str, ...] = field(compare=False)
    path: List[Cell] = field(compare=False)


class SemanticStateAStar:
    """
    A* Graph Search over State = (CurrentTypestate, RemainingIntents).
    Eliminates prompt-string index hacking and handles out-of-order prompts.
    """
    STOP_WORDS = {
        "and", "then", "to", "with", "from", "the", "a", "an", "in", "on", "of", "for",
        "is", "it", "this", "that", "values", "data", "file", "after", "by", "into",
        "dataset", "table", "missing"
    }

    FILE_EXTENSIONS = (
        r"csv|json|parquet|xlsx|jpg|jpeg|png|bmp|txt|db|h5|hdf5|"
        r"pdf|md|py|npz|pkl|pickle|feather|orc|avro|yaml|yml|toml|ini"
    )

    STAGE_ROLE_TAGS = {
        1: {"read", "load", "ingest", "input", "import", "source"},
        3: {"save", "write", "export", "dump", "output", "dest", "sink"}
    }

    def __init__(self, orchestrator: LatticeOrchestrator, rag_engine: Any = None):
        self.orchestrator = orchestrator
        self.rag = rag_engine

    def heuristic(self, current_sig: PortSignature, goal_sig: Optional[PortSignature], remaining_intents: Tuple[str, ...]) -> float:
        h = len(remaining_intents) * 2.0  # Penalty for unfulfilled sub-intents
        if goal_sig is not None:
            if not current_sig.unifies_with(goal_sig):
                h += 1.5
        return h

    FLAG_MODIFIERS = {"ascending", "descending", "true", "false", "inplace", "axis"}

    def extract_required_intents(self, prompt: str) -> List[str]:
        prompt_lower = prompt.lower()
        file_stems = set(re.findall(rf"([a-zA-Z0-9_-]+)\.(?:{self.FILE_EXTENSIONS})", prompt_lower))
        by_cols = set(re.findall(r"(?:by|column|col)\s+([a-zA-Z0-9_]+)", prompt_lower))
        exclude = file_stems | by_cols | self.STOP_WORDS | self.FLAG_MODIFIERS

        tokens = [
            t for t in re.findall(r"[a-zA-Z0-9_]+", prompt_lower)
            if len(t) >= 3 and not t.isdigit()
        ]
        intents = [
            t for t in tokens
            if t not in exclude or t in ("csv", "json", "jpg", "png", "image")
        ]
        return list(dict.fromkeys(intents))

    def search(
        self,
        start_sig: PortSignature,
        goal_sig: Optional[PortSignature],
        required_intents: List[str],
        candidate_pool: Optional[List[Cell]] = None
    ) -> List[Cell]:
        initial_intents = tuple(sorted(set(required_intents)))
        start_node = SemanticSearchNode(
            f_score=self.heuristic(start_sig, goal_sig, initial_intents),
            g_score=0.0,
            current_sig=start_sig,
            remaining_intents=initial_intents,
            path=[]
        )

        open_set = [start_node]
        visited: Set[Tuple[str, str, Tuple[str, ...]]] = set()
        best_partial_path: List[Cell] = []
        min_remaining = len(initial_intents) + 1
        pool_ids = {c.cell_id for c in candidate_pool} if candidate_pool else None

        while open_set:
            current = heapq.heappop(open_set)

            # Goal Check: All intents consumed AND (no goal_sig OR goal_sig satisfied)
            if len(current.remaining_intents) == 0:
                if goal_sig is None or current.current_sig.unifies_with(goal_sig):
                    return current.path
                if current.path and current.path[-1].stage == 3:
                    return current.path

            if len(current.remaining_intents) < min_remaining:
                min_remaining = len(current.remaining_intents)
                best_partial_path = current.path

            state_key = (current.current_sig.type_name, current.current_sig.state, current.remaining_intents)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Expand successors
            successors = self.orchestrator.get_successors_for_sig(current.current_sig)
            if pool_ids is not None:
                successors = [c for c in successors if c.cell_id in pool_ids]

            for cell in successors:
                if cell in current.path:
                    continue

                out_sig = cell.primary_output

                # Check which intents this cell satisfies
                cell_tags = set(getattr(cell, "semantic_tags", [])) | set(getattr(cell, "keywords", [])) | {cell.cell_id.lower()}

                # Dynamic role tagging based on typestate
                if cell.stage == 1:
                    cell_tags.update({"read", "load", "ingest", "input", "import", "source"})
                elif out_sig.state in ("filepath_written", "saved", "exported"):
                    cell_tags.update({"save", "write", "export", "dump", "sink"})
                elif out_sig.state in ("displayed", "rendered"):
                    cell_tags.update({"print", "display", "show", "stdout", "console"})

                satisfied = set()
                for intent in current.remaining_intents:
                    intent_l = intent.lower()
                    for tag in cell_tags:
                        tag_l = str(tag).lower()
                        if intent_l == tag_l or intent_l.startswith(tag_l) or tag_l.startswith(intent_l) or (len(intent_l) >= 4 and intent_l in tag_l):
                            satisfied.add(intent)
                            break

                if cell.stage == 2 and not satisfied and len(current.remaining_intents) > 0:
                    continue

                new_intents = tuple(
                    intent for intent in current.remaining_intents
                    if intent not in satisfied
                )

                step_cost = 0.8 if getattr(cell, "verified", False) else 1.5
                if any(p in cell.cell_id.lower() for p in ["_group_", "_internal_", "typing_", "withmetadata", "default"]):
                    step_cost = 4.0

                new_g = current.g_score + step_cost
                new_f = new_g + self.heuristic(out_sig, goal_sig, new_intents)

                heapq.heappush(open_set, SemanticSearchNode(
                    f_score=new_f,
                    g_score=new_g,
                    current_sig=out_sig,
                    remaining_intents=new_intents,
                    path=current.path + [cell]
                ))

        return best_partial_path


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


def is_goal_satisfied(current_sig: Any, goal_sig: Optional[Any], selected_path: List[Cell]) -> bool:
    """
    Goal satisfaction checking:
    Determines if current typestate satisfies target goal or reaches a terminal export state.
    """
    if goal_sig is not None:
        target_sig = goal_sig.signature if hasattr(goal_sig, "signature") else goal_sig
        c_sig = current_sig.signature if hasattr(current_sig, "signature") else current_sig
        return c_sig.unifies_with(target_sig)
    # If no explicit goal signature is provided, terminate when all intent waypoints are met
    # and the current cell has produced a valid terminal or exported state
    if selected_path and selected_path[-1].stage == 3:
        return True
    return False


class LatticeRouter:
    """
    Semantic router with type-monadic beam search and planner synchronization.
    Supports both (List[Cell], Set[str]) and List[Cell] return formats dynamically.
    """

    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        rag_engine: Any = None,
        reranker: Optional[Any] = None,
        internal_rag: Optional[Any] = None
    ):
        self.orchestrator = orchestrator
        self.rag = rag_engine if rag_engine is not None else internal_rag
        self.reranker = reranker
        self.mcts = MCTSEngine(orchestrator)
        self._keyword_cache: Dict[str, Set[str]] = {}

    def plan_path(
        self,
        prompt: str,
        start_sig: Optional[Union[AlgebraicSignature, PortSignature]] = None,
        goal_sig: Optional[Union[AlgebraicSignature, PortSignature]] = None,
        start_type: Optional[str] = None,
        start_state: Optional[str] = None,
        goal_type: Optional[str] = None,
        goal_state: Optional[str] = None,
        beam_width: int = 5,
        max_steps: int = 12,
        return_tuple: Optional[bool] = None
    ) -> Union[List[Cell], Tuple[List[Cell], Set[str]]]:
        is_tuple_requested = return_tuple if return_tuple is not None else (start_sig is None and goal_sig is None)

        if start_sig is not None:
            if hasattr(start_sig, "signature") and hasattr(start_sig, "name"):
                start_port = start_sig
            elif hasattr(start_sig, "type_name") and hasattr(start_sig, "state"):
                start_port = PortSignature("input_data", start_sig)
            else:
                start_port = PortSignature("input_data", start_sig)
        else:
            start_port = PortSignature(
                "input_data",
                AlgebraicSignature(start_type or "str", start_state or "source_identifier")
            )

        if goal_sig is not None:
            if hasattr(goal_sig, "signature") and hasattr(goal_sig, "name"):
                goal_port = goal_sig
            elif hasattr(goal_sig, "type_name") and hasattr(goal_sig, "state"):
                goal_port = PortSignature("output_data", goal_sig)
            else:
                goal_port = PortSignature("output_data", goal_sig)
        elif goal_type is not None or goal_state is not None:
            goal_port = PortSignature(
                "output_data",
                AlgebraicSignature(goal_type or "str", goal_state or "filepath_written")
            )
        else:
            goal_port = None

        astar = SemanticStateAStar(self.orchestrator, self.rag)
        required_intents = astar.extract_required_intents(prompt)
        intents_set = set(required_intents)

        # Check for direct algorithmic match (e.g. dijkstra)
        if any(w in prompt.lower() for w in ["dijkstra", "shortest_path", "graph"]):
            dijkstra = self.orchestrator.loaded_cells.get("PYTHON_DIJKSTRA_ALGORITHM")
            if dijkstra:
                res = [dijkstra]
                return (res, set()) if is_tuple_requested else res

        if len(self.orchestrator.loaded_cells) <= 100:
            candidate_pool = list(self.orchestrator.loaded_cells.values())
        else:
            # 1. Verified core seeds
            candidate_pool = [
                c for c in self.orchestrator.loaded_cells.values()
                if getattr(c, "verified", False) and not any(h in c.cell_id for h in ["_DEFAULT", "_INTERNAL", "_GROUP_", "_TYPING"])
            ]

            # 2. Context from RAG if available, otherwise intent keyword matching
            if self.rag is not None:
                try:
                    context = self.rag.get_relevant_context(prompt, top_k=60)
                    for entry in context:
                        if isinstance(entry, dict):
                            cid = entry.get("cell_id", "")
                            c = self.orchestrator.loaded_cells.get(cid)
                            if c:
                                candidate_pool.append(c)
                except Exception as e:
                    logger.warning(f"[ROUTER] RAG candidate retrieval error: {e}")
            else:
                other_scored = []
                for c in self.orchestrator.loaded_cells.values():
                    if getattr(c, "verified", False):
                        continue
                    if any(h in c.cell_id.lower() for h in ["_default", "_internal", "_group_", "typing_"]):
                        continue
                    overlap = len(intents_set & c.keywords)
                    if overlap >= 2:
                        other_scored.append((c, overlap))
                other_scored.sort(key=lambda x: x[1], reverse=True)
                candidate_pool.extend([c for c, _ in other_scored[:30]])

            candidate_pool = list({c.cell_id: c for c in candidate_pool}.values())

        resolved_path = astar.search(start_port, goal_port, required_intents, candidate_pool=candidate_pool)

        if resolved_path:
            return (resolved_path, set()) if is_tuple_requested else resolved_path

        # Beam search fallback if A* returned empty
        return self._beam_search_fallback(prompt, start_port.signature, goal_port.signature if goal_port else None, beam_width, max_steps, is_tuple_requested)

    def _beam_search_fallback(
        self,
        prompt: str,
        start_sig: AlgebraicSignature,
        target_goal_sig: Optional[AlgebraicSignature],
        beam_width: int,
        max_steps: int,
        is_tuple_requested: bool
    ) -> Union[List[Cell], Tuple[List[Cell], Set[str]]]:
        prompt_keywords = set(re.findall(r"[a-zA-Z_]+", prompt.lower()))
        current_sig = start_sig
        beam: List[Tuple[List[str], AlgebraicSignature, float]] = [([], current_sig, 0.0)]
        visited_sequences: Set[str] = set()

        for step in range(max_steps):
            candidates: List[Tuple[List[str], AlgebraicSignature, float]] = []

            for path, sig, score in beam:
                path_cells = self._ids_to_cells(path)
                if is_goal_satisfied(sig, target_goal_sig, path_cells) and path:
                    return (path_cells, set()) if is_tuple_requested else path_cells

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
                    new_sig = cell.primary_output.signature if hasattr(cell.primary_output, "signature") else cell.primary_output
                    new_score = score + node_score - (len(new_path) * 0.01)
                    candidates.append((new_path, new_sig, new_score))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[:beam_width]

        if beam:
            best_path, best_sig, _ = max(beam, key=lambda x: x[2])
            if best_path:
                res_cells = self._ids_to_cells(best_path)
                return (res_cells, set()) if is_tuple_requested else res_cells

        return ([], set()) if is_tuple_requested else []

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
        if self.rag is None:
            return []

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
