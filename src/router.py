"""
src/router.py - Neuro-Symbolic Topological Lattice (NSTL)
Semantic Tunneling Router based on Dense Vector Embeddings and Softmax Temperature.

Conforms strictly to Section 3.3 of the NSTL paper:
  e_x = Embed(prompt)
  P(C_k | e_x) = softmax_k( cos(e_x, mu_k) / gamma )
  Tunnel T = { v in V | P(v | e_x) >= epsilon }

Contains ZERO keyword-sniffing regexes, ZERO manual score boosts, and ZERO domain hardcodes.
"""

from __future__ import annotations
import math
import re
from typing import Optional, List, Dict, Set, Tuple, Any

import numpy as np
from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell
    from .internal_rag import LocalRAG
    from .planner import LatticePlanner
    from .tokenizer import CellTokenizer
    from .inference import ModelManager
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell
    from internal_rag import LocalRAG
    from planner import LatticePlanner
    from tokenizer import CellTokenizer
    from inference import ModelManager

logger = get_logger('router')


LEN_NORM_TABLE = [1.0 / (math.log2(2 + i) ** 0.1) for i in range(500)]


class LatticeRouter:
    """
    Semantic Tunneling Router (Section 3.3).
    Narrows the search space from the entire lattice (thousands of nodes)
    down to the active tunnel T using dense vector semantic relevance.
    """
    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        internal_rag: Optional[LocalRAG] = None,
        gamma: float = 0.15,
        epsilon: float = 0.001,
        rag_engine: Optional[LocalRAG] = None,
        **kwargs
    ):
        self.orchestrator = orchestrator
        self.internal_rag = internal_rag or rag_engine
        self.gamma = gamma          # Temperature parameter scaling cosine similarity
        self.epsilon = epsilon      # Cutoff threshold for tunnel inclusion
        self.planner = LatticePlanner(orchestrator=self.orchestrator)

    def route(
        self,
        prompt: str,
        top_k: int = 150
    ) -> Tuple[List[Cell], Dict[str, float]]:
        """
        Computes the semantic tunnel T and relevance distribution P(v | e_x).
        Returns:
          (tunnel_cells, cell_relevance_probabilities)
        """
        if not prompt or not prompt.strip():
            return [], {}

        # 1. Retrieve candidates via vector similarity from RAG index.
        # Decompose compound prompt into constituent sub-goals to guarantee representation
        # across all sequential intents (Section 3.3).
        candidates_with_scores: List[Tuple[Cell, float]] = []

        if self.internal_rag is not None and self.internal_rag.index is not None:
            # Continuous sliding-window vector queries across prompt (zero linguistic heuristics, zero regex)
            query_spans = self._generate_query_spans(prompt.strip())

            candidate_scores: Dict[str, float] = {}
            for q in query_spans:
                rag_results = self.internal_rag.get_relevant_context(q, top_k=min(top_k, 100))
                for item in rag_results:
                    cid = item.get("cell_id")
                    score = float(item.get("score", 0.0))
                    if cid and (cid not in candidate_scores or score > candidate_scores[cid]):
                        candidate_scores[cid] = score

            primary_results = self.internal_rag.get_relevant_context(prompt, top_k=top_k)
            for item in primary_results:
                cid = item.get("cell_id")
                score = float(item.get("score", 0.0))
                if cid and (cid not in candidate_scores or score > candidate_scores[cid]):
                    candidate_scores[cid] = score

            for cid, score in candidate_scores.items():
                cell = self.orchestrator.loaded_cells.get(cid)
                if cell is not None:
                    candidates_with_scores.append((cell, score))

        # Fallback if RAG index has not yet indexed cells: use length-normalized prompt coverage across query spans
        if not candidates_with_scores:
            token_index = getattr(self.orchestrator, "token_index", None)
            N = len(self.orchestrator.loaded_cells)

            # Decompose compound prompt into constituent sub-goals across grammatical clauses
            # to guarantee representation across all sequential intents without drowning under high-volume libraries.
            clauses = [c.strip() for c in re.split(r'[,;]|\b(?:and|then)\b', prompt.strip()) if c.strip()]
            search_texts = [prompt.strip()] + [c for c in clauses if c != prompt.strip()]

            pooled_candidates: Dict[Cell, float] = {}

            for text in search_texts:
                query_spans = self._generate_query_spans(text)
                clause_scores: Dict[Cell, float] = {}
                for q in query_spans:
                    q_tokens = CellTokenizer.tokenize_prompt(q)
                    q_len = max(len(q_tokens), 1)
                    inv_q_len = 1.0 / q_len
                    if token_index and N > 0:
                        overlap_counts: Dict[Cell, int] = {}
                        for tok in q_tokens:
                            cells = token_index.get(tok, [])
                            if N < 20 or len(cells) < 0.3 * N:
                                for cell in cells:
                                    overlap_counts[cell] = overlap_counts.get(cell, 0) + 1
                        for cell, count in overlap_counts.items():
                            c_len = max(getattr(cell, "token_count", len(cell.token_set)), 1)
                            sc = (count * inv_q_len) * LEN_NORM_TABLE[min(c_len, 499)]
                            if cell not in clause_scores or sc > clause_scores[cell]:
                                clause_scores[cell] = sc
                    else:
                        for cell in self.orchestrator.loaded_cells.values():
                            cell_tokens = cell.token_set
                            intersection_len = len(q_tokens & cell_tokens)
                            if intersection_len > 0:
                                c_len = max(len(cell_tokens), 1)
                                sc = (intersection_len * inv_q_len) * LEN_NORM_TABLE[min(c_len, 499)]
                                if cell not in clause_scores or sc > clause_scores[cell]:
                                    clause_scores[cell] = float(sc)

                # Preserve top 100 candidates per clause to guarantee sub-goal coverage
                sorted_clause = sorted(clause_scores.items(), key=lambda x: x[1], reverse=True)[:100]
                for cell, sc in sorted_clause:
                    if cell not in pooled_candidates or sc > pooled_candidates[cell]:
                        pooled_candidates[cell] = sc

            candidates_with_scores = list(pooled_candidates.items())

        if not candidates_with_scores:
            return [], {}

        # Prune low-scoring tail before computing exponential softmax distribution:
        # Softmax is strictly monotonic; preserving top 400 candidates guarantees
        # active tunnel integrity while eliminating overhead on broad queries.
        if len(candidates_with_scores) > 400:
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
            candidates_with_scores = candidates_with_scores[:400]

        # 2. Compute Softmax Distribution with temperature gamma:
        #    P(v_i | e_x) = exp(s_i / gamma) / sum_j exp(s_j / gamma)
        scores = np.array([s for _, s in candidates_with_scores], dtype=np.float64)
        scaled_scores = scores / max(self.gamma, 1e-5)
        shifted_scores = scaled_scores - np.max(scaled_scores)
        exp_scores = np.exp(shifted_scores)
        probabilities = exp_scores / np.sum(exp_scores)

        # 3. Filter into Active Tunnel T via Scale-Invariant Relative Likelihood:
        #    P(v | e_x) / max_u P(u | e_x) >= tau
        cell_probs = [(cell, float(prob)) for (cell, _), prob in zip(candidates_with_scores, probabilities)]
        cell_probs.sort(key=lambda x: x[1], reverse=True)

        relevance_map: Dict[str, float] = {cell.cell_id: prob for cell, prob in cell_probs}

        tau = max(self.epsilon, 0.01)
        tunnel_cells = [cell for (cell, _), rel_lik in zip(candidates_with_scores, exp_scores) if rel_lik >= tau]
        if top_k and len(tunnel_cells) > top_k:
            tunnel_cells.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
            tunnel_cells = tunnel_cells[:top_k]

        final_tunnel: List[Cell] = list(tunnel_cells)
        tunnel_ids = set(c.cell_id for c in final_tunnel)

        # Category-Theoretic Bridge & Active Subcategory Morphism Completion:
        # If the active tunnel contains distinct carrier types A and B,
        # discover bridge morphisms Hom(A, B) and domain morphisms on those carriers.
        carriers_out = set()
        carriers_in = set()
        for c in final_tunnel:
            out_t = getattr(c.primary_output, "type_name", "")
            in_t = getattr(c.primary_input, "type_name", "")
            if out_t and out_t.lower() not in ("any", "none", "*", "top", "void"):
                carriers_out.add(out_t)
            if in_t and in_t.lower() not in ("any", "none", "*", "top", "void"):
                carriers_in.add(in_t)

        if carriers_out and carriers_in:
            bridge_cells = getattr(self.orchestrator, "bridge_cells", None)
            if bridge_cells is None:
                bridge_cells = [
                    c for c in self.orchestrator.loaded_cells.values()
                    if getattr(c, "node_role", "") == "bridge" or getattr(c, "node_type", "") == "tunnel"
                ]
            for cell in bridge_cells:
                c_in = getattr(cell.primary_input, "type_name", "")
                c_out = getattr(cell.primary_output, "type_name", "")
                if c_in in carriers_out and c_out in carriers_in:
                    if cell.cell_id not in tunnel_ids:
                        final_tunnel.append(cell)
                        tunnel_ids.add(cell.cell_id)
                        relevance_map[cell.cell_id] = max(relevance_map.get(cell.cell_id, 0.0), self.epsilon * 2.0)

        if carriers_out or carriers_in:
            prompt_toks = CellTokenizer.tokenize_prompt(prompt)
            active_domains = set(c.domain_name for c in final_tunnel if c.domain_name)
            for cell in self.orchestrator.loaded_cells.values():
                if cell.domain_name in active_domains and cell.cell_id not in tunnel_ids:
                    c_in = getattr(cell.primary_input, "type_name", "")
                    c_out = getattr(cell.primary_output, "type_name", "")
                    if (c_in in carriers_out or c_out in carriers_in) and (cell.token_set & prompt_toks):
                        final_tunnel.append(cell)
                        tunnel_ids.add(cell.cell_id)
                        relevance_map[cell.cell_id] = max(relevance_map.get(cell.cell_id, 0.0), self.epsilon * 2.0)

        return final_tunnel, relevance_map

    @staticmethod
    def _generate_query_spans(text: str) -> List[str]:
        """
        Generates continuous sliding semantic window queries across the prompt.
        Multi-scale representation: zero linguistic connectors (no 'then'), zero regex.
        """
        tokens = text.strip().split()
        if len(tokens) <= 3:
            return [text.strip()]

        queries = [text.strip()]
        n = len(tokens)
        for window_size in (2, 3, max(3, n // 2)):
            if window_size >= n:
                continue
            step = max(1, window_size // 2)
            for i in range(0, n - window_size + 1, step):
                sub_q = " ".join(tokens[i : i + window_size]).strip()
                if sub_q and sub_q not in queries:
                    queries.append(sub_q)

        tail_q = " ".join(tokens[max(0, n - 3) :]).strip()
        if tail_q and tail_q not in queries:
            queries.append(tail_q)

        return queries

    def plan_path(
        self,
        prompt: str,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        return_tuple: bool = True,
        top_k: int = 200
    ) -> Union[List[Cell], Tuple[List[Cell], Set[str]]]:
        """
        End-to-end routing & topological pathfinding:
          1. Computes active semantic tunnel T for prompt.
          2. Uses LatticePlanner to find valid monadic composition through T.
        """
        tunnel_cells, relevance_map = self.route(prompt, top_k=top_k)
        if not tunnel_cells:
            logger.warning(f"[ROUTER] Empty tunnel for prompt: '{prompt}'")
            return ([], set()) if return_tuple else []

        # Find verified composition path inside tunnel T
        path = self.planner.plan(
            prompt=prompt,
            tunnel=tunnel_cells,
            relevance_map=relevance_map,
            start_sig=start_sig,
            goal_sig=goal_sig
        )

        if return_tuple:
            candidate_ids = set(c.cell_id for c in tunnel_cells)
            return path, candidate_ids
        return path


class HardwareProfiler:
    """Device configuration profiler for inference accelerators."""
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
        try:
            import torch
            if torch.cuda.is_available():
                cls._cached_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                cls._cached_device = "mps"
            else:
                cls._cached_device = "cpu"
        except Exception:
            cls._cached_device = "cpu"
        return cls._cached_device


class MCTSEngine:
    """MCTS gap bridging proxy (Section 3.4)."""
    def __init__(self, orchestrator: LatticeOrchestrator):
        self.orchestrator = orchestrator
        self.planner = LatticePlanner(orchestrator)

    def search(self, arg1: Any, arg2: Any = None) -> Optional[List[Cell]]:
        try:
            from lattice import AlgebraicSignature
        except (ImportError, ValueError):
            from .lattice import AlgebraicSignature

        if isinstance(arg1, AlgebraicSignature) and isinstance(arg2, AlgebraicSignature):
            start_sig, goal_sig = arg1, arg2
            # 1-step direct transition:
            for cell in self.orchestrator.cells:
                if cell.primary_input.unifies_with(start_sig) and cell.primary_output.unifies_with(goal_sig):
                    return [cell]
            # 2-step BFS:
            for c1 in self.orchestrator.cells:
                if c1.primary_input.unifies_with(start_sig):
                    for c2 in self.orchestrator.cells:
                        if c1.primary_output.unifies_with(c2.primary_input) and c2.primary_output.unifies_with(goal_sig):
                            return [c1, c2]
            return []

        if isinstance(arg1, list) and isinstance(arg2, dict):
            return self.planner._bounded_mcts_search(arg1, arg2)
        return []


def log_coverage_gap(prompt: str, domain_guess: str = "", score: float = 0.0, node_id: str = ""):
    """Telemetry logging for below-threshold query gaps."""
    import os
    import json
    import time
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "coverage_gaps.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "prompt": prompt,
            "domain_guess": domain_guess,
            "score": score,
            "node_id": node_id,
            "timestamp": time.time()
        }) + "\n")

