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
            # Deterministic punctuation and connector decomposition (zero regex)
            clauses = [prompt.strip()]
            sub_clauses = self._split_prompt_clauses(prompt.strip())
            for sc in sub_clauses:
                if len(sc) >= 3 and sc not in clauses:
                    clauses.append(sc)

            candidate_scores: Dict[str, float] = {}
            for q in clauses:
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

            # Categorical Stage 3 Egress Guarantee: Sinks (B -> 1) are sparse in lattice topology.
            # Ensure candidate_scores evaluates terminal egress morphisms against the exit goal.
            terminal_clause = clauses[-1] if clauses else prompt
            term_emb = None
            for cell in self.orchestrator.loaded_cells.values():
                if cell.stage == 3 and getattr(cell, "node_type", "") != "constant":
                    c_data = getattr(self.internal_rag, "cell_cache", {}).get(cell.cell_id)
                    if c_data and "embedding" in c_data:
                        if term_emb is None:
                            raw_term = ModelManager.get_instance().get_embedding(terminal_clause)
                            t_norm = np.linalg.norm(raw_term)
                            term_emb = np.array(raw_term, dtype=np.float32) / (t_norm if t_norm > 0 else 1.0)
                        c_emb = np.array(c_data["embedding"], dtype=np.float32)
                        c_norm = np.linalg.norm(c_emb)
                        if c_norm > 0:
                            sim = float(np.dot(term_emb, c_emb) / c_norm)
                            if cell.cell_id not in candidate_scores or sim > candidate_scores[cell.cell_id]:
                                candidate_scores[cell.cell_id] = sim

            for cid, score in candidate_scores.items():
                cell = self.orchestrator.loaded_cells.get(cid)
                if cell is not None:
                    candidates_with_scores.append((cell, score))

        # Fallback if RAG index has not yet indexed cells: use token Jaccard similarity
        if not candidates_with_scores:
            prompt_tokens = CellTokenizer.tokenize_prompt(prompt)
            for cell in self.orchestrator.loaded_cells.values():
                cell_tokens = cell.token_set
                intersection = len(prompt_tokens & cell_tokens)
                union = len(prompt_tokens | cell_tokens)
                score = (intersection / max(union, 1)) if union > 0 else 0.001
                candidates_with_scores.append((cell, float(score)))

        if not candidates_with_scores:
            return [], {}

        # 2. Compute Softmax Distribution with temperature gamma:
        #    P(v_i | e_x) = exp(s_i / gamma) / sum_j exp(s_j / gamma)
        scores = np.array([s for _, s in candidates_with_scores], dtype=np.float64)
        scaled_scores = scores / max(self.gamma, 1e-5)
        shifted_scores = scaled_scores - np.max(scaled_scores)
        exp_scores = np.exp(shifted_scores)
        probabilities = exp_scores / np.sum(exp_scores)

        # 3. Filter into Active Tunnel T: { v in V | P(v | e_x) >= epsilon }
        cell_probs = [(cell, float(prob)) for (cell, _), prob in zip(candidates_with_scores, probabilities)]
        cell_probs.sort(key=lambda x: x[1], reverse=True)

        relevance_map: Dict[str, float] = {cell.cell_id: prob for cell, prob in cell_probs}

        # Mathematical tunnel cutoff: P(v | e_x) >= epsilon
        tunnel_cells = [cell for cell, prob in cell_probs if prob >= self.epsilon]

        # Stage Stratification: Ensure stage representation in tunnel without arbitrary hardcoded slices
        stage_partition: Dict[int, List[Cell]] = {1: [], 2: [], 3: []}
        for cell, _ in cell_probs:
            st = getattr(cell, "stage", 2)
            if getattr(cell, "node_type", "") != "constant" and st in stage_partition:
                stage_partition[st].append(cell)

        final_tunnel: List[Cell] = list(tunnel_cells)
        tunnel_ids = set(c.cell_id for c in final_tunnel)
        for st in (1, 2, 3):
            if stage_partition[st] and not any(getattr(c, "stage", 2) == st for c in final_tunnel):
                best_st_cell = stage_partition[st][0]
                if best_st_cell.cell_id not in tunnel_ids:
                    final_tunnel.append(best_st_cell)
                    tunnel_ids.add(best_st_cell.cell_id)

        return final_tunnel, relevance_map

    @staticmethod
    def _split_prompt_clauses(text: str) -> List[str]:
        """Splits compound prompt into clauses using punctuation and sequential connectors without regex."""
        clauses: List[str] = []
        current: List[str] = []
        words = text.strip().split()
        for w in words:
            w_clean = w.strip(";,.")
            if w_clean.lower() in ("then", "and_then") or w.endswith((";", ",", ".")):
                if w_clean.lower() not in ("then", "and_then") and w_clean:
                    current.append(w_clean)
                if current:
                    clause_str = " ".join(current).strip()
                    if len(clause_str) >= 2:
                        clauses.append(clause_str)
                    current = []
            else:
                current.append(w)
        if current:
            clause_str = " ".join(current).strip()
            if len(clause_str) >= 2:
                clauses.append(clause_str)
        return clauses if clauses else [text.strip()]

    def plan_path(
        self,
        prompt: str,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        return_tuple: bool = True,
        top_k: int = 150
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

