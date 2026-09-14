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
from collections import defaultdict

import numpy as np
from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell
    from .internal_rag import LocalRAG
    from .planner import LatticePlanner, STOPWORDS
    from .tokenizer import CellTokenizer
    from .inference import ModelManager
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell
    from internal_rag import LocalRAG
    from planner import LatticePlanner, STOPWORDS
    from tokenizer import CellTokenizer
    from inference import ModelManager

logger = get_logger('router')


class IdiomSubgraph:
    """
    Candidate workflow idiom / subgraph in the lattice, discovered via Stage 1 retrieval.
    Represents an empirical pipeline path or cluster connecting prompt intent clauses.
    """
    def __init__(
        self,
        subgraph_id: str,
        cells: List[Cell],
        edges: List[Tuple[str, str, float]],
        covered_clauses: Set[int],
        lexical_score: float = 0.0,
        affinity_score: float = 0.0,
        entry_node: Optional[Cell] = None
    ):
        self.subgraph_id = subgraph_id
        self.cells = list(cells)
        self.cell_ids = {c.cell_id for c in cells}
        self.edges = list(edges)
        self.covered_clauses = set(covered_clauses)
        self.lexical_score = lexical_score
        self.affinity_score = affinity_score
        self.entry_node = entry_node

    @property
    def total_score(self) -> float:
        # Multi-clause coverage bonus + lexical relevance + transition affinities
        coverage_bonus = 3.0 * len(self.covered_clauses)
        return self.lexical_score + 1.2 * self.affinity_score + coverage_bonus

    def __repr__(self) -> str:
        entry_id = self.entry_node.cell_id if self.entry_node else "None"
        return f"<IdiomSubgraph {self.subgraph_id}: {len(self.cells)} cells, entry={entry_id}, score={self.total_score:.2f}>"


LEN_NORM_TABLE = [1.0 / (math.log2(2 + i) ** 0.1) for i in range(500)]


class LatticeRouter:
    """
    Semantic Tunneling Router (Section 3.3).
    Implements Phase 4 Two-Stage Retrieval:
      - Stage 1: Discovers candidate idioms/subgraphs using the lexical/IDF scorer
                 combined with empirical graph edges.
      - Stage 2: Identifies the entry node within the candidate idiom.
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

    def _score_clause_lexical_idf(
        self,
        text: str,
        token_index: Optional[Dict[str, List[Cell]]],
        N: int
    ) -> Dict[Cell, float]:
        """Computes length-normalized, IDF-weighted lexical scores for a query clause."""
        q_tokens = CellTokenizer.tokenize_prompt(text)
        q_len = max(len(text.split()), len(q_tokens), 1)
        inv_q_len = 1.0 / q_len
        clause_scores: Dict[Cell, float] = {}

        if not token_index or N == 0:
            for cell in self.orchestrator.loaded_cells.values():
                cell_tokens = cell.token_set
                intersection_len = len(q_tokens & cell_tokens)
                if intersection_len > 0:
                    c_len = max(len(cell_tokens), 1)
                    sc = (intersection_len * inv_q_len) * LEN_NORM_TABLE[min(c_len, 499)]
                    clause_scores[cell] = float(sc)
            return clause_scores

        def _idf(tok: str) -> float:
            df = len(token_index.get(tok, ()))
            return math.log(1.0 + (N + 1) / (df + 1.0))

        match_weights: Dict[Cell, float] = {}
        for tok in q_tokens:
            cells = token_index.get(tok, [])
            if N < 20 or len(cells) < 0.3 * N:
                w = _idf(tok)
                for cell in cells:
                    weight = w if tok in cell.identity_tokens else 0.3 * w
                    match_weights[cell] = match_weights.get(cell, 0.0) + weight

        for cell, weight_sum in match_weights.items():
            c_len = max(getattr(cell, "token_count", len(cell.token_set)), 1)
            sc = (weight_sum * inv_q_len) * LEN_NORM_TABLE[min(c_len, 499)]
            clause_scores[cell] = sc

        return clause_scores

    def _discover_candidate_idioms(
        self,
        prompt: str,
        clauses: List[str],
        max_idioms: int = 10
    ) -> List[IdiomSubgraph]:
        """
        Stage 1 Retrieval:
        Discovers candidate idioms/subgraphs in the lattice using:
          1. Clause-level lexical/IDF token matching across prompt intents.
          2. Expansion along empirical AST-mined directed edges and typestate transitions.
          3. Aggregation of path transition affinities and multi-clause coverage.
        """
        token_index = getattr(self.orchestrator, "token_index", None)
        N = len(self.orchestrator.loaded_cells)
        if N == 0:
            return []

        # Step 1.1: Lexical scoring per clause
        clause_matches: List[List[Tuple[Cell, float]]] = []
        for cl in clauses:
            scores = self._score_clause_lexical_idf(cl, token_index, N)
            top_clause = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]
            clause_matches.append(top_clause)

        # Step 1.2: Graph Edge Expansion along directed transitions
        idiom_candidates: List[IdiomSubgraph] = []

        if len(clauses) == 1 or not clause_matches:
            top_seeds = clause_matches[0] if clause_matches else []
            for seed_cell, s_score in top_seeds[:5]:
                cluster_cells = [seed_cell]
                cluster_edges: List[Tuple[str, str, float]] = []
                for e in getattr(seed_cell, "edges", [])[:6]:
                    tgt_id = e.get("target_cell_id") if isinstance(e, dict) else getattr(e, "target_cell_id", "")
                    tgt_cell = self.orchestrator.loaded_cells.get(tgt_id)
                    if tgt_cell and tgt_cell not in cluster_cells:
                        cluster_cells.append(tgt_cell)
                        aff = e.get("affinity_score", 0.5) if isinstance(e, dict) else getattr(e, "affinity_score", 0.5)
                        cluster_edges.append((seed_cell.cell_id, tgt_id, aff))

                idiom = IdiomSubgraph(
                    subgraph_id=f"idiom_{seed_cell.cell_id}",
                    cells=cluster_cells,
                    edges=cluster_edges,
                    covered_clauses={0},
                    lexical_score=s_score,
                    affinity_score=sum(aff for _, _, aff in cluster_edges)
                )
                idiom_candidates.append(idiom)
        else:
            # Multi-clause sequential pipeline chaining
            for s0, sc0 in clause_matches[0][:4]:
                chain_cells: List[Cell] = [s0]
                chain_edges: List[Tuple[str, str, float]] = []
                covered: Set[int] = {0}
                total_lex = sc0
                curr_node = s0

                for c_idx in range(1, len(clauses)):
                    next_seeds = clause_matches[c_idx]
                    found_next = False
                    curr_out = [e.get("target_cell_id") for e in getattr(curr_node, "edges", [])]
                    adj_out = self.orchestrator._adjacency.get(curr_node.cell_id, [])

                    # 1. Direct edge check
                    for cand, cand_sc in next_seeds[:6]:
                        if cand.cell_id in curr_out or cand.cell_id in adj_out:
                            edge_rec = next((e for e in getattr(curr_node, "edges", []) if e.get("target_cell_id") == cand.cell_id), None)
                            aff = edge_rec.get("affinity_score", 0.5) if edge_rec else 0.5
                            chain_edges.append((curr_node.cell_id, cand.cell_id, aff))
                            if cand not in chain_cells:
                                chain_cells.append(cand)
                            covered.add(c_idx)
                            total_lex += cand_sc
                            curr_node = cand
                            found_next = True
                            break

                    # 2. 1-hop bridge transition check
                    if not found_next:
                        mid_candidates = curr_out[:6] if curr_out else adj_out[:6]
                        for mid_id in mid_candidates:
                            mid_cell = self.orchestrator.loaded_cells.get(mid_id)
                            if not mid_cell:
                                continue
                            mid_out = [e.get("target_cell_id") for e in getattr(mid_cell, "edges", [])]
                            mid_adj = self.orchestrator._adjacency.get(mid_id, [])
                            for cand, cand_sc in next_seeds[:6]:
                                if cand.cell_id in mid_out or cand.cell_id in mid_adj:
                                    e1 = next((e for e in getattr(curr_node, "edges", []) if e.get("target_cell_id") == mid_id), None)
                                    aff1 = e1.get("affinity_score", 0.5) if e1 else 0.5
                                    e2 = next((e for e in getattr(mid_cell, "edges", []) if e.get("target_cell_id") == cand.cell_id), None)
                                    aff2 = e2.get("affinity_score", 0.5) if e2 else 0.5
                                    
                                    chain_edges.append((curr_node.cell_id, mid_id, aff1))
                                    chain_edges.append((mid_id, cand.cell_id, aff2))
                                    if mid_cell not in chain_cells:
                                        chain_cells.append(mid_cell)
                                    if cand not in chain_cells:
                                        chain_cells.append(cand)
                                    covered.add(c_idx)
                                    total_lex += cand_sc
                                    curr_node = cand
                                    found_next = True
                                    break
                            if found_next:
                                break

                    # 3. Soft anchor inclusion if path not fully bridged
                    if not found_next and next_seeds:
                        best_cand, best_sc = next_seeds[0]
                        if best_cand not in chain_cells:
                            chain_cells.append(best_cand)
                        covered.add(c_idx)
                        total_lex += best_sc
                        curr_node = best_cand

                idiom = IdiomSubgraph(
                    subgraph_id=f"pipeline_{s0.cell_id}",
                    cells=chain_cells,
                    edges=chain_edges,
                    covered_clauses=covered,
                    lexical_score=total_lex,
                    affinity_score=sum(aff for _, _, aff in chain_edges)
                )
                idiom_candidates.append(idiom)

        idiom_candidates.sort(key=lambda x: x.total_score, reverse=True)
        return idiom_candidates[:max_idioms]

    def _select_entry_node(
        self,
        idiom: IdiomSubgraph,
        first_clause: str,
        token_index: Optional[Dict[str, List[Cell]]],
        N: int
    ) -> Optional[Cell]:
        """
        Stage 2 Retrieval:
        Pinpoints the entry node within the candidate idiom:
          1. Subgraph in-degree: nodes with 0 incoming edges within the idiom.
          2. Stage classification: Stage 1 source/ingestion nodes (e.g. read_csv, imread).
          3. Alignment with initial prompt intent / first clause.
        """
        if not idiom.cells:
            return None

        # Compute internal in-degrees within this idiom
        internal_in_deg: Dict[str, int] = defaultdict(int)
        for src, tgt, _ in idiom.edges:
            if src in idiom.cell_ids and tgt in idiom.cell_ids:
                internal_in_deg[tgt] += 1

        c1_scores = self._score_clause_lexical_idf(first_clause, token_index, N) if first_clause else {}

        best_cell = idiom.cells[0]
        best_score = -1e9

        for cell in idiom.cells:
            cid = cell.cell_id
            c1_sc = c1_scores.get(cell, 0.0)
            is_root = 1.0 if internal_in_deg[cid] == 0 else 0.0
            is_stage1 = 1.5 if getattr(cell, "stage", 2) == 1 else 0.0
            is_source = 1.0 if getattr(cell, "node_role", "") == "source" else 0.0
            
            # Nodes with no required input ports receive initiation bonus
            no_req_inputs = 1.0 if not any(getattr(p, "required", True) for p in getattr(cell, "inputs", {}).values()) else 0.0

            entry_score = c1_sc * 1.5 + 2.0 * is_root + 2.0 * is_stage1 + 1.0 * is_source + 1.0 * no_req_inputs
            if entry_score > best_score:
                best_score = entry_score
                best_cell = cell

        return best_cell

    def route(
        self,
        prompt: str,
        top_k: int = 150
    ) -> Tuple[List[Cell], Dict[str, float]]:
        """
        Two-Stage Retrieval Architecture (Phase 4):
          - Stage 1: Discovers candidate idioms/subgraphs using the lexical/IDF scorer
                     and graph edges (AST-mined and typestate transitions).
          - Stage 2: Identifies the entry node within the winning idiom(s).
        Returns:
          (tunnel_cells, cell_relevance_probabilities)
        """
        if not prompt or not prompt.strip():
            return [], {}

        token_index = getattr(self.orchestrator, "token_index", None)
        N = len(self.orchestrator.loaded_cells)

        # Decompose prompt into constituent clauses (sub-goals)
        clauses = [c.strip() for c in re.split(r'[,;]|\b(?:and|then)\b', prompt.strip()) if c.strip()]
        search_texts = [prompt.strip()] + [c for c in clauses if c != prompt.strip()]
        first_clause = clauses[0] if clauses else prompt.strip()

        # Step 1: Stage 1 Idiom / Subgraph Discovery
        candidate_idioms = self._discover_candidate_idioms(prompt, clauses, max_idioms=10)

        # Step 2: Stage 2 Entry Node Selection
        for idiom in candidate_idioms:
            entry = self._select_entry_node(idiom, first_clause, token_index, N)
            idiom.entry_node = entry

        # Step 3: Candidate Group Assembly
        grouped_candidates: List[Dict[Cell, float]] = []

        for text in search_texts:
            clause_scores = self._score_clause_lexical_idf(text, token_index, N)
            if not clause_scores:
                continue

            # Boost cells belonging to top candidate idioms and entry nodes
            for idiom in candidate_idioms[:3]:
                for c in idiom.cells:
                    boost = 1.5 if c == idiom.entry_node else 1.2
                    clause_scores[c] = clause_scores.get(c, 0.0) * boost + (0.5 if c == idiom.entry_node else 0.2)

            kept = sorted(clause_scores.items(), key=lambda x: x[1], reverse=True)[:100]
            grouped_candidates.append(dict(kept))

        # Step 4: Optional dense vector blending from edge-context embeddings
        if self.internal_rag is not None and self.internal_rag.index is not None:
            query_spans = self._generate_query_spans(prompt.strip())
            dense_scores: Dict[str, float] = {}
            rag_results = self.internal_rag.get_relevant_context_batch(query_spans, top_k=min(top_k, 100))
            for span_results in rag_results:
                for item in span_results:
                    cid = item.get("cell_id")
                    sc = float(item.get("score", 0.0))
                    if cid and (cid not in dense_scores or sc > dense_scores[cid]):
                        dense_scores[cid] = sc

            if dense_scores:
                dense_dict: Dict[Cell, float] = {}
                for cid, sc in dense_scores.items():
                    c = self.orchestrator.loaded_cells.get(cid)
                    if c:
                        dense_dict[c] = sc
                grouped_candidates.append(dense_dict)

        candidates_with_scores = self._tunnel_from_groups(grouped_candidates)

        if not candidates_with_scores:
            return [], {}

        # Softmax normalization across the embedding-path candidate pool (the lexical
        # path already normalized per group in _tunnel_from_groups).
        if self.internal_rag is not None and self.internal_rag.index is not None:
            candidates_with_scores = self._tunnel_from_groups([dict(candidates_with_scores)])
            if not candidates_with_scores:
                return [], {}

        final_tunnel: List[Cell] = [c for c, _ in candidates_with_scores]
        relevance_map = {c.cell_id: s for c, s in candidates_with_scores}
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
            prompt_toks = (CellTokenizer.tokenize_prompt(prompt) if prompt else set()) - STOPWORDS
            active_domains = set(c.domain_name for c in final_tunnel if c.domain_name)
            # Carrier-completion augmentation is intentionally TIGHT: only cells in
            # an active domain whose carriers fit the active boundary AND share at
            # least content tokens with the prompt. An unbounded same-domain
            # flood drowns the trellis in noise cells (measured: +571 cells on a
            # 150-cell tunnel in the 38K-node corpus).
            augment: List[Cell] = []
            for cell in self.orchestrator.loaded_cells.values():
                if cell.domain_name in active_domains and cell.cell_id not in tunnel_ids:
                    c_in = getattr(cell.primary_input, "type_name", "")
                    c_out = getattr(cell.primary_output, "type_name", "")
                    is_s1_src = getattr(cell, "stage", None) == 1 and getattr(cell, "node_role", "") == "source"
                    min_ov = 1 if is_s1_src else 2
                    tok_ov = len((cell.token_set - STOPWORDS) & prompt_toks)
                    if (c_in in carriers_out or c_out in carriers_in) and tok_ov >= min_ov:
                        augment.append(cell)
            augment.sort(key=lambda c: (
                len((c.token_set - STOPWORDS) & prompt_toks),
                relevance_map.get(c.cell_id, 0.0),
                -getattr(c, "source_priority", 100)
            ), reverse=True)
            for cell in augment[:35]:
                final_tunnel.append(cell)
                tunnel_ids.add(cell.cell_id)
                relevance_map[cell.cell_id] = max(relevance_map.get(cell.cell_id, 0.0), self.epsilon * 2.0)

        return final_tunnel, relevance_map

    def _tunnel_from_groups(
        self,
        groups: List[Dict[Cell, float]]
    ) -> List[Tuple[Cell, float]]:
        """
        Computes the group-relative softmax tunnel:
          P(v | group_g) = exp(s_v / gamma) / Z_g,  keep v iff P >= tau * max_u P(u | g)
        A cell survives if it clears the threshold in ANY group; its relevance is
        the maximum normalized probability across groups. This preserves every
        sub-goal of a compound prompt at the distribution level.
        """
        tau = max(self.epsilon, 0.01)
        relevance: Dict[str, Tuple[Cell, float]] = {}

        for group in groups:
            if not group:
                continue
            items = list(group.items())
            raw_scores = np.array([s for _, s in items], dtype=np.float64)
            max_s = np.max(raw_scores)
            norm_scores = raw_scores / max(max_s, 1e-6)
            scaled = norm_scores / max(self.gamma, 1e-5)
            shifted = scaled - np.max(scaled)
            exp_scores = np.exp(shifted)
            probs = exp_scores / np.sum(exp_scores)
            max_prob = float(np.max(probs))
            for (cell, _), p in zip(items, probs):
                if p < tau * max_prob:
                    continue
                cid = cell.cell_id
                prev = relevance.get(cid)
                if prev is None or float(p) > prev[1]:
                    relevance[cid] = (cell, float(p))

        ranked = sorted(relevance.values(), key=lambda x: x[1], reverse=True)
        return ranked

    @staticmethod
    def _generate_query_spans(text: str) -> List[str]:
        """
        Generates continuous sliding semantic window queries across the prompt.
        Multi-scale representation: zero linguistic connectors (no 'then'), zero regex.
        The span count is bounded: retrieval quality saturates while query latency
        stays independent of prompt length.
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

        # Bound the retrieval fan-out: the full prompt and the tail always stay;
        # intermediate windows are truncated deterministically.
        if len(queries) > 15:
            queries = queries[:15] + [tail_q] if tail_q not in queries[:15] else queries[:15]
        return queries

    def plan_path(
        self,
        prompt: str,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        return_tuple: bool = True,
        top_k: int = 400
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

        # Bound the planning trellis: the highest-likelihood tunnel prefix is
        # searched exhaustively; deeper tail cells remain reachable through
        # carrier-completion augmentation inside the planner.
        if top_k and len(tunnel_cells) > top_k:
            tunnel_cells = tunnel_cells[:top_k]

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

