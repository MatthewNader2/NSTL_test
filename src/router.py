"""
src/router.py - Neuro-Symbolic Topological Lattice (NSTL)
Semantic Tunneling Router based on Dense Vector Embeddings and Softmax Temperature.

Conforms strictly to Section 3.3 of the NSTL paper:
  e_x = Embed(prompt)
  P(C_k | e_x) = softmax_k( cos(e_x, mu_k) / gamma )
  Tunnel T = { v in V | P(v | e_x) >= epsilon }

Contains ZERO keyword-sniffing regexes, ZERO manual score boosts, and ZERO domain hardcodes.

Tunables previously baked into this module as magic literals now resolve through
`config.settings` when present (see `_cfg`). In-module fallbacks keep the file
runnable standalone; production profiles should override via settings.
"""

from __future__ import annotations
import math
from typing import Optional, List, Dict, Set, Tuple, Any, Union
from collections import defaultdict

import numpy as np
from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell, MacroCell
    from .internal_rag import LocalRAG
    from .planner import LatticePlanner, STOPWORDS, _segment_prompt_clauses
    from .tokenizer import CellTokenizer
    from .inference import ModelManager
    from .route_methods import get_route_method, RouteMethod
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell, MacroCell
    from internal_rag import LocalRAG
    from planner import LatticePlanner, STOPWORDS, _segment_prompt_clauses
    from tokenizer import CellTokenizer
    from inference import ModelManager
    from route_methods import get_route_method, RouteMethod

logger = get_logger('router')


# ---------------------------------------------------------------------------
# Settings access with in-module fallbacks
# ---------------------------------------------------------------------------
# Everything previously hardcoded in this file (span caps, delimiter sets,
# stratified-pruning caps, macro-promotion scaling, dense-RAG budget) is now
# resolved here. If `config.settings` is unavailable or lacks a given key, the
# documented fallback is used so the router stays runnable in isolation.

def _load_settings():
    try:
        from .config import settings as _s
        return _s
    except (ImportError, ValueError):
        try:
            from config import settings as _s
            return _s
        except Exception:
            return None


_SETTINGS = _load_settings()


def _cfg(name: str, default):
    if _SETTINGS is not None:
        return getattr(_SETTINGS, name, default)
    return default


# --- Query span generation ------------------------------------------------
# Bound on the number of query spans emitted per routing call. This directly
# caps dense-RAG fan-out (one embed + FAISS lookup per span).
MAX_QUERY_SPANS: int = int(_cfg("max_query_spans", 12))
# Prompts with <= this many tokens skip sliding-window generation: clause-level
# decomposition already covers their structure.
SHORT_PROMPT_WINDOW_SKIP: int = int(_cfg("short_prompt_window_skip", 6))

# --- Stratified pruning ---------------------------------------------------
# Caps scale with the number of semantic clauses (n) discovered from the IR
# or clause segmenter. Asymmetry across stages is intentional: stage 2 is the
# transform band where intermediates live and where recall matters most.
PRUNE_S1_MIN: int = int(_cfg("prune_s1_min", 12))
PRUNE_S1_PER_CLAUSE: int = int(_cfg("prune_s1_per_clause", 2))
PRUNE_S2_MIN: int = int(_cfg("prune_s2_min", 38))
PRUNE_S2_PER_CLAUSE: int = int(_cfg("prune_s2_per_clause", 6))
PRUNE_S3_MIN: int = int(_cfg("prune_s3_min", 8))
PRUNE_S3_PER_CLAUSE: int = int(_cfg("prune_s3_per_clause", 2))

# --- Macro promotion (Section 3.5) ----------------------------------------
# IDF-weighted concept coverage gate: a macro must own at least this fraction
# of the prompt's semantic mass before it can be promoted.
MACRO_CONCEPT_COVERAGE_GATE: float = float(_cfg("macro_concept_coverage_gate", 0.35))
MACRO_PRIORITY_MICRO_FACTOR: float = float(_cfg("macro_priority_micro_factor", 1.25))
MACRO_PRIORITY_MICRO_BIAS: float = float(_cfg("macro_priority_micro_bias", 0.05))
MACRO_PRIORITY_IDF_WEIGHT: float = float(_cfg("macro_priority_idf_weight", 0.15))
MACRO_REL_MICRO_FACTOR: float = float(_cfg("macro_rel_micro_factor", 1.1))
MACRO_REL_MICRO_BIAS: float = float(_cfg("macro_rel_micro_bias", 0.02))
MACRO_REL_FLOOR: float = float(_cfg("macro_rel_floor", 0.1))

# --- Dense-RAG gating -----------------------------------------------------
# When Stage-1 idiom discovery produces a strong lexical signal (top idiom's
# total_score >= ratio * n_clauses AND full clause coverage), the dense blend
# is skipped: the lexical evidence already suffices and the embed+FAISS pass
# is redundant.
DENSE_GATE_COVERAGE_RATIO: float = float(_cfg("dense_gate_coverage_ratio", 3.0))
# Per-span top_k bounds when dense blending does run.
DENSE_RAG_MIN_PER_SPAN: int = int(_cfg("dense_rag_min_per_span", 4))
DENSE_RAG_MAX_PER_SPAN: int = int(_cfg("dense_rag_max_per_span", 25))


# --- Clause delimiter fallback -------------------------------------------
# Canonical definition lives on CellTokenizer (see recommended addition to
# tokenizer.py). Kept here ONLY as a compatibility shim so the router remains
# functional before that lands. If CellTokenizer.CLAUSE_DELIMITERS is
# present, it wins and this constant is unused.
_FALLBACK_CLAUSE_DELIMITERS: frozenset = frozenset({
    "on the", "on", "using", "with", "via", "into", "from",
    "then", "and", "of the", "of", "to", "for", "by", "in",
})


def _get_clause_delims() -> frozenset:
    delims = getattr(CellTokenizer, "CLAUSE_DELIMITERS", None)
    if delims:
        return frozenset(delims)
    return _FALLBACK_CLAUSE_DELIMITERS


def _extract_id(item):
    if hasattr(item, "cell_id"):
        return item.cell_id
    if isinstance(item, (tuple, list)):
        for x in item:
            if hasattr(x, "cell_id"):
                return x.cell_id
    return str(item)


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
        # Accept both the current kwarg name (`default_route_method`, used by
        # every call site) and the legacy name (`route_method`) that this
        # line used to read exclusively -- the mismatch meant a caller's
        # explicit default_route_method=... was silently dropped into
        # **kwargs and never read, always falling back to "m0" regardless of
        # what was requested. An explicit method still wins; when neither is
        # given, prefer an LLM-driven default (m4) over the pure-symbolic
        # trellis (m0) whenever the active model profile actually has a
        # usable generation model -- there is no reason to fall back to
        # regex/heuristic-only routing when a model capable of reading the
        # prompt is already loaded. Profiles that intentionally have no
        # usable LLM (0, A) or that intentionally disable synthesis for
        # ablation purposes (e.g. a routing-only benchmark profile) report
        # can_synthesize() == False and are unaffected.
        _explicit_method = kwargs.get("default_route_method", kwargs.get("route_method"))
        if _explicit_method:
            self.default_route_method = str(_explicit_method).strip().lower()
        else:
            try:
                self.default_route_method = "m4" if ModelManager.get_instance().can_synthesize() else "m0"
            except Exception:
                self.default_route_method = "m0"
        # Macro-goal routing (Section 3.5). Defaults to the global settings
        # value; an explicit kwarg wins. Toggling only affects routers created
        # afterwards unless mutated directly (see CLI `set macros on|off`).
        macros_val = kwargs.get("macros_enabled")
        if macros_val is None:
            macros_val = _cfg("macros_enabled", True)
        self._macros_enabled = bool(macros_val)

        topo_val = kwargs.get("topology_mode")
        if topo_val is None:
            topo_val = _cfg("topology_mode", "frontier")
        self._topology_mode = str(topo_val).lower()
        self.planner = LatticePlanner(
            orchestrator=self.orchestrator,
            macros_enabled=self._macros_enabled,
            topology_mode=self._topology_mode
        )
        # (apply_fixes_v7) IR compiler hook
        self._use_ir_compiler = bool(_cfg("use_ir_compiler", True))
        self._ir_compiler = None
        if self._use_ir_compiler:
            try:
                from ir_compiler import IRCompiler
                self._ir_compiler = IRCompiler(self.orchestrator)
            except Exception as _e:
                logger.warning("IR compiler unavailable: %s", _e)
        self._last_ir_steps = 0
        self.priority_map: Dict[str, float] = {}

    @property
    def macros_enabled(self) -> bool:
        return self._macros_enabled

    @macros_enabled.setter
    def macros_enabled(self, val: bool):
        self._macros_enabled = bool(val)
        if hasattr(self, "planner") and self.planner is not None:
            self.planner.macros_enabled = bool(val)

    @property
    def topology_mode(self) -> str:
        return getattr(self.planner, "topology_mode", getattr(self, "_topology_mode", "frontier"))

    @topology_mode.setter
    def topology_mode(self, val: str):
        self._topology_mode = str(val).lower()
        if hasattr(self, "planner") and self.planner is not None:
            self.planner.topology_mode = self._topology_mode

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
                    # 1. Direct edge check across DAG history in chain_cells
                    for anc in reversed(chain_cells):
                        anc_out = [e.get("target_cell_id") for e in getattr(anc, "edges", [])]
                        anc_adj = self.orchestrator._adjacency.get(anc.cell_id, [])
                        for cand, cand_sc in next_seeds[:6]:
                            if cand.cell_id in anc_out or cand.cell_id in anc_adj:
                                edge_rec = next((e for e in getattr(anc, "edges", []) if e.get("target_cell_id") == cand.cell_id), None)
                                aff = edge_rec.get("affinity_score", 0.5) if edge_rec else 0.5
                                chain_edges.append((anc.cell_id, cand.cell_id, aff))
                                if cand not in chain_cells:
                                    chain_cells.append(cand)
                                covered.add(c_idx)
                                total_lex += cand_sc
                                curr_node = cand
                                found_next = True
                                break
                        if found_next:
                            break

                    # 2. 1-hop bridge transition check across DAG history
                    if not found_next:
                        for anc in reversed(chain_cells):
                            anc_out = [e.get("target_cell_id") for e in getattr(anc, "edges", [])]
                            anc_adj = self.orchestrator._adjacency.get(anc.cell_id, [])
                            mid_candidates = anc_out[:6] if anc_out else anc_adj[:6]
                            for mid_id in mid_candidates:
                                mid_cell = self.orchestrator.loaded_cells.get(mid_id)
                                if not mid_cell:
                                    continue
                                mid_out = [e.get("target_cell_id") for e in getattr(mid_cell, "edges", [])]
                                mid_adj = self.orchestrator._adjacency.get(mid_id, [])
                                for cand, cand_sc in next_seeds[:6]:
                                    if cand.cell_id in mid_out or cand.cell_id in mid_adj:
                                        e1 = next((e for e in getattr(anc, "edges", []) if e.get("target_cell_id") == mid_id), None)
                                        aff1 = e1.get("affinity_score", 0.5) if e1 else 0.5
                                        e2 = next((e for e in getattr(mid_cell, "edges", []) if e.get("target_cell_id") == cand.cell_id), None)
                                        aff2 = e2.get("affinity_score", 0.5) if e2 else 0.5

                                        chain_edges.append((anc.cell_id, mid_id, aff1))
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
        if not prompt or not prompt.strip() or len(self.orchestrator.loaded_cells) == 0:
            return [], {}

        # IR compilation: when the loaded model successfully compiles the
        # prompt into a typed IR, that IR -- not a regex/punctuation split of
        # the raw text -- is the source of truth for the pipeline's semantic
        # clauses. The IR was produced by a model that actually read the
        # prompt; the previous approach ran the LLM IR compiler ONLY to
        # bump a clause *count* and to stuff its op/library words into the
        # embedding query text, then still regex-segmented that mutated
        # text for the actual clauses used everywhere else (search_texts,
        # first_clause, idiom discovery). That defeated the point of having
        # an IR at all. Here, a successfully compiled IR with >=1 step
        # supplies the clauses directly (one per step); the embedding text
        # itself is left untouched. The regex/punctuation segmenter is kept
        # ONLY as the fallback for when no model is loaded or IR compilation
        # did not produce anything usable.
        num_semantic_clauses = 0
        ir_clauses: List[str] = []
        if getattr(self, "_use_ir_compiler", False) and self._ir_compiler is not None:
            try:
                _ir = self._ir_compiler.compile(prompt)
                if _ir is not None and _ir.steps:
                    for s in _ir.steps:
                        op = str(s.get("op", "")).strip()
                        lib = str(s.get("library", "")).strip()
                        params = s.get("params") or {}
                        param_words = " ".join(
                            str(v) for v in params.values()
                            if isinstance(v, (str, int, float)) and str(v).strip()
                        )
                        piece = " ".join(w for w in (op, lib, param_words) if w).strip()
                        if piece:
                            ir_clauses.append(piece)
                    num_semantic_clauses = len(_ir.steps)
                    self._last_ir_steps = num_semantic_clauses
            except Exception as _e:
                logger.debug("[IR] compilation failed: %s", _e)

        token_index = getattr(self.orchestrator, "token_index", None)
        N = len(self.orchestrator.loaded_cells)

        # Decompose prompt into constituent clauses (sub-goals). Prefer the
        # model-compiled IR steps; fall back to the punctuation/vocabulary
        # segmenter only when no IR was available.
        clauses = ir_clauses if ir_clauses else _segment_prompt_clauses(prompt)
        if not num_semantic_clauses:
            num_semantic_clauses = len(clauses)
        search_texts = [prompt.strip()] + [c for c in clauses if c != prompt.strip()]
        for q in self._generate_query_spans(prompt.strip()):
            if q not in search_texts:
                search_texts.append(q)
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

            kept = sorted(clause_scores.items(), key=lambda x: x[1], reverse=True)[:100]
            grouped_candidates.append(dict(kept))

        # Step 3.1: Candidate Subgraph / Idiom Group Preservation
        # Discovered idioms represent coherent topological paths across clauses and hidden clauses.
        # Adding each idiom as its own candidate group ensures bridging nodes survive the group-relative softmax.
        for idiom in candidate_idioms[:3]:
            if not idiom.cells:
                continue
            idiom_group: Dict[Cell, float] = {}
            for c in idiom.cells:
                idiom_group[c] = 1.0 + (0.5 if c == idiom.entry_node else 0.0)
            if idiom_group:
                grouped_candidates.append(idiom_group)

        # Step 4: Optional dense vector blending from edge-context embeddings.
        # Gated on weak lexical signal: if Stage-1 idiom discovery already
        # produced a strongly-covered candidate (full clause coverage and a
        # total_score at least DENSE_GATE_COVERAGE_RATIO x clause count), the
        # lexical evidence is sufficient and the embed + FAISS pass is skipped.
        # This removes the dominant latency cost on Profile A/C/D prompts whose
        # intent is already unambiguous from tokens alone.
        strong_lexical = bool(candidate_idioms) and (
            candidate_idioms[0].total_score
                >= DENSE_GATE_COVERAGE_RATIO * max(num_semantic_clauses, 1)
            and len(candidate_idioms[0].covered_clauses) >= num_semantic_clauses
        )

        if (not strong_lexical) and self.internal_rag is not None and self.internal_rag.index is not None:
            clause_spans = [c.strip() for c in search_texts if c and c.strip()]
            query_spans = clause_spans + [
                q for q in self._generate_query_spans(prompt.strip()) if q not in clause_spans
            ]
            # Divide the retrieval budget across spans instead of granting every
            # span the full top_k, so total dense work is bounded by top_k
            # regardless of span count.
            n_spans = max(len(query_spans), 1)
            per_span_k = max(
                DENSE_RAG_MIN_PER_SPAN,
                min(DENSE_RAG_MAX_PER_SPAN, top_k // n_spans),
            )
            dense_scores: Dict[str, float] = {}
            rag_results = self.internal_rag.get_relevant_context_batch(query_spans, top_k=per_span_k)
            for span_results in rag_results:
                for item in span_results:
                    cid = item.get("cell_id")
                    sc = float(item.get("score", 0.0))
                    if cid and sc >= 0.25 and (cid not in dense_scores or sc > dense_scores[cid]):
                        dense_scores[cid] = sc

            if dense_scores:
                sorted_dense = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)[:60]
                dense_dict: Dict[Cell, float] = {}
                for cid, sc in sorted_dense:
                    c = self.orchestrator.loaded_cells.get(cid)
                    if c:
                        dense_dict[c] = sc
                if dense_dict:
                    grouped_candidates.append(dense_dict)

        candidates_with_scores = self._tunnel_from_groups(grouped_candidates)

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
            prompt_toks = CellTokenizer.tokenize_prompt(prompt) if prompt else set()
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
                    tok_ov = len(getattr(cell, "identity_tokens", cell.token_set) & prompt_toks)
                    if (c_in in carriers_out or c_out in carriers_in) and tok_ov >= min_ov:
                        augment.append(cell)
            augment.sort(key=lambda c: (
                len(getattr(c, "identity_tokens", c.token_set) & prompt_toks),
                relevance_map.get(c.cell_id, 0.0),
                -getattr(c, "source_priority", 100)
            ), reverse=True)
            for cell in augment[:35]:
                final_tunnel.append(cell)
                tunnel_ids.add(cell.cell_id)
                relevance_map[cell.cell_id] = max(relevance_map.get(cell.cell_id, 0.0), self.epsilon * 2.0)

        # Macro-goal routing toggle: macro-goal cells participate in the
        # tunnel as ordinary lexical members when enabled; when disabled they
        # are excluded entirely, restoring the pre-macro routing distribution
        # exactly (clean A/B baseline for benchmarks).
        if not self.macros_enabled:
            final_tunnel = [c for c in final_tunnel if not isinstance(c, MacroCell)]
            tunnel_ids = {c.cell_id for c in final_tunnel}
            relevance_map = {
                cid: s for cid, s in relevance_map.items()
                if not isinstance(self.orchestrator.loaded_cells.get(cid), MacroCell)
            }
        # Macro-goal promotion (Section 3.5): known-good composite paths.
        # When the prompt is relevant to a MacroCell (a predefined, verified
        # path over existing micro-cells — no new functionality), the macro
        # enters the tunnel and scores ABOVE every one of its constituent
        # micro-cells, sharpening the routing distribution toward the
        # known-good path. Skipped entirely when macros are disabled.
        if self.macros_enabled:
            self._promote_macro_goals(prompt, final_tunnel, tunnel_ids, relevance_map)
        final_tunnel = self._prune_tunnel_stratified(
            final_tunnel, relevance_map,
            num_semantic_clauses=num_semantic_clauses,
        )

        for cid, s in relevance_map.items():
            if cid not in self.priority_map:
                self.priority_map[cid] = s

        return final_tunnel, relevance_map


    def _prune_tunnel_stratified(self, tunnel_cells, relevance_map,
                                 num_semantic_clauses: int = 1,
                                 max_s1=None, max_s2=None, max_s3=None):
        """(apply_fixes_v7) Stratified pruning; caps scale with task size and
        resolve through config (see PRUNE_* module constants)."""
        n = max(int(num_semantic_clauses or 1), 1)
        max_s1 = max_s1 if max_s1 is not None else max(PRUNE_S1_MIN, n * PRUNE_S1_PER_CLAUSE)
        max_s2 = max_s2 if max_s2 is not None else max(PRUNE_S2_MIN, n * PRUNE_S2_PER_CLAUSE)
        max_s3 = max_s3 if max_s3 is not None else max(PRUNE_S3_MIN, n * PRUNE_S3_PER_CLAUSE)

        def _stage_num(c):
            st = getattr(c, 'stage', -1)
            if isinstance(st, int): return st
            st_str = str(st).strip().upper()
            return 1 if st_str in ('1', 'S1', 'INGRESS') else (
                   2 if st_str in ('2', 'S2', 'TRANSFORM') else (
                   3 if st_str in ('3', 'S3', 'EGRESS', 'SINK') else 0))

        protected_ids = {c.cell_id for c in tunnel_cells if relevance_map.get(c.cell_id, 0.0) >= 0.05}
        s1 = sorted([c for c in tunnel_cells if _stage_num(c) == 1],
                    key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)[:max_s1]
        s2 = sorted([c for c in tunnel_cells if _stage_num(c) == 2],
                    key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)[:max_s2]
        s3 = sorted([c for c in tunnel_cells if _stage_num(c) == 3],
                    key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)[:max_s3]
        bridge_ids = {c.cell_id for c in tunnel_cells if getattr(c, "node_role", "") == "bridge" or getattr(c, "node_type", "") == "tunnel"}
        keep_ids = {c.cell_id for c in s1 + s2 + s3} | protected_ids | bridge_ids
        return [c for c in tunnel_cells if c.cell_id in keep_ids] or tunnel_cells

    def _promote_macro_goals(
        self,
        prompt: str,
        final_tunnel: List[Cell],
        tunnel_ids: Set[str],
        relevance_map: Dict[str, float]
    ) -> None:
        """
        Promotes matched MacroCells above their constituent micro-cells.

        A macro is a macro-goal: an empirically verified path over existing
        micro-cells (synapses forming a known-good macro-path). Promotion is
        gated by IDF-weighted CONCEPT COVERAGE: the macro's declared identity
        vocabulary (cell id + keywords) must cover a substantial fraction of
        the prompt's semantic mass. Generic I/O vocabulary ('csv', 'save')
        carries near-zero IDF and can therefore never promote a macro on its
        own — only the macro's composite concept (e.g. ranking, contour
        annotation) can. This mirrors the planner's idf-weighted coverage
        philosophy and keeps macros from hijacking prompts they do not express.

        When promoted, the macro's relevance is strictly greater than the
        maximum relevance of its own micro-cells (the known-good path outranks
        its constituents). The macro carries no new functionality: downstream,
        UnificationGate expands it back into its micro-cell sequence.

        All scaling constants resolve through config (see MACRO_* module
        constants); see settings for tuning semantics.
        """
        prompt_toks = CellTokenizer.tokenize_prompt(prompt) if prompt else set()
        if not prompt_toks:
            return

        token_index = getattr(self.orchestrator, "token_index", None) or {}
        n_cells = max(len(self.orchestrator.loaded_cells), 1)

        def _idf(tok: str) -> float:
            return math.log(1.0 + (n_cells + 1) / (len(token_index.get(tok, ())) + 1.0))

        prompt_mass = sum(_idf(t) for t in prompt_toks)
        if prompt_mass <= 0:
            return

        promoted: List[Tuple[float, float, Cell]] = []
        for cell in self.orchestrator.loaded_cells.values():
            if not (isinstance(cell, MacroCell) and getattr(cell, "sub_cells", None)):
                continue
            identity_hits = getattr(cell, "identity_tokens", set()) & prompt_toks
            if not identity_hits:
                continue

            idf_mass = sum(_idf(t) for t in identity_hits)
            concept_coverage = idf_mass / prompt_mass
            if concept_coverage < MACRO_CONCEPT_COVERAGE_GATE:
                continue

            micro_scores = [
                float(relevance_map.get(sid, 0.0))
                for sid in cell.sub_cells
                if sid in relevance_map
            ]
            micro_max = max(micro_scores) if micro_scores else 0.0

            # Strictly-above guarantee: the macro outranks its strongest micro
            # by a margin, scaled by the discriminative concept mass.
            # D2 Fix: We compute an unbounded priority score for priority_map,
            # but calibrate relevance_map so it remains strictly in [0.0, 1.0].
            raw_priority = max(
                micro_max * MACRO_PRIORITY_MICRO_FACTOR + MACRO_PRIORITY_MICRO_BIAS,
                float(self.epsilon) * 4.0,
            ) + MACRO_PRIORITY_IDF_WEIGHT * idf_mass

            calibrated_macro_rel = min(
                1.0,
                max(micro_max * MACRO_REL_MICRO_FACTOR + MACRO_REL_MICRO_BIAS, MACRO_REL_FLOOR),
            )

            promoted.append((raw_priority, calibrated_macro_rel, cell))

        for raw_priority, calibrated_rel, cell in promoted:
            if cell.cell_id not in tunnel_ids:
                final_tunnel.append(cell)
                tunnel_ids.add(cell.cell_id)
            if calibrated_rel > relevance_map.get(cell.cell_id, 0.0):
                relevance_map[cell.cell_id] = calibrated_rel
            self.priority_map[cell.cell_id] = max(self.priority_map.get(cell.cell_id, 0.0), raw_priority)

            # Synaptic micro-cell promotion: bring constituent micro-cells into the tunnel
            # with boosted relevance so they participate in routing and allow intermediate nodes
            # (such as ranking, slicing, or custom filtering) to be spliced in the middle.
            for sid in getattr(cell, "sub_cells", []):
                micro_cell = self.orchestrator.loaded_cells.get(sid)
                if micro_cell:
                    if sid not in tunnel_ids:
                        final_tunnel.append(micro_cell)
                        tunnel_ids.add(sid)
                    boosted_score = min(1.0, max(relevance_map.get(sid, 0.0), calibrated_rel * 0.85))
                    relevance_map[sid] = boosted_score
                    self.priority_map[sid] = max(self.priority_map.get(sid, 0.0), raw_priority * 0.85)

        if promoted:
            logger.debug(
                f"[ROUTER] Macro-goal promotion: "
                f"{[_extract_id(c) for c in promoted]}"
            )

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
            group_size_factor = min(1.0, (len(items) / 5.0) ** 0.5)
            calibrated_probs = probs * group_size_factor
            cutoff = tau * max_prob * group_size_factor
            sorted_indices = np.argsort(raw_scores)[::-1]
            top_indices = set(sorted_indices[:8])

            for i, ((cell, _), p) in enumerate(zip(items, calibrated_probs)):
                if p < cutoff and i not in top_indices:
                    continue
                effective_p = max(float(p), float(cutoff)) if i in top_indices else float(p)
                cid = cell.cell_id
                prev = relevance.get(cid)
                if prev is None or effective_p > prev[1]:
                    relevance[cid] = (cell, effective_p)

        ranked = sorted(relevance.values(), key=lambda x: x[1], reverse=True)
        return ranked

    @staticmethod
    def _generate_query_spans(text: str) -> List[str]:
        """
        Generates continuous sliding semantic window queries across the prompt.
        Multi-scale representation: decomposes compound sentences and spans across
        the entire prompt uniformly so trailing and embedded operations are never truncated.

        Bounded: the returned list is capped at MAX_QUERY_SPANS so the downstream
        dense-RAG stage cannot blow up on long prompts. Dedup uses a set for
        O(1) membership rather than O(n) list scans. Short prompts
        (<= SHORT_PROMPT_WINDOW_SKIP tokens) skip the sliding-window pass
        entirely: their clause-level decomposition already covers their
        structure, and sliding windows on 4-6 tokens just re-emit substrings.

        The delimiter set used to break compound clauses comes from
        CellTokenizer.CLAUSE_DELIMITERS when present; a module-local fallback
        keeps this function usable before that lands.
        """
        tokens = text.strip().split()
        if len(tokens) <= 3:
            return [text.strip()]

        seen: Set[str] = set()
        queries: List[str] = []

        def _add(q: str) -> None:
            q = q.strip(" ,.;:'\"")
            if len(q) >= 2 and q not in seen:
                seen.add(q)
                queries.append(q)

        _add(text.strip())

        # 1. Clause-level decomposition and sub-action split on delimiters
        clauses = _segment_prompt_clauses(text)
        delims = _get_clause_delims()
        for cl in clauses:
            _add(cl)
            current_chunk: List[str] = []
            for w in cl.split():
                cw = w.lower().strip(" ,.;:'\"")
                if cw in delims:
                    if current_chunk:
                        _add(" ".join(current_chunk))
                        current_chunk = []
                else:
                    current_chunk.append(w)
            if current_chunk:
                _add(" ".join(current_chunk))

        # Short prompts: clause-level decomposition is sufficient; sliding
        # windows would only re-emit overlapping substrings of the same few
        # tokens.
        if len(tokens) <= SHORT_PROMPT_WINDOW_SKIP:
            return queries[:MAX_QUERY_SPANS]

        # 2. Sliding multi-scale windows spanning uniformly across the full prompt
        n = len(tokens)
        for window_size in (2, 3, 4):
            if window_size >= n:
                continue
            step = max(1, (n - window_size) // 8)
            for i in range(0, n - window_size + 1, step):
                _add(" ".join(tokens[i : i + window_size]))

        _add(" ".join(tokens[max(0, n - 3) :]))

        # Bounded retrieval fan-out; see MAX_QUERY_SPANS.
        return queries[:MAX_QUERY_SPANS]

    def plan_path(
        self,
        prompt: str,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        return_tuple: bool = True,
        top_k: int = 400,
        route_method: Optional[str] = None,
        ctx: Optional[Any] = None,
        **kwargs
    ) -> Union[List[Cell], Tuple[List[Cell], Set[str]]]:
        """
        End-to-end routing & topological pathfinding:
          1. Computes active semantic tunnel T for prompt.
          2. Uses selected RouteMethod (M0 - M6) to find valid monadic composition through T.
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

        # Dispatch to selected RouteMethod
        method_name = str(route_method or self.default_route_method or "m0").strip().lower()
        # Registry-driven dispatch decision (no alias duplication here):
        # the M0 trellis planner is the in-router baseline; everything else
        # delegates to its registered RouteMethod class.
        from route_methods import ROUTE_METHOD_REGISTRY as _ROUTE_REGISTRY
        _method_cls = _ROUTE_REGISTRY.get(method_name)
        if _method_cls is not None and _method_cls.__name__ != "M0TrellisRouteMethod":
            try:
                method = get_route_method(method_name, orchestrator=self.orchestrator)
                path = method.plan(
                    prompt=prompt,
                    tunnel=tunnel_cells,
                    relevance_map=relevance_map,
                    orchestrator=self.orchestrator,
                    ctx=ctx,
                    start_sig=start_sig,
                    goal_sig=goal_sig,
                    rag=kwargs.get("rag", self.internal_rag),
                    **kwargs
                )
            except Exception as e:
                logger.error(
                    f"[ROUTER] RouteMethod '{method_name}' failed: {e}",
                    exc_info=True,
                )
                path = self.planner.plan(
                    prompt=prompt,
                    tunnel=tunnel_cells,
                    relevance_map=relevance_map,
                    start_sig=start_sig,
                    goal_sig=goal_sig
                )
        else:
            # Find verified composition path inside tunnel T using baseline Trellis
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
                        if c2.cell_id == c1.cell_id:
                            continue  # no self-composition: a cell cannot consume its own output
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
