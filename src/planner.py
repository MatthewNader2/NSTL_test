"""
src/planner.py - Neuro-Symbolic Topological Lattice (NSTL)
Topological Pathfinding, Multi-Stage Progression, and Formal Type-Monadic Verification.

Conforms strictly to Sections 3.1, 3.2, and 3.4 of the NSTL paper:
  A path through the lattice is a sequence of monadic binds. Any step that
  would produce bottom is rejected the moment it is proposed.
  Finds maximum-likelihood type-valid composition paths inside the semantic tunnel T.
"""

from __future__ import annotations
import copy
import math
import os
import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any

from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry, is_path_port as _lattice_is_path_port
    from .unification import unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch, substitute_generics, ExecutionContext, UnificationGate, Success
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry, is_path_port as _lattice_is_path_port
    from unification import unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch, substitute_generics, ExecutionContext, UnificationGate, Success
    from tokenizer import CellTokenizer

logger = get_logger('planner')

registry = TypeRegistry.get_instance()

class DynamicStopwords(frozenset):
    """Dynamic stopword set backed by document-frequency corpus function words."""
    def __contains__(self, item):
        return item in TypeRegistry.get_instance().get_function_words()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_function_words())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_function_words())
    def __sub__(self, other):
        return TypeRegistry.get_instance().get_function_words() - (set(other) if not isinstance(other, set) else other)
    def __rsub__(self, other):
        return set(other) - TypeRegistry.get_instance().get_function_words()
    def __and__(self, other):
        return TypeRegistry.get_instance().get_function_words() & (set(other) if not isinstance(other, set) else other)
    def __rand__(self, other):
        return (set(other) if not isinstance(other, set) else other) & TypeRegistry.get_instance().get_function_words()
    def __or__(self, other):
        return TypeRegistry.get_instance().get_function_words() | (set(other) if not isinstance(other, set) else other)
    def __ror__(self, other):
        return (set(other) if not isinstance(other, set) else other) | TypeRegistry.get_instance().get_function_words()

STOPWORDS = DynamicStopwords()

class DynamicWildcardCarriers(frozenset):
    """Dynamic top/wildcard carrier checker backed by poset and tree declarations."""
    def __contains__(self, item):
        s = str(item or "").strip().lower()
        return not s or s in ("none", "unknown", "*", "top", "any", "object") or TypeRegistry.get_instance().is_declared_top(s)
    def __iter__(self):
        return iter(TypeRegistry.get_instance()._declared_top | {"any", "", "none", "*", "top", "unknown"})

_WILDCARD_CARRIERS = DynamicWildcardCarriers()

# --------------------------------------------------------------------- #
# Path-scoring weight table (the ONLY place these weights are declared).
# Individually they are calibrated design constants, not prompt/domain
# sniffing: they weight STRUCTURAL terms of the path objective (coverage,
# affinity, parsimony, intent completion). Keeping them in one declared
# table makes the objective auditable and tunable without touching logic.
# --------------------------------------------------------------------- #
PATH_SCORE_WEIGHTS = {
    "coverage": 10.0,          # idf-weighted prompt-token coverage of the path
    "alignment": 10.0,         # clause-alignment (monotonic DP) term
    "affinity": 25.0,          # declared/AST-mined edge affinity (dominant)
    "literal_consumption": 15.0,  # ratio of L0 literals bound to declared ports
    "macro_log_prob_weight": 1.3, # macro-goal tunnel relevance carries full weight
    "flat_log_prob_weight": 0.3,  # flat-cell softmax log-prob is a weak tiebreaker
    "macro_affinity_floor": 0.5,  # single-step baseline for verified sub-compositions
    "dead_ctor": 25.0,
    "dead_expansion_step": 10.0,   # macro middle step matching zero prompt tokens
    "unbindable": 50.0,
    "gap": 1.5,
    "dispersion": 5.0,
    "intent_deficit_final": 35.0,
    "intent_deficit_partial": 15.0,
    "weak_edge": 0.75,
    "wildcarrier": 1.0,
    "parsimony_step": 1.2,
    "parsimony_base": 0.2,
    "sink_bonus": 8.0,
    "non_sink_penalty": 6.0,
    "domain_coherence": 1.5,
    "file_port_bonus": 2.0,
    "file_port_penalty": 3.0,
}

class DynamicEgressTokens(frozenset):
    """Dynamic egress intent verbs harvested from stage-3 sink cells across loaded domain trees."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_egress_tokens()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_egress_tokens())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_egress_tokens())
    def __and__(self, other):
        return (set(other) if not isinstance(other, set) else other) & TypeRegistry.get_instance().get_egress_tokens()
    def __rand__(self, other):
        return (set(other) if not isinstance(other, set) else other) & TypeRegistry.get_instance().get_egress_tokens()

EGRESS_INTENT_TOKENS = DynamicEgressTokens()

class DynamicMaterializationStates(frozenset):
    """Dynamic materialization states harvested from output typestates of sink cells."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_materialization_states()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_materialization_states())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_materialization_states())

MATERIALIZATION_OUTPUT_STATES = DynamicMaterializationStates()


def _is_col_projection_port(p_sig: Any) -> bool:
    """Structural column-projection port test: declared state or declared
    list-carrier with projection vocabulary in the DECLARED state tokens.
    No cell-id substrings."""
    st = str(getattr(getattr(p_sig, "signature", p_sig), "state", "")).lower()
    if st == "column_projection":
        return True
    tname = str(getattr(getattr(p_sig, "signature", p_sig), "type_name", "")).lower()
    if registry.is_subtype(tname, "list") or registry.is_subtype(tname, "sequence"):
        st_tokens = set(CellTokenizer.tokenize_identifier(st))
        if st_tokens & {"column", "columns", "col", "feature", "features", "projection", "fields"}:
            return True
    return False


def _safe_slots_items(cell: Any):
    s = getattr(cell, "slots", None)
    if isinstance(s, dict):
        return list(s.items())
    elif isinstance(s, (list, tuple, set)):
        return [(item, {}) for item in s]
    return []



def _segment_prompt_clauses(prompt: str) -> List[str]:
    """
    Language-level clause segmentation:
    Partitions user prompt on major punctuation (;), sequencing ('then'), and clause boundaries (,).
    Merges operand fragments that are list continuations (where non-stopword tokens are a subset
    of the preceding clause) so parameter coordinate lists like 'X column, Y column and Z column'
    or coordinated noun phrases 'on X and Y' do not artificially fragment into spurious clauses.
    """
    if not prompt:
        return []
    raw_parts = [p.strip() for p in re.split(r'[;]|\b(?:then)\b|,', prompt.strip()) if p.strip()]
    clauses: List[str] = []
    for p in raw_parts:
        p_toks = CellTokenizer.tokenize_prompt(p) - STOPWORDS
        if not p_toks:
            continue
        if clauses:
            prev_toks = CellTokenizer.tokenize_prompt(clauses[-1]) - STOPWORDS
            if p_toks.issubset(prev_toks):
                clauses[-1] = clauses[-1] + ", " + p
                continue
        clauses.append(p)
    return clauses or [prompt.strip()]


class LatticePlanner:
    """
    Topological Planner & Gap Bridging Engine (Sections 3.1-3.4).
    Operates strictly within the active semantic tunnel T.
    Finds maximum-likelihood type-valid composition paths.
    """
    def __init__(self, orchestrator: LatticeOrchestrator, rag: Optional[Any] = None, macros_enabled: Optional[bool] = None):
        self.orchestrator = orchestrator
        self.rag = rag
        if macros_enabled is not None:
            self.macros_enabled = bool(macros_enabled)
        else:
            try:
                from config import settings
                self.macros_enabled = bool(getattr(settings, "macros_enabled", True))
            except Exception as e:
                self.macros_enabled = True
        self.current_relevance_map: Dict[str, float] = {}

    def _calculate_edge_affinity(self, src_cell: Cell, dst_cell: Cell) -> float:
        """
        Calculates empirical edge affinity score between two cells.
        AST-mined edges from real code snippets receive the highest affinity,
        followed by LLM seed edges, synaptic macro reinforcement, and topological reachability.
        """
        # Synaptic Macro-Goal Edge Reinforcement (Synapses in the Brain):
        # When macro routing is enabled, a known-good path pre-wired inside an active
        # macro reinforces the synaptic connection between its constituent micro-cells.
        if getattr(self, "macros_enabled", True) and hasattr(self.orchestrator, "loaded_cells"):
            for m in self.orchestrator.loaded_cells.values():
                if getattr(m, "cell_type", "") == "macro" or getattr(m, "sub_cells", None):
                    topo = getattr(m, "internal_topology", {}) or {}
                    is_macro_edge = dst_cell.cell_id in topo.get(src_cell.cell_id, ())
                    if not is_macro_edge and getattr(m, "sub_cells", None):
                        subs = m.sub_cells
                        for i in range(len(subs) - 1):
                            if subs[i] == src_cell.cell_id and subs[i + 1] == dst_cell.cell_id:
                                is_macro_edge = True
                                break
                    if is_macro_edge:
                        macro_rel = getattr(self, "current_relevance_map", {}).get(m.cell_id, 0.5)
                        synaptic_boost = 0.4 * max(macro_rel, 0.5)
                        return min(1.0, max(0.8, 0.6 + synaptic_boost))

        # 1. Forward declared edges on src_cell
        dst_id_lower = dst_cell.cell_id.lower()
        for edge in getattr(src_cell, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id and (tgt_id == dst_cell.cell_id or str(tgt_id).lower() == dst_id_lower):
                aff = float(edge.get("affinity_score", 0.5) if isinstance(edge, dict) else getattr(edge, "affinity_score", 0.5))
                prov = edge.get("score_provenance") if isinstance(edge, dict) else getattr(edge, "score_provenance", "")
                if prov == "ast_mined":
                    return min(1.0, aff * 1.25)
                return aff

        # 2. Reverse declared edges on dst_cell (bidirectional idiom affinity)
        src_id_lower = src_cell.cell_id.lower()
        for edge in getattr(dst_cell, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id and (tgt_id == src_cell.cell_id or str(tgt_id).lower() == src_id_lower):
                aff = float(edge.get("affinity_score", 0.3) if isinstance(edge, dict) else getattr(edge, "affinity_score", 0.3))
                return aff * 0.75

        # 3. Stage 2 -> Stage 3 Egress Completion:
        # A data transformer transitioning to a matching Stage 3 sink in the same domain
        # represents a canonical pipeline conclusion (e.g. dropna -> to_csv, canny -> imwrite).
        src_stage = getattr(src_cell, "stage", None)
        dst_stage = getattr(dst_cell, "stage", None)
        src_domain = getattr(src_cell, "domain_name", "")
        dst_domain = getattr(dst_cell, "domain_name", "")
        if src_stage == 2 and dst_stage == 3 and src_domain and dst_domain and src_domain == dst_domain:
            return 0.75

        # 4. Topological reachability in orchestrator if built
        adj = getattr(self.orchestrator, "_adjacency", None) or getattr(self.orchestrator, "adjacency", None)
        if adj and dst_cell.cell_id in adj.get(src_cell.cell_id, ()):
            return 0.35

        # 5. Same-domain morphism continuity
        if src_domain and dst_domain and src_domain == dst_domain:
            return 0.20

        return 0.0

    def compute_edge_score(
        self,
        src_cell: Cell,
        dst_cell: Cell,
        relevance_map: Optional[Dict[str, float]] = None,
        weights: Tuple[float, float, float, float] = (0.35, 0.30, 0.25, 0.10)
    ) -> float:
        """
        Formal 4-Term Edge Score Model (T3.1):
        E(u, v) = w1 * AST_affinity(u, v) + w2 * semantic_relevance(v) + w3 * role_progress(u, v) + w4 * (1 / distance(u, v))
        """
        w1, w2, w3, w4 = weights

        # Term 1: AST-affinity
        ast_aff = self._calculate_edge_affinity(src_cell, dst_cell)

        # Term 2: Semantic Relevance
        rel = (relevance_map or {}).get(dst_cell.cell_id, 0.0)

        # Term 3: Role Progress
        src_stage = getattr(src_cell, "stage", 1) or 1
        dst_stage = getattr(dst_cell, "stage", 2) or 2
        if src_stage == 1 and dst_stage == 2:
            stage_prog = 0.5
        elif src_stage == 2 and dst_stage == 2:
            stage_prog = 0.3
        elif src_stage == 2 and dst_stage == 3:
            stage_prog = 0.8
        elif src_stage == 1 and dst_stage == 3:
            stage_prog = 0.6
        else:
            stage_prog = 0.2

        src_roles = {getattr(p, "port_role", None) or getattr(p, "derived_role", "") for p in src_cell.outputs.values()}
        dst_roles = {getattr(p, "port_role", None) or getattr(p, "derived_role", "") for p in dst_cell.inputs.values()}
        role_prog = stage_prog
        if ("source_data" in src_roles or "data_input" in src_roles) and ("feature_input" in dst_roles or "data_input" in dst_roles):
            role_prog += 0.3
        if ("feature_input" in src_roles or "model_input" in src_roles) and ("model_sink" in dst_roles or "data_input" in dst_roles):
            role_prog += 0.3
        role_prog = min(1.0, role_prog)

        # Term 4: Graph Distance (1 / dist)
        adj = getattr(self.orchestrator, "_adjacency", {})
        if dst_cell.cell_id in adj.get(src_cell.cell_id, ()):
            inv_dist = 1.0
        else:
            two_hop = any(dst_cell.cell_id in adj.get(mid, ()) for mid in adj.get(src_cell.cell_id, ()))
            inv_dist = 0.5 if two_hop else 0.2

        return (w1 * ast_aff) + (w2 * rel) + (w3 * role_prog) + (w4 * inv_dist)

    def plan(
        self,
        prompt: str,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        max_transforms: int = 6
    ) -> List[Cell]:
        """
        Plans a type-valid compositional pipeline:
          [Entry (Stage 1)] -> [Transforms / Bridges (Stage 2)]* -> [Terminal (Stage 3)]
        Conforms strictly to Sections 3.1, 3.2, and 3.4 of the NSTL paper:
        Viterbi Trellis Dynamic Programming over the typed category G|_T.
        Monadic Unification Gate rejects any invalid edge proposals.

        Goal-directed extensions (Section 3.4 objective):
        - Terminal bias: paths terminating at a Stage-3 egress morphism are
          preferred whenever the tunnel declares sinks (dataflow must land).
        - Asset absorption: file-asset literals extracted from the prompt must be
          consumed by a path-typed input port somewhere on the path.
        - Wildcarrier penalty: transitions whose producer output carrier is the
          untyped wildcard are weak evidence and are penalized, so fully-untyped
          utility morphisms never beat typed equivalents.
        - Zero-ary constructor morphisms (node_type 'constructor') are insertable
          mid-chain to complete instance lifecycles (construct -> fit -> score).
        """
        self.current_relevance_map = dict(relevance_map or {})
        if not tunnel:
            return []

        # Single standalone node case
        if len(tunnel) == 1:
            return [tunnel[0]]

        # Candidate pool excluding constants
        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"]
        if not candidates:
            return [tunnel[0]]

        # Compute log-likelihood log P(v | e_x) from tunnel relevance
        log_probs: Dict[str, float] = {}
        for c in candidates:
            p = max(relevance_map.get(c.cell_id, 0.0), 1e-6)
            log_probs[c.cell_id] = math.log(p)

        # Set NSTL_DEBUG_PLAN=1 to dump the ranked candidate paths after planning.
        _debug_plan = os.environ.get("NSTL_DEBUG_PLAN", "") in ("1", "true", "True", "on")
        _component_trace: Dict[Tuple[str, ...], Dict[str, float]] = {}

        # ---- Goal-directed objective data (all derived from DECLARED structure) ----
        registry = TypeRegistry.get_instance()
        l0_extracted_literals = ExecutionContext._extract_universal_literals(prompt or "")
        universal_literals = [
            (kind, val) for _, kind, val in l0_extracted_literals
            if kind in ("file_asset", "identifier", "quoted_str")
        ]
        literal_positions = {
            (kind, val): pos
            for pos, kind, val in l0_extracted_literals
            if kind in ("file_asset", "identifier", "quoted_str")
        }
        file_literals = [
            v for kind, v in universal_literals
            if kind == "file_asset"
        ]

        def _has_path_port(cell: Cell) -> bool:
            for p_name, p_sig in cell.inputs.items():
                if _is_path_port_name(p_name, p_sig):
                    return True
            return False

        def _is_path_port_name(p_name: str, p_sig: Any) -> bool:
            return _lattice_is_path_port(p_sig)

        def _is_path_port_sig(p_sig: Any) -> bool:
            sig = getattr(p_sig, "signature", p_sig)
            return _is_path_port_name(str(getattr(p_sig, "name", "") or ""), sig)

        # Numeric literals extracted from prompt
        numeric_literals = [
            v for _, t, v in ExecutionContext._extract_universal_literals(prompt or "")
            if t in ("numeric", "int", "float", "number")
        ]

        def _can_be_entry_source(cell: Cell) -> bool:
            for p_name, p_sig in cell.inputs.items():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
                if _lattice_is_path_port(p_sig):
                    if not file_literals:
                        return False
                    continue
                is_num = (
                    registry.is_subtype(t_name, "numeric")
                    or registry.is_subtype(t_name, "int")
                    or registry.is_subtype(t_name, "float")
                    or t_name in ("int", "float", "number", "dim_size")
                )
                if is_num:
                    if not numeric_literals:
                        return False
                    continue
                if not (
                    registry.is_subtype(t_name, "str")
                    or registry.is_subtype(t_name, "bool")
                    or registry.is_subtype(t_name, "scalar")
                ):
                    return False
            return True

        tunnel_has_sinks = any(getattr(c, "stage", None) == 3 for c in candidates)
        tunnel_absorbs_assets = any(
            getattr(c, "stage", None) == 1 and _has_path_port(c) for c in candidates
        ) if file_literals else False

        # Candidate starting cells (Stage 1 sources or cells matching start_sig)
        candidate_entries = list(candidates)
        if start_sig is not None:
            s_sig = start_sig.signature if hasattr(start_sig, "signature") else start_sig
            matching = [c for c in candidate_entries if unify(s_sig, c.primary_input.signature) is not None]
            if matching:
                candidate_entries = matching
        else:
            viable_entries = [c for c in candidate_entries if _can_be_entry_source(c)]
            if file_literals:
                s1_entries = [c for c in viable_entries if getattr(c, "stage", None) == 1 and _has_path_port(c)]
                if s1_entries:
                    first_clause = re.split(r'[,;]|\b(?:and|then)\b', prompt.strip())[0].strip()
                    clause_tokens = (CellTokenizer.tokenize_prompt(first_clause) if first_clause else set()) - STOPWORDS
                    if clause_tokens:
                        matching_s1 = [c for c in s1_entries if len(clause_tokens & (c.token_set - STOPWORDS)) > 0]
                        if matching_s1:
                            s1_entries = matching_s1
                    s1_entries.sort(key=lambda c: (
                        getattr(c, "source_priority", 100),
                        -(relevance_map.get(c.cell_id, 0.0) * (1.0 + (len(clause_tokens & (c.token_set - STOPWORDS)) if clause_tokens else 0)))
                    ))
                    candidate_entries = s1_entries[:30]
                else:
                    viable_entries.sort(key=lambda c: -relevance_map.get(c.cell_id, 0.0))
                    candidate_entries = viable_entries[:30]
            else:
                viable_entries.sort(key=lambda c: -relevance_map.get(c.cell_id, 0.0))
                candidate_entries = viable_entries[:30]

        # Clauses and tokens for sequential alignment and concept coverage
        clauses = _segment_prompt_clauses(prompt)
        clause_tokens_list = [(CellTokenizer.tokenize_prompt(cl) - STOPWORDS) for cl in clauses]
        clause_tokens_list = [t for t in clause_tokens_list if t]
        content_prompt_tokens = set().union(*clause_tokens_list) if clause_tokens_list else ((CellTokenizer.tokenize_prompt(prompt) if prompt else set()) - STOPWORDS)
        p_len = max(len(content_prompt_tokens), 1)
        num_clauses = max(len(clause_tokens_list), 1)

        # Corpus-derived IDF over prompt tokens (df from the lattice token index).
        # Used to weight coverage and clause alignment: generic tokens ('data',
        # 'column') carry near-zero objective mass, while discriminative tokens
        # ('csv', 'train', 'split') dominate. Trivial duplicate clauses ("Y
        # column") can then neither inflate the clause count nor be farmed by
        # cells that merely share generic vocabulary.
        token_index_for_idf = getattr(self.orchestrator, "token_index", None) or {}
        corpus_size_for_idf = max(len(self.orchestrator.loaded_cells), 1)

        def _idf(tok: str) -> float:
            df = len(token_index_for_idf.get(tok, ()))
            return math.log(1.0 + (corpus_size_for_idf + 1) / (df + 1.0))

        idf_of_prompt = {tok: _idf(tok) for tok in content_prompt_tokens}
        total_prompt_idf = sum(idf_of_prompt.values()) or 1.0
        clause_weights = [sum(_idf(t) for t in cl_toks) for cl_toks in clause_tokens_list]
        total_clause_weight = sum(clause_weights) or 1.0

        # Token provenance weighting: a cell's IDENTITY tokens (cell_id + declared
        # keywords) describe what it IS; its docstring prose merely describes what
        # it says. Identity matches carry full idf mass, docstring matches are
        # down-weighted — descriptive vocabulary (e.g. a splitter's docstring
        # mentioning "train/test") must not outrank another cell's identifier.
        def _identity_tokens(cell: Cell) -> Set[str]:
            toks = CellTokenizer.tokenize_identifier(cell.cell_id)
            for kw in getattr(cell, "keywords", ()) or ():
                toks.update(CellTokenizer.tokenize_identifier(kw))
            return toks

        identity_cache: Dict[str, Set[str]] = {}
        for c in candidates:
            identity_cache[c.cell_id] = _identity_tokens(c)

        def _match_mass(cl_toks: Set[str], c_toks: Set[str], id_toks: Set[str]) -> float:
            strong = sum(idf_of_prompt.get(t, _idf(t)) for t in (cl_toks & (c_toks & id_toks)))
            weak = sum(idf_of_prompt.get(t, _idf(t)) for t in (cl_toks & (c_toks - id_toks)))
            return strong + 0.3 * weak

        def _is_wildcarrier(cell: Cell) -> bool:
            t = str(getattr(getattr(cell, "primary_output", None), "type_name", "") or "").lower()
            return t in _WILDCARD_CARRIERS

        # ---- Precomputed per-cell scoring tables (make path scoring O(k)) ----
        # clause_mass[cell]: per-clause match masses, plus the covered-clause set.
        # coverage is computed on the UNION of path token matches (see below).
        cell_cov_strong: Dict[str, Set[str]] = {}
        cell_cov_weak: Dict[str, Set[str]] = {}
        cell_cov_mass_bonus: Dict[str, float] = {}
        cell_clause_mass: Dict[str, List[float]] = {}
        cell_covered: Dict[str, Set[int]] = {}
        for c in candidates:
            c_toks = c.token_set - STOPWORDS
            id_toks = identity_cache.get(c.cell_id, c_toks)
            cell_cov_strong[c.cell_id] = content_prompt_tokens & (c_toks & id_toks)
            cell_cov_weak[c.cell_id] = content_prompt_tokens & (c_toks - id_toks)
            cell_cov_mass_bonus[c.cell_id] = 10.0 * (
                sum(idf_of_prompt.get(t, _idf(t)) for t in cell_cov_strong[c.cell_id])
                + 0.3 * sum(idf_of_prompt.get(t, _idf(t)) for t in cell_cov_weak[c.cell_id])
            ) / total_prompt_idf
            masses: List[float] = []
            covered: Set[int] = set()
            for cl_toks in clause_tokens_list:
                m = _match_mass(cl_toks, c_toks, id_toks)
                masses.append(m)
                # Intent coverage: With empirical edge affinity dominating path selection,
                # the legacy id_mass clamp is replaced with calibrated clause matching.
                cl_idx = len(masses) - 1
                if m >= 0.15 * clause_weights[cl_idx]:
                    covered.add(cl_idx)
            cell_clause_mass[c.cell_id] = masses
            cell_covered[c.cell_id] = covered

        # Macro-goal expansion tables: a MacroCell is scored by its EXPANDED
        # sub-cell composition, with the SAME objective as flat paths (coverage,
        # clause alignment, parsimony evidence). This is what lets a macro
        # outrank its own micro-cells legitimately — its expansion explains the
        # prompt at least as well as the constituents do, plus whatever
        # composite concept vocabulary only the macro declares — and never lets
        # it win on generic I/O vocabulary alone (the expansion exposes the
        # sub-path's uncovered prompt tokens just as honestly).
        macro_expansion: Dict[str, List[Cell]] = {}
        for c in candidates:
            if getattr(c, "cell_type", "") != "macro" or not getattr(c, "sub_cells", None):
                continue
            subs = []
            for sid in c.sub_cells:
                sub = self.orchestrator.loaded_cells.get(sid) if self.orchestrator else None
                if sub is not None:
                    subs.append(sub)
                else:
                    subs = []
                    break
            if len(subs) >= 2:
                macro_expansion[c.cell_id] = subs

        def _expansion_ids(c_id: str) -> List[str]:
            subs = macro_expansion.get(c_id)
            return [s.cell_id for s in subs] if subs else [c_id]

        # Expanded coverage tables: for a macro, union its sub-cells' coverage.
        # Sub-cells may lie outside the tunnel candidates, so their coverage is
        # computed here directly with the same formulas as the main table.
        for c_id, subs in macro_expansion.items():
            strong: Set[str] = set()
            weak: Set[str] = set()
            mass_bonus = 0.0
            masses: List[float] = []
            covered: Set[int] = set()
            for sub in subs:
                s_id = sub.cell_id
                if s_id in cell_cov_strong:
                    strong |= cell_cov_strong[s_id]
                    weak |= cell_cov_weak.get(s_id, set())
                    mass_bonus += cell_cov_mass_bonus.get(s_id, 0.0)
                    for gi, m in enumerate(cell_clause_mass.get(s_id, [])):
                        if len(masses) <= gi:
                            masses.append(0.0)
                        masses[gi] = max(masses[gi], m)
                    covered |= cell_covered.get(s_id, set())
                    continue
                s_toks = sub.token_set - STOPWORDS
                s_id_toks = identity_cache.get(s_id)
                if s_id_toks is None:
                    s_id_toks = _identity_tokens(sub)
                s_strong = content_prompt_tokens & (s_toks & s_id_toks)
                s_weak = content_prompt_tokens & (s_toks - s_id_toks)
                strong |= s_strong
                weak |= s_weak
                mass_bonus += 10.0 * (
                    sum(idf_of_prompt.get(t, _idf(t)) for t in s_strong)
                    + 0.3 * sum(idf_of_prompt.get(t, _idf(t)) for t in s_weak)
                ) / total_prompt_idf
                for gi, cl_toks in enumerate(clause_tokens_list):
                    m = _match_mass(cl_toks, s_toks, s_id_toks)
                    if len(masses) <= gi:
                        masses.append(0.0)
                    masses[gi] = max(masses[gi], m)
                    if m >= 0.15 * clause_weights[gi]:
                        covered.add(gi)
            cell_cov_strong[c_id] = strong
            cell_cov_weak[c_id] = weak
            cell_cov_mass_bonus[c_id] = mass_bonus
            cell_clause_mass[c_id] = masses
            cell_covered[c_id] = covered

        # Concrete (non-wildcard) input port signatures per cell — used to judge
        # whether a zero-ary constructor is actually CONSUMED downstream.
        cell_concrete_in_sigs: Dict[str, List[Any]] = {}
        for c in candidates:
            sigs = [
                p.signature for p in c.inputs.values()
                if str(p.signature.type_name).lower() not in _WILDCARD_CARRIERS
            ]
            cell_concrete_in_sigs[c.cell_id] = sigs

        def _ctor_justified(ctor: Cell, nxt: Cell) -> bool:
            """A zero-ary constructor is justified iff the following cell declares a
            concrete port that unifies with the constructed type (real lifecycle)."""
            out_sig = ctor.primary_output.signature
            for p_sig in cell_concrete_in_sigs.get(nxt.cell_id, ()):
                if unify(out_sig, p_sig) is not None:
                    return True
            return False

        # Receiver-bindability: a required concrete port whose carrier is a domain
        # class must be produced somewhere earlier in the path (typically by the
        # class's constructor morphism) or by the environment's start signature.
        # Ports whose declared carrier is literal-groundable are exempt.
        def _port_literal_groundable(type_name: str) -> bool:
            t = type_name.lower()
            return (
                registry.is_subtype(t, "str")
                or registry.is_subtype(t, "numeric")
                or registry.is_subtype(t, "bool")
                or registry.is_subtype(t, "filepath")
                or registry.is_subtype(t, "uri")
                or registry.is_subtype(t, "scalar")
            )

        # Per-cell: required concrete non-groundable receiver signatures (the ports
        # that need an in-path producer such as a constructor morphism).
        cell_receiver_sigs: Dict[str, List[Any]] = {}
        for c in candidates:
            if getattr(c, "node_type", "") == "constructor":
                cell_receiver_sigs[c.cell_id] = []
                continue
            sigs = []
            for p_name, p_sig in c.inputs.items():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                desc = getattr(p_sig, "description", None) or getattr(p_sig, "doc", "") or ""
                declared_role = getattr(p_sig, "port_role", None) or ""
                is_instance_receiver = (
                    declared_role == "receiver"
                    or (not declared_role and (p_name in ("data", "self") or "receiver" in str(desc).lower()))
                )
                t = str(p_sig.signature.type_name)
                if not is_instance_receiver:
                    if t.lower() in _WILDCARD_CARRIERS or _port_literal_groundable(t):
                        continue
                sigs.append(p_sig.signature)
            cell_receiver_sigs[c.cell_id] = sigs

        def _is_col_proj_cell(cell: Cell) -> bool:
            # Structural: a cell is a column projection iff it DECLARES a
            # column-projection input port (state / list-carrier vocabulary).
            # No cell-id substrings.
            return any(_is_col_projection_port(p_s) for p_s in cell.inputs.values())

        def _new_unbindable(cand: Cell, prev_path: List[Cell]) -> int:
            produced = [out_sig.signature for prev in prev_path for out_sig in prev.outputs.values()]
            count = 0
            for p_sig in cell_receiver_sigs.get(cand.cell_id, ()):
                if not any(unify(prod, p_sig) is not None for prod in produced):
                    count += 1

            # Path port capacity check: required path inputs across the entire pipeline
            # must not exceed the available file literals from the prompt (unless produced in-pipeline).
            full_path = prev_path + [cand]
            required_path_ports = 0
            for c in full_path:
                for p_name, p_sig in c.inputs.items():
                    if not p_sig.required or p_sig.default_value is not None:
                        continue
                    t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
                    if _lattice_is_path_port(p_sig):
                        required_path_ports += 1
                        break
            if required_path_ports > len(file_literals):
                count += (required_path_ports - len(file_literals))

            # Role-carrier tracking (R2.5):
            # Check role-bearing inputs (e.g. target_input, feature_input, data_input)
            ROLE_CARRIERS = TypeRegistry.get_instance().get_declared_role_carriers()
            matched_producers: Set[Tuple[int, str]] = set()

            for p_name, p_sig in cand.inputs.items():
                p_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                if not p_role and hasattr(p_sig, "derive_port_role"):
                    p_role = p_sig.derive_port_role()

                if p_role in ROLE_CARRIERS:
                    t_name = str(p_sig.signature.type_name)
                    def_val = p_sig.default_value
                    if def_val is not None and str(def_val).strip() not in ("None", "none", "null", ""):
                        continue
                    if _port_literal_groundable(t_name):
                        continue

                    # Search for an unallocated matching producer output in prev_path (newest to oldest)
                    found_out = None
                    for idx in range(len(prev_path) - 1, -1, -1):
                        prev = prev_path[idx]
                        for out_name, out_sig in prev.outputs.items():
                            if (idx, out_name) in matched_producers:
                                continue
                            if unify(out_sig.signature, p_sig.signature) is not None:
                                found_out = (idx, out_name)
                                break
                        if found_out is not None:
                            matched_producers.add(found_out)
                            break

                    if found_out is None:
                        already_penalized = (
                            p_sig.required
                            and p_sig.default_value is None
                            and p_sig.signature in cell_receiver_sigs.get(cand.cell_id, ())
                            and not any(unify(prod, p_sig.signature) is not None for prod in produced)
                        )
                        if not already_penalized:
                            if p_role == "target_input":
                                count += 1
                            elif not p_sig.required:
                                if p_name in getattr(cand, "slots", []) and f"{{{p_name}}}" in getattr(cand, "code_template", ""):
                                    count += 1

            return count

        def _edge_is_weak(prev: Cell, cand: Cell) -> bool:
            out_sig = prev.primary_output.signature
            has_strong = False
            has_any = False
            for p in cand.inputs.values():
                if unify(out_sig, p.signature) is None:
                    continue
                t = str(p.signature.type_name).lower()
                if t in _WILDCARD_CARRIERS:
                    has_any = True
                else:
                    has_strong = True
                    break
            return (not has_strong) and has_any

        _edge_affinity = self._calculate_edge_affinity

        def compute_path_score(item: Tuple[List[Cell], Substitution, float, int, int], is_final: bool = False) -> float:
            path, _, sc, weak_edges, unbindable = item
            k = len(path)

            # Prompt token coverage across the composition path:
            # Edge affinity dominates path ranking, so token coverage cleanly
            # reflects path coverage without artificial multi-count clamps.
            strong_tokens: Set[str] = set()
            weak_tokens: Set[str] = set()
            for c in path:
                strong_tokens |= cell_cov_strong.get(c.cell_id, set())
                weak_tokens |= cell_cov_weak.get(c.cell_id, set())
            coverage = (
                sum(idf_of_prompt.get(t, _idf(t)) for t in strong_tokens)
                + 0.5 * sum(idf_of_prompt.get(t, _idf(t)) for t in (weak_tokens - strong_tokens))
            ) / total_prompt_idf

            # Clause coverage is a SET relation: a cell whose declared vocabulary
            # intersects several clauses covers all of them (a partitioning cell
            # covers both the 'train' and the 'test/allocate' clause), while the
            # monotonic best-match chain preserves sequential ordering.
            # Clause coverage and monotonic alignment:
            # A cell covering multiple clauses (e.g. train_test_split covering both
            # data partitioning and training preparation) can validly align with any
            # clause where it holds significant mass. Monotonic alignment DP finds the
            # assignment of cells to candidate clauses that minimizes sequence inversions.
            covered_clauses: Set[int] = set()
            steps_candidate_clauses: List[Set[int]] = []
            for c in path:
                covered_clauses |= cell_covered.get(c.cell_id, set())
                masses = cell_clause_mass.get(c.cell_id, [])
                if not masses:
                    continue
                max_m = max(masses)
                if max_m <= 0:
                    continue
                cands = {g for g, m in enumerate(masses) if m >= 0.10 * max_m and m > 0}
                if cands:
                    steps_candidate_clauses.append(cands)

            distinct_matched_clauses = len(covered_clauses)
            matched_clause_weight = sum(clause_weights[i] for i in covered_clauses)
            clause_cov = matched_clause_weight / total_clause_weight

            inversions = 0
            if len(steps_candidate_clauses) >= 2:
                dp = {g: 0 for g in steps_candidate_clauses[0]}
                for step_cands in steps_candidate_clauses[1:]:
                    next_dp = {}
                    for g in step_cands:
                        next_dp[g] = min(dp[prev_g] + (1 if prev_g > g else 0) for prev_g in dp)
                    dp = next_dp
                inversions = min(dp.values())
            # Structural hole: an UNCOVERED clause sandwiched between covered ones
            # means the path skipped an intermediate intent stage — the connector
            # between two satisfied sub-goals is missing.
            gap_penalty = 0.0
            if covered_clauses:
                ordered = sorted(covered_clauses)
                lo, hi = ordered[0], ordered[-1]
                holes = sum(1 for g in range(lo, hi + 1) if g not in covered_clauses)
                gap_penalty = holes  # weighted by PATH_SCORE_WEIGHTS["gap"] at combination

            align_factor = max(0.85, 1.0 - 0.05 * inversions)
            alignment = clause_cov * align_factor

            # Parsimony: every step must pay for itself. A per-step cost makes
            # chains of same-domain endomorphisms (DataFrame -> DataFrame utility
            # hops, which all type-check) unattractive unless each hop covers a
            # new clause; the excess penalty handles structural padding on top.
            consumed_ctors = sum(
                1 for i, c in enumerate(path)
                if getattr(c, "node_type", "") == "constructor"
                and any(
                    unify(c.primary_output.signature, p.signature) is not None
                    for downstream_cell in path[i + 1:]
                    for p in downstream_cell.inputs.values()
                )
            )
            # Prerequisite bridging morphisms: essential bridges whose predecessor
            # cannot directly connect to its successor without this carrier conversion.
            prerequisite_bridges = 0
            for i in range(1, k - 1):
                c = path[i]
                if not cell_covered.get(c.cell_id):
                    pred = path[i - 1]
                    succ = path[i + 1]
                    pred_out = pred.primary_output.signature
                    direct_compat = any(
                        unify(pred_out, p.signature) is not None
                        for p in succ.inputs.values()
                    )
                    if not direct_compat:
                        is_consumed = any(
                            unify(c.primary_output.signature, p.signature) is not None
                            for p in succ.inputs.values()
                        )
                        if is_consumed:
                            prerequisite_bridges += 1

            # Parsimony measures PLANNING cost: a macro-goal is genuinely one
            # beam step (its expansion is a declared, pre-verified composition),
            # so k counts planned nodes, not expansion internals. Honesty about
            # what the expansion contributes is enforced on the COVERAGE side
            # (expansion-based coverage + dead-expansion-step penalty below).
            effective_k = max(1, k - consumed_ctors - prerequisite_bridges)
            excess_steps = max(0, effective_k - max(distinct_matched_clauses, 1))
            parsimony_penalty = (excess_steps * PATH_SCORE_WEIGHTS["parsimony_step"]
                                 + effective_k * PATH_SCORE_WEIGHTS["parsimony_base"])

            # Tunnel likelihood is a WEAK tiebreaker for flat cells: the
            # group-relative softmax already guarantees every surviving cell is
            # within the same likelihood window of its clause's maximum, so the
            # raw log-prob gap between clause-rank-1 cells and correct-but-rank-2
            # cells is distribution noise, not semantic evidence. Full-weight
            # log-probs structurally prefer whichever garbage cell happened to
            # rank #1 in a weak clause. A promoted macro-goal cell is the
            # exception: its relevance is a deliberate router verdict (it cleared
            # the concept-coverage gate and outranks its own micro-cells), so it
            # carries meaningful weight instead of distribution noise.
            if k == 1 and getattr(path[0], "cell_type", "") == "macro":
                mean_log_prob = PATH_SCORE_WEIGHTS["macro_log_prob_weight"] * (sc / max(k, 1))
            else:
                mean_log_prob = PATH_SCORE_WEIGHTS["flat_log_prob_weight"] * (sc / max(k, 1))

            # Goal-directed terms
            goal_bonus = 0.0
            # Terminal-sink preference (Section 3.4 objective) fires ONLY when the
            # prompt implies materialization: file-asset literals exist and the
            # terminal declares a path-typed port to receive one. Without an
            # egress intent, a Stage-2 ending (trained model, drawn image, final
            # value) is an equally complete dataflow, and a blanket sink bonus
            # merely rewards whatever compute function old data mislabeled as a
            # sink.
            terminal = path[-1]
            has_egress_intent = bool(content_prompt_tokens & EGRESS_INTENT_TOKENS) or (
                len(file_literals) > 1 and any(getattr(c, "stage", None) == 1 for c in path)
            ) or (goal_sig is not None)
            # Materialization is a property of the terminal's OUTPUT STATE, not
            # merely of its stage field: composite macro-goal cells inherit the
            # stage of their first (ingress) sub-cell, yet their output declares
            # egress (written_to_disk / saved / exported / ...) because their
            # final sub-cell is a sink. Mirrors the egress-derivation contract
            # in UnificationGate._derive_egress_paths.
            output_declares_materialization = any(
                str(getattr(out_p, "state", "")).lower()
                in MATERIALIZATION_OUTPUT_STATES
                for out_p in getattr(terminal, "outputs", {}).values()
            )
            terminal_is_materializing = (
                (terminal.stage == 3 or output_declares_materialization)
                and has_egress_intent
                and unbindable == 0
                and (
                    not file_literals
                    or _has_path_port(terminal)
                )
            )
            if tunnel_has_sinks:
                if terminal_is_materializing:
                    goal_bonus += PATH_SCORE_WEIGHTS["sink_bonus"]
                elif has_egress_intent and not terminal_is_materializing:
                    goal_bonus -= PATH_SCORE_WEIGHTS["non_sink_penalty"]
            # Domain coherence: the terminal morphism should belong to the
            # pipeline's own declared domains. A terminal imported from a foreign
            # domain (e.g. a plotting library bolted onto a vision pipeline)
            # exists to harvest bonuses — it pays a coherence tax, while a
            # home-domain terminal earns one. Composite macro-goal terminals are
            # coherent with the domains of their EXPANDED composition: their
            # sub-cells are the pipeline the macro stands for.
            terminal_domain = getattr(terminal, "domain_name", "")
            if terminal_domain:
                effective_other_cells: List[Cell] = list(path[:-1])
                if getattr(terminal, "cell_type", "") == "macro" and getattr(terminal, "sub_cells", None) and self.orchestrator is not None:
                    for sid in terminal.sub_cells:
                        sub_cell = self.orchestrator.loaded_cells.get(sid)
                        if sub_cell is not None:
                            effective_other_cells.append(sub_cell)
                other_domains = {getattr(c, "domain_name", "") for c in effective_other_cells}
                if terminal_domain not in other_domains:
                    goal_bonus -= PATH_SCORE_WEIGHTS["domain_coherence"]
                else:
                    goal_bonus += PATH_SCORE_WEIGHTS["domain_coherence"]

            if file_literals:
                if any(_has_path_port(c) for c in path):
                    goal_bonus += PATH_SCORE_WEIGHTS["file_port_bonus"]
                elif not tunnel_absorbs_assets:
                    goal_bonus -= PATH_SCORE_WEIGHTS["file_port_penalty"]
            weak_total = (sum(1 for c in path if _is_wildcarrier(c)) * PATH_SCORE_WEIGHTS["wildcarrier"]
                          + weak_edges * PATH_SCORE_WEIGHTS["weak_edge"])

            # Dead expansion steps: a MIDDLE step of a macro expansion that
            # matches NONE of the prompt's tokens is dead weight for THIS
            # prompt — the macro's known-good path includes it for other
            # intents. It pays so the macro cannot ride its mined sub-pair
            # affinities through prompts its expansion does not serve.
            # (Ingress/egress steps are exempt: they may match nothing on
            # prompts that express no I/O vocabulary.)
            dead_expansion_steps = 0
            for c in path:
                subs = macro_expansion.get(c.cell_id)
                if not subs:
                    continue
                for j, sub in enumerate(subs):
                    if j == 0 or j == len(subs) - 1:
                        continue
                    s_strong = cell_cov_strong.get(sub.cell_id, set())
                    s_weak = cell_cov_weak.get(sub.cell_id, set())
                    if not s_strong and not s_weak:
                        # Prerequisite bridging morphism check: if predecessor cannot directly
                        # connect to successor without this step, it is an essential type/state bridge.
                        pred = subs[j - 1]
                        succ = subs[j + 1]
                        pred_out = getattr(pred, "primary_output", None)
                        succ_in = getattr(succ, "primary_input", None)
                        pred_sig = getattr(pred_out, "signature", pred_out)
                        succ_sig = getattr(succ_in, "signature", succ_in)
                        if pred_sig and succ_sig and unify(pred_sig, succ_sig) is None:
                            continue
                        dead_expansion_steps += 1

            # Dead-constructor penalty: a constructor whose output is not
            # consumed by ANY downstream cell is dead code inserted purely
            # to harvest coverage tokens — it must pay heavily.
            dead_ctors = 0
            for i, c in enumerate(path):
                if getattr(c, "node_type", "") == "constructor":
                    if not is_final and i == len(path) - 1:
                        continue  # newly instantiated constructor at beam tip awaiting downstream receiver
                    out_sig = c.primary_output.signature
                    consumed = any(
                        unify(out_sig, p.signature) is not None
                        for downstream_cell in path[i + 1:]
                        for p in downstream_cell.inputs.values()
                    )
                    if not consumed:
                        dead_ctors += 1

            # Domain dispersion: pipelines should be domain-coherent. While 1 or 2
            # cooperating domains (e.g. pandas + sklearn) are common, gratuitous domain
            # hopping (e.g. inserting cv2 or nltk into tabular data pipelines) pays
            # a dispersion penalty per foreign domain.
            pipeline_domains = {
                getattr(c, "domain_name", "")
                for c in path
                if getattr(c, "domain_name", "") and getattr(c, "domain_name", "") not in ("generic", "python_core", "builtins")
            }
            domain_dispersion = max(0, len(pipeline_domains) - 2) * PATH_SCORE_WEIGHTS["dispersion"]

            # Edge affinity term: AST-mined and declared topological transitions
            # are the dominant score term for idiomatic composition.
            if k <= 1:
                affinity_score = 0.5
                # A single macro-goal cell embodies a verified internal
                # composition: it earns the affinity evidence of its own
                # sub-cell sequence (declared edges, egress completion,
                # adjacency, same-domain continuity), floored at the
                # single-step baseline because every internal transition was
                # type-verified at macro definition/harvest time. DEAD steps
                # (middle sub-cells matching zero prompt tokens) forfeit the
                # affinity evidence of the pairs they participate in — mined
                # idioms around a step this prompt never asked for are not
                # evidence for THIS composition.
                if k == 1 and getattr(path[0], "cell_type", "") == "macro" and getattr(path[0], "sub_cells", None) and self.orchestrator is not None:
                    subs = [self.orchestrator.loaded_cells.get(sid) for sid in path[0].sub_cells]
                    subs = [s for s in subs if s is not None]
                    if len(subs) >= 2:
                        def _alive(idx: int) -> bool:
                            if idx == 0 or idx == len(subs) - 1:
                                return True
                            s_id = subs[idx].cell_id
                            return bool(cell_cov_strong.get(s_id) or cell_cov_weak.get(s_id))
                        affs = [
                            max(self._calculate_edge_affinity(subs[i], subs[i + 1]), PATH_SCORE_WEIGHTS["macro_affinity_floor"])
                            for i in range(len(subs) - 1)
                            if _alive(i) and _alive(i + 1)
                        ]
                        affinity_score = sum(affs) / len(affs) if affs else 0.5
            else:
                total_aff = sum(_edge_affinity(path[i], path[i + 1]) for i in range(k - 1))
                affinity_score = total_aff / max(k - 1, 1)

            # Intent deficit objective: penalize incomplete pipelines that abandon requested clauses
            intent_deficit = ((PATH_SCORE_WEIGHTS["intent_deficit_final"] if is_final
                               else PATH_SCORE_WEIGHTS["intent_deficit_partial"])
                              * max(0.0, 1.0 - clause_cov)) if num_clauses > 1 else 0.0

            # Literal consumption term (T1.3 & R2.7):
            # Measures ratio of L0 universal literals bound to at least one port on the path.
            if universal_literals:
                path_ports = set()
                path_file_ports = 0
                path_has_col_proj = any(_is_col_proj_cell(c) for c in path)
                for c in path:
                    for p_name, p_sig in c.inputs.items():
                        path_ports.add(p_name.lower())
                        role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                        if role:
                            path_ports.add(role.lower())
                        # Count each declared required path port: a cell (or a
                        # composite macro-goal with separate ingest/egress path
                        # ports) can absorb as many file literals as it declares
                        # destinations for.
                        if p_sig.required and p_sig.default_value is None and _is_path_port_sig(p_sig):
                            path_file_ports += 1
                    for s_k in getattr(c, "bound_slots", {}).keys():
                        path_ports.add(s_k.lower())

                consumed_count = 0
                used_file_ports = 0
                # Column-projection ports declared on the path: an identifier
                # literal counts as consumed only when its ROLE context (e.g.
                # "X column") matches the projection port's declared vocabulary.
                proj_port_tokens: Set[str] = set()
                if path_has_col_proj:
                    for c in path:
                        for p_sig in c.inputs.values():
                            if _is_col_projection_port(p_sig):
                                st = str(getattr(p_sig.signature, "state", "")).lower()
                                proj_port_tokens |= set(CellTokenizer.tokenize_identifier(st))
                                proj_port_tokens |= {t for t in CellTokenizer.tokenize_identifier(str(getattr(c, "cell_id", "")).lower())}
                # Declared relational-trigger ports: a port NAMED for the
                # preposition that binds its value ("by") can consume the
                # prepositional object ("by age") through the semantic-slot
                # channel. A path with no such port leaves the referent
                # unconsumed — the objective then prefers paths that honor it.
                trigger_port_tokens: Set[str] = set()
                for c in path:
                    for p_name, p_sig in c.inputs.items():
                        p_toks = set(CellTokenizer.tokenize_identifier(p_name))
                        p_toks |= set(CellTokenizer.tokenize_identifier(str(getattr(p_sig.signature, "state", ""))))
                        if p_toks & {"by", "on", "per", "of", "for", "with"}:
                            trigger_port_tokens |= p_toks
                identifier_role_map = getattr(self, "_last_identifier_roles", None)
                for kind, lit in universal_literals:
                    lit_clean = str(lit).lower().strip("'\"")
                    if kind == "file_asset":
                        if used_file_ports < path_file_ports:
                            consumed_count += 1
                            used_file_ports += 1
                    else:
                        if lit_clean in path_ports or any(lit_clean in p for p in path_ports):
                            consumed_count += 1
                        elif proj_port_tokens:
                            lit_role = set()
                            if identifier_role_map:
                                lit_role = set(identifier_role_map.get(str(lit), frozenset()))
                            if (lit_role & proj_port_tokens) or (proj_port_tokens & {t for t in CellTokenizer.tokenize_identifier(lit_clean)}):
                                consumed_count += 1
                        elif trigger_port_tokens:
                            # Prepositional-object referent: consumed iff the
                            # path declares a relational-trigger port whose
                            # binding preposition matches the literal's.
                            lit_pos = literal_positions.get((kind, lit))
                            if lit_pos is not None:
                                prev_chunk = (prompt or "")[:lit_pos].rstrip()
                                import re as _re
                                m_prev = _re.search(r"([A-Za-z]+)$", prev_chunk)
                                prep = m_prev.group(1).lower() if m_prev else ""
                                if prep and prep in trigger_port_tokens:
                                    consumed_count += 1
                literal_consumption = consumed_count / len(universal_literals)
            else:
                literal_consumption = 1.0

            # Dominant score combination: edge affinity (w_aff = 25.0) heavily rewards
            # AST-mined canonical transitions, rendering legacy junk exploit patches obsolete.
            total = (coverage * PATH_SCORE_WEIGHTS["coverage"] + alignment * PATH_SCORE_WEIGHTS["alignment"]
                     + affinity_score * PATH_SCORE_WEIGHTS["affinity"] - parsimony_penalty
                     + mean_log_prob + goal_bonus - weak_total
                     - dead_ctors * PATH_SCORE_WEIGHTS["dead_ctor"]
                     - dead_expansion_steps * PATH_SCORE_WEIGHTS["dead_expansion_step"]
                     - unbindable * PATH_SCORE_WEIGHTS["unbindable"]
                     - domain_dispersion - gap_penalty * PATH_SCORE_WEIGHTS["gap"]
                     - intent_deficit + literal_consumption * PATH_SCORE_WEIGHTS["literal_consumption"])
            if _debug_plan:
                _component_trace[tuple(c.cell_id for c in path)] = {
                    "coverage*10": round(coverage * 10.0, 2),
                    "alignment*10": round(alignment * 10.0, 2),
                    "affinity*25": round(affinity_score * 25.0, 2),
                    "parsimony": round(-parsimony_penalty, 2),
                    "log_prob": round(mean_log_prob, 2),
                    "goal_bonus": round(goal_bonus, 2),
                    "weak": round(-weak_total, 2),
                    "gap": round(-gap_penalty, 2),
                    "intent_deficit": round(-intent_deficit, 2),
                    "literal*15": round(literal_consumption * 15.0, 2),
                }
            return total

        # Type-gated adjacency: index candidates by the DECLARED input carrier they
        # expose. Expansion enumerates distinct port carriers and gates them through
        # the registered poset (mirroring unify's subtyping direction: producer's
        # output carrier must be a subtype of the consumer's port carrier), so each
        # trellis expansion iterates only type-compatible successors.
        cells_by_in_type: Dict[str, List[Cell]] = {}
        distinct_in_states: Dict[str, Set[str]] = {}
        for cand in candidates:
            for p_name, p_sig in cand.inputs.items():
                t_key = str(getattr(p_sig.signature, "type_name", ""))
                cells_by_in_type.setdefault(t_key, []).append(cand)
                distinct_in_states.setdefault(t_key, set()).add(str(getattr(p_sig.signature, "state", "")))

        candidate_map = {c.cell_id: c for c in candidates}
        candidate_map_lower = {c.cell_id.lower(): c for c in candidates}

        def _successors(prev_cell: Cell) -> List[Cell]:
            if (getattr(prev_cell, "node_role", "") == "macro" or getattr(prev_cell, "cell_type", "") == "macro") and getattr(prev_cell, "endable", False):
                return []
            macro_subs = set(getattr(prev_cell, "sub_cells", ()) or ())

            out_sig = prev_cell.primary_output.signature if hasattr(prev_cell.primary_output, "signature") else prev_cell.primary_output
            out_t = str(getattr(out_sig, "type_name", ""))
            out_s = str(getattr(out_sig, "state", ""))

            acc: Dict[str, Cell] = {}

            # Prioritize/include explicit graph edges declared on prev_cell
            for edge in getattr(prev_cell, "edges", []):
                tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                if tgt_id:
                    tgt_cell = candidate_map.get(tgt_id) or candidate_map_lower.get(str(tgt_id).lower())
                    if tgt_cell:
                        acc.setdefault(tgt_cell.cell_id, tgt_cell)

            # Generic carriers ("Sequence[T]", products) and TYPE VARIABLES ("T",
            # "S") unify by binding, not by poset subtyping: enumerate all
            # candidates and let exact unification gate them.
            is_type_var = out_t.isalpha() and len(out_t) == 1 and out_t.isupper()
            if "[" in out_t or is_type_var:
                for cand in candidates:
                    if cand.cell_id in macro_subs or prev_cell.cell_id in getattr(cand, "sub_cells", ()):
                        continue
                    if any(unify(out_sig, p_sig.signature) is not None
                           for p_sig in cand.inputs.values()):
                        acc.setdefault(cand.cell_id, cand)
                res_cells = [c for c in acc.values() if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
                return sorted(res_cells, key=lambda c: _edge_affinity(prev_cell, c), reverse=True)

            for in_t, cell_list in cells_by_in_type.items():
                if not registry.is_subtype(out_t, in_t):
                    continue
                for c in cell_list:
                    if c.cell_id in macro_subs or prev_cell.cell_id in getattr(c, "sub_cells", ()):
                        continue
                    # Typestate compatibility with accepted_states and parent_state walking
                    if any(registry.is_state_compatible(
                        producer_state=out_s,
                        consumer_state=getattr(p_sig.signature, "state", "any"),
                        producer_accepted=getattr(out_sig, "accepted_states", frozenset()),
                        consumer_accepted=getattr(p_sig.signature, "accepted_states", frozenset()),
                        consumer_parent=getattr(p_sig.signature, "parent_state", None),
                        producer_parent=getattr(out_sig, "parent_state", None),
                    ) for p_sig in c.inputs.values()):
                        acc.setdefault(c.cell_id, c)
            res_cells = [c for c in acc.values() if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
            return sorted(res_cells, key=lambda c: _edge_affinity(prev_cell, c), reverse=True)

        # Viterbi Trellis: paths of length t = 1 ... T_max
        # Item layout: (path, sigma, cumulative log-prob, weak_edge_count, unbindable_count)
        all_valid_paths: List[Tuple[List[Cell], Substitution, float, int, int]] = []

        # Step t = 1: Initialize beam
        current_beam: List[Tuple[List[Cell], Substitution, float, int, int]] = []
        for entry in candidate_entries:
            sc = log_probs.get(entry.cell_id, -10.0)
            p_tuple = ([entry], Substitution(), sc, 0, _new_unbindable(entry, []))
            current_beam.append(p_tuple)
            all_valid_paths.append(p_tuple)

        # Edge compatibility cache: signature-level gate results are sigma-invariant
        # for ground signatures; cell-pair results are cached per (pair, sigma fingerprint).
        edge_compat_cache: Dict[Tuple[str, str, str], Optional[Substitution]] = {}

        def _sigma_fingerprint(sigma: Substitution) -> str:
            try:
                return tuple(sorted((k, str(v)) for k, v in sigma.mappings.items()))
            except Exception as e:
                return ()

        def _required_ports_bindable(cand: Cell, prev_path: List[Cell], sigma: Substitution) -> Optional[Substitution]:
            """All required ports must be wire-satisfiable from the path or literal-groundable."""
            sub = sigma
            for p_name, p_sig in cand.inputs.items():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                satisfied = False
                for earlier_cell in reversed(prev_path):
                    for out_name, out_sig in earlier_cell.outputs.items():
                        s_wire = unify(out_sig.signature, p_sig.signature, sub)
                        if s_wire is not None:
                            sub = s_wire
                            satisfied = True
                            break
                    if satisfied:
                        break
                if not satisfied:
                    desc = getattr(p_sig, "description", None) or getattr(p_sig, "doc", "") or ""
                    is_instance_receiver = p_name in ("data", "self") or ("receiver" in str(desc).lower())
                    if not is_instance_receiver:
                        t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
                        if (
                            registry.is_subtype(t_name, "str")
                            or registry.is_subtype(t_name, "numeric")
                            or registry.is_subtype(t_name, "bool")
                            or registry.is_subtype(t_name, "filepath")
                            or registry.is_subtype(t_name, "uri")
                            or registry.is_subtype(t_name, "scalar")
                            or bool(getattr(p_sig, "domain", ""))
                        ):
                            satisfied = True
                if not satisfied:
                    return None
            return sub

        # Zero-ary constructor morphisms: insertable after ANY cell (they consume
        # no incoming wire), completing instance lifecycles (construct -> fit).
        zero_ary_ctors = [
            c for c in candidates
            if getattr(c, "node_type", "") == "constructor"
            and not any(p.required for p in c.inputs.values())
        ]

        # Sequential Trellis extensions for t = 2 ... max_steps
        max_steps = max(2, min(8, max_transforms + 2))
        for step in range(2, max_steps + 1):
            candidates_for_next: List[Tuple[List[Cell], Substitution, float, int, int]] = []

            for prev_path, prev_sigma, prev_score, prev_weak, prev_unbind in current_beam:
                prev_cell = prev_path[-1]

                # Terminal morphisms (Stage 3) cannot have outgoing arrows unless they have slots
                if getattr(prev_cell, "stage", None) == 3 and not getattr(prev_cell, "slots", None):
                    continue

                prev_path_ids = {c.cell_id for c in prev_path}

                successor_cells = _successors(prev_cell)
                seen_ids = {c.cell_id for c in successor_cells}
                for ctor in zero_ary_ctors:
                    if ctor.cell_id not in seen_ids:
                        successor_cells.append(ctor)
                        seen_ids.add(ctor.cell_id)

                for cand in successor_cells:
                    # Acyclic: cell cannot repeat in pipeline
                    if cand.cell_id in prev_path_ids:
                        continue

                    cand_node_type = getattr(cand, "node_type", "")
                    cand_stage = getattr(cand, "stage", None)

                    # Zero-ary constructor morphisms: insertable mid-chain, consume
                    # no incoming wire; every required port must still be bindable.
                    if cand_node_type == "constructor":
                        new_sigma = _required_ports_bindable(cand, prev_path, prev_sigma)
                        if new_sigma is None:
                            continue
                    else:
                        # Stage 1 cells cannot be appended as intermediate transitions
                        if cand_stage == 1:
                            continue

                        # Monadic Unification Gate: edge exists iff unify(tau_out, tau_in, sigma) != bottom
                        prev_out_sigs = tuple(sorted((out_sig.signature.type_name, str(getattr(out_sig.signature, "state", "any"))) for c in prev_path for out_sig in c.outputs.values()))
                        pair_key = (prev_cell.cell_id, cand.cell_id, _sigma_fingerprint(prev_sigma), prev_out_sigs)
                        if pair_key in edge_compat_cache:
                            new_sigma = edge_compat_cache[pair_key]
                        else:
                            new_sigma = self._verify_transition(prev_path, cand, prev_sigma)
                            edge_compat_cache[pair_key] = new_sigma

                        if new_sigma is None:
                            continue

                    cand_unbind = _new_unbindable(cand, prev_path)
                    if cand_unbind > 0:
                        continue

                    cand_sc = log_probs.get(cand.cell_id, -10.0)
                    total_sc = prev_score + cand_sc
                    step_weak = prev_weak + (
                        1 if (cand_node_type != "constructor" and _edge_is_weak(prev_cell, cand)) else 0
                    )
                    step_unbind = prev_unbind + cand_unbind
                    new_tuple = (prev_path + [cand], new_sigma, total_sc, step_weak, step_unbind)
                    candidates_for_next.append(new_tuple)
                    all_valid_paths.append(new_tuple)

            if not candidates_for_next:
                break

            # Bound the trellis memory: keep the strongest half of discovered paths
            if len(all_valid_paths) > 6000:
                all_valid_paths.sort(key=lambda x: compute_path_score(x, is_final=False), reverse=True)
                all_valid_paths = all_valid_paths[:3000]

            # Beam pruning with endpoint diversity (max 5 per endpoint, beam width 250)
            candidates_for_next.sort(key=lambda x: compute_path_score(x, is_final=False), reverse=True)
            endpoint_counts: Dict[str, int] = {}
            next_beam = []
            for item in candidates_for_next:
                endpoint = item[0][-1].cell_id
                if endpoint_counts.get(endpoint, 0) < 5:
                    next_beam.append(item)
                    endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
                    if len(next_beam) >= 250:
                        break
            current_beam = next_beam

        # Filter and rank valid composition paths
        if all_valid_paths:
            valid_candidates = list(all_valid_paths)
            zero_unbind = [item for item in valid_candidates if item[4] == 0]
            if zero_unbind:
                valid_candidates = zero_unbind

            # 1. Filter by goal_sig if provided
            if goal_sig is not None:
                g_sig = goal_sig.signature if hasattr(goal_sig, "signature") else goal_sig
                matching_goals = [
                    item for item in valid_candidates
                    if unify(item[0][-1].primary_output.signature, g_sig) is not None
                ]
                if matching_goals:
                    valid_candidates = matching_goals

            # 2. Valid terminal boundary filter (Endable Nodes):
            # A pipeline cannot terminate at an intermediate data transformer (Stage 2)
            # if the prompt contains downstream unfulfilled action clauses.
            def _is_valid_terminal(cand_path: List[Cell]) -> bool:
                terminal = cand_path[-1]
                if getattr(terminal, "endable", None) is True:
                    return True
                if getattr(terminal, "is_endable", False):
                    covered = set().union(*(cell_covered.get(c.cell_id, set()) for c in cand_path))
                    if num_clauses > 1 and covered and max(covered) < num_clauses - 1:
                        if getattr(terminal, "stage", None) != 3:
                            return False
                    return True
                t_stage = getattr(terminal, "stage", None)
                if t_stage == 3:
                    return True
                t_id = terminal.cell_id.lower()
                t_role = getattr(terminal, "node_role", "")
                is_transformer = (t_role == "transformer" or t_id.endswith(".transform") or t_id.endswith(".fit_transform"))
                if is_transformer:
                    covered = set().union(*(cell_covered.get(c.cell_id, set()) for c in cand_path))
                    if num_clauses > 1 and covered and max(covered) < num_clauses - 1:
                        return False

                # Declared-goal terminals: a node whose DECLARED role is an
                # estimator, or whose output state reaches a declared terminal
                # family (trained model, materialized artifact), is a valid
                # pipeline goal. No cell-id substring whitelists: roles and
                # states are declared tree data.
                if t_role in ("estimator", "terminal", "evaluator", "sink", "consumer"):
                    return True
                out_states = {
                    str(getattr(op, "state", "")).lower()
                    for op in getattr(terminal, "outputs", {}).values()
                }
                if any(
                    registry.state_ancestry_reaches(st, {"trained", "fit_estimator", "written_to_disk", "exported", "saved"})
                    for st in out_states if st and st not in ("any", "*")
                ):
                    return True

                covered = set().union(*(cell_covered.get(c.cell_id, set()) for c in cand_path))
                if num_clauses > 1 and covered and max(covered) < num_clauses - 1:
                    return False
                return True

            endable_candidates = [item for item in valid_candidates if _is_valid_terminal(item[0])]
            if endable_candidates:
                valid_candidates = endable_candidates

            scored_candidates = [(item, compute_path_score(item, is_final=True)) for item in valid_candidates]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            if _debug_plan:
                dbg_top = [
                    (round(score, 2), " -> ".join(c.cell_id for c in item[0]))
                    for item, score in scored_candidates[:12]
                ]
                logger.info(f"[PLANNER-DEBUG] top candidates for prompt '{prompt[:60]}...':")
                for s, p in dbg_top:
                    comps = _component_trace.get(tuple(p.split(" -> ")), {})
                    logger.info(f"[PLANNER-DEBUG]   {s}  {p}  {comps}")
                macro_paths = [(round(score, 2), " -> ".join(c.cell_id for c in item[0]))
                               for item, score in scored_candidates
                               if any(getattr(c, "cell_type", "") == "macro" for c in item[0])][:5]
                if macro_paths:
                    logger.info(f"[PLANNER-DEBUG] best macro paths:")
                    for s, p in macro_paths:
                        comps = _component_trace.get(tuple(p.split(" -> ")), {})
                        logger.info(f"[PLANNER-DEBUG]   {s}  {p}  {comps}")
                else:
                    logger.info("[PLANNER-DEBUG] NO macro path survived the candidate filters")

            # Slot-aware re-ranking: a macro's planned sub-lattice is part of the
            # pipeline's semantics (a loop body calling contourArea covers the
            # "minimum area" clause). Plan the slots of the strongest candidates
            # and re-rank with the slot coverage included, so macro-based
            # pipelines are compared against flat pipelines with their bodies
            # filled in — not as bare skeletons.
            slot_augmented: List[Tuple[Tuple, float]] = []
            seen_prefixes: Set[Tuple[str, ...]] = set()
            trials = 0
            for it, base_score in scored_candidates:
                if trials >= 6:
                    break
                cand_path = it[0]
                prefix = tuple(c.cell_id for c in cand_path)
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                trials += 1
                slot_cells: List[Cell] = []
                has_slots = False
                for c in cand_path:
                    for slot_cells_list in (getattr(c, "bound_slots", {}) or {}).values():
                        slot_cells.extend(slot_cells_list)
                    if getattr(c, "slots", None):
                        has_slots = True
                if not has_slots:
                    slot_augmented.append((it, base_score))
                    continue
                # Plan slots for this candidate (mutates bound_slots for trial)
                trial_sigma = it[1]
                for c in cand_path:
                    for slot_name, slot_contract in _safe_slots_items(c):
                        if slot_name not in getattr(c, "bound_slots", {}):
                            sub = self.plan_sublattice(
                                c, slot_name, slot_contract, tunnel, relevance_map, trial_sigma, prompt
                            )
                            if sub:
                                c.bound_slots[slot_name] = sub
                                slot_cells.extend(sub)
                aug_score = base_score
                for sc_cell in slot_cells:
                    aug_score += cell_cov_mass_bonus.get(sc_cell.cell_id, 0.0)
                    aug_score += 10.0 * sum(
                        clause_weights[g] for g in cell_covered.get(sc_cell.cell_id, set())
                        if g not in set().union(*(cell_covered.get(pc.cell_id, set()) for pc in cand_path))
                    ) / total_clause_weight
                slot_augmented.append((it, aug_score))

            if slot_augmented:
                slot_augmented.sort(key=lambda x: x[1], reverse=True)
                scored_candidates = [(it, aug) for it, aug in slot_augmented]

            if os.environ.get("NSTL_DEBUG_PLAN"):
                import sys as _sys
                print("[PLAN-DEBUG] top scored paths:", file=_sys.stderr)
                for it, s in scored_candidates[:30]:
                    ids = [c.cell_id for c in it[0]]
                    dbg_path, _, dbg_sc, dbg_weak, dbg_unbind = it
                    dbg_k = len(dbg_path)
                    dbg_s: Set[str] = set()
                    dbg_w: Set[str] = set()
                    for c in dbg_path:
                        dbg_s |= cell_cov_strong.get(c.cell_id, set())
                        dbg_w |= cell_cov_weak.get(c.cell_id, set())
                    dbg_cov = (sum(idf_of_prompt.get(t, _idf(t)) for t in dbg_s)
                               + 0.3 * sum(idf_of_prompt.get(t, _idf(t)) for t in (dbg_w - dbg_s))) / total_prompt_idf
                    dbg_covd: Set[int] = set()
                    dbg_idx = []
                    dbg_cur = 0
                    for c in dbg_path:
                        dbg_m = cell_clause_mass.get(c.cell_id, [])
                        b, bc = -1, 0.0
                        for di, dm in enumerate(dbg_m):
                            if dm > bc or (dm == bc and dm > 0 and di >= dbg_cur):
                                bc, b = dm, di
                        if b >= 0:
                            dbg_idx.append(b); dbg_cur = max(dbg_cur, b)
                        dbg_covd |= cell_covered.get(c.cell_id, set())
                    dbg_align_w = sum(clause_weights[i] for i in dbg_covd) / total_clause_weight
                    dbg_mlb = 0.3 * (dbg_sc / max(dbg_k, 1))
                    print(f"  {s:.3f} cov={dbg_cov:.2f} alignW={dbg_align_w:.2f} k={dbg_k} weak={dbg_weak} unbind={dbg_unbind} mlb={dbg_mlb:.2f}  {' -> '.join(ids)}", file=_sys.stderr)
            # The acceptance gate must share this planner's orchestrator so that
            # macro-goal cells can resolve their string sub-cell ids during
            # pre-unification expansion (otherwise macros stay unexpanded here
            # and are accepted as opaque single nodes).
            gate = UnificationGate(orchestrator=self.orchestrator)
            chosen_candidate = None
            for it, sc in scored_candidates:
                cand_p = it[0]
                cand_test = self._expand_identifier_multiplicity(cand_p, prompt)
                try:
                    res = gate.unify_pipeline(cand_test, ExecutionContext(prompt=prompt))
                    if isinstance(res, Success) and not res.is_bottom():
                        chosen_candidate = it
                        break
                except Exception as e:
                    continue
            if chosen_candidate is None and scored_candidates:
                # Semantic repair stage (Section 3.4): attempt dynamic repair of candidate cells
                try:
                    from semantic_repair_engine import repair_cell_semantics
                    for it, _ in scored_candidates:
                        cand_p = it[0]
                        repaired_cells = []
                        any_repaired = False
                        for c in cand_p:
                            c_dict = c.to_dict() if hasattr(c, "to_dict") else dict(c.__dict__)
                            if repair_cell_semantics(c_dict, domain=getattr(c, "domain_name", "generic")):
                                any_repaired = True
                                repaired_cells.append(Cell.from_dict(c_dict))
                            else:
                                repaired_cells.append(c)
                        if any_repaired:
                            cand_test = self._expand_identifier_multiplicity(repaired_cells, prompt)
                            res = gate.unify_pipeline(cand_test, ExecutionContext(prompt=prompt))
                            if isinstance(res, Success) and not res.is_bottom():
                                chosen_candidate = (repaired_cells, res.sigma, it[2], it[3], it[4])
                                break
                except Exception as e:
                    logger.debug(f"[PLANNER] Semantic repair pass: {e}")

            if chosen_candidate is None:
                chosen_candidate = scored_candidates[0][0]
            best_candidate = chosen_candidate
            best_path, best_sigma, _, _, _ = best_candidate

            # For-each multiplicity expansion (Section 3.4 goal decomposition):
            # a clause that names an enumerable set of referential identifiers
            # sharing one role ("normalize X column, Y column and Z column")
            # demands one application of the witnessing transform PER MEMBER,
            # not a single best-effort witness. The witnessing cell is the path
            # cell whose DECLARED identity vocabulary intersects the group's
            # role context and which declares a defaultless reference port;
            # replicas re-consume the receiver from the environment and bind
            # the remaining members in prompt order at unification time.
            best_path = self._expand_identifier_multiplicity(best_path, prompt)

            # Sub-Lattice recursive planning for macro/control-flow cells with slots
            for cell in best_path:
                if getattr(cell, "slots", None):
                    for slot_name, slot_contract in _safe_slots_items(cell):
                        if slot_name not in getattr(cell, "bound_slots", {}):
                            sub_plan = self.plan_sublattice(
                                cell, slot_name, slot_contract, tunnel, relevance_map, best_sigma, prompt
                            )
                            if sub_plan:
                                cell.bound_slots[slot_name] = sub_plan

            return best_path

        # Step 4: Bounded MCTS Fallback (Section 3.4) if Trellis was disconnected
        logger.info("[PLANNER] Running bounded MCTS search...")
        mcts_path = self._bounded_mcts_search(tunnel, relevance_map)
        if mcts_path:
            return mcts_path

        return [max(tunnel, key=lambda c: relevance_map.get(c.cell_id, 0.0))]

    def _expand_identifier_multiplicity(
        self,
        path: List[Cell],
        prompt: str
    ) -> List[Cell]:
        """
        Expands for-each sub-goals: for every identifier role group in the
        prompt (>= 2 members sharing a role noun) and every witnessing cell on
        the path, inserts N-1 replicas of the witness directly after it.
        Gating is DECLARED-structure only: role tokens must intersect the
        cell's own token vocabulary, and the cell must own a defaultless
        required port of the reference value class (strict str, or any-typed
        with a declared semantic role state). Mirrors the binding gate in
        ExecutionContext.resolve_literal_for_port exactly.
        """
        if not prompt or len(path) < 1:
            return path
        try:
            groups = ExecutionContext.extract_identifier_groups(prompt)
        except Exception as e:
            return path
        if not groups:
            return path

        registry = TypeRegistry.get_instance()

        def _is_witness(cell: Cell, role_tokens: FrozenSet[str]) -> bool:
            # Only Stage 2 transform cells can be identifier multiplicity witnesses (never sinks or constructors)
            if getattr(cell, "stage", None) != 2:
                return False
            if getattr(cell, "node_type", "") == "constructor":
                return False
            for p_sig in cell.inputs.values():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                t_name = str(p_sig.signature.type_name).lower()
                strict_str = registry.is_subtype(t_name, "str") and t_name not in ("any", "*", "top", "")
                state = str(getattr(p_sig.signature, "state", "") or "").lower()
                role_state = state not in ("any", "default", "") and bool(
                    role_tokens & CellTokenizer.tokenize_identifier(state)
                )
                if strict_str or role_state:
                    return True
            return False

        expanded: List[Cell] = []
        replicas_added = 0
        MAX_REPLICAS = 8
        for cell in path:
            expanded.append(cell)
            if replicas_added >= MAX_REPLICAS:
                continue
            for group in groups:
                if len(group.members) < 2 or not group.role_tokens:
                    continue
                if not _is_witness(cell, group.role_tokens):
                    continue
                for _pos, _tok in group.members[1:]:
                    if replicas_added >= MAX_REPLICAS:
                        break
                    replica = copy.copy(cell)
                    replica.replica_of = cell.cell_id
                    replica.replica_role = ",".join(sorted(group.role_tokens))
                    expanded.append(replica)
                    replicas_added += 1
        return expanded

    def _verify_transition(
        self,
        prev_path: List[Cell],
        cand: Cell,
        prev_sigma: Substitution
    ) -> Optional[Substitution]:
        """
        Verifies monadic transition from prev_path to cand.
        Supports:
          1. Flat sequential 1D morphism: prev_cell.primary_output -> cand.primary_input.
          2. Multi-port monoidal product & port sharing (⊗, Δ):
             prev_cell.primary_output binds to some input of cand,
             and other required inputs are satisfied by earlier cells in prev_path or defaults.
        """
        prev_cell = prev_path[-1]
        prev_out = prev_cell.primary_output.signature

        def _extract_ndim(sc: Any) -> Optional[int]:
            if isinstance(sc, dict):
                val = sc.get("ndim")
                if isinstance(val, int):
                    return val
                try:
                    return int(val) if val is not None else None
                except (ValueError, TypeError):
                    return None
            elif isinstance(sc, str):
                sc = sc.strip()
                if sc.startswith("(") and sc.endswith(")"):
                    inner = sc[1:-1].strip()
                    if not inner:
                        return 0
                    parts = [p.strip() for p in inner.split(",") if p.strip()]
                    return len(parts)
            return None

        def _shape_compatible(p_out: Any, p_in: Any) -> bool:
            out_sc = getattr(p_out, "shape_contract", None)
            in_sc = getattr(p_in, "shape_contract", None)
            if out_sc and in_sc:
                o_ndim = _extract_ndim(out_sc)
                i_ndim = _extract_ndim(in_sc)
                if o_ndim is not None and i_ndim is not None and o_ndim != i_ndim:
                    return False
            return True

        # 1. Check primary input first
        sub = None
        if _shape_compatible(prev_cell.primary_output, cand.primary_input):
            sub = unify(prev_out, cand.primary_input.signature, prev_sigma)
        bound_in: Optional[str] = cand.primary_input.name if sub is not None else None

        # 2. If primary input did not match, check other input ports (required ports take precedence)
        if sub is None:
            req_ports = [(k, v) for k, v in cand.inputs.items() if v.required]
            candidate_ports = req_ports if req_ports else list(cand.inputs.items())
            for p_name, p_sig in candidate_ports:
                if not _shape_compatible(prev_cell.primary_output, p_sig):
                    continue
                s_try = unify(prev_out, p_sig.signature, prev_sigma)
                if s_try is not None:
                    sub = s_try
                    bound_in = p_name
                    break

        if sub is None:
            return None

        # 3. Check that all remaining REQUIRED inputs of cand can be satisfied
        # from earlier cells in prev_path (Port Sharing Δ) or have default values/literals
        for p_name, p_sig in cand.inputs.items():
            if p_name == bound_in:
                continue
            if not p_sig.required or p_sig.default_value is not None:
                continue

            satisfied = False
            for earlier_cell in reversed(prev_path[:-1]):
                for out_name, out_sig in earlier_cell.outputs.items():
                    if not _shape_compatible(out_sig, p_sig):
                        continue
                    s_wire = unify(out_sig.signature, p_sig.signature, sub)
                    if s_wire is not None:
                        sub = s_wire
                        satisfied = True
                        break
                if satisfied:
                    break

            if not satisfied:
                desc = getattr(p_sig, "description", None) or getattr(p_sig, "doc", "") or ""
                is_instance_receiver = p_name in ("data", "self") or ("receiver" in str(desc).lower())
                if not is_instance_receiver:
                    registry = TypeRegistry.get_instance()
                    t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
                    # Type-driven literal groundability: a required port is satisfiable at
                    # synthesis time iff its DECLARED carrier is literal-groundable (scalar
                    # family, textual/path family, logical, or an explicitly untyped carrier)
                    # or the port declares an enum domain for reflection-based grounding.
                    # Zero port-name heuristics: naming is data, typing is semantics.
                    is_literal_groundable = (
                        registry.is_subtype(t_name, "str")
                        or registry.is_subtype(t_name, "numeric")
                        or registry.is_subtype(t_name, "bool")
                        or registry.is_subtype(t_name, "filepath")
                        or registry.is_subtype(t_name, "uri")
                        or registry.is_subtype(t_name, "scalar")
                        or bool(getattr(p_sig, "domain", ""))
                    )
                    if is_literal_groundable:
                        satisfied = True

            if not satisfied:
                return None

        return sub

    def plan_sublattice(
        self,
        parent_cell: Cell,
        slot_name: str,
        slot_contract: Any,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        active_sigma: Substitution,
        prompt: str = ""
    ) -> Optional[List[Cell]]:
        """
        Synthesizes a type-verified sub-pipeline for a macro slot.
        Categorically verifies traced loop invariants (Tr^U) and coproduct branch joins (⊕).
        """
        topology = getattr(parent_cell, "topology_type", "sequential")

        # 1. Traced Loop Slot Planning (Tr^U) — dispatched by the DECLARED topology
        # of the macro cell, never by cell identifier substrings.
        if topology == "traced_loop":
            # Recover the container-typed input port structurally (generic carrier
            # C[T]); the item carrier T is extracted from its generic argument.
            coll_sig = None
            for p_sig in parent_cell.inputs.values():
                if "[" in str(getattr(p_sig.signature, "type_name", "")):
                    coll_sig = p_sig
                    break
            item_type: Any = "any"
            if coll_sig is not None:
                c_type_str = str(coll_sig.signature.type_name)
                c_concrete = substitute_generics(c_type_str, active_sigma)
                if "[" in c_concrete and c_concrete.endswith("]"):
                    b_idx = c_concrete.index("[")
                    item_type = c_concrete[b_idx + 1 : -1].strip()
                elif "T" in active_sigma.mappings:
                    item_type = str(active_sigma.mappings["T"])

            u_raw = getattr(parent_cell, "feedback_state_type", None) or "S"
            u_concrete = substitute_generics(u_raw, active_sigma)

            pool = [c for c in tunnel if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]
            if not pool:
                pool = [c for c in self.orchestrator.loaded_cells.values() if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]

            # Dynamic clause targeting: the clause(s) that describe the loop are the
            # ones sharing vocabulary with the loop morphism's DECLARED token set.
            # Zero hardcoded connector/loop keyword lists.
            parent_toks = getattr(parent_cell, "token_set", set()) - STOPWORDS
            clauses = _segment_prompt_clauses(prompt)
            if clauses and parent_toks:
                related = [
                    cl for cl in clauses
                    if (CellTokenizer.tokenize_prompt(cl) - STOPWORDS) & parent_toks
                ]
                slot_clause = " ".join(related)
            else:
                slot_clause = ""
            target_text = slot_clause.strip() or prompt
            target_tokens = (CellTokenizer.tokenize_prompt(target_text) if target_text else set()) - STOPWORDS
            if not target_tokens:
                target_tokens = (CellTokenizer.tokenize_prompt(prompt) if prompt else set()) - STOPWORDS

            child_candidates = []
            for cand in pool:
                if getattr(cand, "stage", None) not in (2, 3):
                    continue
                for p_name, p_sig in cand.inputs.items():
                    u_cand = unify(item_type, p_sig.signature, active_sigma) or unify(p_sig.signature, item_type, active_sigma)
                    if u_cand is not None:
                        rel = relevance_map.get(cand.cell_id, 0.0)
                        cand_content_toks = cand.token_set - STOPWORDS
                        tok_ov = len(target_tokens & cand_content_toks)
                        domain_bonus = 0.5 if cand.domain_name and any(c.domain_name == cand.domain_name for c in tunnel) else 0.0
                        score = tok_ov * 1.0 + rel + domain_bonus
                        child_candidates.append((cand, score, u_cand))
                        break

            if child_candidates:
                child_candidates.sort(key=lambda x: x[1], reverse=True)
                best_child, _, child_sigma = child_candidates[0]

                valid, _ = verify_traced_loop_invariant(u_concrete, u_concrete, child_sigma)
                if valid:
                    return [best_child]

        # 2. Coproduct Branch Slot Planning (⊕) — declared topology dispatch.
        elif topology == "coproduct_branch":
            pool = [c for c in tunnel if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]
            if pool:
                pool.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
                return [pool[0]]

        return None

    def _bounded_mcts_search(
        self,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        max_simulations: int = 50
    ) -> Optional[List[Cell]]:
        """
        Bounded Monte Carlo Tree Search over tunnel T (Section 3.4).
        Treats partial chains as tree nodes and explores indirect combinations.
        """
        entry_nodes = [c for c in tunnel if c.stage == 1] or list(tunnel)

        for entry in entry_nodes:
            chain = [entry]
            current_sigma = Substitution()

            for _ in range(max_simulations):
                curr = chain[-1]
                if curr.stage == 3:
                    return chain

                # Find valid unifiable candidates
                out_sig = curr.primary_output.signature
                valid_next = []
                for cand in tunnel:
                    if cand.cell_id in (c.cell_id for c in chain):
                        continue
                    new_sigma = unify(out_sig, cand.primary_input.signature, current_sigma)
                    if new_sigma is not None:
                        valid_next.append((cand, new_sigma))

                if not valid_next:
                    break

                # Bias exploration by semantic relevance probability combined with edge affinity
                valid_next.sort(key=lambda x: relevance_map.get(x[0].cell_id, 0.0) + 2.0 * self._calculate_edge_affinity(curr, x[0]), reverse=True)
                chosen_cand, chosen_sigma = valid_next[0]
                chain.append(chosen_cand)
                current_sigma = chosen_sigma

                if chosen_cand.stage == 3 and not getattr(chosen_cand, "slots", None):
                    for cell in chain:
                        if getattr(cell, "slots", None):
                            for slot_name, slot_contract in _safe_slots_items(cell):
                                if slot_name not in getattr(cell, "bound_slots", {}):
                                    sub_plan = self.plan_sublattice(
                                        cell, slot_name, slot_contract, tunnel, relevance_map, current_sigma, ""
                                    )
                                    if sub_plan:
                                        cell.bound_slots[slot_name] = sub_plan
                    return chain

            if len(chain) > 1:
                for cell in chain:
                    if getattr(cell, "slots", None):
                        for slot_name, slot_contract in _safe_slots_items(cell):
                            if slot_name not in getattr(cell, "bound_slots", {}):
                                sub_plan = self.plan_sublattice(
                                    cell, slot_name, slot_contract, tunnel, relevance_map, current_sigma, ""
                                )
                                if sub_plan:
                                    cell.bound_slots[slot_name] = sub_plan
                return chain

        return None


ZeroShotPlanner = LatticePlanner

