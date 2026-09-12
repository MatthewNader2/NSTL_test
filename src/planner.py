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
    from .lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry
    from .unification import unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch, substitute_generics, ExecutionContext
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry
    from unification import unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch, substitute_generics, ExecutionContext
    from tokenizer import CellTokenizer

logger = get_logger('planner')

STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "from", "by", "with",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "them", "they", "their", "this", "that", "these", "those",
    "make", "sure", "some", "get", "do", "using", "use", "into", "onto",
    "all", "each", "also", "just", "named", "have", "has", "having"
})

_WILDCARD_CARRIERS = frozenset(("any", "", "none", "*", "top", "unknown"))


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
    def __init__(self, orchestrator: LatticeOrchestrator, rag: Optional[Any] = None):
        self.orchestrator = orchestrator
        self.rag = rag

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

        # ---- Goal-directed objective data (all derived from DECLARED structure) ----
        registry = TypeRegistry.get_instance()
        file_literals = [
            v for _, t, v in ExecutionContext._extract_universal_literals(prompt or "")
            if t == "file_asset"
        ]

        def _has_path_port(cell: Cell) -> bool:
            for p_name, p_sig in cell.inputs.items():
                if p_name.lower() in ("filepath", "filename", "file_path", "path", "file", "savepath", "pathname", "fname"):
                    return True
                t_name = str(getattr(p_sig, "type_name", "")).lower()
                if registry.is_subtype(t_name, "filepath") or registry.is_subtype(t_name, "path") or registry.is_subtype(t_name, "uri"):
                    return True
            return False

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
            # If Stage 1 sources exist in tunnel, prioritize them as initial state entries
            s1_entries = [c for c in candidate_entries if getattr(c, "stage", None) == 1]
            if s1_entries:
                first_clause = re.split(r'[,;]|\b(?:and|then)\b', prompt.strip())[0].strip()
                clause_tokens = (CellTokenizer.tokenize_prompt(first_clause) if first_clause else set()) - STOPWORDS
                if clause_tokens:
                    overlaps = [len(clause_tokens & (c.token_set - STOPWORDS)) for c in s1_entries]
                    max_overlap = max(overlaps) if overlaps else 0
                    if max_overlap > 0:
                        s1_entries = [c for c, ov in zip(s1_entries, overlaps) if ov == max_overlap]
                # Asset absorption at the entry: sources declaring path-typed ports
                # are the categorical ingestion points for file-asset literals.
                if file_literals:
                    absorbing = [c for c in s1_entries if _has_path_port(c)]
                    if absorbing:
                        s1_entries = absorbing
                s1_entries.sort(key=lambda c: (getattr(c, "source_priority", 100), -relevance_map.get(c.cell_id, 0.0)))
                candidate_entries = s1_entries[:30]

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
                # A clause is CLAIMED only by identity matches (cell_id/keywords/
                # port names). Docstring prose may rank a cell but never claims
                # intent coverage — measured junk exploit: a triangle estimator's
                # docstring mentioning "area" claimed the loop clause.
                id_mass = sum(idf_of_prompt.get(t, _idf(t)) for t in (cl_toks & (c_toks & id_toks)))
                if id_mass >= 0.4 * clause_weights[len(masses) - 1]:
                    covered.add(len(masses) - 1)
            cell_clause_mass[c.cell_id] = masses
            cell_covered[c.cell_id] = covered

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
                or t in ("any", "*", "top", "scalar", "color", "enum")
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
                is_instance_receiver = p_name in ("data", "self") or ("receiver" in str(desc).lower())
                t = str(p_sig.signature.type_name)
                if not is_instance_receiver:
                    if t.lower() in _WILDCARD_CARRIERS or _port_literal_groundable(t):
                        continue
                sigs.append(p_sig.signature)
            cell_receiver_sigs[c.cell_id] = sigs

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
                    if (
                        p_name.lower() in ("filepath", "filename", "file_path", "path", "file", "savepath", "pathname", "fname", "path_or_buf")
                        or registry.is_subtype(t_name, "filepath")
                        or registry.is_subtype(t_name, "path")
                        or registry.is_subtype(t_name, "uri")
                    ):
                        required_path_ports += 1
                        break
            if required_path_ports > len(file_literals):
                count += (required_path_ports - len(file_literals))

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

        def compute_path_score(item: Tuple[List[Cell], Substitution, float, int, int], is_final: bool = False) -> float:
            path, _, sc, weak_edges, unbindable = item
            k = len(path)

            # Identity-weighted UNION coverage: each prompt token counts at most
            # once, at its strongest provenance across the path (identity match by
            # any cell > prose match). Per-cell summation lets overlapping cells
            # multi-count their dominant tokens and inflate coverage above the
            # total prompt mass — a measured junk-path exploit.
            strong_tokens: Set[str] = set()
            weak_tokens: Set[str] = set()
            for c in path:
                strong_tokens |= cell_cov_strong.get(c.cell_id, set())
                weak_tokens |= cell_cov_weak.get(c.cell_id, set())
            coverage = (
                sum(idf_of_prompt.get(t, _idf(t)) for t in strong_tokens)
                + 0.3 * sum(idf_of_prompt.get(t, _idf(t)) for t in (weak_tokens - strong_tokens))
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
                gap_penalty = holes * 1.5

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
            effective_k = k - consumed_ctors
            excess_steps = max(0, effective_k - max(distinct_matched_clauses, 1))
            parsimony_penalty = excess_steps * 1.2 + effective_k * 0.2

            # Tunnel likelihood is a WEAK tiebreaker: the group-relative softmax
            # already guarantees every surviving cell is within the same likelihood
            # window of its clause's maximum, so the raw log-prob gap between
            # clause-rank-1 cells and correct-but-rank-2 cells is distribution
            # noise, not semantic evidence. Full-weight log-probs structurally
            # prefer whichever garbage cell happened to rank #1 in a weak clause.
            mean_log_prob = 0.3 * (sc / max(k, 1))

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
            egress_tokens = frozenset({"save", "export", "write", "dump", "persist", "store", "plot", "show", "display"})
            has_egress_intent = bool(content_prompt_tokens & egress_tokens) or (
                len(file_literals) > 1 and any(getattr(c, "stage", None) == 1 for c in path)
            ) or (goal_sig is not None)
            terminal_is_materializing = (
                terminal.stage == 3
                and has_egress_intent
                and unbindable == 0
                and (
                    not file_literals
                    or any(
                        registry.is_subtype(str(p.signature.type_name).lower(), "filepath")
                        or registry.is_subtype(str(p.signature.type_name).lower(), "path")
                        or registry.is_subtype(str(p.signature.type_name).lower(), "uri")
                        for p in terminal.inputs.values()
                    )
                )
            )
            if tunnel_has_sinks and terminal_is_materializing:
                goal_bonus += 2.5
            # Domain coherence: the terminal morphism should belong to the
            # pipeline's own declared domains. A terminal imported from a foreign
            # domain (e.g. a plotting library bolted onto a vision pipeline)
            # exists to harvest bonuses — it pays a coherence tax, while a
            # home-domain terminal earns one.
            terminal_domain = getattr(terminal, "domain_name", "")
            if terminal_domain:
                other_domains = {getattr(c, "domain_name", "") for c in path[:-1]}
                if terminal_domain not in other_domains:
                    goal_bonus -= 1.5
                else:
                    goal_bonus += 1.5

            if file_literals:
                if any(_has_path_port(c) for c in path):
                    goal_bonus += 2.0
                elif not tunnel_absorbs_assets:
                    goal_bonus -= 3.0
            weak_total = sum(1 for c in path if _is_wildcarrier(c)) * 1.0 + weak_edges * 0.75

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
            domain_dispersion = max(0, len(pipeline_domains) - 2) * 5.0

            # Receiver-bindability (threaded incrementally): chains containing
            # fit-like morphisms whose instance receiver cannot be produced by any
            # earlier cell would fail at synthesis with an unresolved placeholder.
            return (coverage * 10.0 + alignment * 10.0 - parsimony_penalty
                    + mean_log_prob + goal_bonus - weak_total
                    - dead_ctors * 25.0 - unbindable * 50.0 - domain_dispersion - gap_penalty)

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

        def _successors(prev_cell: Cell) -> List[Cell]:
            out_t = str(getattr(prev_cell.primary_output, "type_name", ""))
            out_s = str(getattr(prev_cell.primary_output, "state", ""))

            acc: Dict[str, Cell] = {}

            # Generic carriers ("Sequence[T]", products) and TYPE VARIABLES ("T",
            # "S") unify by binding, not by poset subtyping: enumerate all
            # candidates and let exact unification gate them.
            is_type_var = out_t.isalpha() and len(out_t) == 1 and out_t.isupper()
            if "[" in out_t or is_type_var:
                for cand in candidates:
                    if any(unify(prev_cell.primary_output.signature, p_sig.signature) is not None
                           for p_sig in cand.inputs.values()):
                        acc.setdefault(cand.cell_id, cand)
                return list(acc.values())

            for in_t, cell_list in cells_by_in_type.items():
                if not registry.is_subtype(out_t, in_t):
                    continue
                states = distinct_in_states.get(in_t, set())
                state_ok = (out_s == "any") or any(s == "any" or s == out_s for s in states)
                if not state_ok:
                    continue
                for c in cell_list:
                    acc.setdefault(c.cell_id, c)
            return list(acc.values())

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
            except Exception:
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
                            or t_name in ("any", "*", "top", "scalar", "color", "enum")
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

                _dbg = os.environ.get("NSTL_DEBUG_PLAN")
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
                        prev_out_types = tuple(sorted(out_sig.signature.type_name for c in prev_path[:-1] for out_sig in c.outputs.values()))
                        pair_key = (prev_cell.cell_id, cand.cell_id, _sigma_fingerprint(prev_sigma), prev_out_types)
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
                    (p, s, sc) for p, s, sc in valid_candidates
                    if unify(p[-1].primary_output.signature, g_sig) is not None
                ]
                if matching_goals:
                    valid_candidates = matching_goals

            scored_candidates = [(item, compute_path_score(item, is_final=True)) for item in valid_candidates]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

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
                    for slot_name, slot_contract in (getattr(c, "slots", {}) or {}).items():
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
            best_candidate = scored_candidates[0][0]
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
                    for slot_name, slot_contract in cell.slots.items():
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
        except Exception:
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

        def _shape_compatible(p_out: Any, p_in: Any) -> bool:
            out_sc = getattr(p_out, "shape_contract", None)
            in_sc = getattr(p_in, "shape_contract", None)
            if out_sc and in_sc:
                o_ndim = out_sc.get("ndim")
                i_ndim = in_sc.get("ndim")
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
                        or t_name in ("any", "*", "top", "scalar", "color", "enum")
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
                    u_cand = unify(item_type, p_sig.signature, active_sigma)
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

                # Bias exploration by semantic relevance probability
                valid_next.sort(key=lambda x: relevance_map.get(x[0].cell_id, 0.0), reverse=True)
                chosen_cand, chosen_sigma = valid_next[0]
                chain.append(chosen_cand)
                current_sigma = chosen_sigma

                if chosen_cand.stage == 3 and not getattr(chosen_cand, "slots", None):
                    for cell in chain:
                        if getattr(cell, "slots", None):
                            for slot_name, slot_contract in cell.slots.items():
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
                        for slot_name, slot_contract in cell.slots.items():
                            if slot_name not in getattr(cell, "bound_slots", {}):
                                sub_plan = self.plan_sublattice(
                                    cell, slot_name, slot_contract, tunnel, relevance_map, current_sigma, ""
                                )
                                if sub_plan:
                                    cell.bound_slots[slot_name] = sub_plan
                return chain

        return None


ZeroShotPlanner = LatticePlanner

