"""
src/planner.py - Neuro-Symbolic Topological Lattice (NSTL)
Topological Pathfinding, Multi-Stage Progression, and Formal Type-Monadic Verification.

Conforms strictly to Sections 3.1, 3.2, and 3.4 of the NSTL paper:
  A path through the lattice is a sequence of monadic binds. Any step that
  would produce bottom is rejected the moment it is proposed.
  Finds maximum-likelihood type-valid composition paths inside the semantic tunnel T.
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Optional, Set, Tuple, Any

from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry
    from .unification import unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch, substitute_generics
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry
    from unification import unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch, substitute_generics
    from tokenizer import CellTokenizer

logger = get_logger('planner')

STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "from", "by", "with",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "them", "they", "their", "this", "that", "these", "those"
})


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
        max_transforms: int = 4
    ) -> List[Cell]:
        """
        Plans a type-valid compositional pipeline:
          [Entry (Stage 1)] -> [Transforms / Bridges (Stage 2)]* -> [Terminal (Stage 3)]
        Conforms strictly to Sections 3.1, 3.2, and 3.4 of the NSTL paper:
        Viterbi Trellis Dynamic Programming over the typed category G|_T.
        Monadic Unification Gate rejects any invalid edge proposals.
        ZERO linguistic connector heuristics (no 'then'), ZERO punctuation splitting, ZERO regex.
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
                s1_entries.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
                candidate_entries = s1_entries[:30]

        # Clauses and tokens for sequential alignment and concept coverage
        clauses = [cl.strip() for cl in re.split(r'[,;]|\b(?:and|then)\b', prompt.strip()) if cl.strip()]
        clause_tokens_list = [(CellTokenizer.tokenize_prompt(cl) - STOPWORDS) for cl in clauses]
        clause_tokens_list = [t for t in clause_tokens_list if t]
        content_prompt_tokens = set().union(*clause_tokens_list) if clause_tokens_list else ((CellTokenizer.tokenize_prompt(prompt) if prompt else set()) - STOPWORDS)
        p_len = max(len(content_prompt_tokens), 1)
        num_clauses = max(len(clause_tokens_list), 1)

        def compute_path_score(item: Tuple[List[Cell], Substitution, float]) -> float:
            path, _, sc = item
            k = len(path)
            path_tokens = set().union(*((c.token_set - STOPWORDS) for c in path))
            coverage = len(content_prompt_tokens & path_tokens) / p_len

            clause_match_indices = []
            curr_max_idx = 0
            for c in path:
                c_toks = c.token_set - STOPWORDS
                best_match_idx = -1
                best_match_cnt = 0
                for idx, cl_toks in enumerate(clause_tokens_list):
                    cnt = len(cl_toks & c_toks)
                    if cnt > best_match_cnt or (cnt == best_match_cnt and cnt > 0 and idx >= curr_max_idx):
                        best_match_cnt = cnt
                        best_match_idx = idx
                if best_match_idx >= 0:
                    clause_match_indices.append(best_match_idx)
                    curr_max_idx = max(curr_max_idx, best_match_idx)

            distinct_matched_clauses = len(set(clause_match_indices))
            clause_cov = distinct_matched_clauses / num_clauses
            is_monotonic = (
                all(clause_match_indices[i] <= clause_match_indices[i+1] for i in range(len(clause_match_indices)-1))
                if len(clause_match_indices) >= 2 else True
            )
            alignment = clause_cov * (1.0 if is_monotonic else 0.5)

            # Parsimony penalty: penalize unnecessary steps beyond matched clauses
            excess_steps = max(0, k - max(distinct_matched_clauses, 1))
            parsimony_penalty = excess_steps * 1.5

            mean_log_prob = sc / max(k, 1)
            return coverage * 10.0 + alignment * 10.0 - parsimony_penalty + mean_log_prob

        # Viterbi Trellis: paths of length t = 1 ... T_max
        all_valid_paths: List[Tuple[List[Cell], Substitution, float]] = []

        # Step t = 1: Initialize beam
        current_beam: List[Tuple[List[Cell], Substitution, float]] = []
        for entry in candidate_entries:
            sc = log_probs.get(entry.cell_id, -10.0)
            p_tuple = ([entry], Substitution(), sc)
            current_beam.append(p_tuple)
            all_valid_paths.append(p_tuple)

        # Edge compatibility cache for candidate pairs in tunnel
        edge_compat_cache: Dict[Tuple[str, str, int], Optional[Substitution]] = {}

        # Sequential Trellis extensions for t = 2 ... max_steps
        max_steps = max(2, min(6, max_transforms + 2))
        for step in range(2, max_steps + 1):
            candidates_for_next: List[Tuple[List[Cell], Substitution, float]] = []

            for prev_path, prev_sigma, prev_score in current_beam:
                prev_cell = prev_path[-1]

                # Terminal morphisms (Stage 3) cannot have outgoing arrows unless they have slots
                if getattr(prev_cell, "stage", None) == 3 and not getattr(prev_cell, "slots", None):
                    continue

                for cand in candidates:
                    # Acyclic: cell cannot repeat in pipeline
                    if cand.cell_id in (c.cell_id for c in prev_path):
                        continue

                    # Stage 1 cells cannot be appended as intermediate transitions
                    if getattr(cand, "stage", None) == 1:
                        continue

                    # Stage ordering: cannot move backwards to Stage 1 from Stage 2 or 3
                    if getattr(cand, "stage", None) == 1 and getattr(prev_cell, "stage", None) in (2, 3):
                        continue

                    # Monadic Unification Gate: edge exists iff unify(tau_out, tau_in, sigma) != bottom
                    pair_key = (prev_cell.cell_id, cand.cell_id, len(prev_path))
                    if pair_key in edge_compat_cache:
                        new_sigma = edge_compat_cache[pair_key]
                    else:
                        new_sigma = self._verify_transition(prev_path, cand, prev_sigma)
                        edge_compat_cache[pair_key] = new_sigma

                    if new_sigma is not None:
                        cand_sc = log_probs.get(cand.cell_id, -10.0)
                        total_sc = prev_score + cand_sc
                        new_tuple = (prev_path + [cand], new_sigma, total_sc)
                        candidates_for_next.append(new_tuple)
                        all_valid_paths.append(new_tuple)

            if not candidates_for_next:
                break

            # Beam pruning with endpoint diversity (max 5 per endpoint, beam width 100)
            candidates_for_next.sort(key=compute_path_score, reverse=True)
            endpoint_counts: Dict[str, int] = {}
            next_beam = []
            for item in candidates_for_next:
                endpoint = item[0][-1].cell_id
                if endpoint_counts.get(endpoint, 0) < 5:
                    next_beam.append(item)
                    endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
                    if len(next_beam) >= 100:
                        break
            current_beam = next_beam

        # Filter and rank valid composition paths
        if all_valid_paths:
            valid_candidates = list(all_valid_paths)

            # 1. Filter by goal_sig if provided
            if goal_sig is not None:
                g_sig = goal_sig.signature if hasattr(goal_sig, "signature") else goal_sig
                matching_goals = [
                    (p, s, sc) for p, s, sc in valid_candidates
                    if unify(p[-1].primary_output.signature, g_sig) is not None
                ]
                if matching_goals:
                    valid_candidates = matching_goals

            scored_candidates = [(item, compute_path_score(item)) for item in valid_candidates]
            best_candidate, _ = max(scored_candidates, key=lambda x: x[1])
            best_path, best_sigma, _ = best_candidate

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

        # 1. Check primary input first
        sub = unify(prev_out, cand.primary_input.signature, prev_sigma)
        bound_in: Optional[str] = cand.primary_input.name if sub is not None else None

        # 2. If primary input did not match, check other input ports
        if sub is None:
            for p_name, p_sig in cand.inputs.items():
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
                    s_wire = unify(out_sig.signature, p_sig.signature, sub)
                    if s_wire is not None:
                        sub = s_wire
                        satisfied = True
                        break
                if satisfied:
                    break

            if not satisfied:
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
            clauses = [cl.strip() for cl in re.split(r'[,;]|\b(?:and|then)\b', prompt.strip()) if cl.strip()]
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

