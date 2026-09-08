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
from typing import Dict, List, Optional, Set, Tuple, Any

from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry
    from .unification import unify, Substitution
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry
    from unification import unify, Substitution
    from tokenizer import CellTokenizer

logger = get_logger('planner')


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
                candidate_entries = s1_entries

        # Viterbi Trellis: paths of length t = 1 ... T_max
        all_valid_paths: List[Tuple[List[Cell], Substitution, float]] = []

        # Step t = 1: Initialize trellis
        current_trellis: Dict[str, Tuple[List[Cell], Substitution, float]] = {}
        for entry in candidate_entries:
            sc = log_probs.get(entry.cell_id, -10.0)
            p_tuple = ([entry], Substitution(), sc)
            current_trellis[entry.cell_id] = p_tuple
            all_valid_paths.append(p_tuple)

        # Edge compatibility cache for candidate pairs in tunnel
        edge_compat_cache: Dict[Tuple[str, str], Optional[Substitution]] = {}

        # Sequential Trellis extensions for t = 2 ... max_steps
        max_steps = max(2, min(6, max_transforms + 2))
        for step in range(2, max_steps + 1):
            next_trellis: Dict[str, Tuple[List[Cell], Substitution, float]] = {}

            for prev_cid, (prev_path, prev_sigma, prev_score) in current_trellis.items():
                prev_cell = prev_path[-1]

                # Terminal morphisms (Stage 3) cannot have outgoing arrows
                if getattr(prev_cell, "stage", None) == 3:
                    continue

                for cand in candidates:
                    # Acyclic: cell cannot repeat in pipeline
                    if cand.cell_id in (c.cell_id for c in prev_path):
                        continue

                    # Stage ordering: cannot move backwards to Stage 1 from Stage 2 or 3
                    if getattr(cand, "stage", None) == 1 and getattr(prev_cell, "stage", None) in (2, 3):
                        continue

                    # Monadic Unification Gate: edge exists iff unify(tau_out, tau_in, sigma) != bottom
                    pair_key = (prev_cell.cell_id, cand.cell_id)
                    if pair_key in edge_compat_cache:
                        new_sigma = edge_compat_cache[pair_key]
                    else:
                        new_sigma = unify(prev_cell.primary_output.signature, cand.primary_input.signature, prev_sigma)
                        edge_compat_cache[pair_key] = new_sigma

                    if new_sigma is not None:
                        cand_sc = log_probs.get(cand.cell_id, -10.0)
                        total_sc = prev_score + cand_sc

                        if cand.cell_id not in next_trellis or total_sc > next_trellis[cand.cell_id][2]:
                            new_tuple = (prev_path + [cand], new_sigma, total_sc)
                            next_trellis[cand.cell_id] = new_tuple
                            all_valid_paths.append(new_tuple)

            if not next_trellis:
                break
            if len(next_trellis) > 40:
                sorted_entries = sorted(next_trellis.items(), key=lambda x: x[1][2], reverse=True)[:40]
                current_trellis = dict(sorted_entries)
            else:
                current_trellis = next_trellis

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

            # Normalized path score: balance concept coverage and mean log-likelihood
            prompt_tokens = CellTokenizer.tokenize_prompt(prompt)
            p_len = max(len(prompt_tokens), 1)

            def compute_path_score(item: Tuple[List[Cell], Substitution, float]) -> float:
                path, _, sc = item
                k = len(path)
                path_tokens = set().union(*(c.token_set for c in path))
                coverage = len(prompt_tokens & path_tokens) / p_len
                mean_log_prob = sc / k
                return coverage * 10.0 + mean_log_prob

            scored_candidates = [(item, compute_path_score(item)) for item in valid_candidates]
            best_candidate, _ = max(scored_candidates, key=lambda x: x[1])
            best_path = best_candidate[0]
            return best_path

        # Step 4: Bounded MCTS Fallback (Section 3.4) if Trellis was disconnected
        logger.info("[PLANNER] Running bounded MCTS search...")
        mcts_path = self._bounded_mcts_search(tunnel, relevance_map)
        if mcts_path:
            return mcts_path

        return [max(tunnel, key=lambda c: relevance_map.get(c.cell_id, 0.0))]

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

                if chosen_cand.stage == 3:
                    return chain

            if len(chain) > 1:
                return chain

        return None


ZeroShotPlanner = LatticePlanner

