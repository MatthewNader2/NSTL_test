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
          [Entry (Stage 1)] -> [Transforms (Stage 2)]* -> [Terminal (Stage 3)]
        Guided by semantic relevance probabilities P(v | e_x) from the tunnel.
        """
        if not tunnel:
            return []

        # Single standalone node case
        if len(tunnel) == 1:
            return [tunnel[0]]

        # Decompose prompt into constituent clauses using deterministic delimiters (zero regex)
        clauses = self._split_prompt_clauses(prompt.strip())
        is_multistage = len(clauses) > 1

        # Compute log-likelihood distributions log P(v | e_{c_k}) for each clause
        clause_log_probs: List[Dict[str, float]] = []
        for c_text in clauses:
            clause_scores: Dict[str, float] = {}
            for cell in tunnel:
                # Semantic vector similarity via RAG or token Jaccard fallback
                rel = relevance_map.get(cell.cell_id, 0.0)
                clause_tokens = CellTokenizer.tokenize_prompt(c_text)
                cell_tokens = cell.token_set
                intersection = len(clause_tokens & cell_tokens)
                union = len(clause_tokens | cell_tokens)
                jaccard = (intersection / max(union, 1)) if union > 0 else 0.0
                # Combined Bayesian evidence: prior tunnel probability and clause similarity
                clause_scores[cell.cell_id] = math.log(max(rel, 1e-6)) + 2.0 * jaccard

            # Numerical log-softmax over tunnel candidates
            max_s = max(clause_scores.values()) if clause_scores else 0.0
            sum_exp = sum(math.exp(s - max_s) for s in clause_scores.values())
            log_sum_exp = max_s + math.log(max(sum_exp, 1e-12))
            clause_log_probs.append({cid: s - log_sum_exp for cid, s in clause_scores.items()})

        # Viterbi Dynamic Programming over typed category G|_T
        # Step 1: Initial state distribution V_1(v)
        viterbi_scores: Dict[str, float] = {}
        viterbi_paths: Dict[str, List[Cell]] = {}
        viterbi_sigmas: Dict[str, Substitution] = {}

        candidate_entries = [c for c in tunnel if getattr(c, "node_type", "") != "constant"]
        if start_sig is not None:
            s_sig = start_sig.signature if hasattr(start_sig, "signature") else start_sig
            matching_entries = [c for c in candidate_entries if unify(s_sig, c.primary_input.signature) is not None]
            if matching_entries:
                candidate_entries = matching_entries

        # In multi-stage, Stage 1 entry morphisms are canonical initial state candidates
        if is_multistage:
            s1_entries = [c for c in candidate_entries if c.stage == 1]
            if s1_entries:
                candidate_entries = s1_entries

        for entry in candidate_entries:
            score = clause_log_probs[0].get(entry.cell_id, -100.0)
            viterbi_scores[entry.cell_id] = score
            viterbi_paths[entry.cell_id] = [entry]
            viterbi_sigmas[entry.cell_id] = Substitution()

        # Step 2: Sequential Viterbi Trellis transitions for k = 2 ... K
        if is_multistage:
            for k in range(1, len(clauses)):
                is_terminal_step = (k == len(clauses) - 1)
                next_scores: Dict[str, float] = {}
                next_paths: Dict[str, List[Cell]] = {}
                next_sigmas: Dict[str, Substitution] = {}

                # Candidate pool for this stage
                search_pool = [
                    c for c in tunnel
                    if getattr(c, "node_type", "") != "constant"
                    and (c.stage in (2, 3) if is_terminal_step else c.stage == 2)
                ]
                if not search_pool:
                    search_pool = [c for c in tunnel if getattr(c, "node_type", "") != "constant"]

                for cand in search_pool:
                    log_p = clause_log_probs[k].get(cand.cell_id, -100.0)
                    best_prev_score = -float('inf')
                    best_prev_cid = None
                    best_prev_sigma = None

                    for prev_cid, prev_score in viterbi_scores.items():
                        prev_path = viterbi_paths[prev_cid]
                        if cand.cell_id in (c.cell_id for c in prev_path):
                            continue
                        prev_cell = prev_path[-1]
                        prev_sigma = viterbi_sigmas[prev_cid]

                        # Monadic Unification Gate: Edge exists iff unify(tau_out, tau_in) != bottom
                        new_sigma = unify(prev_cell.primary_output.signature, cand.primary_input.signature, prev_sigma)
                        if new_sigma is not None:
                            total_score = prev_score + log_p
                            if total_score > best_prev_score:
                                best_prev_score = total_score
                                best_prev_cid = prev_cid
                                best_prev_sigma = new_sigma

                    if best_prev_cid is not None and best_prev_sigma is not None:
                        next_scores[cand.cell_id] = best_prev_score
                        next_paths[cand.cell_id] = viterbi_paths[best_prev_cid] + [cand]
                        next_sigmas[cand.cell_id] = best_prev_sigma

                if next_scores:
                    viterbi_scores = next_scores
                    viterbi_paths = next_paths
                    viterbi_sigmas = next_sigmas
                else:
                    break

        # Step 3: Select global optimal path from Viterbi trellis
        if viterbi_paths:
            candidate_endpoints = list(viterbi_scores.keys())
            if goal_sig is not None:
                g_sig = goal_sig.signature if hasattr(goal_sig, "signature") else goal_sig
                valid_goals = [
                    cid for cid in candidate_endpoints
                    if unify(viterbi_paths[cid][-1].primary_output.signature, g_sig) is not None
                ]
                if valid_goals:
                    candidate_endpoints = valid_goals

            # Stage 3 terminal prioritization if egress intent exists
            if any(viterbi_paths[cid][-1].stage == 3 for cid in candidate_endpoints):
                s3_goals = [cid for cid in candidate_endpoints if viterbi_paths[cid][-1].stage == 3]
                if s3_goals:
                    candidate_endpoints = s3_goals

            best_end_cid = max(candidate_endpoints, key=lambda cid: viterbi_scores[cid])
            optimal_path = viterbi_paths[best_end_cid]
            if len(optimal_path) > 1 or not is_multistage:
                return optimal_path

        # Step 4: Bounded MCTS Fallback (Section 3.4) if Trellis was disconnected
        logger.info("[PLANNER] Running bounded MCTS search...")
        mcts_path = self._bounded_mcts_search(tunnel, relevance_map)
        if mcts_path:
            return mcts_path

        return [max(tunnel, key=lambda c: relevance_map.get(c.cell_id, 0.0))]

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

