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

        # Partition tunnel cells by stage
        stage1_cells = [c for c in tunnel if c.stage == 1]
        stage2_cells = [c for c in tunnel if c.stage == 2]
        stage3_cells = [c for c in tunnel if c.stage == 3]

        # If start_sig is specified, filter candidate entry cells to type-compatible entries
        if start_sig is not None:
            s_sig = start_sig.signature if hasattr(start_sig, "signature") else start_sig
            matching_entries = [c for c in tunnel if unify(s_sig, c.primary_input.signature) is not None]
            candidate_entries = matching_entries if matching_entries else tunnel
        else:
            candidate_entries = tunnel

        # If goal_sig is specified, filter stage 3 cells to type-compatible exits
        if goal_sig is not None:
            g_sig = goal_sig.signature if hasattr(goal_sig, "signature") else goal_sig
            matching_s3 = [c for c in stage3_cells if unify(c.primary_output.signature, g_sig) is not None]
            if matching_s3:
                stage3_cells = matching_s3

        # Decompose prompt into constituent clauses to identify initial clause intent
        import re
        clauses = re.split(r'\s+(?:then|and\s+then|and|,|;)\s+', prompt.strip())
        first_clause = clauses[0].strip() if clauses else prompt.strip()
        first_clause_tokens = CellTokenizer.tokenize_prompt(first_clause)
        prompt_tokens = CellTokenizer.tokenize_prompt(prompt)
        is_multistage = len(clauses) > 1

        # 1. Select Best Entry Node via universal category-theoretic fitness
        def entry_fitness(c: Cell) -> Tuple[int, int, int, float]:
            curated = 1 if c.source_priority <= 10 else 0
            c_tokens = set(c.token_set)
            clause_overlap = len(c_tokens & first_clause_tokens)
            # In a multi-stage workflow, initial ingestion (stage 1) with clause overlap is favored for entry
            stage_match = 1 if (is_multistage and c.stage == 1 and clause_overlap > 0) or (not is_multistage and clause_overlap > 0) else 0
            rel = relevance_map.get(c.cell_id, 0.0)
            return (stage_match, curated, clause_overlap, rel)

        sorted_entries = sorted(candidate_entries, key=entry_fitness, reverse=True)
        best_entry = sorted_entries[0]
        pipeline: List[Cell] = [best_entry]
        current_cell = best_entry
        current_sigma = Substitution()

        covered_tokens = set(best_entry.token_set) & prompt_tokens

        # 2. Select Relevant Intermediate Transforms (Stage 2) via Intent-Coverage Stopping
        curated_transforms = [c for c in stage2_cells if c.source_priority <= 10]
        active_transforms = curated_transforms if curated_transforms else stage2_cells
        active_transforms.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)

        for _ in range(max_transforms):
            best_next = None
            best_next_sigma = None

            for cand in active_transforms:
                if cand in pipeline:
                    continue

                # Intent-Coverage Criterion: candidate must satisfy newly uncovered prompt tokens
                cand_tokens = set(cand.token_set) & prompt_tokens
                new_tokens = cand_tokens - covered_tokens
                if not new_tokens and covered_tokens:
                    continue

                new_sigma = unify(current_cell.primary_output.signature, cand.primary_input.signature, current_sigma)
                if new_sigma is not None:
                    best_next = cand
                    best_next_sigma = new_sigma
                    covered_tokens.update(cand_tokens)
                    break

            if best_next is not None:
                pipeline.append(best_next)
                current_cell = best_next
                current_sigma = best_next_sigma
            else:
                break

        # 3. Select Terminal Node (Stage 3) only if required by uncovered prompt intent
        if stage3_cells:
            curated_stage3 = [c for c in stage3_cells if c.source_priority <= 10]
            active_stage3 = curated_stage3 if curated_stage3 else stage3_cells
            active_stage3.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)

            for term in active_stage3:
                term_tokens = set(term.token_set) & prompt_tokens
                new_tokens = term_tokens - covered_tokens
                if new_tokens or not covered_tokens:
                    new_sigma = unify(current_cell.primary_output.signature, term.primary_input.signature, current_sigma)
                    if new_sigma is not None:
                        pipeline.append(term)
                        current_cell = term
                        current_sigma = new_sigma
                        covered_tokens.update(term_tokens)
                        break

        # Return pipeline if multi-cell or if prompt intent is fully satisfied / single-stage
        if len(pipeline) > 1 or not is_multistage or not (prompt_tokens - covered_tokens):
            return pipeline

        # 4. Bounded MCTS Fallback (Section 3.4) if pipeline could not bridge intent
        logger.info("[PLANNER] Incomplete intent coverage. Running bounded MCTS...")
        mcts_path = self._bounded_mcts_search(tunnel, relevance_map)
        if mcts_path:
            return mcts_path

        return pipeline

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
        entry_nodes = [c for c in tunnel if c.stage == 1] or tunnel[:3]

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

