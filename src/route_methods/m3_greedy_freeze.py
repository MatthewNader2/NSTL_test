"""
src/route_methods/m3_greedy_freeze.py - Neuro-Symbolic Topological Lattice (NSTL)
M3: Forward Greedy Beam with Frozen Committed Prefixes.
Greedily commits the best next morphism at each step without backtracking.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, STOPWORDS

try:
    from ..lattice import Cell, LatticeOrchestrator
    from ..tokenizer import CellTokenizer
    from ..unification import unify, ExecutionContext
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from tokenizer import CellTokenizer
    from unification import unify, ExecutionContext


class M3GreedyFreezeRouteMethod(RouteMethod):
    """
    RouteMethod M3: Forward Greedy Freeze.
    Iteratively extends the pipeline forward, permanently committing the best
    valid continuation cell at each step without backtracking across earlier stages.
    """
    name: str = "m3_greedy_freeze"

    def plan(
        self,
        prompt: str,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        orchestrator: Optional[LatticeOrchestrator] = None,
        ctx: Optional[ExecutionContext] = None,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        max_transforms: int = 6,
        **kwargs
    ) -> List[Cell]:
        orch = orchestrator or self.orchestrator
        if not tunnel:
            return []
        if len(tunnel) == 1:
            return [tunnel[0]]

        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"]
        if not candidates:
            return [tunnel[0]]

        prompt_tokens = CellTokenizer.tokenize_prompt(prompt) - STOPWORDS
        l0_extracted = ExecutionContext._extract_universal_literals(prompt or "") if ctx or prompt else []
        file_literals = [v for _, t, v in l0_extracted if t == "file_asset"]

        # 1. Select initial entry cell C_0
        stage1_cands = [c for c in candidates if getattr(c, "stage", None) == 1]
        entry_pool = stage1_cands if stage1_cands else candidates[:5]

        def _score_entry(c: Cell) -> float:
            sc = relevance_map.get(c.cell_id, 0.0) * 5.0 + len(prompt_tokens & c.token_set) * 3.0
            is_path_consumer = any(
                getattr(p, "abstract_type", None) == "path"
                or getattr(p, "port_role", None) in ("source_data", "model_sink")
                or getattr(p, "derived_role", None) in ("source_data", "model_sink")
                or getattr(p.signature, "abstract_type", None) == "path"
                for p in c.inputs.values()
            )
            if file_literals and is_path_consumer:
                sc += 5.0
            return sc

        best_entry = max(entry_pool, key=_score_entry)

        committed_path: List[Cell] = [best_entry]
        visited_ids: Set[str] = {best_entry.cell_id}
        covered_tokens: Set[str] = set(best_entry.token_set & prompt_tokens)

        # 2. Greedily commit steps forward
        for step in range(max_transforms + 1):
            curr = committed_path[-1]

            # Terminal condition: Stage 3 reached
            if getattr(curr, "stage", None) == 3 and len(committed_path) > 1:
                break

            # Find all valid next candidates that unify with curr
            valid_next = [
                c for c in candidates
                if c.cell_id not in visited_ids and self.step_unifies(curr, c, prev_path=committed_path)
            ]

            if not valid_next:
                break

            # Score each candidate greedily
            def _score_cand(cand: Cell) -> float:
                rel = relevance_map.get(cand.cell_id, 0.0)
                aff = self.calculate_edge_affinity(curr, cand, orch)
                uncovered_toks = (cand.token_set & prompt_tokens) - covered_tokens
                tok_bonus = len(uncovered_toks) * 3.0
                
                # Favor stage progression (1 -> 2 -> 3)
                curr_stage = getattr(curr, "stage", 1) or 1
                cand_stage = getattr(cand, "stage", 2) or 2
                progression_bonus = 1.0 if cand_stage >= curr_stage else -2.0

                is_path_consumer = any(
                    getattr(p, "abstract_type", None) == "path"
                    or getattr(p, "port_role", None) in ("source_data", "model_sink")
                    or getattr(p, "derived_role", None) in ("source_data", "model_sink")
                    or getattr(p.signature, "abstract_type", None) == "path"
                    for p in cand.inputs.values()
                )
                path_bonus = 6.0 if (file_literals and is_path_consumer and cand_stage == 3) else 0.0

                return (rel * 10.0) + (aff * 5.0) + tok_bonus + progression_bonus + path_bonus

            valid_next.sort(key=_score_cand, reverse=True)
            best_next = valid_next[0]

            # Commit next cell (freeze)
            committed_path.append(best_next)
            visited_ids.add(best_next.cell_id)
            covered_tokens.update(best_next.token_set & prompt_tokens)

            # Check if all prompt content tokens are covered and cell is stage 3
            if (prompt_tokens - covered_tokens) == set() and getattr(best_next, "stage", None) == 3:
                break

        return committed_path
