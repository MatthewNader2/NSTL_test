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

        candidates = [
            c for c in tunnel
            if getattr(c, "node_type", "") not in ("constant", "macro")
            and not c.cell_id.startswith("MACRO_")
            and not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
        ]
        if not candidates:
            return [tunnel[0]]

        prompt_tokens = CellTokenizer.tokenize_prompt(prompt)
        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)

        # 1. Select initial entry cell C_0
        stage1_cands = [c for c in candidates if getattr(c, "stage", None) == 1]
        entry_pool = stage1_cands if stage1_cands else candidates[:5]

        clauses = self.segment_prompt_clauses(prompt)
        first_clause = clauses[0] if clauses else prompt
        src_lit = src_file_literals[0] if src_file_literals else None
        dst_lit = dest_file_literals[0] if dest_file_literals else None
        best_entry = max(
            entry_pool,
            key=lambda c: (self.file_compat(c, src_lit), self.clause_fit(first_clause, c), relevance_map.get(c.cell_id, 0.0)),
        )

        committed_path: List[Cell] = [best_entry]
        visited_ids: Set[str] = {best_entry.cell_id}
        _ev = self._evidence()
        prompt_tokens = _ev.clause_tokens(prompt) if _ev is not None else prompt_tokens
        covered_tokens: Set[str] = set(getattr(best_entry, "identity_tokens", best_entry.token_set) & prompt_tokens)

        num_clauses = len(clauses) if clauses else 1
        max_steps = max(32, num_clauses * 4 + 4, max_transforms + 8)

        # 2. Greedily commit steps forward
        for step in range(max_steps):
            curr = committed_path[-1]

            # Terminal condition: Stage 3 or terminal evaluator reached after covering clauses
            if (getattr(curr, "stage", None) == 3 or getattr(curr, "is_endable", False)) and len(committed_path) > 1:
                if step >= max(1, num_clauses - 1) or not (prompt_tokens - covered_tokens):
                    break

            # Find all valid next candidates that unify with curr or DAG scope
            valid_next = [
                c for c in candidates
                if c.cell_id not in visited_ids and (self.step_unifies(curr, c, prev_path=committed_path) or self.step_unifies_dag(c, committed_path, ctx=ctx))
            ]

            if not valid_next:
                break

            # Target clause for sequential alignment
            cl_idx = min(step + 1, num_clauses - 1)
            target_cl = clauses[cl_idx] if cl_idx < num_clauses else prompt
            _requested_sink = self.requested_sink(prompt, candidates)

            # Greedy key, lexicographic (no magic weights): never take a path-writing/terminal cell the
            # prompt did not ask for > evidence for the clause this step realises > newly explained prompt
            # tokens > stage progression (forward, same, backward) > retrieval relevance + edge affinity.
            def _score_cand(cand: Cell):
                rel = relevance_map.get(cand.cell_id, 0.0)
                aff = self.calculate_edge_affinity(curr, cand, orch, relevance_map=relevance_map)
                c_toks = getattr(cand, "identity_tokens", cand.token_set)
                new_mass = _ev.mass((c_toks & prompt_tokens) - covered_tokens) if _ev is not None else 0.0
                curr_stage = getattr(curr, "stage", 1) or 1
                cand_stage = getattr(cand, "stage", 2) or 2
                progress = 2 if cand_stage > curr_stage else (1 if cand_stage == curr_stage else 0)
                unrequested_egress = (
                    cand_stage == 3 and self._is_path_consumer(cand)
                    and not (dst_lit is not None and self.file_compat(cand, dst_lit) > 0)
                ) or (_requested_sink is None and cand_stage == 3 and curr_stage != 3 and self.clause_fit(target_cl, cand)[0] <= 0.0)
                return (0 if unrequested_egress else 1, self.clause_fit(target_cl, cand), new_mass, progress, rel + aff)

            valid_next.sort(key=_score_cand, reverse=True)
            best_next = valid_next[0]

            # Commit next cell (freeze)
            committed_path.append(best_next)
            visited_ids.add(best_next.cell_id)
            covered_tokens.update(getattr(best_next, "identity_tokens", best_next.token_set) & prompt_tokens)

            # Goal test shared with preflight: once every prompt clause is served by the path (after each
            # clause position has had its turn) the plan is complete; extending it only adds unrequested cells.
            if step >= num_clauses - 1:
                from coverage_audit import audit as _audit
                _res = _audit(committed_path, prompt, orch)
                if _res.get("available") and not any(i["kind"] == "clause_unserved" for i in _res["issues"]):
                    self._trace("m3_goal_reached", step=step, path_len=len(committed_path))
                    break

            # Check if all prompt content tokens are covered and cell is stage 3
            if (prompt_tokens - covered_tokens) == set() and (getattr(best_next, "stage", None) == 3 or getattr(best_next, "is_endable", False)):
                break

        return committed_path
