"""
src/route_methods/m11_latent_cognitive.py - Neuro-Symbolic Topological Lattice (NSTL)

M11: Cognitive Latent Pathfinding over Lattice DAG.
Combines category-theoretic type-unification constraints with continuous
latent residual stream projections extracted from local sub-1B models
(EmbeddingGemma-300M / Qwen2.5-Coder-0.5B).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from .base import RouteMethod, STOPWORDS

try:
    from ..lattice import Cell, LatticeOrchestrator
    from ..tokenizer import CellTokenizer
    from ..unification import unify, ExecutionContext
    from ..cognitive_pathfinder import CognitivePathfinder
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from tokenizer import CellTokenizer
    from unification import unify, ExecutionContext
    from cognitive_pathfinder import CognitivePathfinder


class M11LatentCognitiveRouteMethod(RouteMethod):
    """
    RouteMethod M11: Cognitive Latent Pathfinding.
    Evaluates candidate morphisms by projecting directional progress vectors
    in the latent representation space of local sub-1B models toward the prompt goal,
    strictly constrained by algebraic typestate unification gates.
    """
    name: str = "m11_latent_cognitive"

    def __init__(self, orchestrator: Optional[LatticeOrchestrator] = None, **kwargs):
        super().__init__(orchestrator=orchestrator, **kwargs)
        self.pathfinder = CognitivePathfinder()

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
            if getattr(c, "node_type", "") != "constant"
            and (
                bool(getattr(c, "slots", None))
                or (
                    getattr(c, "node_type", "") != "macro"
                    and not c.cell_id.startswith("MACRO_")
                    and not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
                )
            )
        ]
        if not candidates:
            return [tunnel[0]]

        clauses = self.segment_prompt_clauses(prompt)
        if not clauses:
            clauses = [prompt]

        # 1. Continuous Latent Goal and Clause Representations
        goal_vec = self.pathfinder.extractor.encode(prompt)
        clause_vecs = self.pathfinder.extractor.encode_batch(clauses)
        current_vec = np.zeros_like(goal_vec)

        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)
        src_lit = src_file_literals[0] if src_file_literals else None
        dst_lit = dest_file_literals[0] if dest_file_literals else None

        # 2. Select initial entry cell C_0
        stage1_cands = [c for c in candidates if getattr(c, "stage", None) == 1]
        entry_pool = stage1_cands if stage1_cands else candidates[:5]

        first_target_vec = clause_vecs[0] if clause_vecs else goal_vec
        best_entry = max(
            entry_pool,
            key=lambda c: (
                self.file_compat(c, src_lit),
                self.pathfinder.compute_latent_heuristic(current_vec, self.pathfinder.get_cell_vector(c), first_target_vec),
                relevance_map.get(c.cell_id, 0.0),
            ),
        )

        committed_path: List[Cell] = [best_entry]
        visited_ids: Set[str] = {best_entry.cell_id}
        current_vec = self.pathfinder.get_cell_vector(best_entry)

        _ev = self._evidence()
        prompt_tokens = _ev.clause_tokens(prompt) if _ev is not None else CellTokenizer.tokenize_prompt(prompt)
        covered_tokens: Set[str] = set(getattr(best_entry, "identity_tokens", best_entry.token_set) & prompt_tokens)

        num_clauses = len(clauses)
        max_steps = max(32, num_clauses * 4 + 4, max_transforms + 8)

        # 3. Latent Geodesic Path Progression
        for step in range(max_steps):
            curr = committed_path[-1]

            # Terminal check: stage 3 or endable after covering requirements
            if (getattr(curr, "stage", None) == 3 or getattr(curr, "is_endable", False)) and len(committed_path) > 1:
                if step >= max(1, num_clauses - 1) or not (prompt_tokens - covered_tokens):
                    break

            # Find all valid next candidates that unify with curr or DAG scope
            valid_next = [
                c for c in candidates
                if c.cell_id not in visited_ids and (
                    self.step_unifies(curr, c, prev_path=committed_path)
                    or self.step_unifies_dag(c, committed_path, ctx=ctx)
                )
            ]

            if not valid_next:
                break

            cl_idx = min(step + 1, num_clauses - 1)
            target_cl_vec = clause_vecs[cl_idx] if cl_idx < len(clause_vecs) else goal_vec
            target_cl = clauses[cl_idx] if cl_idx < num_clauses else prompt
            _requested_sink = self.requested_sink(prompt, candidates)

            def _score_cand(cand: Cell) -> Tuple[Any, ...]:
                c_vec = self.pathfinder.get_cell_vector(cand)
                latent_goal_h = self.pathfinder.compute_latent_heuristic(current_vec, c_vec, goal_vec)
                latent_cl_h = self.pathfinder.compute_latent_heuristic(current_vec, c_vec, target_cl_vec)
                aff = self.calculate_edge_affinity(curr, cand, orch, relevance_map=relevance_map)
                rel = relevance_map.get(cand.cell_id, 0.0)

                curr_stage = getattr(curr, "stage", 1) or 1
                cand_stage = getattr(cand, "stage", 2) or 2
                progress = 2 if cand_stage > curr_stage else (1 if cand_stage == curr_stage else 0)

                is_cand_sink = (
                    cand_stage == 3
                    or str(getattr(cand, "node_role", "") or getattr(cand, "role", "")).lower() in ("sink", "terminal", "egress", "consumer")
                )
                is_curr_sink = (
                    curr_stage == 3
                    or str(getattr(curr, "node_role", "") or getattr(curr, "role", "")).lower() in ("sink", "terminal", "egress", "consumer")
                )
                unrequested_egress = (
                    is_cand_sink and self._is_path_consumer(cand)
                    and not (dst_lit is not None and self.file_compat(cand, dst_lit) > 0)
                ) or (_requested_sink is None and is_cand_sink and not is_curr_sink and self.clause_fit(target_cl, cand)[0] <= 0.0)

                composite_latent = 0.5 * latent_cl_h + 0.3 * latent_goal_h + 0.2 * (aff + rel)
                return (0 if unrequested_egress else 1, composite_latent, progress)

            valid_next.sort(key=_score_cand, reverse=True)
            best_next = valid_next[0]

            committed_path.append(best_next)
            visited_ids.add(best_next.cell_id)
            covered_tokens.update(getattr(best_next, "identity_tokens", best_next.token_set) & prompt_tokens)

            # Update running latent state vector
            next_vec = self.pathfinder.get_cell_vector(best_next)
            current_vec = 0.7 * current_vec + 0.3 * next_vec

            if step >= num_clauses - 1:
                try:
                    from coverage_audit import audit as _audit
                    _res = _audit(committed_path, prompt, orch)
                    if _res.get("available") and not any(i["kind"] == "clause_unserved" for i in _res.get("issues", [])):
                        self._trace("m11_goal_reached", step=step, path_len=len(committed_path))
                        break
                except Exception:
                    pass

            if (prompt_tokens - covered_tokens) == set() and (getattr(best_next, "stage", None) == 3 or getattr(best_next, "is_endable", False)):
                break

        self._trace("m11_plan_complete", cells=[c.cell_id for c in committed_path])
        return committed_path
