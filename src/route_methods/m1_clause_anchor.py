"""
src/route_methods/m1_clause_anchor.py - Neuro-Symbolic Topological Lattice (NSTL)
M1: Clause-Anchored Insertion.
Anchors key morphisms for each linguistic clause in the prompt and bridges type gaps.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, STOPWORDS

try:
    from ..lattice import Cell, LatticeOrchestrator, TypeRegistry
    from ..tokenizer import CellTokenizer
    from ..unification import unify, ExecutionContext
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator, TypeRegistry
    from tokenizer import CellTokenizer
    from unification import unify, ExecutionContext


def _clause_has_egress_intent(clause: str) -> bool:
    """Language-level materialization intent for a clause: declared egress
    verbs (planner.EGRESS_INTENT_TOKENS) or an explicit file asset literal —
    the two ways English marks an output destination. No ad-hoc fragment
    lists ("to_")."""
    try:
        from .planner import EGRESS_INTENT_TOKENS
    except (ImportError, ValueError):
        from planner import EGRESS_INTENT_TOKENS
    toks = set(CellTokenizer.tokenize_prompt(clause.lower()))
    if toks & EGRESS_INTENT_TOKENS:
        return True
    literals = ExecutionContext._extract_universal_literals(clause)
    return any(kind == "file_asset" for _, kind, _ in literals)


class M1ClauseAnchorRouteMethod(RouteMethod):
    """
    RouteMethod M1: Clause-Anchored Insertion.
    Segments user prompt into sequential intent clauses, extracts high-confidence anchor cells
    for each clause, and synthesizes/routes type-valid bridges between consecutive anchors.
    """
    name: str = "m1_clause_anchor"

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

        clauses = self.segment_prompt_clauses(prompt)
        anchors: List[Cell] = []

        # 1. Identify best anchor cell for each clause with stage awareness
        for idx, cl in enumerate(clauses):
            cl_tokens = CellTokenizer.tokenize_prompt(cl)
            if not cl_tokens:
                continue

            if idx == 0 and len(clauses) > 1:
                pool = [c for c in candidates if getattr(c, "stage", None) == 1] or candidates
            elif idx == len(clauses) - 1 and len(clauses) > 1 and _clause_has_egress_intent(cl):
                pool = [c for c in candidates if getattr(c, "stage", None) == 3] or candidates
            else:
                pool = [c for c in candidates if getattr(c, "stage", None) != 1] or candidates

            best_c: Optional[Cell] = None
            best_score = -1.0

            for c in pool:
                c_toks = c.token_set
                id_toks = getattr(c, "identity_tokens", c_toks)
                
                strong_overlap = len(cl_tokens & id_toks)
                weak_overlap = len((cl_tokens & c_toks) - id_toks)
                rel_score = relevance_map.get(c.cell_id, 0.0)
                aff_score = self.calculate_edge_affinity(anchors[-1], c, orch) if anchors else 0.0
                unif_bonus = 5.0 if (anchors and self.step_unifies(anchors[-1], c)) else 0.0
                domain_bonus = 2.0 if (anchors and c.domain_name == anchors[-1].domain_name) else 0.0

                score = (
                    (strong_overlap * 4.0)
                    + (weak_overlap * 1.5)
                    + (rel_score * 5.0)
                    + (aff_score * 4.0)
                    + unif_bonus
                    + domain_bonus
                )

                if score > best_score:
                    best_score = score
                    best_c = c

            if best_c and (not anchors or anchors[-1].cell_id != best_c.cell_id):
                anchors.append(best_c)

        if not anchors:
            return [candidates[0]]

        # 2. Bridge gaps between consecutive anchors
        routed_path: List[Cell] = [anchors[0]]

        for next_anchor in anchors[1:]:
            curr_cell = routed_path[-1]

            if self.step_unifies(curr_cell, next_anchor, prev_path=routed_path):
                routed_path.append(next_anchor)
            else:
                bridge = self.find_bridge(curr_cell, next_anchor, candidates, orch)
                if bridge:
                    routed_path.append(bridge)
                    routed_path.append(next_anchor)
                else:
                    # If bridge not found, skip incompatible anchor to avoid pipeline breakage
                    pass

        # 3. Handle file-asset endpoint completion if declared in prompt
        l0_extracted = ExecutionContext._extract_universal_literals(prompt or "") if ctx or prompt else []
        file_literals = [v for _, t, v in l0_extracted if t == "file_asset"]

        if file_literals and len(file_literals) >= 2:
            last_cell = routed_path[-1]
            if getattr(last_cell, "stage", None) != 3:
                sinks = [c for c in candidates if getattr(c, "stage", None) == 3]
                best_sink = None
                best_s_score = -1.0
                for s in sinks:
                    if self.step_unifies(last_cell, s, prev_path=routed_path):
                        sc = relevance_map.get(s.cell_id, 0.0) + self.calculate_edge_affinity(last_cell, s, orch)
                        if sc > best_s_score:
                            best_s_score = sc
                            best_sink = s
                if best_sink:
                    routed_path.append(best_sink)

        # 4. Monadic pipeline validation check: if broken, fallback to LatticePlanner
        if orch:
            try:
                from ..unification import UnificationGate
                gate = UnificationGate(orchestrator=orch)
                test_res = gate.unify_pipeline(routed_path, ExecutionContext(prompt=prompt))
                if test_res.is_bottom():
                    from ..planner import LatticePlanner
                    planner = LatticePlanner(orchestrator=orch)
                    return planner.plan(
                        prompt=prompt,
                        tunnel=tunnel,
                        relevance_map=relevance_map,
                        start_sig=start_sig,
                        goal_sig=goal_sig,
                        max_transforms=max_transforms
                    )
            except Exception:
                pass

        return routed_path[:max_transforms + 2]
