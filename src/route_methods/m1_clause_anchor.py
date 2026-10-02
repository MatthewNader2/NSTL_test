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
    verbs (planner.EGRESS_INTENT_TOKENS) when not an in-memory variable target."""
    try:
        from .planner import EGRESS_INTENT_TOKENS
    except (ImportError, ValueError):
        from planner import EGRESS_INTENT_TOKENS
    target = ExecutionContext._extract_target_sink(clause)
    toks = set(CellTokenizer.tokenize_prompt(clause.lower()))
    return bool(toks & EGRESS_INTENT_TOKENS) and not target


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
        self._current_prompt = prompt
        orch = orchestrator or self.orchestrator
        if not tunnel:
            return []
        if len(tunnel) == 1:
            return [tunnel[0]]

        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant" and not c.cell_id.startswith("MACRO_")]
        if not candidates:
            return [tunnel[0]]

        clauses = self.segment_prompt_clauses(prompt)
        anchors: List[Cell] = []

        # 1. Identify anchor cells for each clause via semantic set-covering and stage awareness
        for idx, cl in enumerate(clauses):
            cl_tokens = CellTokenizer.tokenize_prompt(cl) - STOPWORDS
            if not cl_tokens:
                continue

            cl_literals = [
                v for _, k, v in ExecutionContext._extract_universal_literals(cl)
                if k in ("identifier", "quoted_str")
            ]

            if idx == 0 and len(clauses) > 1:
                pool = [c for c in candidates if getattr(c, "stage", None) == 1] or candidates
            elif idx == len(clauses) - 1 and len(clauses) > 1 and _clause_has_egress_intent(cl):
                pool = [c for c in candidates if getattr(c, "stage", None) == 3] or candidates
            else:
                pool = [c for c in candidates if getattr(c, "stage", None) != 1] or candidates

            pool = [
                c for c in pool
                if not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
            ]

            # Semantic set-covering over distinct operational concepts in the clause
            clause_cells: List[Cell] = []
            uncovered_tokens = set(cl_tokens)

            while uncovered_tokens:
                best_c: Optional[Cell] = None
                best_score = -1.0
                best_covered: Set[str] = set()

                for c in pool:
                    if c in clause_cells:
                        continue
                    c_toks = c.token_set
                    id_toks = getattr(c, "identity_tokens", c_toks)
                    strong_overlap = len(uncovered_tokens & id_toks)
                    if strong_overlap == 0 and clause_cells:
                        continue
                    weak_overlap = len((uncovered_tokens & c_toks) - id_toks)
                    rel_score = relevance_map.get(c.cell_id, 0.0)
                    last_anc = clause_cells[-1] if clause_cells else (anchors[-1] if anchors else None)
                    aff_score = self.calculate_edge_affinity(last_anc, c, orch, relevance_map=relevance_map) if last_anc else 0.0
                    scope_so_far = anchors + clause_cells
                    unif_bonus = 5.0 if (last_anc and (self.step_unifies(last_anc, c, prev_path=scope_so_far) or self.step_unifies_dag(c, scope_so_far, ctx=ctx))) else 0.0
                    domain_bonus = 2.0 if (last_anc and c.domain_name == last_anc.domain_name) else 0.0

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
                        best_covered = (uncovered_tokens & id_toks) or (uncovered_tokens & c_toks)

                if best_c and best_covered:
                    c_inst = best_c.clone() if hasattr(best_c, "clone") else best_c
                    c_inst.matched_clause_idx = idx
                    c_inst.clause_literals = cl_literals
                    clause_cells.append(c_inst)
                    uncovered_tokens -= best_covered
                else:
                    break

            if len(clause_cells) >= 2:
                from functools import cmp_to_key
                def _reach_cmp(a: Cell, b: Cell) -> int:
                    if a.cell_id == b.cell_id:
                        return 0
                    aff_ab = self.calculate_edge_affinity(a, b, orch)
                    aff_ba = self.calculate_edge_affinity(b, a, orch)
                    if aff_ab > aff_ba:
                        return -1
                    if aff_ba > aff_ab:
                        return 1
                    return 0
                clause_cells = sorted(clause_cells, key=cmp_to_key(_reach_cmp))

            for c in clause_cells:
                if not anchors or anchors[-1].cell_id != c.cell_id or getattr(anchors[-1], "clause_literals", None) != getattr(c, "clause_literals", None):
                    anchors.append(c)

        if not anchors:
            return [candidates[0]]

        # 2. Bridge gaps between consecutive anchors
        routed_path: List[Cell] = [anchors[0]]

        for next_anchor in anchors[1:]:
            curr_cell = routed_path[-1]

            if self.step_unifies(curr_cell, next_anchor, prev_path=routed_path):
                routed_path.append(next_anchor)
            elif self.step_unifies_dag(next_anchor, routed_path, ctx=ctx):
                routed_path.append(next_anchor)
            else:
                bridge = self.find_bridge(curr_cell, next_anchor, candidates, orch, prev_path=routed_path)
                if bridge:
                    routed_path.append(bridge)
                    routed_path.append(next_anchor)
                else:
                    # In a DAG, if next_anchor does not unify directly with curr_cell,
                    # check if next_anchor unifies or bridges from any earlier ancestor in routed_path
                    connected = False
                    for ancestor in reversed(routed_path[:-1]):
                        if self.step_unifies(ancestor, next_anchor, prev_path=routed_path) or self.step_unifies_dag(next_anchor, routed_path, ctx=ctx):
                            routed_path.append(next_anchor)
                            connected = True
                            break
                        anc_bridge = self.find_bridge(ancestor, next_anchor, candidates, orch, prev_path=routed_path)
                        if anc_bridge:
                            routed_path.extend([anc_bridge, next_anchor])
                            connected = True
                            break

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
                        sc = relevance_map.get(s.cell_id, 0.0) + self.calculate_edge_affinity(last_cell, s, orch, relevance_map=relevance_map)
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

        max_allowed = max(32, len(clauses) * 4 + 4, max_transforms + 8)
        return routed_path[:max_allowed]
