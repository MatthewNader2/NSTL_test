"""
src/route_methods/m6_hybrid_anchors.py - Neuro-Symbolic Topological Lattice (NSTL)
M6: Flagship Hybrid — Clause Anchors + Endpoint Anchors + Beam Bridging + Pre-Flight Linting.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, STOPWORDS

try:
    from ..lattice import Cell, LatticeOrchestrator
    from ..tokenizer import CellTokenizer
    from ..unification import unify, UnificationGate, ExecutionContext
    from ..preflight import PreflightLinter
    from ..planner import LatticePlanner
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from tokenizer import CellTokenizer
    from unification import unify, UnificationGate, ExecutionContext
    from preflight import PreflightLinter
    from planner import LatticePlanner


def _prompt_has_egress_intent(prompt: str) -> bool:
    """Declared-egress-vocabulary intent test (see planner.EGRESS_INTENT_TOKENS);
    when not an in-memory variable target."""
    try:
        from .planner import EGRESS_INTENT_TOKENS
    except (ImportError, ValueError):
        from planner import EGRESS_INTENT_TOKENS
    target = ExecutionContext._extract_target_sink(prompt or "")
    toks = set(CellTokenizer.tokenize_prompt((prompt or "").lower()))
    return bool(toks & EGRESS_INTENT_TOKENS) and not target


class M6HybridAnchorsRouteMethod(RouteMethod):
    """
    RouteMethod M6: Flagship Hybrid.
    Integrates clause-level waypoint anchoring, source/sink endpoint binding,
    Viterbi beam bridging, and static pre-flight linting into a unified routing strategy.
    """
    name: str = "m6_hybrid_anchors"

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

        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant" and not c.cell_id.startswith("MACRO_")]
        if not candidates:
            return [tunnel[0]]

        # 1. Extract universal literals (files, identifiers, etc.)
        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)

        # 2. Identify Source & Sink Endpoints
        source_cell: Optional[Cell] = None
        sink_cell: Optional[Cell] = None

        stage1_cands = [c for c in candidates if getattr(c, "stage", None) == 1]
        stage3_cands = [c for c in candidates if getattr(c, "stage", None) == 3]

        if stage1_cands:
            def _score_s1(c: Cell) -> float:
                score = relevance_map.get(c.cell_id, 0.0)
                if src_file_literals:
                    is_pc = any(
                        getattr(p, "abstract_type", None) == "path"
                        or getattr(p, "port_role", None) == "source_data"
                        or getattr(getattr(p, "signature", None), "abstract_type", None) == "path"
                        for p in c.inputs.values()
                    )
                    if is_pc and self.is_file_format_compatible(c, str(src_file_literals[0])):
                        score += 10.0
                    elif is_pc and not self.is_file_format_compatible(c, str(src_file_literals[0])):
                        score -= 20.0
                    elif not is_pc:
                        score -= 10.0
                return score
            source_cell = max(stage1_cands, key=_score_s1)
        if stage3_cands and (_prompt_has_egress_intent(prompt) or dest_file_literals):
            def _score_s3(c: Cell) -> float:
                score = relevance_map.get(c.cell_id, 0.0) * 5.0
                prompt_toks = CellTokenizer.tokenize_prompt(prompt)
                c_toks = getattr(c, "identity_tokens", c.token_set)
                score += len(prompt_toks & c_toks) * 3.0
                if dest_file_literals:
                    is_pc = any(
                        getattr(p, "abstract_type", None) == "path"
                        or getattr(getattr(p, "signature", None), "abstract_type", None) == "path"
                        for p in c.inputs.values()
                    )
                    if is_pc and self.is_file_format_compatible(c, str(dest_file_literals[0])):
                        score += 15.0
                    elif is_pc and not self.is_file_format_compatible(c, str(dest_file_literals[0])):
                        score -= 25.0
                return score
            sink_cell = max(stage3_cands, key=_score_s3)

        # 3. Identify Clause-Level Waypoint Anchors
        clauses = self.segment_prompt_clauses(prompt)
        clause_anchors: List[Cell] = []

        for idx, cl in enumerate(clauses):
            cl_tokens = CellTokenizer.tokenize_prompt(cl) - STOPWORDS
            if not cl_tokens:
                continue

            cl_literals = [
                v for _, k, v in ExecutionContext._extract_universal_literals(cl)
                if k in ("identifier", "quoted_str")
            ]

            clause_cells: List[Cell] = []
            uncovered_tokens = set(cl_tokens)

            while uncovered_tokens:
                best_c: Optional[Cell] = None
                best_score = -1.0
                best_covered: Set[str] = set()

                for c in candidates:
                    if c in clause_cells:
                        continue
                    if getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator":
                        continue
                    c_toks = c.token_set
                    id_toks = getattr(c, "identity_tokens", c_toks)
                    strong_overlap = len(uncovered_tokens & id_toks)
                    if strong_overlap == 0 and clause_cells:
                        continue
                    weak_overlap = len((uncovered_tokens & c_toks) - id_toks)
                    rel_score = relevance_map.get(c.cell_id, 0.0)
                    last_anc = clause_cells[-1] if clause_cells else (clause_anchors[-1] if clause_anchors else None)
                    aff_score = self.calculate_edge_affinity(last_anc, c, orch, relevance_map=relevance_map) if last_anc else 0.0
                    scope_so_far = clause_anchors + clause_cells
                    unif_bonus = 5.0 if (last_anc and (self.step_unifies(last_anc, c, prev_path=scope_so_far) or self.step_unifies_dag(c, scope_so_far, ctx=ctx))) else 0.0
                    domain_bonus = 2.0 if (last_anc and c.domain_name == last_anc.domain_name) else 0.0
                    sc = (
                        (strong_overlap * 4.0)
                        + (weak_overlap * 1.5)
                        + (rel_score * 5.0)
                        + (aff_score * 4.0)
                        + unif_bonus
                        + domain_bonus
                    )
                    if sc > best_score:
                        best_score = sc
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
                if not clause_anchors or clause_anchors[-1].cell_id != c.cell_id or getattr(clause_anchors[-1], "clause_literals", None) != getattr(c, "clause_literals", None):
                    clause_anchors.append(c)

        # 4. Assemble Waypoints
        waypoints: List[Cell] = []
        if source_cell:
            waypoints.append(source_cell)
        for ca in clause_anchors:
            if not waypoints or waypoints[-1].cell_id != ca.cell_id or getattr(waypoints[-1], "clause_literals", None) != getattr(ca, "clause_literals", None):
                if not sink_cell or ca.cell_id != sink_cell.cell_id:
                    waypoints.append(ca)
        if sink_cell:
            if not waypoints or waypoints[-1].cell_id != sink_cell.cell_id:
                waypoints.append(sink_cell)

        if not waypoints:
            waypoints = [candidates[0]]

        # 5. Bridge gaps between waypoints using lattice search
        bridged_path: List[Cell] = [waypoints[0]]
        for nxt in waypoints[1:]:
            curr = bridged_path[-1]
            if curr.cell_id == nxt.cell_id:
                continue
            if self.step_unifies(curr, nxt, prev_path=bridged_path) or self.step_unifies_dag(nxt, bridged_path, ctx=ctx):
                bridged_path.append(nxt)
            else:
                bridge = self.find_bridge(curr, nxt, candidates, orch, prev_path=bridged_path)
                if bridge:
                    bridged_path.append(bridge)
                    bridged_path.append(nxt)
                else:
                    connected = False
                    for ancestor in reversed(bridged_path[:-1]):
                        if self.step_unifies(ancestor, nxt, prev_path=bridged_path) or self.step_unifies_dag(nxt, bridged_path, ctx=ctx):
                            bridged_path.append(nxt)
                            connected = True
                            break
                        anc_bridge = self.find_bridge(ancestor, nxt, candidates, orch, prev_path=bridged_path)
                        if anc_bridge:
                            bridged_path.extend([anc_bridge, nxt])
                            connected = True
                            break

        # 6. Pre-flight verification & path return
        num_clauses = len(clauses) if clauses else 1
        max_allowed = max(32, num_clauses * 4 + 4, max_transforms + 8)

        if bridged_path and len(bridged_path) >= 2:
            return bridged_path[:max_allowed]

        # If waypoints failed to assemble, fallback to Trellis
        if orch:
            planner = LatticePlanner(orchestrator=orch)
            m0_path = planner.plan(
                prompt=prompt,
                tunnel=tunnel,
                relevance_map=relevance_map,
                start_sig=start_sig,
                goal_sig=goal_sig,
                max_transforms=max_transforms
            )
            if m0_path:
                return m0_path

        return bridged_path[:max_allowed]
