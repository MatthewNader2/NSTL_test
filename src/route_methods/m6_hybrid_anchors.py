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

        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"]
        if not candidates:
            return [tunnel[0]]

        # 1. Extract universal literals (files, identifiers, etc.)
        extracted_literals = ExecutionContext._extract_universal_literals(prompt or "")
        file_literals = [v for _, t, v in extracted_literals if t == "file_asset"]

        # 2. Identify Source & Sink Endpoints
        source_cell: Optional[Cell] = None
        sink_cell: Optional[Cell] = None

        stage1_cands = [c for c in candidates if getattr(c, "stage", None) == 1]
        stage3_cands = [c for c in candidates if getattr(c, "stage", None) == 3]

        if stage1_cands:
            source_cell = max(stage1_cands, key=lambda c: relevance_map.get(c.cell_id, 0.0))
        if stage3_cands and (file_literals or any(w in prompt.lower() for w in ("save", "write", "to_", "output"))):
            sink_cell = max(stage3_cands, key=lambda c: relevance_map.get(c.cell_id, 0.0))

        # 3. Identify Clause-Level Waypoint Anchors
        clauses = self.segment_prompt_clauses(prompt)
        clause_anchors: List[Cell] = []
        for cl in clauses:
            cl_tokens = CellTokenizer.tokenize_prompt(cl) - STOPWORDS
            if not cl_tokens:
                continue
            best_c = None
            best_score = -1.0
            for c in candidates:
                c_toks = c.token_set - STOPWORDS
                id_toks = getattr(c, "identity_tokens", c_toks)
                strong_overlap = len(cl_tokens & c_toks & id_toks)
                weak_overlap = len((cl_tokens & c_toks) - id_toks)
                sc = (strong_overlap * 4.0) + (weak_overlap * 1.5) + (relevance_map.get(c.cell_id, 0.0) * 5.0)
                if sc > best_score:
                    best_score = sc
                    best_c = c
            if best_c and (not clause_anchors or clause_anchors[-1].cell_id != best_c.cell_id):
                clause_anchors.append(best_c)

        # 4. Assemble Waypoints
        waypoints: List[Cell] = []
        if source_cell:
            waypoints.append(source_cell)
        for ca in clause_anchors:
            if not waypoints or waypoints[-1].cell_id != ca.cell_id:
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
            if self.step_unifies(curr, nxt):
                bridged_path.append(nxt)
            else:
                bridge = self.find_bridge(curr, nxt, candidates, orch)
                if bridge:
                    bridged_path.append(bridge)
                bridged_path.append(nxt)

        # 6. Pre-flight verification & alternative exploration
        gate = UnificationGate(orchestrator=getattr(self, "orchestrator", orch))
        exec_ctx = ctx or ExecutionContext(prompt=prompt)
        
        # Test if bridged_path lints cleanly
        try:
            bindings = []
            test_ctx = exec_ctx.clone()
            for c in bridged_path:
                bound_code = gate.unify_cell(c, test_ctx)
                # Recover bindings for cell from test_ctx or gate
                bindings.append((c, dict(test_ctx.variables)))
            lint_res = PreflightLinter.lint(bindings, prompt=prompt)
            if lint_res.is_valid:
                return bridged_path[:max_transforms + 2]
        except Exception:
            pass

        # If waypoints had type break or lint issues, fallback to Viterbi Trellis (M0)
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

        return bridged_path[:max_transforms + 2]
