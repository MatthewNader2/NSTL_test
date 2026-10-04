"""
src/route_methods/m2_endpoint_anchor.py - Neuro-Symbolic Topological Lattice (NSTL)
M2: Endpoint-Anchored Coverage (Bidirectional).
Anchors source entry and sink exit morphisms and searches bidirectionally.
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


class M2EndpointAnchorRouteMethod(RouteMethod):
    """
    RouteMethod M2: Endpoint-Anchored Coverage.
    Identifies source and destination anchors from prompt assets and constraints,
    then executes bidirectional meeting-in-the-middle pathfinding.
    """
    name: str = "m2_endpoint_anchor"

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

        # 1. Identify Source Anchor (Stage 1)
        source_candidates = [c for c in candidates if getattr(c, "stage", None) == 1]
        best_source: Optional[Cell] = None
        best_src_score = -999.0
        for sc in (source_candidates or candidates[:5]):
            score = (
                relevance_map.get(sc.cell_id, 0.0) * 5.0
                + len(prompt_tokens & getattr(sc, "identity_tokens", sc.token_set)) * 3.0
            )
            is_path_consumer = any(
                getattr(p, "abstract_type", None) == "path"
                or getattr(p, "port_role", None) == "source_data"
                or getattr(p, "derived_role", None) == "source_data"
                or getattr(getattr(p, "signature", None), "abstract_type", None) == "path"
                for p in sc.inputs.values()
            )
            if src_file_literals and is_path_consumer:
                if self.is_file_format_compatible(sc, str(src_file_literals[0])):
                    score += 6.0
                else:
                    score -= 20.0
            if score > best_src_score:
                best_src_score = score
                best_source = sc

        # 2. Identify Sink Anchor (Stage 3 or Endable Evaluator)
        clauses = self.segment_prompt_clauses(prompt)
        final_clause_tokens = (CellTokenizer.tokenize_prompt(clauses[-1]) - STOPWORDS) if clauses else prompt_tokens
        sink_candidates = [c for c in candidates if getattr(c, "stage", None) == 3 or getattr(c, "is_endable", False)]
        best_sink: Optional[Cell] = None
        best_sink_score = -999.0
        for snk in (sink_candidates or candidates[-5:]):
            score = (
                relevance_map.get(snk.cell_id, 0.0) * 5.0
                + len(prompt_tokens & getattr(snk, "identity_tokens", snk.token_set)) * 3.0
                + len(final_clause_tokens & getattr(snk, "identity_tokens", snk.token_set)) * 5.0
            )
            is_path_consumer = any(
                getattr(p, "abstract_type", None) == "path"
                or getattr(p, "port_role", None) in ("source_data", "model_sink")
                or getattr(p, "derived_role", None) in ("source_data", "model_sink")
                or getattr(p.signature, "abstract_type", None) == "path"
                for p in snk.inputs.values()
            )
            if dest_file_literals and is_path_consumer:
                if self.is_file_format_compatible(snk, str(dest_file_literals[0])):
                    score += 6.0
                else:
                    score -= 20.0
            elif not dest_file_literals and is_path_consumer and getattr(snk, "stage", None) == 3:
                # Egress file writer without destination file requested in prompt
                score -= 25.0

            if score > best_sink_score:
                best_sink_score = score
                best_sink = snk

        if not best_source:
            best_source = candidates[0]

        # Single-node pipeline if source is also sink or only one anchor makes sense
        if best_sink and best_source.cell_id == best_sink.cell_id:
            return [best_source]

        clauses = self.segment_prompt_clauses(prompt)
        num_clauses = len(clauses) if clauses else 1

        # 3. For multi-clause compound prompts (> 2 clauses), chain forward through intermediate transformations
        if num_clauses > 2:
            path = [best_source]
            curr = best_source
            max_steps = max(2, min(16, max(max_transforms + 2, num_clauses + 3)))
            for cl_idx in range(1, num_clauses + 2):
                if getattr(curr, "stage", None) == 3 or (best_sink and curr.cell_id == best_sink.cell_id):
                    break
                valid_next = [
                    c for c in candidates
                    if c.cell_id != curr.cell_id and c.cell_id not in [p.cell_id for p in path]
                    and (self.step_unifies(curr, c, prev_path=path) or self.step_unifies_dag(c, path, ctx=ctx))
                ]
                if not valid_next:
                    if best_sink and (self.step_unifies(curr, best_sink, prev_path=path) or self.step_unifies_dag(best_sink, path, ctx=ctx)) and best_sink.cell_id not in [p.cell_id for p in path]:
                        path.append(best_sink)
                    break

                target_cl = clauses[min(cl_idx, num_clauses - 1)] if cl_idx < num_clauses else ""
                target_toks = (CellTokenizer.tokenize_prompt(target_cl) - STOPWORDS) if target_cl else prompt_tokens

                def _score_step(c: Cell) -> float:
                    rel = relevance_map.get(c.cell_id, 0.0)
                    aff = self.calculate_edge_affinity(curr, c, orch, relevance_map=relevance_map)
                    c_toks = getattr(c, "identity_tokens", c.token_set)
                    match_sc = len(target_toks & c_toks) * 4.0 + len(prompt_tokens & c_toks) * 1.5
                    is_snk = 3.0 if (best_sink and c.cell_id == best_sink.cell_id and cl_idx >= num_clauses - 1) else 0.0
                    return (rel * 8.0) + (aff * 4.0) + match_sc + is_snk

                valid_next.sort(key=_score_step, reverse=True)
                nxt = valid_next[0]
                path.append(nxt)
                curr = nxt

            if best_sink and path[-1].cell_id != best_sink.cell_id:
                if self.step_unifies(path[-1], best_sink, prev_path=path) or self.step_unifies_dag(best_sink, path, ctx=ctx):
                    path.append(best_sink)
            return path

        # Bidirectional Meeting-in-the-Middle through Stage 2 transforms (short prompts)
        transforms = [c for c in candidates if getattr(c, "stage", None) == 2]

        if best_sink and transforms:
            # Look for 1-hop bridge: source -> T -> sink
            best_mid: Optional[Cell] = None
            best_mid_score = -1.0
            for t in transforms:
                if self.step_unifies(best_source, t) and self.step_unifies(t, best_sink, prev_path=[best_source, t]):
                    sc = (
                        relevance_map.get(t.cell_id, 0.0) * 5.0
                        + len(prompt_tokens & getattr(t, "identity_tokens", t.token_set)) * 3.0
                        + self.calculate_edge_affinity(best_source, t, orch, relevance_map=relevance_map) * 4.0
                        + self.calculate_edge_affinity(t, best_sink, orch, relevance_map=relevance_map) * 4.0
                        + (3.0 if t.domain_name == best_source.domain_name else 0.0)
                    )
                    if sc > best_mid_score:
                        best_mid_score = sc
                        best_mid = t
            if best_mid:
                return [best_source, best_mid, best_sink]

            # Look for 2-hop bridge: source -> T1 -> T2 -> sink
            best_pair: Optional[Tuple[Cell, Cell]] = None
            best_pair_score = -1.0
            for t1 in transforms:
                if not self.step_unifies(best_source, t1):
                    continue
                for t2 in transforms:
                    if t1.cell_id == t2.cell_id:
                        continue
                    if self.step_unifies(t1, t2, prev_path=[best_source, t1]) and self.step_unifies(t2, best_sink, prev_path=[best_source, t1, t2]):
                        sc = (
                            relevance_map.get(t1.cell_id, 0.0) * 5.0
                            + relevance_map.get(t2.cell_id, 0.0) * 5.0
                            + len(prompt_tokens & (t1.token_set | t2.token_set)) * 3.0
                            + self.calculate_edge_affinity(best_source, t1, orch, relevance_map=relevance_map) * 3.0
                            + self.calculate_edge_affinity(t1, t2, orch, relevance_map=relevance_map) * 3.0
                            + self.calculate_edge_affinity(t2, best_sink, orch, relevance_map=relevance_map) * 3.0
                        )
                        if sc > best_pair_score:
                            best_pair_score = sc
                            best_pair = (t1, t2)
            if best_pair:
                return [best_source, best_pair[0], best_pair[1], best_sink]

        # If direct unification is valid when no transform found and prompt is trivial
        if best_sink and self.step_unifies(best_source, best_sink) and num_clauses <= 1:
            return [best_source, best_sink]

        # Fallback forward chaining
        path = [best_source]
        curr = best_source
        max_steps = max(32, num_clauses * 4 + 4, max_transforms + 8)
        for _ in range(max_steps):
            valid_next = [c for c in candidates if c.cell_id != curr.cell_id and self.step_unifies(curr, c)]
            if not valid_next:
                break
            valid_next.sort(
                key=lambda c: relevance_map.get(c.cell_id, 0.0) + self.calculate_edge_affinity(curr, c, orch, relevance_map=relevance_map),
                reverse=True
            )
            nxt = valid_next[0]
            path.append(nxt)
            curr = nxt
            if getattr(curr, "stage", None) == 3:
                break

        return path
