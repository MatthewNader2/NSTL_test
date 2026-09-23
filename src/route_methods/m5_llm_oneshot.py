from __future__ import annotations
import json, re
from typing import Dict, List, Optional, Any
from .base import RouteMethod
from .m1_clause_anchor import M1ClauseAnchorRouteMethod
from lattice import Cell, LatticeOrchestrator
from unification import ExecutionContext
from inference import ModelManager


def _ir_filter_m5(candidates, ir_step):
    if not ir_step: return candidates
    op  = str(ir_step.get("op", "")).strip().lower()
    lib = str(ir_step.get("library", "")).strip().lower()
    if not op and not lib: return candidates
    out = []
    for c in candidates:
        cid = c.cell_id.lower(); dom = (getattr(c, "domain_name", "") or "").lower()
        toks = set(str(k).lower() for k in getattr(c, "keywords", []) or [])
        toks |= set(str(k).lower() for k in getattr(c, "semantic_tags", []) or [])
        if (not op or op in cid or any(op in t for t in toks)) and (not lib or lib in dom or lib in cid):
            out.append(c)
    return out or candidates


class M5LLMOneShotRouteMethod(RouteMethod):
    name = "m5_llm_oneshot"

    def plan(self, prompt, tunnel, relevance_map, orchestrator=None, ctx=None,
             start_sig=None, goal_sig=None, max_transforms=6, **kwargs):
        orch = orchestrator or self.orchestrator
        if not tunnel: return []
        if len(tunnel) == 1: return [tunnel[0]]
        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"] or [tunnel[0]]
        cand_map = {c.cell_id.lower(): c for c in candidates}
        mm = ModelManager.get_instance()
        has_llm = mm.profile is not None and getattr(mm.profile, "llm", None) is not None

        proposed = []
        if has_llm:
            try:
                from ir_compiler import IRCompiler
                compiler = getattr(self, "_cached_ir", None) or IRCompiler(orch)
                self._cached_ir = compiler
                ir = compiler.compile(prompt)
                if ir is not None and ir.steps:
                    for step in ir.steps:
                        pool = _ir_filter_m5(candidates, step)
                        pool.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0) * 5.0, reverse=True)
                        if pool and (not proposed or pool[0].cell_id != proposed[-1].cell_id):
                            proposed.append(pool[0])
            except Exception:
                proposed = []

        if not proposed and has_llm:
            try:
                top_str = "\n".join(f"- {c.cell_id}" for c in sorted(candidates, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)[:20])
                resp = mm.generate_text(f"Intent: {prompt}\nCells:\n{top_str}\nReturn JSON list of cell IDs in order:", max_tokens=128, system_prompt="Output ONLY a JSON list.").strip()
                m = re.search(r"\[.*?\]", resp, re.DOTALL)
                if m:
                    for cid in json.loads(m.group(0)):
                        if str(cid).strip().lower() in cand_map: proposed.append(cand_map[str(cid).strip().lower()])
            except Exception:
                proposed = []

        if not proposed:
            return M1ClauseAnchorRouteMethod(orchestrator=orch).plan(
                prompt, tunnel, relevance_map, orchestrator=orch, ctx=ctx,
                start_sig=start_sig, goal_sig=goal_sig, max_transforms=max_transforms)

        chain = [proposed[0]]
        for nxt in proposed[1:]:
            curr = chain[-1]
            if self.step_unifies(curr, nxt): chain.append(nxt)
            else:
                bridge = self.find_bridge(curr, nxt, candidates, orch)
                if bridge: chain.extend([bridge, nxt])
                else: return M1ClauseAnchorRouteMethod(orchestrator=orch).plan(
                    prompt, tunnel, relevance_map, orchestrator=orch, ctx=ctx,
                    start_sig=start_sig, goal_sig=goal_sig, max_transforms=max_transforms)
        return chain[:min(16, max(max_transforms + 2, 4))]
