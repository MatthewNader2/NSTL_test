from __future__ import annotations
import json
from typing import Dict, List, Optional, Any
from .base import RouteMethod
from .m1_clause_anchor import M1ClauseAnchorRouteMethod
from lattice import Cell, LatticeOrchestrator
from unification import ExecutionContext
from inference import ModelManager
try:
    from ..tokenizer import CellTokenizer
except (ImportError, ValueError):
    from tokenizer import CellTokenizer


def _ir_filter_m5(candidates, ir_step):
    if not ir_step: return candidates
    op  = str(ir_step.get("op", "")).strip().lower()
    lib = str(ir_step.get("library", "")).strip().lower()
    if not op and not lib: return candidates
    out = []
    for c in candidates:
        cid = c.cell_id.lower(); dom = (getattr(c, "domain_name", "") or getattr(c, "domain", "") or "").lower()
        toks = set(str(k).lower() for k in getattr(c, "keywords", []) or [])
        toks |= set(str(k).lower() for k in getattr(c, "semantic_tags", []) or [])
        if (not op or op in cid or any(op in t for t in toks)) and (not lib or lib in dom or lib in cid):
            out.append(c)
    if out:
        return out
    op_tokens = CellTokenizer.tokenize_identifier(op)
    if op_tokens:
        for c in candidates:
            c_toks = getattr(c, "identity_tokens", c.token_set)
            if op_tokens & c_toks:
                out.append(c)
    return out


class M5LLMOneShotRouteMethod(RouteMethod):
    name = "m5_llm_oneshot"

    def plan(self, prompt, tunnel, relevance_map, orchestrator=None, ctx=None,
             start_sig=None, goal_sig=None, max_transforms=6, **kwargs):
        orch = orchestrator or self.orchestrator
        if not tunnel: return []
        candidates = [
            c for c in tunnel
            if getattr(c, "node_type", "") not in ("constant", "macro")
            and not c.cell_id.startswith("MACRO_")
            and not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
        ] or [tunnel[0]]
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
                        if pool:
                            chosen = None
                            for cand in pool:
                                if not any(p.cell_id == cand.cell_id for p in proposed):
                                    chosen = cand
                                    break
                            if chosen is not None:
                                proposed.append(chosen)
            except Exception:
                proposed = []

        if not proposed and has_llm:
            try:
                top_str = "\n".join(f"- {c.cell_id}" for c in sorted(candidates, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)[:20])
                resp = mm.generate_text(f"Intent: {prompt}\nCells:\n{top_str}\nReturn JSON list of cell IDs in order:", max_tokens=128, system_prompt="Output ONLY a JSON list.").strip()
                parsed_list = None
                if "[" in resp and "]" in resp:
                    lbracket = resp.find("[")
                    rbracket = resp.rfind("]")
                    try:
                        parsed_list = json.loads(resp[lbracket : rbracket + 1])
                    except Exception:
                        pass
                if isinstance(parsed_list, list):
                    for cid in parsed_list:
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
            if self.step_unifies(curr, nxt, prev_path=chain) or self.step_unifies_dag(nxt, chain, ctx=ctx):
                chain.append(nxt)
            else:
                bridge = self.find_bridge(curr, nxt, candidates, orch, prev_path=chain)
                if bridge:
                    chain.extend([bridge, nxt])
                else:
                    connected = False
                    for anc in reversed(chain[:-1]):
                        if self.step_unifies(anc, nxt, prev_path=chain) or self.step_unifies_dag(nxt, chain, ctx=ctx):
                            chain.append(nxt)
                            connected = True
                            break
                        anc_brg = self.find_bridge(anc, nxt, candidates, orch, prev_path=chain)
                        if anc_brg:
                            chain.extend([anc_brg, nxt])
                            connected = True
                            break
                    if not connected:
                        return M1ClauseAnchorRouteMethod(orchestrator=orch).plan(
                            prompt, tunnel, relevance_map, orchestrator=orch, ctx=ctx,
                            start_sig=start_sig, goal_sig=goal_sig, max_transforms=max_transforms
                        )
        clauses = self.segment_prompt_clauses(prompt)
        max_allowed = max(32, len(clauses) * 4 + 4, max_transforms + 8)
        return chain[:max_allowed]

