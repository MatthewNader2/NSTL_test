from __future__ import annotations
import re
from typing import Dict, List, Optional, Set, Any
from .base import RouteMethod
from lattice import Cell, LatticeOrchestrator, TypeRegistry
from tokenizer import CellTokenizer
from unification import ExecutionContext
from inference import ModelManager
from planner import _is_terminal_sink_cell


def _ir_filter(candidates, ir_step):
    """(apply_fixes_v7) Constrain candidates to those matching an IR step."""
    if not ir_step: return candidates
    op  = str(ir_step.get("op", "")).strip().lower()
    lib = str(ir_step.get("library", "")).strip().lower()
    if not op and not lib: return candidates
    matched = []
    for c in candidates:
        cid = c.cell_id.lower()
        dom = (getattr(c, "domain_name", "") or "").lower()
        toks = set(str(k).lower() for k in getattr(c, "keywords", []) or [])
        toks |= set(str(k).lower() for k in getattr(c, "semantic_tags", []) or [])
        op_hit = (not op) or (op in cid) or any(op in t for t in toks)
        lib_hit = (not lib) or (lib in dom) or (lib in cid)
        if op_hit and lib_hit: matched.append(c)
    return matched or candidates


class M4LLMStepwiseRouteMethod(RouteMethod):
    name = "m4_llm_stepwise"

    def plan(self, prompt, tunnel, relevance_map, orchestrator=None, ctx=None,
             start_sig=None, goal_sig=None, max_transforms=6, **kwargs):
        orch = orchestrator or self.orchestrator
        if not tunnel: return []
        if len(tunnel) == 1: return [tunnel[0]]
        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"] or [tunnel[0]]
        mm = ModelManager.get_instance()
        has_llm = mm.profile is not None and getattr(mm.profile, "llm", None) is not None

        ir_steps = []
        if has_llm:
            try:
                from ir_compiler import IRCompiler
                compiler = getattr(self, "_cached_ir", None) or IRCompiler(orch)
                self._cached_ir = compiler
                ir = compiler.compile(prompt)
                if ir is not None: ir_steps = list(ir.steps)
            except Exception: ir_steps = []

        prompt_tokens = CellTokenizer.tokenize_prompt(prompt)
        file_literals = [v for _, t, v in (ExecutionContext._extract_universal_literals(prompt or "") if ctx or prompt else []) if t == "file_asset"]
        def _is_pc(c):
            return any(getattr(p, "abstract_type", None) == "path"
                       or getattr(p, "port_role", None) in ("source_data", "model_sink")
                       or getattr(p.signature, "abstract_type", None) == "path"
                       for p in c.inputs.values())
        entry_pool = [c for c in candidates if getattr(c, "stage", None) == 1] or candidates[:5]
        best_entry = max(entry_pool,
                         key=lambda c: relevance_map.get(c.cell_id, 0.0) * 5.0
                         + len(prompt_tokens & getattr(c, "identity_tokens", c.token_set)) * 3.0
                         + (5.0 if file_literals and _is_pc(c) else 0.0))

        path, visited = [best_entry], {best_entry.cell_id}
        clauses = self.segment_prompt_clauses(prompt)
        max_steps = max(2, min(16, max(max_transforms + 2, (len(clauses) if clauses else 1) + 3)))

        has_reg = bool(prompt_tokens & {"regression", "regressor", "continuous"}) and not bool(prompt_tokens & {"classification", "classifier"})
        has_cls = bool(prompt_tokens & {"classification", "classifier"}) and not bool(prompt_tokens & {"regression", "regressor"})

        def _cov_cnt(pc):
            return sum(1 for cl in clauses
                       if any(len(CellTokenizer.tokenize_prompt(cl) & getattr(c, "identity_tokens", c.token_set)) > 0
                              for c in pc))
        prev_cov, stall = _cov_cnt(path), 0

        for step_idx in range(max_steps):
            curr = path[-1]
            if (getattr(curr, "stage", None) == 3 and len(path) > 1) or any(_is_terminal_sink_cell(c) for c in path):
                break
            curr_out_st = str(getattr(curr.primary_output, "state", "")).lower()
            is_pred = bool(TypeRegistry.get_instance().get_state_properties(curr_out_st).get("is_prediction")) or curr_out_st.startswith("predicted_")

            valid = []
            for c in candidates:
                if c.cell_id in visited or not self.step_unifies(curr, c, prev_path=path): continue
                c_out = str(getattr(c.primary_output, "state", "")).lower()
                if has_reg and ("label" in c_out or c_out == "predicted_labels"): continue
                if has_cls and c_out == "predicted_values": continue
                if is_pred and (getattr(c, "node_role", "") in ("estimator", "model") or "fit" in c.cell_id.lower()): continue
                valid.append(c)
            if not valid: break

            if ir_steps and step_idx < len(ir_steps):
                filtered = _ir_filter(valid, ir_steps[step_idx])
                if filtered: valid = filtered

            valid.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0) * 10.0
                       + self.calculate_edge_affinity(curr, c, orch, relevance_map=relevance_map) * 5.0
                       + len(c.token_set & prompt_tokens) * 3.0, reverse=True)
            top_valid = valid[:5]
            selected = None
            if has_llm:
                try:
                    summary = "\n".join(f"- {c.cell_id}" for c in top_valid)
                    ir_hint = ""
                    if ir_steps and step_idx < len(ir_steps):
                        s = ir_steps[step_idx]
                        ir_hint = f"\nTarget op: {s.get('op','')} (library: {s.get('library','')})"
                    resp = mm.generate_text(
                        f"Intent: {prompt}\nPipeline: {[c.cell_id for c in path]}{ir_hint}\nCandidates:\n{summary}\nSelect next CELL_ID:",
                        max_tokens=64,
                        system_prompt="Output ONLY the exact cell ID from the candidate list, or FINISH.",
                    ).strip()
                    toks = set(re.findall(r"[A-Za-z0-9_]+", resp.lower()))
                    if "finish" in toks: break
                    selected = next((c for c in top_valid if c.cell_id.lower() in toks), None)
                except Exception:
                    selected = None
            selected = selected or top_valid[0]
            path.append(selected); visited.add(selected.cell_id)
            new_cov = _cov_cnt(path)
            if new_cov > prev_cov: prev_cov, stall = new_cov, 0
            else:
                stall += 1
                if stall >= 2: break
        return path
