"""
src/route_methods/m8_llm_stepwise.py - Neuro-Symbolic Topological Lattice (NSTL)

M8: LLM Stepwise Edge Pathfinder.

Unlike M4 (which only showed 5 bare cell names at each step) and M7 (which
plans the entire path in one global shot), M8 performs iterative per-node
decision making with full type context and explicit edge visibility:

At each step:
1. Shows the current pipeline prefix built so far.
2. Identifies all topologically valid next-hop transitions from the current
   node (both direct unifying edges and 1-step bridged transitions).
3. Shows the rest of the relevant candidate nodes in the pool with full
   port type signatures so the model maintains foresight of overall goals.
4. Asks the model to choose the next node from the valid transitions, or
   FINISH when the pipeline is complete.

Every step is verified with Robinson unification (step_unifies / find_bridge).
Falls back to M1 (clause-anchor) when no LLM is loaded or parsing fails.
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional, Any, Tuple

from .base import RouteMethod
from .m1_clause_anchor import M1ClauseAnchorRouteMethod
from lattice import Cell, LatticeOrchestrator
from unification import ExecutionContext
from inference import ModelManager

try:
    from utils import extract_json_object, tokenize_alphanumeric
except ImportError:
    from ..utils import extract_json_object, tokenize_alphanumeric

try:
    from node_resolver import DynamicNodeResolver
except (ImportError, ValueError):
    try:
        from ..node_resolver import DynamicNodeResolver
    except (ImportError, ValueError):
        DynamicNodeResolver = None

MAX_SHOWN_CANDIDATES = 35
MAX_TRANSITIONS_SHOWN = 15

STEP_SCHEMA = {
    "type": "object",
    "required": ["next_node"],
    "properties": {
        "next_node": {"type": "string"},
    },
}

SYSTEM_PROMPT_ENTRY = (
    "You are a pipeline planner. Given a user request and a list of available "
    "typed entry operations (cells), select the best operation to BEGIN the "
    "pipeline.\n"
    "Respond with ONLY a JSON object of the form: {\"next_node\": \"CELL_ID\"}."
)

SYSTEM_PROMPT_STEP = (
    "You are an iterative pipeline planner. You are given a user request, the "
    "current pipeline prefix, a list of topologically VALID next transitions from "
    "the current node, and the remaining available candidate operations in the pool.\n"
    "Select the NEXT operation from the 'Valid Next Transitions' list to advance "
    "toward fulfilling the request, or output 'FINISH' if the pipeline has achieved "
    "all required operations.\n"
    "Respond with ONLY a JSON object of the form: {\"next_node\": \"CELL_ID\"} or "
    "{\"next_node\": \"FINISH\"}. No prose, no markdown."
)


def _fmt_port_sig(p: Any) -> str:
    sig = getattr(p, "signature", None)
    tn = str(getattr(sig, "type_name", "") or getattr(p, "abstract_type", "") or "any")
    st = str(getattr(sig, "state", "") or "")
    return f"{tn}[{st}]" if st and st not in ("any", "none") else tn


def _format_cell_signature(cell: Cell) -> str:
    req_in = [f"{n}:{_fmt_port_sig(p)}" for n, p in cell.inputs.items() if getattr(p, "required", False)]
    opt_in = [f"{n}:{_fmt_port_sig(p)}?" for n, p in cell.inputs.items() if not getattr(p, "required", False)]
    outs = [f"{n}:{_fmt_port_sig(p)}" for n, p in cell.outputs.items()]
    in_str = ", ".join(req_in + opt_in) or "none"
    out_str = ", ".join(outs) or "none"
    dom = getattr(cell, "domain_name", "") or "?"
    stage = getattr(cell, "stage", "?")
    doc = getattr(cell, "docstring", "") or ""
    summary = doc.split(".")[0].strip() if doc else ""
    desc_str = f" ({summary})" if summary else ""
    return f"{cell.cell_id} [{dom} s{stage}]{desc_str} in({in_str}) -> out({out_str})"


def _parse_next_node(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    data = extract_json_object(raw_text)
    if data and isinstance(data.get("next_node"), str):
        return data["next_node"].strip()
    # Fallback token search without regex
    toks = tokenize_alphanumeric(raw_text)
    if "finish" in toks:
        return "FINISH"
    return toks[0] if toks else None


class M8LLMStepwisePathfinderRouteMethod(RouteMethod):
    name = "m8_llm_stepwise"

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

        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"] or [tunnel[0]]
        mm = ModelManager.get_instance()
        has_llm = mm.profile is not None and getattr(mm.profile, "llm", None) is not None

        def _fallback():
            return M1ClauseAnchorRouteMethod(orchestrator=orch).plan(
                prompt, tunnel, relevance_map, orchestrator=orch, ctx=ctx,
                start_sig=start_sig, goal_sig=goal_sig, max_transforms=max_transforms,
            )

        if not has_llm:
            return _fallback()

        shown = sorted(candidates, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
        shown = shown[:MAX_SHOWN_CANDIDATES]
        cand_map = {c.cell_id.lower(): c for c in shown}

        # Step 0: Choose entry node
        entry_candidates = [
            c for c in shown
            if getattr(c, "stage", None) == 1 or getattr(c, "node_role", "") == "source"
        ]
        if not entry_candidates:
            entry_candidates = shown[:5]

        entry_cell = None
        try:
            entry_lines = "\n".join(f"- {_format_cell_signature(c)}" for c in entry_candidates[:8])
            entry_prompt = f"Request: {prompt}\n\nAvailable Entry Operations:\n{entry_lines}"
            raw = mm.generate_text(
                entry_prompt, max_tokens=128, schema=STEP_SCHEMA, system_prompt=SYSTEM_PROMPT_ENTRY
            )
            node_id = _parse_next_node(raw)
            if node_id:
                entry_cell = cand_map.get(node_id.lower())
                if entry_cell is None:
                    try:
                        from node_resolver import DynamicNodeResolver
                        entry_cell = DynamicNodeResolver.resolve_node(
                            raw_id=node_id,
                            prompt=prompt,
                            orchestrator=orch,
                            rag=kwargs.get("rag"),
                        )
                    except Exception:
                        entry_cell = None
        except Exception:
            entry_cell = None

        if entry_cell is None:
            entry_cell = entry_candidates[0]

        path: List[Cell] = [entry_cell]
        visited_ids = {entry_cell.cell_id}
        max_steps = min(16, max_transforms + 4)

        # Stepwise expansion loop
        for _ in range(max_steps):
            curr = path[-1]
            curr_stage = getattr(curr, "stage", None)
            curr_role = str(getattr(curr, "node_role", "") or "").lower()

            # Identify valid next transitions from curr
            # Tuple: (target_cell, bridge_cell_or_None, transition_type_str)
            transitions: List[Tuple[Cell, Optional[Cell], str]] = []

            for c in shown:
                if c.cell_id in visited_ids:
                    continue
                if getattr(c, "stage", None) == 1:
                    continue

                if self.step_unifies(curr, c, prev_path=path):
                    transitions.append((c, None, "DIRECT"))
                elif self.step_unifies_dag(c, path, ctx=ctx):
                    transitions.append((c, None, "DAG_FRONTIER"))
                else:
                    bridge = self.find_bridge(curr, c, candidates, orch)
                    if bridge and bridge.cell_id not in visited_ids:
                        transitions.append((c, bridge, f"BRIDGED via {bridge.cell_id}"))

            if not transitions:
                break

            # Sort transitions by candidate relevance and edge affinity
            def _trans_score(t: Tuple[Cell, Optional[Cell], str]) -> float:
                tgt, brg, _ = t
                score = relevance_map.get(tgt.cell_id, 0.0) * 2.0
                score += self.calculate_edge_affinity(curr, brg if brg else tgt, orch, relevance_map=relevance_map)
                if brg:
                    score += self.calculate_edge_affinity(brg, tgt, orch, relevance_map=relevance_map) * 0.8
                return score

            transitions.sort(key=_trans_score, reverse=True)
            active_transitions = transitions[:MAX_TRANSITIONS_SHOWN]
            trans_by_id = {t[0].cell_id.lower(): t for t in active_transitions}

            # Remaining unvisited relevant nodes (for model foresight)
            remaining_nodes = [
                c for c in shown
                if c.cell_id not in visited_ids and c.cell_id.lower() not in trans_by_id
            ][:10]

            # Format prompt for the step
            path_str = " -> ".join([c.cell_id for c in path])
            valid_lines = []
            for tgt, brg, mode in active_transitions:
                valid_lines.append(f"- {tgt.cell_id} [{mode}]: {_format_cell_signature(tgt)}")
            valid_block = "\n".join(valid_lines)

            rem_block = ""
            if remaining_nodes:
                rem_lines = [f"- {_format_cell_signature(c)}" for c in remaining_nodes]
                rem_block = f"\n\nOther Candidate Nodes in Pool (Goals / Rest of Pipeline):\n" + "\n".join(rem_lines)

            finish_hint = " (or 'FINISH' if pipeline is complete)" if (curr_stage == 3 or curr_role == "sink" or len(path) >= 2) else ""
            step_msg = (
                f"Request: {prompt}\n\n"
                f"Current Pipeline: {path_str}\n\n"
                f"Valid Next Transitions from {curr.cell_id}:\n{valid_block}"
                f"{rem_block}\n\n"
                f"Choose the next node ID from 'Valid Next Transitions'{finish_hint}:"
            )

            try:
                raw_step = mm.generate_text(
                    step_msg, max_tokens=128, schema=STEP_SCHEMA, system_prompt=SYSTEM_PROMPT_STEP
                )
                chosen_id = _parse_next_node(raw_step)
            except Exception:
                chosen_id = None

            if chosen_id and chosen_id.upper() in ("FINISH", "DONE", "STOP", "END"):
                break

            matched_trans = trans_by_id.get(chosen_id.lower()) if chosen_id else None
            if matched_trans is None and chosen_id:
                try:
                    from node_resolver import DynamicNodeResolver
                    res_cell = DynamicNodeResolver.resolve_node(
                        raw_id=chosen_id,
                        prompt=prompt,
                        orchestrator=orch,
                        rag=kwargs.get("rag"),
                    )
                    if res_cell is not None and res_cell.cell_id not in visited_ids:
                        if self.step_unifies(curr, res_cell, prev_path=path):
                            matched_trans = (res_cell, None, "DYNAMIC_DIRECT")
                        elif self.step_unifies_dag(res_cell, path, ctx=ctx):
                            matched_trans = (res_cell, None, "DYNAMIC_DAG_FRONTIER")
                        else:
                            brg = self.find_bridge(curr, res_cell, candidates, orch)
                            if brg and brg.cell_id not in visited_ids:
                                matched_trans = (res_cell, brg, f"DYNAMIC_BRIDGED via {brg.cell_id}")
                except Exception:
                    pass

            if matched_trans is None:
                # If current node is a sink or stage 3, complete
                if curr_stage == 3 or curr_role == "sink":
                    break
                # Default to top scored transition
                matched_trans = active_transitions[0]

            target_cell, bridge_cell, _ = matched_trans
            if bridge_cell:
                path.append(bridge_cell)
                visited_ids.add(bridge_cell.cell_id)
            path.append(target_cell)
            visited_ids.add(target_cell.cell_id)

            if getattr(target_cell, "stage", None) == 3 or getattr(target_cell, "node_role", "") == "sink":
                # Egress reached
                break

        # Verification pass: ensure consecutive unification
        if len(path) > 1:
            verified = [path[0]]
            for nxt in path[1:]:
                c_prev = verified[-1]
                if self.step_unifies(c_prev, nxt, prev_path=verified) or self.step_unifies_dag(nxt, verified, ctx=ctx):
                    verified.append(nxt)
                else:
                    brg = self.find_bridge(c_prev, nxt, candidates, orch)
                    if brg:
                        verified.extend([brg, nxt])
                    else:
                        # DAG branch check: can nxt unify with earlier ancestor in verified?
                        anc_ok = False
                        for anc in reversed(verified[:-1]):
                            if self.step_unifies(anc, nxt, prev_path=verified) or self.step_unifies_dag(nxt, verified, ctx=ctx):
                                verified.append(nxt)
                                anc_ok = True
                                break
                        if not anc_ok:
                            try:
                                from node_resolver import DynamicNodeResolver
                                dyn_brg = DynamicNodeResolver.synthesize_adapter(c_prev, nxt, orch, rag=kwargs.get("rag"))
                                if dyn_brg and self.step_unifies(c_prev, dyn_brg, prev_path=verified) and self.step_unifies(dyn_brg, nxt, prev_path=verified + [dyn_brg]):
                                    verified.extend([dyn_brg, nxt])
                                else:
                                    return _fallback()
                            except Exception:
                                return _fallback()
            path = verified

        return path[:min(20, max(max_transforms + 4, 6))]
