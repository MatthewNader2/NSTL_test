"""
src/route_methods/m5_llm_oneshot.py - Neuro-Symbolic Topological Lattice (NSTL)
M5: One-shot LLM full-pipeline generation with Monadic Lattice Validation and Gap Bridging.
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, STOPWORDS
from .m1_clause_anchor import M1ClauseAnchorRouteMethod

try:
    from ..lattice import Cell, LatticeOrchestrator
    from ..tokenizer import CellTokenizer
    from ..unification import unify, ExecutionContext
    from ..inference import ModelManager
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from tokenizer import CellTokenizer
    from unification import unify, ExecutionContext
    from inference import ModelManager


class M5LLMOneShotRouteMethod(RouteMethod):
    """
    RouteMethod M5: One-Shot LLM Generation.
    Queries the LLM once for an entire pipeline sequence, then validates
    all transitions in the lattice and automatically bridges any type gaps.
    """
    name: str = "m5_llm_oneshot"

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

        cand_map = {c.cell_id.lower(): c for c in candidates}
        mm = ModelManager.get_instance()
        has_llm = mm.active_profile is not None and mm.can_synthesize()

        proposed_cells: List[Cell] = []

        if has_llm:
            try:
                # Provide top 20 candidate cells
                top_cands = sorted(candidates, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)[:20]
                cand_list_str = "\n".join(f"- {c.cell_id}" for c in top_cands)

                sys_prompt = "You are a software pipeline synthesizer. Output a JSON list of cell IDs composing the solution pipeline, e.g. [\"CELL_A\", \"CELL_B\"]."
                user_prompt = (
                    f"User Intent: {prompt}\n"
                    f"Available Component Cells:\n{cand_list_str}\n"
                    f"Return ONLY the JSON list of cell IDs in execution order:"
                )
                raw_resp = mm.generate_text(user_prompt, max_tokens=128, system_prompt=sys_prompt).strip()

                # Parse JSON list
                match = re.search(r"\[.*?\]", raw_resp, re.DOTALL)
                if match:
                    parsed_ids = json.loads(match.group(0))
                    for cid in parsed_ids:
                        cid_clean = str(cid).strip().lower()
                        if cid_clean in cand_map:
                            proposed_cells.append(cand_map[cid_clean])
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"[M5] LLM one-shot proposal unparseable: {e}")
                proposed_cells = []

        # Fallback to M1 if LLM failed or produced empty list
        if not proposed_cells:
            fallback = M1ClauseAnchorRouteMethod(orchestrator=orch)
            return fallback.plan(
                prompt=prompt,
                tunnel=tunnel,
                relevance_map=relevance_map,
                orchestrator=orch,
                ctx=ctx,
                start_sig=start_sig,
                goal_sig=goal_sig,
                max_transforms=max_transforms
            )

        # Monadic verification & gap bridging over proposed chain
        validated_chain: List[Cell] = [proposed_cells[0]]
        for next_c in proposed_cells[1:]:
            curr_c = validated_chain[-1]
            if self.step_unifies(curr_c, next_c):
                validated_chain.append(next_c)
            else:
                bridge = self.find_bridge(curr_c, next_c, candidates, orch)
                if bridge:
                    validated_chain.append(bridge)
                    validated_chain.append(next_c)
                else:
                    # No bridge found: inserting next_c anyway would knowingly
                    # ship a type break. Repair honestly via the M1 fallback.
                    fallback = M1ClauseAnchorRouteMethod(orchestrator=orch)
                    return fallback.plan(
                        prompt=prompt,
                        tunnel=tunnel,
                        relevance_map=relevance_map,
                        orchestrator=orch,
                        ctx=ctx,
                        start_sig=start_sig,
                        goal_sig=goal_sig,
                        max_transforms=max_transforms
                    )
        clauses = self.segment_prompt_clauses(prompt)
        num_clauses = len(clauses) if clauses else 1
        dynamic_cap = max(max_transforms + 2, num_clauses + 3)
        return validated_chain[:min(16, dynamic_cap)]
