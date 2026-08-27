"""
src/planner.py - Zero-Shot Planner with algorithmic-task bypass and robust JSON.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set

from log_config import get_logger
from lattice import LatticeOrchestrator, Cell, MacroCell, MicroCell, AlgebraicSignature

logger = get_logger('planner')


class ZeroShotPlanner:
    ALGO_PATTERNS = re.compile(
        r"\b(dijkstra|bfs|dfs|a\s*star|astar|quicksort|mergesort|heapsort|"
        r"binary\s*search|topological\s*sort|bellman.?ford|floyd.?warshall|"
        r"kruskal|prim|kmp|rsa|sha|md5|algorithm)\b",
        re.IGNORECASE
    )

    def __init__(self, orchestrator: LatticeOrchestrator, rag: Any):
        self.orchestrator = orchestrator
        self.rag = rag

    def run_planning_pass(self, prompt: str, profile: str = "C") -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        if self.ALGO_PATTERNS.search(prompt):
            # Check if a verified algorithmic seed exists in the lattice
            matching_algo_cells = [
                c.cell_id for c in self.orchestrator.loaded_cells.values()
                if (c.domain_name in ("algorithms", "python_core") or c.stage == 2)
                and any(k.lower() in prompt_lower for k in c.keywords if len(k) > 3)
            ]
            if matching_algo_cells:
                logger.info(f"[PLANNER] Algorithmic seed found in lattice: {matching_algo_cells[0]}")
                return {
                    "cells": [{
                        "cell_id": "macro_algo_seeded",
                        "type": "macro",
                        "stage": 2,
                        "sub_cells": [matching_algo_cells[0]]
                    }]
                }
            logger.info("[PLANNER] Algorithmic task detected with no exact seed; routing to synthesis.")
            return {
                "cells": [{
                    "cell_id": "macro_algo_synth",
                    "type": "macro",
                    "stage": 2,
                    "sub_cells": [f"SYNTH_ALGO_{self._slugify(prompt)[:30]}"]
                }]
            }

        context = self.rag.get_relevant_context(prompt, top_k=25)
        context_str = self._format_context(context)

        system_prompt = self._build_system_prompt(context_str)
        user_prompt = f"User Request: {prompt}\n\nOutput ONLY valid JSON."

        raw = None
        try:
            from inference import ModelManager
            mm = ModelManager.get_instance()
            if mm.can_synthesize():
                raw = mm.generate_text(system_prompt + "\n" + user_prompt, max_tokens=1024)
        except Exception as e:
            logger.warning(f"[PLANNER] LLM generation unavailable: {e}")

        parsed = self._safe_parse_json(raw) if raw else None
        if parsed is None:
            logger.warning("[PLANNER] Invalid or missing JSON from LLM. Falling back to deterministic planner.")
            return self._deterministic_fallback(prompt, context)

        for cell_block in parsed.get("cells", []):
            grounded = []
            for sub in cell_block.get("sub_cells", []):
                existing = self._find_closest_existing_cell(sub)
                if existing:
                    grounded.append(existing)
                else:
                    grounded.append(sub)
            cell_block["sub_cells"] = grounded

        return parsed

    def _build_system_prompt(self, context_str: str) -> str:
        return f"""You are a Software Architect. Decompose the user request into a strict sequence of verified computational node IDs.
Output ONLY valid JSON matching the schema below. No markdown formatting, no commentary.

Schema:
{{
  "cells": [
    {{
      "cell_id": "macro_dynamic_task",
      "type": "macro",
      "stage": 1,
      "sub_cells": ["<node_id_1>", "<node_id_2>", "<node_id_3>"]
    }}
  ]
}}

Available Verified Micro-Nodes:
{context_str}

RULES:
1. Select exclusively from the Available Verified Micro-Nodes when a suitable node exists.
2. If an essential step has no matching node, invent a new node ID prefixed with 'SYNTH_' (e.g. 'SYNTH_CALCULATE_METRIC'). The engine will synthesize it at runtime.
3. Order sub_cells sequentially from data source -> transformations -> output/sink.
4. For algorithmic tasks (sorting, graph search, etc.), use a single SYNTH_ node rather than forcing library primitives.
"""

    def _format_context(self, context: List[Any]) -> str:
        lines = []
        for entry in context:
            cid = ""
            if isinstance(entry, dict):
                cid = entry.get("cell_id", "UNKNOWN")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
                cid = str(entry[0])
            elif isinstance(entry, str):
                m = re.search(r"ID:\s*([A-Z0-9_]+)", entry)
                cid = m.group(1) if m else "UNKNOWN"

            cell = self.orchestrator.loaded_cells.get(cid)
            if not cell:
                continue
            in_sig = f"{cell.primary_input.type_name}[{cell.primary_input.state}]"
            out_sig = f"{cell.primary_output.type_name}[{cell.primary_output.state}]"
            lines.append(f"- ID: {cid} | In: {in_sig} -> Out: {out_sig} | Domain: {cell.domain_name}")
        return "\n".join(lines)

    def _safe_parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        return None

    def _deterministic_fallback(self, prompt: str, context: List[Any]) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        keywords = set(re.findall(r"[a-zA-Z_]+", prompt_lower))

        # Dynamic domain inference from top context candidate
        goal_domain = None
        if context:
            top = context[0]
            if isinstance(top, dict) and top.get("domain") and top.get("domain") not in ("generic", "python_core"):
                goal_domain = top.get("domain")

        path = []
        current_sig = AlgebraicSignature("str", "source_identifier")

        for stage in [1, 2, 3]:
            best_match = None
            best_score = -1.0
            for entry in context:
                cid = ""
                score = 0.0
                if isinstance(entry, dict):
                    cid = entry.get("cell_id", "")
                    score = entry.get("score", 0.0)
                elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
                    cid = str(entry[0])
                    score = float(entry[1]) if len(entry) > 1 else 0.0
                elif isinstance(entry, str):
                    m = re.search(r"ID:\s*([A-Z0-9_]+)", entry)
                    cid = m.group(1) if m else ""

                cell = self.orchestrator.loaded_cells.get(cid)
                if not cell or cell.stage != stage:
                    continue
                if not current_sig.unifies_with(cell.primary_input):
                    continue

                score += len(keywords & set(k.lower() for k in cell.keywords)) * 0.2
                if goal_domain and cell.domain_name != goal_domain:
                    score *= 0.5

                if score > best_score:
                    best_score = score
                    best_match = cid

            if best_match:
                path.append(best_match)
                cell = self.orchestrator.loaded_cells[best_match]
                current_sig = cell.primary_output

        return {
            "cells": [{
                "cell_id": "macro_fallback",
                "type": "macro",
                "stage": 1,
                "sub_cells": path
            }]
        }

    def _find_closest_existing_cell(self, cell_id: str) -> Optional[str]:
        if cell_id in self.orchestrator.loaded_cells:
            return cell_id
        cid_upper = cell_id.upper()
        for cid in self.orchestrator.loaded_cells:
            if cid_upper in cid or cid in cid_upper:
                return cid
        return None

    @staticmethod
    def _slugify(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").upper()
