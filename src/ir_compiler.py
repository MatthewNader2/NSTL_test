"""
ir_compiler.py — Prompt -> typed IR via the local LLM.

Robust JSON extraction:
  - Balanced-brace scanner (respects strings and escapes).
  - Strips markdown fences.
  - Retries once with a corrective prompt on parse failure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from log_config import get_logger

try:
    from .inference import ModelManager
    from .utils import extract_json_object
except (ImportError, ValueError):
    from inference import ModelManager
    from utils import extract_json_object

logger = get_logger("ir_compiler")

IR_SCHEMA = {
    "type": "object",
    "required": ["steps"],
    "properties": {
        "libraries": {"type": "array", "items": {"type": "string"}},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op"],
                "properties": {
                    "op":      {"type": "string"},
                    "params":  {"type": "object"},
                    "library": {"type": "string"},
                },
            },
        },
        "literals": {"type": "object"},
    },
}

PROMPT_TEMPLATE = """\
You are a semantic compiler. Convert the user request into a typed IR.

Allowed `op` values: {vocab}

Allowed libraries: {libs}

Rules:
- One `op` per semantic action. Do NOT split on commas inside lists.
- Use only ops from the allowed list.
- Extract EVERY literal into `literals`.
- Return ONLY a JSON object. No prose, no markdown.

User prompt:
{prompt}
"""

RETRY_TEMPLATE = """\
Your previous response was not valid JSON. Respond with ONLY a JSON object
matching this schema:

{{"steps": [{{"op": "str", "params": {{}}, "library": "str"}}], "libraries": ["..."], "literals": {{}}}}

User prompt:
{prompt}
"""


def _extract_json_object(text: str) -> Optional[dict]:
    """Return the first balanced JSON object found in ``text``."""
    return extract_json_object(text)


@dataclass
class IR:
    steps: List[Dict[str, Any]]
    literals: Dict[str, Any] = field(default_factory=dict)
    libraries: List[str] = field(default_factory=list)
    raw: str = ""

    @classmethod
    def parse(cls, text: str) -> "IR":
        data = _extract_json_object(text)
        if data is None:
            raise ValueError("IR compiler: no valid JSON object in output")
        return cls(
            steps=data.get("steps", []),
            literals=dict(data.get("literals") or {}),
            libraries=list(data.get("libraries") or []),
            raw=text,
        )

    def validate(self, vocab: Set[str]) -> List[str]:
        errs = []
        for i, s in enumerate(self.steps):
            op = str(s.get("op", "")).strip().lower().replace("-", "_").replace(" ", "_")
            if not op:
                continue
            if op in vocab:
                continue
            if len(op) >= 3 and any((op in v) or (v in op) for v in vocab if len(v) >= 3):
                continue
            errs.append(f"step {i}: unknown op {s.get('op')!r}")
        return errs


class IRCompiler:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.vocab: Set[str] = self._derive_vocab()
        self.libs: List[str] = self._derive_libs()

    def _derive_vocab(self) -> Set[str]:
        """Derives available operations dynamically from loaded cells or orchestrator configuration."""
        out: Set[str] = set()
        loaded = getattr(self.orchestrator, "loaded_cells", {}) or {}

        for c in loaded.values():
            # 1. Prefer explicit operation identifiers if exposed on the cell
            op_cand = (
                getattr(c, "op_name", None)
                or getattr(c, "operation", None)
                or getattr(c, "callable_name", None)
            )
            if op_cand:
                tail = str(op_cand).strip().lower()
            else:
                cid = str(getattr(c, "cell_id", "")).lower()
                dom = str(getattr(c, "domain_name", "") or "").lower()

                # Strip domain prefix across multiple possible delimiters (_, ., :, /, -)
                tail = cid
                if dom and tail.startswith(dom):
                    remainder = tail[len(dom):]
                    if remainder and remainder[0] in ("_", ".", ":", "/", "-"):
                        tail = remainder.lstrip("_.:/-")

                # If still delimited by hierarchical paths, take the terminal operation name
                for delim in (":", "/", "."):
                    if delim in tail:
                        tail = tail.split(delim)[-1]

            if 2 <= len(tail) <= 60 and " " not in tail and not tail.isdigit():
                out.add(tail)

            # Extract semantic annotations and keywords
            for src in (getattr(c, "keywords", []) or [],
                        getattr(c, "semantic_tags", []) or []):
                for kw in src:
                    k = str(kw).strip().lower()
                    if 2 < len(k) < 40 and " " not in k and not k.isdigit():
                        out.add(k)

        if not out:
            # Fall back to orchestrator-level configured operation vocabulary if present
            orch_vocab = (
                getattr(self.orchestrator, "default_vocab", None)
                or getattr(self.orchestrator, "allowed_ops", None)
            )
            if orch_vocab:
                return set(orch_vocab)

        return out

    def _derive_libs(self) -> List[str]:
        loaded = getattr(self.orchestrator, "loaded_cells", {}) or {}
        return sorted({
            c.domain_name for c in loaded.values()
            if getattr(c, "domain_name", None)
        })

    def _vocab_for_prompt(self, prompt: str = "", max_entries: int = 300) -> str:
        """
        Formats vocabulary for the LLM prompt (Solution 7).

        Selection is PROMPT-RELEVANCE based, never stride sampling: the
        previous uniform stride sampling arbitrarily dropped up to ~70% of
        operations, so any operation falling between sample steps was
        unrepresentable in the compiled IR. Now:
          1. Entries whose declared tokens overlap (or prefix-match) the
             prompt's tokens always rank first — the operations the task
             needs are guaranteed to be offered to the LLM.
          2. Remaining budget is filled by stable, deterministic order so the
             prompt stays within the context window without semantic loss
             being concentrated on any lexical range.
        """
        if not self.vocab:
            return "(dynamically infer based on prompt and library context)"

        entries = sorted(self.vocab)
        if len(entries) <= max_entries:
            return ", ".join(entries)

        prompt_toks: Set[str] = set()
        try:
            from .tokenizer import CellTokenizer as _CT
        except (ImportError, ValueError):
            try:
                from tokenizer import CellTokenizer as _CT
            except Exception:
                _CT = None
        if _CT is not None and prompt:
            try:
                prompt_toks = {t.lower() for t in _CT.tokenize_prompt(prompt)}
            except Exception:
                prompt_toks = set()
        if not prompt_toks and prompt:
            prompt_toks = {
                w.strip("_'\",.;:()[]{}").lower()
                for w in prompt.split()
                if len(w) >= 3
            }

        def _split_entry(entry: str) -> Set[str]:
            parts = entry.replace("-", "_").split("_")
            return {p.lower() for p in parts if len(p) >= 3}

        def _relevance(entry: str) -> Tuple[int, int, str]:
            if not prompt_toks:
                return (0, 0, entry)
            entry_toks = _split_entry(entry)
            overlap = len(entry_toks & prompt_toks)
            prefix_hit = any(pt.startswith(entry) or entry.startswith(pt) for pt in prompt_toks if len(entry) >= 3)
            return (overlap, 1 if prefix_hit else 0, entry)

        scored = sorted(entries, key=_relevance, reverse=True)
        relevant = [e for e in scored if _relevance(e)[:2] != (0, 0)]
        filler = [e for e in scored if _relevance(e)[:2] == (0, 0)]
        selected = relevant[:max_entries]
        if len(selected) < max_entries:
            selected.extend(filler[: max_entries - len(selected)])
        # Restore deterministic (alphabetical) order within the selected set
        # so the prompt is stable across runs for identical inputs.
        return ", ".join(sorted(selected))

    def compile(self, prompt: str) -> Optional[IR]:
        mm = ModelManager.get_instance()
        prof = getattr(mm, "profile", None)
        if prof is None or getattr(prof, "llm", None) is None:
            return None

        vocab_str = self._vocab_for_prompt(prompt)
        libs_str = ", ".join(sorted(self.libs))
        msg = PROMPT_TEMPLATE.format(vocab=vocab_str, libs=libs_str, prompt=prompt)

        try:
            raw = mm.generate_text(msg, max_tokens=768, schema=IR_SCHEMA)
            ir = IR.parse(raw)
            errs = ir.validate(self.vocab)
            if errs:
                logger.debug("[IR] vocab strict-miss: %s", errs)

            _ops = [s.get('op', '?') + '/' + str(s.get('library', '?')) for s in ir.steps]
            logger.info("[IR] compiled %d steps: %s", len(ir.steps), _ops)
            if ir.literals:
                logger.info("[IR] literals: %s", ir.literals)
            return ir
        except Exception as e1:
            logger.debug("[IR] first attempt failed: %s", e1)
            # Corrective retry
            try:
                raw2 = mm.generate_text(
                    RETRY_TEMPLATE.format(prompt=prompt),
                    max_tokens=768,
                    schema=IR_SCHEMA,
                )
                ir = IR.parse(raw2)
                logger.info("[IR] corrective retry succeeded")
                return ir
            except Exception as e2:
                logger.warning("[IR] compilation failed after retry: %s", e2)
                return None
