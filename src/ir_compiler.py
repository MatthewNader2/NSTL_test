"""
ir_compiler.py (apply_fixes_v9) — Prompt -> typed IR via the local LLM.

Robust JSON extraction:
  - Balanced-brace scanner (respects strings and escapes).
  - Strips markdown fences.
  - Retries once with a corrective prompt on parse failure.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from log_config import get_logger
try:
    from .inference import ModelManager
except (ImportError, ValueError):
    from inference import ModelManager

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
    """Return the first balanced JSON object found in ``text``.
    Ignores markdown fences and surrounding prose. Respects strings."""
    if not text:
        return None
    # strip code fences
    t = re.sub(r"```(?:json)?\s*", "", text)
    t = t.replace("```", "")
    i = 0
    while True:
        start = t.find("{", i)
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(t)):
            ch = t[j]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = t[start:j + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # advance past this start
                        i = start + 1
                        break
        else:
            return None


@dataclass
class IR:
    steps: List[Dict[str, Any]]
    literals: Dict[str, int] = field(default_factory=dict)
    libraries: List[str] = field(default_factory=list)
    raw: str = ""

    @classmethod
    def parse(cls, text: str) -> "IR":
        data = _extract_json_object(text)
        if data is None:
            raise ValueError("IR compiler: no valid JSON object in output")
        return cls(
            steps=data.get("steps", []),
            literals={str(k): int(v) for k, v in (data.get("literals") or {}).items()},
            libraries=list(data.get("libraries") or []),
            raw=text,
        )

    def validate(self, vocab: set) -> List[str]:
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
        self.vocab = self._derive_vocab()
        self.libs  = self._derive_libs()

    def _derive_vocab(self) -> set:
        out = set()
        for c in self.orchestrator.loaded_cells.values():
            cid = c.cell_id.lower()
            dom = (getattr(c, "domain_name", "") or "").lower()
            # op-name from cell_id tail (domain-stripped)
            if dom and cid.startswith(dom + "_"):
                tail = cid[len(dom) + 1:]
            elif "." in cid:
                tail = cid.split(".")[-1]
            else:
                tail = cid
            if 3 <= len(tail) <= 40 and " " not in tail and not tail.isdigit():
                out.add(tail)
            for src in (getattr(c, "keywords", []) or [],
                        getattr(c, "semantic_tags", []) or []):
                for kw in src:
                    k = str(kw).strip().lower()
                    if 3 < len(k) < 25 and " " not in k and not k.isdigit():
                        out.add(k)
        return out or {"read", "transform", "filter", "aggregate", "plot", "save"}

    def _derive_libs(self) -> list:
        return sorted({c.domain_name for c in self.orchestrator.loaded_cells.values()
                       if getattr(c, "domain_name", None)})

    def _vocab_for_prompt(self) -> str:
        # shortest entries first — they're the semantic ops; long
        # qualified names blow the token budget.
        entries = sorted(self.vocab, key=lambda v: (len(v), v))[:200]
        return ", ".join(entries)

    def compile(self, prompt: str) -> Optional[IR]:
        mm = ModelManager.get_instance()
        prof = getattr(mm, "profile", None)
        if prof is None:
            return None
        if getattr(prof, "llm", None) is None:
            return None
        vocab_str = self._vocab_for_prompt()
        libs_str = ", ".join(sorted(self.libs))
        msg = PROMPT_TEMPLATE.format(vocab=vocab_str, libs=libs_str, prompt=prompt)
        try:
            raw = mm.generate_text(msg, max_tokens=768, schema=IR_SCHEMA)
            ir = IR.parse(raw)
            errs = ir.validate(self.vocab)
            if errs:
                logger.debug("[IR] vocab strict-miss: %s", errs)
            # (apply_fixes_v10) always show the compiled IR
            import os as _os
            if _os.environ.get("NSTL_IR_LOG", "1") not in ("0", "false", "False"):
                _ops = [s.get('op','?') + '/' + str(s.get('library','?')) for s in ir.steps]
                print(f"[IR] compiled {len(ir.steps)} steps: {_ops}")
                if ir.literals:
                    print(f"[IR] literals: {ir.literals}")
            return ir
        except Exception as e1:
            logger.debug("[IR] first attempt failed: %s", e1)
            # One corrective retry
            try:
                raw2 = mm.generate_text(
                    RETRY_TEMPLATE.format(prompt=prompt),
                    max_tokens=768, schema=IR_SCHEMA,
                )
                ir = IR.parse(raw2)
                logger.info("[IR] corrective retry succeeded")
                return ir
            except Exception as e2:
                logger.warning("[IR] compilation failed after retry: %s", e2)
                return None
