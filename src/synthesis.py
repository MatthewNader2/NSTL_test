import json
import re
from log_config import get_logger

from external_rag import LiveDocFetcher
from inference import ModelManager

logger = get_logger('synthesis')

# Placeholder names the synthesis prompt instructs the LLM to use.
# Derived from the template strings — update this tuple if placeholders ever change.
_CODE_PLACEHOLDERS = frozenset(
    token
    for template in ("{input_var}", "{output_var}")
    for token in re.findall(r"\{(\w+)\}", template)
)


class SynthesisEngine:
    def __init__(self):
        pass

    @staticmethod
    def _repair_synthesized_code(
        code: str,
        placeholder_names: frozenset = _CODE_PLACEHOLDERS,
    ) -> str:
        """
        AST-based structural repair of LLM-synthesized code. Runs in this order:

        1. Strip bare Return nodes at module scope — the LLM often writes code as if
           it is inside a function body.  Detected via ast.Return in ast.Module.body
           and removed structurally (no text matching).

        2. Compilation gate — reject code that still cannot compile as a module-level
           script after the strip.  Caller catches ValueError and falls back to MCTS.

        3. Normalize bare placeholder names (e.g. input_var → {input_var}) so that
           UnificationGate.unify() can substitute them at runtime.  Uses word-boundary
           regex so partial matches inside longer identifiers are never touched.
        """
        import ast as _ast

        temp_code = code
        # Strip trailing JSON delimiters (e.g. }, }, ], ", ",) leaked by LLM string formatting
        lines = temp_code.rstrip().splitlines()
        while lines and lines[-1].strip() in ("}", "},", "]", "],", '"', '",'):
            lines.pop()
        temp_code = "\n".join(lines)

        # ── Pre-step: Canonicalize all placeholder references to valid Python identifiers ──
        # Both braced ({input_var}) and bare (input_var) forms are replaced with
        # a unique temp name (_nstl_ph_name_) so the code is valid Python for
        # ast.parse() and compile().  The braces in {input_var} make it a Python
        # set display, causing "cannot assign to set display" SyntaxErrors.
        _TEMP_PREFIX = "_nstl_ph_"
        for name in placeholder_names:
            # Replace braced form first: {name} → _nstl_ph_name_
            temp_code = temp_code.replace('{' + name + '}', _TEMP_PREFIX + name + '_')
            # Replace bare form: word-boundary match only
            temp_code = re.sub(
                rf'(?<!\{{)\b{re.escape(name)}\b(?!\}})',
                _TEMP_PREFIX + name + '_',
                temp_code,
            )

        # ── Step 1: Strip module-level Return nodes ───────────────────────────────
        try:
            tree = _ast.parse(temp_code)
            original_len = len(tree.body)
            tree.body = [node for node in tree.body if not isinstance(node, _ast.Return)]
            if len(tree.body) != original_len:
                _ast.fix_missing_locations(tree)
                temp_code = _ast.unparse(tree)
                logger.info(
                    f"[SYNTHESIS REPAIR] Stripped "
                    f"{original_len - len(tree.body)} module-level Return node(s)."
                )
        except SyntaxError:
            # Cannot parse yet — proceed; compile gate below catches hard failures.
            pass

        # ── Step 2: Compilation gate (temp_code is valid Python at this point) ───
        try:
            compile(temp_code, "<synthesis>", "exec")
        except SyntaxError as e:
            raise ValueError(
                f"[SYNTHESIS REPAIR] Code failed compilation gate after structural repair: {e}\n"
                f"Code:\n{temp_code}"
            )

        # ── Step 3: Restore {braced} placeholder form ─────────────────────────────
        # Converts _nstl_ph_name_ → {name} so UnificationGate.unify() can substitute.
        result = temp_code
        for name in placeholder_names:
            result = result.replace(_TEMP_PREFIX + name + '_', '{' + name + '}')

        return result

    def synthesize_micro_cell(
        self,
        gap_concept: str,
        expected_input: str,
        expected_output: str,
        fetcher: LiveDocFetcher,
        context_hint: str = "",
    ) -> dict:
        """
        Dynamically generates a MicroCell to bridge a topological gap using live docs.
        Enforces strict VRAM/RAM purging and JSON grammar (A↑, R↓).

        Args:
            gap_concept:     Logical description of the bridging operation.
            expected_input:  Required input typestate name.
            expected_output: Required output typestate name.
            fetcher:         Live documentation fetcher for the target domain.
            context_hint:    Optional grounding string (domain, file hints, user intent)
                             assembled by the caller from runtime context — no parsing
                             happens inside this method.
        """
        logger.info(f"Fetching live docs for concept: {gap_concept}")
        live_docs = fetcher.fetch(gap_concept)
        if not live_docs:
            logger.warning(f"No live docs found for {gap_concept}. Proceeding with zero-shot knowledge.")
            live_docs = "No external documentation available."

        # Inject caller-supplied grounding block when available.
        grounding_block = (
            f"\nContextual Grounding (infer the correct library and domain from this):\n{context_hint}\n"
            if context_hint else ""
        )

        system_prompt = f"""You are an expert Software Engineer. You must write a Phase 2 MicroCell JSON that implements the concept '{gap_concept}'.
You MUST output ONLY valid JSON matching the schema. No markdown, no explanations.
{grounding_block}
Schema:
{{
  "cell_id": "micro_synthesized_<concept>",
  "type": "micro",
  "stage": 1,
  "keywords": ["<concept>"],
  "inputs": {{ "type_name": "{expected_input}", "state": "raw" }},
  "outputs": {{ "type_name": "{expected_output}", "state": "computed" }},
  "domain_implementations": {{
    "Python_Core": {{
      "code": "# Functional Python statements operating on {{input_var}}\n{{output_var}} = {{input_var}}",
      "dependencies": []
    }}
  }}
}}

CRITICAL INSTRUCTION: The `code` field MUST contain ACTUAL, working Python code. Import any required libraries (e.g. pandas as pd, cv2, numpy as np, math) directly in the code. Do NOT output placeholder text like `module_name` or `...`.
CRITICAL INSTRUCTION 2: You MUST use the exact placeholder strings `{{input_var}}` and `{{output_var}}` in your python logic for the main input and output of the cell. The engine will dynamically replace these at runtime. Do NOT use hardcoded variable names for the input or output.
CRITICAL INSTRUCTION 3: Do NOT call interactive input functions like `input()`, `sys.stdin`, or `raw_input()`. Assume input data or filenames are already available in `{{input_var}}`.

Use the following Official Documentation as your absolute ground truth:"""

        prompt = f"{system_prompt}\n\nTask: {gap_concept}\nLive Context:\n{live_docs}"

        logger.info("Executing generation via ModelManager...")
        result_text = ModelManager.get_instance().generate_text(prompt, max_tokens=2048)

        cleaned_text = result_text.strip()
        if "```json" in cleaned_text:
            cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_text:
            cleaned_text = cleaned_text.split("```")[1].split("```")[0].strip()

        # ── Parse with layered repair fallbacks (handles unescaped control chars, raw # comments, truncation) ──
        micro_json = None

        # Stage 1: Try strict=False (permits raw unescaped newlines/tabs inside string fields)
        try:
            micro_json = json.loads(cleaned_text, strict=False)
        except Exception as e:
            logger.warning(f"Initial JSON parse failed ({e}); attempting multi-stage repair...")

        # Stage 2: Strip Python comments (# ...) outside string literals & trailing commas
        if micro_json is None:
            no_comments = re.sub(r'^\s*#.*$', '', cleaned_text, flags=re.MULTILINE)
            repaired = re.sub(r',\s*([\}\]])', r'\1', no_comments)
            try:
                micro_json = json.loads(repaired, strict=False)
            except Exception:
                pass

        # Stage 3: Handle truncated/unterminated JSON (auto-close quotes and braces)
        if micro_json is None:
            first_brace = cleaned_text.find('{')
            if first_brace != -1:
                sub = cleaned_text[first_brace:]
                # Strip raw comments
                sub = re.sub(r'^\s*#.*$', '', sub, flags=re.MULTILINE)
                # Count open/close quotes and braces
                if sub.count('"') % 2 != 0:
                    sub += '"'
                open_b = sub.count('{') - sub.count('}')
                if open_b > 0:
                    sub += '}' * open_b
                try:
                    micro_json = json.loads(sub, strict=False)
                except Exception:
                    pass

        # Stage 4: Robust DOTALL regex extraction for "code" block fallback (handles partial/truncated code)
        if micro_json is None:
            code_match = re.search(r'"code"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*\}|\s*$)', cleaned_text, re.DOTALL)
            if not code_match:
                code_match = re.search(r'"code"\s*:\s*"(.*)', cleaned_text, re.DOTALL)
            if code_match:
                code_str = code_match.group(1).replace('\\"', '"').replace('\\n', '\n').rstrip('"\n\r\t ')
                safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', gap_concept).lower()[:40]
                micro_json = {
                    "cell_id": f"micro_synthesized_{safe_id}",
                    "type": "micro",
                    "stage": 1,
                    "keywords": [gap_concept],
                    "inputs": {"type_name": expected_input, "state": "raw"},
                    "outputs": {"type_name": expected_output, "state": "computed"},
                    "domain_implementations": {
                        "Python_Core": {
                            "code": code_str,
                            "dependencies": []
                        }
                    }
                }
            else:
                logger.error(
                    f"Failed to parse Synthesis LLM output.\n"
                    f"Raw output:\n{result_text[:500]}"
                )
                raise ValueError(
                    f"Synthesized output was not valid JSON. Raw output: {result_text[:200]}"
                )

        # ── AST structural repair on the extracted code string ───────────────────
        impl = micro_json.get("domain_implementations", {}).get("Python_Core", {})
        raw_code = impl.get("code", "")
        if raw_code:
            try:
                impl["code"] = SynthesisEngine._repair_synthesized_code(raw_code)
                logger.info("[SYNTHESIS REPAIR] Code passed structural repair and compilation gate.")
            except ValueError as repair_err:
                logger.error(f"[SYNTHESIS REPAIR FAILED] {repair_err}")
                # Re-raise so router.py / main.py catch it and fall back to MCTS.
                raise

        return micro_json
