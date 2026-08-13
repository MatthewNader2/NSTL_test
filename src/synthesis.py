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

        # ── Step 1: Strip module-level Return nodes ───────────────────────────────
        try:
            tree = _ast.parse(code)
            original_len = len(tree.body)
            tree.body = [node for node in tree.body if not isinstance(node, _ast.Return)]
            if len(tree.body) != original_len:
                _ast.fix_missing_locations(tree)
                code = _ast.unparse(tree)
                logger.info(
                    f"[SYNTHESIS REPAIR] Stripped "
                    f"{original_len - len(tree.body)} module-level Return node(s)."
                )
        except SyntaxError:
            # Cannot parse yet — proceed; compile gate below catches hard failures.
            pass

        # ── Step 2: Compilation gate (before placeholder bracing) ─────────────────
        try:
            compile(code, "<synthesis>", "exec")
        except SyntaxError as e:
            raise ValueError(
                f"[SYNTHESIS REPAIR] Code failed compilation gate after structural repair: {e}\n"
                f"Code:\n{code}"
            )

        # ── Step 3: Normalize bare placeholder names → {braced} form ─────────────
        # Word-boundary match ensures identifiers like `my_input_var` are untouched.
        for name in placeholder_names:
            code = re.sub(
                rf'(?<!\{{)\b{re.escape(name)}\b(?!\}})',
                '{' + name + '}',
                code,
            )

        return code

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
      "code": "# actual implementation of the concept\\n{{output_var}} = ...",
      "dependencies": []
    }}
  }}
}}

CRITICAL INSTRUCTION: The `code` field MUST contain ACTUAL, functional Python code that implements the concept. DO NOT output placeholder text like `<library>` or `...`. You MUST replace `...` with real library calls and real logic appropriate for the concept.
CRITICAL INSTRUCTION 2: You MUST use the exact placeholder strings `{{input_var}}` and `{{output_var}}` in your python logic for the main input and output of the cell. The engine will dynamically replace these at runtime. Do NOT use hardcoded variable names for the input or output.

Use the following Official Documentation as your absolute ground truth:"""

        prompt = f"{system_prompt}\n\nTask: {gap_concept}\nLive Context:\n{live_docs}"

        logger.info("Executing generation via ModelManager...")
        result_text = ModelManager.get_instance().generate_text(prompt, max_tokens=1024)

        cleaned_text = result_text.strip()
        if "```json" in cleaned_text:
            cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_text:
            cleaned_text = cleaned_text.split("```")[1].split("```")[0].strip()

        # ── Parse with layered repair fallbacks ──────────────────────────────────
        micro_json = None

        try:
            micro_json = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed ({e}); attempting repair...")

            # Attempt 1: truncate to last closing brace (handles unterminated outputs)
            try:
                repaired = cleaned_text.rsplit("}", 1)[0] + "}"
                micro_json = json.loads(repaired)
            except Exception:
                pass

            # Attempt 2: extract code field and reconstruct minimal JSON
            if micro_json is None:
                code_match = re.search(r'["\']code["\']\s*:\s*["\']([^"\']+)["\']', cleaned_text)
                if code_match:
                    code_str = code_match.group(1).replace("\\n", "\n")
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
                        f"Failed to parse Synthesis LLM output: {e}\n"
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
