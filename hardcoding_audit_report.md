# NSTL Comprehensive Hardcoding & Overfitting Audit Report

## Executive Summary

**Total violations found: 62**

| Category | Count | Critical | High | Medium |
|---|---|---|---|---|
| Prompt Sniffing & Keyword Dispatch | 12 | 4 | 5 | 3 |
| Mock Data, Dummy Graphs & Synthetic Literals | 10 | 3 | 4 | 3 |
| Benchmark String & Literal Fallbacks | 16 | 3 | 8 | 5 |
| AST Regex Hacks & String-Replacement Patching | 9 | 3 | 4 | 2 |
| Cell Scoring Biases & Evaluation Harness Leaks | 15 | 5 | 6 | 4 |

**Files affected: 12**
- `src/main.py`, `src/router.py`, `src/planner.py`, `src/unification.py`, `src/synthesis.py`, `src/internal_rag.py`, `src/eval_runner.py`, `src/main_backup.py`, `run_comprehensive_eval.py`, `harvesting/pattern_harvester.py`, `tools/verify_cells.py`, `temp_eval_code.py`

> [!CAUTION]
> The most critical systemic issue is that **prompt-sniffing keyword dispatch**, **hardcoded scoring biases**, and **relaxed evaluation assertions** form a mutually reinforcing loop: keyword dispatch steers routing toward benchmark-tailored cells, manual score boosts ensure they rank first, and empty/relaxed validation scripts guarantee a 100% pass rate regardless of output correctness. This triangle of overfitting makes benchmark scores unreliable indicators of actual system capability.

---

## Detailed Violation Inventory

---

### Finding ID: VIO-01
* **Category**: Prompt Sniffing
* **File & Line**: `src/main.py:97-111`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  def infer_goal_output_type(prompt: str) -> str:
      p_lower = prompt.lower()
      if any(k in p_lower for k in ["csv", "dataframe", "pandas", "table", "clean"]):
          return "DataFrame"
      elif any(k in p_lower for k in ["image", "opencv", "gray", "jpg", "png", "cv2"]):
          return "Mat"
      elif any(k in p_lower for k in ["model", "classifier", "predict", "fit"]):
          return "DataFrame"
      elif any(k in p_lower for k in ["dijkstra", "graph", "shortest", "dict"]):
          return "dict"
      return "any"
  ```
* **Why it breaks generalization**:
  Direct if/elif keyword dispatch on hardcoded word lists forces output typestates based on prompt substrings instead of structural analysis. A prompt containing "clean" in any context (e.g., "clean up my environment variables") would incorrectly force `DataFrame` output type. Novel domains receive `any`, bypassing soundness guarantees.
* **Recommended Principled Fix**:
  Replace with embedding-based type inference: embed the prompt and compare against typestate archetype embeddings in the RAG index, or propagate types backward from the matched cells' declared output typestates.

---

### Finding ID: VIO-02
* **Category**: Prompt Sniffing
* **File & Line**: `src/main.py:308-309`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  if expected_outputs in ("any", "Any", "*", "", "object"):
      expected_outputs = infer_goal_output_type(req.prompt)
  ```
* **Why it breaks generalization**:
  Invokes VIO-01's keyword-sniffing function during gap bridging, overriding the lattice's own typestate propagation with hardcoded prompt heuristics.
* **Recommended Principled Fix**:
  Use the matched cells' declared output typestates from the MCTS path or vector routing results.

---

### Finding ID: VIO-03
* **Category**: Prompt Sniffing
* **File & Line**: `src/unification.py:232-241`
* **Severity**: High
* **Code Snippet**:
  ```python
  if "descending" in prompt_lower or "desc" in prompt_lower or "highest to lowest" in prompt_lower:
      heuristics.append("ascending=False")
  elif "ascending" in prompt_lower:
      heuristics.append("ascending=True")
  if "grayscale" in prompt_lower or "gray" in prompt_lower or "bgr2gray" in prompt_lower:
      heuristics.append("cv2.COLOR_BGR2GRAY")
  ```
* **Why it breaks generalization**:
  Directly injects hardcoded Python code expressions into parameter heuristics based on substring matching. The keyword `"desc"` also matches "description" or "describe".
* **Recommended Principled Fix**:
  Move parameter inference to `ParameterExtractor.extract_slots()` which produces semantic slot bindings resolved by the cell's declared parameter schema.

---

### Finding ID: VIO-04
* **Category**: Prompt Sniffing
* **File & Line**: `src/unification.py:800-804`
* **Severity**: High
* **Code Snippet**:
  ```python
  slots.operational_flags["descending"] = any(w in prompt_lower for w in ["descending", "desc", "highest to lowest", "largest to smallest"])
  slots.operational_flags["is_grayscale"] = any(w in prompt_lower for w in ["grayscale", "gray", "bgr2gray", "gray_convert"])
  slots.operational_flags["is_hsv"] = any(w in prompt_lower for w in ["hsv", "bgr2hsv"])
  slots.operational_flags["is_rgb"] = any(w in prompt_lower for w in ["rgb", "bgr2rgb"])
  ```
* **Why it breaks generalization**:
  Hardcoded boolean flags for exactly four operations extracted via substring matching. Any other operational intent is ignored.
* **Recommended Principled Fix**:
  Replace with domain-agnostic intent extraction matching against target cell's declared parameter schema.

---

### Finding ID: VIO-05
* **Category**: Prompt Sniffing
* **File & Line**: `src/planner.py:62-69`
* **Severity**: High
* **Code Snippet**:
  ```python
  alias_map = {"opencv": "cv2", "pd": "pandas", "np": "numpy", "plt": "matplotlib", "pytorch": "torch", "sklearn": "scikit"}
  for d in available_domains:
      if d and d in prompt_lower:
          active_domains.add(d)
  ```
* **Why it breaks generalization**:
  Hardcoded alias dictionary and substring matching. "np" matches in "input", "pd" matches in "updated".
* **Recommended Principled Fix**:
  Build alias map dynamically from loaded cells' `domain_name` fields. Use word-boundary-aware matching.

---

### Finding ID: VIO-06
* **Category**: Prompt Sniffing
* **File & Line**: `src/planner.py:100-132`
* **Severity**: High
* **Code Snippet**:
  ```python
  if "imread" in tok: c_tokens.update(["im", "read", "load"])
  if "imwrite" in tok: c_tokens.update(["im", "write", "save"])
  if "cvtcolor" in tok: c_tokens.update(["cvt", "color", "convert", "grayscale"])
  if "dropna" in tok: c_tokens.update(["drop", "na", "rows", "missing", "values"])
  if "to_csv" in tok: c_tokens.update(["to", "csv", "save", "write"])
  if "read_csv" in tok: c_tokens.update(["read", "csv", "load"])
  if "sort_values" in tok: c_tokens.update(["sort", "values", "order", "descending", "ascending"])
  ```
* **Why it breaks generalization**:
  Hand-tuned token expansion rules for exactly 8 specific OpenCV/Pandas API methods. Other libraries get no expansion.
* **Recommended Principled Fix**:
  Use a general-purpose sub-word tokenizer or build synonym tables from cell keyword metadata at index time.

---

### Finding ID: VIO-07
* **Category**: Prompt Sniffing
* **File & Line**: `src/router.py:737-746`
* **Severity**: High
* **Code Snippet**:
  ```python
  SYNONYM_EXPANSIONS = {
      "missing": ["na", "null", "nan", "dropna"],
      "grayscale": ["gray", "cvtcolor", "bgr2gray"],
      "read": ["imread", "read_csv", "load", "open"],
      "save": ["imwrite", "to_csv", "save"],
      "write": ["imwrite", "to_csv", "save"],
  }
  ```
* **Why it breaks generalization**:
  Injects specific API method names into the token stream, directly biasing token overlap scoring toward benchmark cells.
* **Recommended Principled Fix**:
  Build synonym expansions dynamically from cells' keyword metadata or use embedding similarity.

---

### Finding ID: VIO-08
* **Category**: Prompt Sniffing
* **File & Line**: `src/router.py:819-822`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  if cell.cell_id in ("BUILTINS_INPUT", "SYS_STDIN_READ"):
      if not any(k in prompt_lower for k in ["prompt user", "ask user", "interactive", "console input", "stdin"]):
          domain_factor *= 0.05
  ```
* **Why it breaks generalization**:
  Exact cell ID string comparison + keyword sniffing. Tag cells with semantic metadata instead.
* **Recommended Principled Fix**:
  Use cell metadata tags (e.g., `"requires_user_interaction": true`) and tag-based filtering.

---

### Finding ID: VIO-09
* **Category**: Prompt Sniffing
* **File & Line**: `src/router.py:795-805`
* **Severity**: High
* **Code Snippet**:
  ```python
  if current_type == "DataFrame": expected_domains = {"pandas", "pd"}
  elif current_type in ("Mat", "Image"): expected_domains = {"opencv", "cv2"}
  elif current_type in ("ndarray", "Array"): expected_domains = {"numpy", "np"}
  ```
* **Why it breaks generalization**:
  Hardcoded type-to-domain bindings. Polars DataFrames, PyTorch Tensors, PIL Images cannot participate.
* **Recommended Principled Fix**:
  Store type-to-domain mappings in lattice cell metadata. Build a `type_name -> {domains}` index dynamically.

---

### Finding ID: VIO-10
* **Category**: Prompt Sniffing
* **File & Line**: `src/router.py:761-764`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  if sp.startswith("im") and len(sp) > 3: cell_sub_tokens.add(sp[2:])
  elif sp.startswith("to") and len(sp) > 3: cell_sub_tokens.add(sp[2:])
  ```
* **Why it breaks generalization**:
  Prefix stripping tailored to OpenCV `im*` and Pandas `to_*`. "implement" and "tokenize" are incorrectly decomposed.
* **Recommended Principled Fix**:
  Apply a general-purpose camelCase/snake_case tokenizer uniformly.

---

### Finding ID: VIO-11
* **Category**: Prompt Sniffing
* **File & Line**: `src/planner.py:368-380`
* **Severity**: High
* **Code Snippet**:
  ```python
  sub_matches = re.findall(r'["\']( PANDAS_[A-Z0-9_]+|CV2_[A-Z0-9_]+|NUMPY_[A-Z0-9_]+|SYNTH_[A-Z0-9_]+)["\']', clean_text)
  ```
* **Why it breaks generalization**:
  Regex recovery only extracts cell IDs with 4 hardcoded prefixes. All other domains are dropped.
* **Recommended Principled Fix**:
  Match against actual loaded cell ID prefixes dynamically.

---

### Finding ID: VIO-12
* **Category**: Prompt Sniffing
* **File & Line**: `src/internal_rag.py:457-489`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  if "imread" in tok: c_tokens.update(["im", "read", "load"])
  if "dropna" in tok: c_tokens.update(["drop", "na", "rows", "missing", "values"])
  # ... (duplicate of VIO-06)
  ```
* **Why it breaks generalization**:
  Duplicate of VIO-06 in the RAG engine. Benchmark-specific API token expansions.
* **Recommended Principled Fix**:
  Centralize tokenization strategy. Both `planner.py` and `internal_rag.py` should share a single implementation.

---

### Finding ID: VIO-13
* **Category**: Mock Data
* **File & Line**: `temp_eval_code.py:18-19`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  graph_input = input_source if input_source is not None else {'A': {'B': 1, 'C': 4}, 'B': {'C': 2, 'D': 5}, 'C': {'D': 1}, 'D': {}}
  python_dijkstra_shortest_path = dijkstra(graph_input, 'A')
  ```
* **Why it breaks generalization**:
  Synthetic 4-node graph injected when `input_source` is `None`. Start node `'A'` is hardcoded. This is the exact benchmark graph.
* **Recommended Principled Fix**:
  Raise `ValueError("No input source provided")` instead of silently falling back to dummy data.

---

### Finding ID: VIO-14
* **Category**: Mock Data
* **File & Line**: `harvesting/pattern_harvester.py:53`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  "{output_var} = dijkstra({input_var}, start_node if 'start_node' in locals() else list({input_var}.keys())[0] ...)"
  ```
* **Why it breaks generalization**:
  `locals()` sniffing for `start_node`, then fallback to first dict key. Circumvents proper parameter binding.
* **Recommended Principled Fix**:
  Add `start_node` as a declared input parameter in the cell's schema.

---

### Finding ID: VIO-15
* **Category**: Mock Data
* **File & Line**: `harvesting/pattern_harvester.py:184`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  "{output_var}.fit({input_var}, df_clean.iloc[:, -1] if 'df_clean' in locals() else df.iloc[:, -1])"
  ```
* **Why it breaks generalization**:
  `locals()` sniffing for `df_clean`/`df` and hardcoded `.iloc[:, -1]` assumption.
* **Recommended Principled Fix**:
  Add `target_column` as a declared parameter. Remove `locals()` introspection.

---

### Finding ID: VIO-16
* **Category**: Mock Data
* **File & Line**: `harvesting/pattern_harvester.py:197`
* **Severity**: High
* **Code Snippet**:
  ```python
  "{output_var} = {input_var}.predict(X_scaled if 'X_scaled' in locals() else X)"
  ```
* **Why it breaks generalization**:
  Sniffs `locals()` for hardcoded variable names from the evaluation pipeline.
* **Recommended Principled Fix**:
  Use `{input_var}` consistently. Scaling should be a separate upstream cell.

---

### Finding ID: VIO-17
* **Category**: Mock Data
* **File & Line**: `harvesting/pattern_harvester.py:24`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  "code": "def add(a, b):\n    return a + b\n\n{output_var} = add(5, 7)\nprint({output_var})"
  ```
* **Why it breaks generalization**: Hardcodes literal arguments `(5, 7)`.
* **Recommended Principled Fix**: Use parameterized inputs via `{input_var}`.

---

### Finding ID: VIO-18
* **Category**: Mock Data
* **File & Line**: `tools/verify_cells.py:34-48`
* **Severity**: High
* **Code Snippet**:
  ```python
  if p_name_lower == "code": return cv2.COLOR_BGR2GRAY
  if p_name_lower in ("ksize", "dsize", "size"): return (3, 3)
  if p_name_lower in ("thresh", "threshold", "maxval"): return 128.0
  ```
* **Why it breaks generalization**: Hardcoded OpenCV parameter-to-value mappings applied globally.
* **Recommended Principled Fix**: Use cell's declared parameter schema with defaults and type constraints.

---

### Finding ID: VIO-19
* **Category**: Mock Data
* **File & Line**: `tools/verify_cells.py:51-86`
* **Severity**: High
* **Code Snippet**:
  ```python
  return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})  # DataFrame
  return "synthetic_data.csv"  # str/file
  return {"a": 1}  # dict
  return np.zeros((8, 8, 3), dtype=np.uint8)  # Fallback for ALL unknown types
  ```
* **Why it breaks generalization**: Fixed schemas and dimensions; unknown types all get 8x8 uint8 array.
* **Recommended Principled Fix**: Generate synthetic inputs from cell parameter schemas.

---

### Finding ID: VIO-20
* **Category**: Mock Data
* **File & Line**: `tools/verify_cells.py:137-148`
* **Severity**: High
* **Code Snippet**:
  ```python
  elif result is not None:
      type_match = True  # Catch-all
  ```
* **Why it breaks generalization**: Bypasses all type checking if result is not None.
* **Recommended Principled Fix**: Remove catch-all. Enforce strict type checking for non-`any` types.

---

### Finding ID: VIO-21
* **Category**: Mock Data
* **File & Line**: `harvesting/pattern_harvester.py:210`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  "pd.DataFrame({'prediction': {input_var}}).to_csv({output_filename}, index=False)"
  ```
* **Why it breaks generalization**: Hardcoded column name `'prediction'`.
* **Recommended Principled Fix**: Use parameterized column name or infer from upstream model.

---

### Finding ID: VIO-22
* **Category**: Mock Data
* **File & Line**: `src/main.py:215-224`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  context.declare_variable(name="input_source", signature=AlgebraicSignature(type_name="str", state="source_identifier"))
  ```
* **Why it breaks generalization**: Hardcoded variable name and initial typestate.
* **Recommended Principled Fix**: Derive from entry cell requirements.

---

### Finding ID: VIO-23
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:18-32`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  CANONICAL_IMPORT_MAP = {"pd": ("import pandas as pd", "pandas"), "np": ("import numpy as np", "numpy"), ...}
  ```
* **Why it breaks generalization**: Only 13 libraries mapped. Others fail import resolution.
* **Recommended Principled Fix**: Build dynamically from cells' `dependencies` fields.

---

### Finding ID: VIO-24
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:34-41`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  KNOWN_PLACEHOLDERS = {"input_var", "output_var", ..., "newcascade", "oldcascade", "winname", "csgraph", ...}
  ```
* **Why it breaks generalization**: API-specific names from OpenCV/SciPy/Pandas. Novel placeholders fail validation.
* **Recommended Principled Fix**: Build dynamically from cell code templates at load time.

---

### Finding ID: VIO-25
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:71-77`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  SLOT_ROLE_MAP = {"SOURCE_URI": {...}, "DEST_URI": {...}, "COLUMN_NAME": {...}, "SORT_ORDER": {...}, "COLOR_CONV": {...}}
  ```
* **Why it breaks generalization**: Only 5 slot roles. Novel parameter roles cannot be expressed.
* **Recommended Principled Fix**: Define slot roles in cell parameter schemas as metadata.

---

### Finding ID: VIO-26
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:142-172`
* **Severity**: High
* **Code Snippet**:
  ```python
  bindings[p] = "input_path"       # SOURCE_URI fallback
  bindings[p] = "df.columns[0]"    # COLUMN_NAME fallback
  bindings[p] = "cv2.COLOR_BGR2GRAY"  # COLOR_CONV default
  ```
* **Why it breaks generalization**: Assumes `input_path`, `df`, and BGR→GRAY defaults exist at runtime.
* **Recommended Principled Fix**: Use `context.find_compatible_variable()`. Raise error if unresolvable.

---

### Finding ID: VIO-27
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:205-222`
* **Severity**: High
* **Code Snippet**:
  ```python
  file_matches = re.findall(r"\b([\w\-_.]+\.(?:csv|json|xlsx|parquet|...))\\b", user_prompt)
  self.extracted_parameters["input_filename"] = file_matches[0]  # First = input
  self.extracted_parameters["output_filename"] = file_matches[-1]  # Last = output
  ```
* **Why it breaks generalization**: Closed extension set; first/last positional heuristic breaks on inverted syntax.
* **Recommended Principled Fix**: Use `ParameterExtractor.extract_slots()` with save-keyword-relative positioning.

---

### Finding ID: VIO-28
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:338-365`
* **Severity**: High
* **Code Snippet**:
  ```python
  is_read_func = any(rk in func_name for rk in ["read", "load", "imread", "open"])
  is_write_func = any(wk in func_name for wk in ["write", "save", "export", "to", "imwrite"])
  if kw_key in ["ascending", "by", "axis", "inplace"] and any(sk in func_name for sk in ["sort", "order", "rank"]):
  ```
* **Why it breaks generalization**: Hardcoded function name substrings and keyword arguments for specific APIs.
* **Recommended Principled Fix**: Use cell's declared parameter schema for injection decisions.

---

### Finding ID: VIO-29
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:420-428`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  if len(parts) > 1 and parts[0] in ["pandas", "opencv", "scikit"]:
      parts = parts[1:]
  ```
* **Why it breaks generalization**: Only 3 library prefixes stripped from cell IDs.
* **Recommended Principled Fix**: Use cell's `domain_name` field dynamically.

---

### Finding ID: VIO-30
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:472-476`
* **Severity**: High
* **Code Snippet**:
  ```python
  if any(kw in cell_id.lower() for kw in ["imwrite", "savefig", "to_csv", "to_json", "to_parquet", "to_excel"]):
      compiled_snippet = compiled_snippet.replace("({input_var})", f"({repr(out_fname)}, {{input_var}})")
  ```
* **Why it breaks generalization**: Cell ID substring matching + blind string replacement tailored to specific function signatures.
* **Recommended Principled Fix**: Use AST-based parameter injection with signature awareness.

---

### Finding ID: VIO-31
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:522-525`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  if any(wk in func_name for wk in ["to_csv", "imwrite", "savefig", "to_json", "to_parquet", "to_excel"]):
  ```
* **Why it breaks generalization**: Hardcoded 6 save function names. `torch.save`, `json.dump`, `pickle.dump` not handled.
* **Recommended Principled Fix**: Use cell metadata to determine sink status.

---

### Finding ID: VIO-32
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:675-677`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  for std_mod in ["heapq", "json", "math", "re", "os", "sys", "time", "random"]:
  ```
* **Why it breaks generalization**: Only 8 stdlib modules. `collections`, `itertools`, `pathlib`, `datetime`, etc. ignored.
* **Recommended Principled Fix**: Detect stdlib usage via `sys.stdlib_module_names`.

---

### Finding ID: VIO-33
* **Category**: Literal Fallback
* **File & Line**: `src/unification.py:697-700`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  SINK_METHODS = {"to_csv", "to_json", "to_parquet", "to_feather", "to_sql", "imwrite", "imsave", "savefig", "dump", "export", "save"}
  ```
* **Why it breaks generalization**: Only 11 methods tracked as sinks.
* **Recommended Principled Fix**: Determine sink status from cell metadata (`output_type == "None"`).

---

### Finding ID: VIO-34
* **Category**: AST Hack
* **File & Line**: `src/unification.py:488-501`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  compiled_snippet = re.sub(
      r"(['\"])(?:input\.(?:jpg|png|jpeg|csv|json|parquet|txt)|input_file|dummy_input)\1",
      repr(in_fname), compiled_snippet, flags=re.IGNORECASE)
  compiled_snippet = re.sub(
      r"(['\"])(?:output\.(?:jpg|png|jpeg|csv|json|parquet|txt)|output_file|export\.\w+|dummy_output)\1",
      repr(out_fname), compiled_snippet, flags=re.IGNORECASE)
  ```
* **Why it breaks generalization**: Post-hoc regex searching for hardcoded dummy filename patterns in generated code. Corrupts legitimate strings containing these patterns.
* **Recommended Principled Fix**: Use formal `{input_filename}`/`{output_filename}` placeholders in all cell templates. Remove post-hoc regex patching.

---

### Finding ID: VIO-35
* **Category**: AST Hack
* **File & Line**: `src/unification.py:546-558`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  known_modules = {"cv2", "pd", "np", "plt", "sns", "tf", "torch", "sk", "sklearn", "os", "sys", "math", "re", "json"}
  snippet = re.sub(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\(", _rebind_caller, snippet)
  ```
* **Why it breaks generalization**: Regex-based method call rebinding. Any identifier not in the 14-module whitelist is forcibly rewritten to `latest_var`, corrupting sub-module access, custom classes, and chained expressions.
* **Recommended Principled Fix**: Use `ast.NodeTransformer` to identify unbound `Name` nodes and rebind using context's variable registry with type checking.

---

### Finding ID: VIO-36
* **Category**: AST Hack
* **File & Line**: `src/synthesis.py:46-49`
* **Severity**: High
* **Code Snippet**:
  ```python
  while lines and lines[-1].strip() in ("}", "},", "]", "],", '"', '",'):
      lines.pop()
  ```
* **Why it breaks generalization**: Strips valid Python lines ending with `}`, `]`, or `"`.
* **Recommended Principled Fix**: Parse LLM output using JSON/code boundary detection, not character matching.

---

### Finding ID: VIO-37
* **Category**: AST Hack
* **File & Line**: `src/synthesis.py:56-65`
* **Severity**: High
* **Code Snippet**:
  ```python
  temp_code = temp_code.replace('{' + name + '}', _TEMP_PREFIX + name + '_')
  temp_code = re.sub(rf'(?<!\{{)\b{re.escape(name)}\b(?!\}})', _TEMP_PREFIX + name + '_', temp_code)
  ```
* **Why it breaks generalization**: Blind string+regex substitution mangles valid internal variables named `input_var`/`output_var`.
* **Recommended Principled Fix**: Use AST-based placeholder detection via `ast.NodeTransformer`.

---

### Finding ID: VIO-38
* **Category**: AST Hack
* **File & Line**: `src/synthesis.py:199-228`
* **Severity**: High
* **Code Snippet**:
  ```python
  no_comments = re.sub(r'^\s*#.*$', '', cleaned_text, flags=re.MULTILINE)
  repaired = re.sub(r',\s*([\}\]])', r'\1', no_comments)
  code_match = re.search(r'"code"\s*:\s*"(.*?)"', cleaned_text, re.DOTALL)
  ```
* **Why it breaks generalization**: Multi-layer regex heuristics to repair LLM JSON output. Comment stripping corrupts code with `#` in URLs/regex.
* **Recommended Principled Fix**: Use a lenient JSON parser (`json5` or `pyjson5`).

---

### Finding ID: VIO-39
* **Category**: AST Hack
* **File & Line**: `src/main_backup.py:163-179`
* **Severity**: High
* **Code Snippet**:
  ```python
  compiled_snippet = re.sub(r"['\"]export\.(?:csv|json|html|feather|parquet)['\"]", f"'{user_assigned_name}'", compiled_snippet)
  compiled_snippet = re.sub(r"by=['\"]\\w+['\"]", f"by='{user_assigned_col}'", compiled_snippet)
  ```
* **Why it breaks generalization**: Regex replaces ALL bracket-access and by= patterns, corrupting unrelated code.
* **Recommended Principled Fix**: Use AST transformation to find specific `ast.Constant` nodes.

---

### Finding ID: VIO-40
* **Category**: AST Hack
* **File & Line**: `src/unification.py:380-382`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  dummy_code = f"dummy({p})"
  dummy_tree = ast.parse(dummy_code)
  ```
* **Why it breaks generalization**: Synthetic wrapper fails for multi-line or syntactically complex parameters.
* **Recommended Principled Fix**: Parse parameters with `ast.parse(p, mode='eval')`.

---

### Finding ID: VIO-41
* **Category**: AST Hack
* **File & Line**: `src/unification.py:770-784`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  for kw in ["save", "to", "write", "output", "export", "destination"]:
      pos = prompt.lower().find(kw)
  ```
* **Why it breaks generalization**: Naive string-index comparison. "to" appears in many non-output contexts.
* **Recommended Principled Fix**: Use compound pattern matching (`"save/write/export" + <filename>`).

---

### Finding ID: VIO-42
* **Category**: Score Bias
* **File & Line**: `src/internal_rag.py:432`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  for dist, idx in zip([0.70] * len(expanded_indices), expanded_indices):
  ```
* **Why it breaks generalization**: Replaces FAISS vector distances with a hardcoded constant `0.70`, erasing the neural embedding's ranking.
* **Recommended Principled Fix**: Preserve original FAISS distances. Recompute actual vector distance for expanded candidates.

---

### Finding ID: VIO-43
* **Category**: Score Bias
* **File & Line**: `src/internal_rag.py:344-350`
* **Severity**: High
* **Code Snippet**:
  ```python
  overlap = len(prompt_tokens.intersection(kws)) * 0.2 + len(prompt_tokens.intersection(id_parts)) * 0.1
  ```
* **Why it breaks generalization**: Arbitrary numerical bonuses override learned embedding space.
* **Recommended Principled Fix**: Normalize keyword overlap to the same scale as embedding distance.

---

### Finding ID: VIO-44
* **Category**: Score Bias
* **File & Line**: `src/internal_rag.py:447-453`
* **Severity**: High
* **Code Snippet**:
  ```python
  if cell_dom in active_domains: domain_weight += 0.35
  elif cell_dom in available_domains: domain_weight -= 0.45
  ```
* **Why it breaks generalization**: Hardcoded ±0.35/0.45 domain weights tuned for benchmarks.
* **Recommended Principled Fix**: Use multiplicative domain affinity from lattice type hierarchy.

---

### Finding ID: VIO-45
* **Category**: Score Bias
* **File & Line**: `src/internal_rag.py:504-520`
* **Severity**: High
* **Code Snippet**:
  ```python
  action_syns = {"read": {"read", "load", "open", "imread", ...}, "write": {"write", "save", "imwrite", ...}, ...}
  action_bonus += 0.35 / action_bonus -= 0.35
  ```
* **Why it breaks generalization**: Hand-crafted verb synonyms with heavy ±0.35 adjustments. 7 covered actions create bias.
* **Recommended Principled Fix**: Use embedding similarity or WordNet for verb matching.

---

### Finding ID: VIO-46
* **Category**: Score Bias
* **File & Line**: `src/internal_rag.py:524-532`
* **Severity**: High
* **Code Snippet**:
  ```python
  if "dataframe" in cid_lower: container_boost += 0.25
  elif "series" in cid_lower: container_boost -= 0.15
  ```
* **Why it breaks generalization**: Hardcoded container-specific boosts/penalties.
* **Recommended Principled Fix**: Use typestate compatibility from lattice type hierarchy.

---

### Finding ID: VIO-47
* **Category**: Score Bias
* **File & Line**: `src/internal_rag.py:535-537`
* **Severity**: High
* **Code Snippet**:
  ```python
  if any(obs in cid_lower for obs in ["cuda", "gpumat", "ocl", "gapi", "multi", "randu", "reshape", "metadata", "list_like", "common_convert"]):
      obscure_penalty += 0.35
  ```
* **Why it breaks generalization**: Penalizes GPU, reshape, and metadata cells. Prevents legitimate advanced usage.
* **Recommended Principled Fix**: Remove penalty. Use cell metadata tags for filtering.

---

### Finding ID: VIO-48
* **Category**: Score Bias
* **File & Line**: `src/internal_rag.py:539-543`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  score = (float(dist) * 0.40) + (concept_coverage * 0.35) + (seq_ratio * 0.25) + domain_weight + action_bonus + container_boost - id_len_penalty - obscure_penalty
  ```
* **Why it breaks generalization**: 6 layers of manual heuristic adjustments (~±2.0 total range) on top of ~1.0-scale embedding distance. Debug logging hardcodes specific cell names.
* **Recommended Principled Fix**: Replace with a learned re-ranker. Remove hardcoded debug filters.

---

### Finding ID: VIO-49
* **Category**: Score Bias
* **File & Line**: `src/internal_rag.py:420-430`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  if "SERIES_" in cid or "INDEX_" in cid:
      alt_cid = cid.replace("SERIES_", "DATAFRAME_").replace("INDEX_", "DATAFRAME_")
  ```
* **Why it breaks generalization**: String replacement targeting Pandas container types only.
* **Recommended Principled Fix**: Define container hierarchies in lattice metadata.

---

### Finding ID: VIO-50
* **Category**: Score Bias
* **File & Line**: `src/router.py:32-37`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  SIMILARITY_THRESHOLDS = {"data_engineering": 0.25, "image_processing": 0.25, "algorithms": 0.30, "default": 0.25}
  ```
* **Why it breaks generalization**: Static thresholds for 3 domains. Unseen domains default to 0.25.
* **Recommended Principled Fix**: Use adaptive thresholds or configure per-domain in lattice metadata.

---

### Finding ID: VIO-51
* **Category**: Score Bias
* **File & Line**: `src/router.py:472-474, 511`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  MIN_CONFIDENCE = 0.30; TUNNELING_MARGIN = 0.15; MACRO_THRESHOLD = 0.40
  ```
* **Why it breaks generalization**: Fixed magic numbers tuned for benchmark distributions.
* **Recommended Principled Fix**: Make configurable via `config.py`. Consider adaptive thresholds.

---

### Finding ID: VIO-52
* **Category**: Score Bias
* **File & Line**: `src/router.py:810-816`
* **Severity**: High
* **Code Snippet**:
  ```python
  is_domain_match = (cell_domain in expected_domains or in_input_type in ("DataFrame", "Mat") or out_output_type in ("DataFrame", "Mat"))
  if not is_domain_match: domain_factor = 0.1
  ```
* **Why it breaks generalization**: DataFrame/Mat exemptions bypass domain filtering. 90% penalty for mismatch.
* **Recommended Principled Fix**: Use lattice type hierarchy for cross-domain compatibility.

---

### Finding ID: VIO-53
* **Category**: Score Bias
* **File & Line**: `src/router.py:851-863`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  overlap = len(prompt_tokens.intersection(kws)) * 0.2 + len(prompt_tokens.intersection(id_parts)) * 0.1
  penalty = 0.0
  if ... == 'any': penalty += 0.3
  if ... != current_type: penalty += 0.5
  ```
* **Why it breaks generalization**: Heuristic additive score adjustments with manually tuned weights.
* **Recommended Principled Fix**: Consolidate into a single principled ranking function.

---

### Finding ID: VIO-54
* **Category**: Score Bias
* **File & Line**: `src/main_backup.py:503-504`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  best_local_score += 0.08
  ```
* **Why it breaks generalization**: Arbitrary +0.08 boost to local neighbor scores.
* **Recommended Principled Fix**: Use principled locality discount based on graph distance.

---

### Finding ID: VIO-55
* **Category**: Eval Leak
* **File & Line**: `src/eval_runner.py:142-143`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  else:
      passed = True  # No validation script = pass if it didn't crash
  ```
* **Why it breaks generalization**: Any code that exits cleanly passes, including empty/no-op code.
* **Recommended Principled Fix**: Mark tasks without validation as `"unvalidated"`, not `passed`.

---

### Finding ID: VIO-56
* **Category**: Eval Leak
* **File & Line**: `run_comprehensive_eval.py:65-67, 94-96`
* **Severity**: Critical
* **Code Snippet**:
  ```python
  "validate": """
  # Check that code parses and runs without fatal exceptions
  """
  ```
* **Why it breaks generalization**: Comment-only validation scripts contain zero assertions.
* **Recommended Principled Fix**: Write substantive assertions for every benchmark task.

---

### Finding ID: VIO-57
* **Category**: Eval Leak
* **File & Line**: `run_comprehensive_eval.py:50-52`
* **Severity**: High
* **Code Snippet**:
  ```python
  assert os.path.exists('output.jpg'), 'output.jpg was not created'
  out = cv2.imread('output.jpg')
  assert out is not None
  ```
* **Why it breaks generalization**: Never verifies image is actually grayscale.
* **Recommended Principled Fix**: Add `assert len(out.shape) == 2 or out.shape[2] == 1`.

---

### Finding ID: VIO-58
* **Category**: Eval Leak
* **File & Line**: `run_comprehensive_eval.py:83-85`
* **Severity**: High
* **Code Snippet**:
  ```python
  assert os.path.exists('predictions.csv')
  assert len(df_p) > 0
  ```
* **Why it breaks generalization**: Only checks file existence and non-empty. No ML pipeline verification.
* **Recommended Principled Fix**: Validate predictions count, value range, and model existence.

---

### Finding ID: VIO-59
* **Category**: Eval Leak
* **File & Line**: `run_comprehensive_eval.py:12-13`
* **Severity**: High
* **Code Snippet**:
  ```python
  PROJECT_ROOT = "/media/matthew/New Volume/grad_test/nstl_prototype"
  os.chdir(PROJECT_ROOT)
  ```
* **Why it breaks generalization**: Hardcoded absolute path to specific machine.
* **Recommended Principled Fix**: Use `os.path.dirname(os.path.abspath(__file__))`.

---

### Finding ID: VIO-60
* **Category**: Eval Leak
* **File & Line**: `src/eval_runner.py:14,19,31,67,118,170,173`
* **Severity**: High
* **Code Snippet**:
  ```python
  cwd="/media/matthew/New Volume/grad_test/nstl_prototype"
  "http://127.0.0.1:58102/api/..."
  subprocess.run("pkill -9 -f 'python3 src/main.py' || true", shell=True)
  ```
* **Why it breaks generalization**: Fixed paths, ports, and sweeping pkill commands.
* **Recommended Principled Fix**: Use configurable port/host, relative paths, and proper process management.

---

### Finding ID: VIO-61
* **Category**: Literal Fallback
* **File & Line**: `src/synthesis.py:123-124, 156-169`
* **Severity**: High
* **Code Snippet**:
  ```python
  cache_path = os.path.join("trees", "micro", "synthesized_nodes.json")
  "domain_implementations": {{"Python_Core": {{"code": "...", "dependencies": []}}}}
  ```
* **Why it breaks generalization**: Hardcoded cache path and `Python_Core` only domain. Multi-stage or non-Python synthesis impossible.
* **Recommended Principled Fix**: Use `trees_dir` argument for paths. Determine domain from gap context.

---

### Finding ID: VIO-62
* **Category**: Score Bias
* **File & Line**: `src/internal_rag.py:379-399`
* **Severity**: Medium
* **Code Snippet**:
  ```python
  common_words = {"read", "write", "file", "into", "from", ...}
  alias_map = {"opencv": "cv2", "pd": "pandas", ...}
  ```
* **Why it breaks generalization**: Triplicated stop words and alias maps across 3 files create maintenance burden and inconsistency.
* **Recommended Principled Fix**: Centralize in a single configuration file.

---

## Summary Table of All Findings

| ID | File | Line(s) | Category | Severity | Root Cause Summary |
|---|---|---|---|---|---|
| VIO-01 | src/main.py | 97-111 | Prompt Sniffing | Critical | Keyword dispatch for output type inference |
| VIO-02 | src/main.py | 308-309 | Prompt Sniffing | Critical | Invokes keyword sniffing during gap bridging |
| VIO-03 | src/unification.py | 232-241 | Prompt Sniffing | High | Keyword injection of ascending/grayscale code |
| VIO-04 | src/unification.py | 800-804 | Prompt Sniffing | High | Hardcoded boolean flags from prompt substrings |
| VIO-05 | src/planner.py | 62-69 | Prompt Sniffing | High | Hardcoded alias map + substring domain matching |
| VIO-06 | src/planner.py | 100-132 | Prompt Sniffing | High | API method token expansion for 8 methods |
| VIO-07 | src/router.py | 737-746 | Prompt Sniffing | High | Synonym expansions injecting API method names |
| VIO-08 | src/router.py | 819-822 | Prompt Sniffing | Medium | Cell ID + keyword sniffing for input() penalty |
| VIO-09 | src/router.py | 795-805 | Prompt Sniffing | High | Hardcoded type-to-domain string mapping |
| VIO-10 | src/router.py | 761-764 | Prompt Sniffing | Medium | Prefix stripping tailored to im*/to* |
| VIO-11 | src/planner.py | 368-380 | Prompt Sniffing | High | Regex recovery with hardcoded prefixes |
| VIO-12 | src/internal_rag.py | 457-489 | Prompt Sniffing | Critical | Duplicate benchmark-specific token expansion |
| VIO-13 | temp_eval_code.py | 18-19 | Mock Data | Critical | Hardcoded dummy graph + start node 'A' |
| VIO-14 | harvesting/pattern_harvester.py | 53 | Mock Data | Critical | locals() sniffing for start_node |
| VIO-15 | harvesting/pattern_harvester.py | 184 | Mock Data | Critical | locals() sniffing for df_clean + iloc[:,-1] |
| VIO-16 | harvesting/pattern_harvester.py | 197 | Mock Data | High | locals() sniffing for X_scaled/X |
| VIO-17 | harvesting/pattern_harvester.py | 24 | Mock Data | Medium | Hardcoded literal arguments (5, 7) |
| VIO-18 | tools/verify_cells.py | 34-48 | Mock Data | High | Hardcoded CV parameter-to-value mappings |
| VIO-19 | tools/verify_cells.py | 51-86 | Mock Data | High | Fixed synthetic DataFrames and arrays |
| VIO-20 | tools/verify_cells.py | 137-148 | Mock Data | High | Catch-all type match override |
| VIO-21 | harvesting/pattern_harvester.py | 210 | Mock Data | Medium | Hardcoded 'prediction' column name |
| VIO-22 | src/main.py | 215-224 | Mock Data | Medium | Hardcoded input_source bootstrap |
| VIO-23 | src/unification.py | 18-32 | Literal Fallback | Medium | Static import map for 13 libraries |
| VIO-24 | src/unification.py | 34-41 | Literal Fallback | Medium | Hand-curated placeholder whitelist |
| VIO-25 | src/unification.py | 71-77 | Literal Fallback | Medium | Fixed 5-role slot mapping |
| VIO-26 | src/unification.py | 142-172 | Literal Fallback | High | Fallback bindings assuming df/input_path |
| VIO-27 | src/unification.py | 205-222 | Literal Fallback | High | Closed extension set + first/last heuristic |
| VIO-28 | src/unification.py | 338-365 | Literal Fallback | High | Function name substring matching |
| VIO-29 | src/unification.py | 420-428 | Literal Fallback | Medium | Hardcoded library prefix stripping |
| VIO-30 | src/unification.py | 472-476 | Literal Fallback | High | Cell ID match + string replacement |
| VIO-31 | src/unification.py | 522-525 | Literal Fallback | Medium | Hardcoded 6 save function names |
| VIO-32 | src/unification.py | 675-677 | Literal Fallback | Medium | 8 hardcoded stdlib modules |
| VIO-33 | src/unification.py | 697-700 | Literal Fallback | Medium | 11 hardcoded sink methods |
| VIO-34 | src/unification.py | 488-501 | AST Hack | Critical | Regex replacement of dummy filenames |
| VIO-35 | src/unification.py | 546-558 | AST Hack | Critical | Regex rebinding with 14-module whitelist |
| VIO-36 | src/synthesis.py | 46-49 | AST Hack | High | Brittle JSON delimiter line popping |
| VIO-37 | src/synthesis.py | 56-65 | AST Hack | High | Blind placeholder text substitution |
| VIO-38 | src/synthesis.py | 199-228 | AST Hack | High | Multi-layer regex JSON repair |
| VIO-39 | src/main_backup.py | 163-179 | AST Hack | High | Regex column/filename replacement |
| VIO-40 | src/unification.py | 380-382 | AST Hack | Medium | dummy() wrapper for param parsing |
| VIO-41 | src/unification.py | 770-784 | AST Hack | Medium | Save keyword position heuristic |
| VIO-42 | src/internal_rag.py | 432 | Score Bias | Critical | FAISS distance replaced with literal 0.70 |
| VIO-43 | src/internal_rag.py | 344-350 | Score Bias | High | Arbitrary keyword overlap bonuses |
| VIO-44 | src/internal_rag.py | 447-453 | Score Bias | High | ±0.35/0.45 domain weight manipulation |
| VIO-45 | src/internal_rag.py | 504-520 | Score Bias | High | Action verb synonyms with ±0.35 scoring |
| VIO-46 | src/internal_rag.py | 524-532 | Score Bias | High | Container-specific boost/penalty |
| VIO-47 | src/internal_rag.py | 535-537 | Score Bias | High | Obscure module blacklist penalty |
| VIO-48 | src/internal_rag.py | 539-543 | Score Bias | Critical | Arbitrary score formula + debug filters |
| VIO-49 | src/internal_rag.py | 420-430 | Score Bias | Medium | Series/Index→DataFrame replacement |
| VIO-50 | src/router.py | 32-37 | Score Bias | Medium | Static domain thresholds |
| VIO-51 | src/router.py | 472-474 | Score Bias | Medium | Magic confidence/threshold literals |
| VIO-52 | src/router.py | 810-816 | Score Bias | High | DataFrame/Mat exemptions + 90% penalty |
| VIO-53 | src/router.py | 851-863 | Score Bias | Medium | Heuristic additive fallback scoring |
| VIO-54 | src/main_backup.py | 503-504 | Score Bias | Medium | +0.08 arbitrary local neighbor boost |
| VIO-55 | src/eval_runner.py | 142-143 | Eval Leak | Critical | Auto-pass on missing validation script |
| VIO-56 | run_comprehensive_eval.py | 65-67, 94-96 | Eval Leak | Critical | Comment-only validation scripts |
| VIO-57 | run_comprehensive_eval.py | 50-52 | Eval Leak | High | No grayscale verification |
| VIO-58 | run_comprehensive_eval.py | 83-85 | Eval Leak | High | Existence-only check for ML predictions |
| VIO-59 | run_comprehensive_eval.py | 12-13 | Eval Leak | High | Hardcoded absolute project root path |
| VIO-60 | src/eval_runner.py | 14,19,67,118,170 | Eval Leak | High | Hardcoded paths, ports, pkill commands |
| VIO-61 | src/synthesis.py | 123-169 | Literal Fallback | High | Hardcoded cache path + Python_Core only |
| VIO-62 | src/internal_rag.py | 379-399 | Score Bias | Medium | Triplicated stop words + alias maps |

---

## Architectural Root Causes

The 62 violations trace back to five systemic architectural gaps:

### 1. No Cell-Declared Parameter Schema
Cells declare `inputs` and `outputs` typestates but have no formal parameter schema (names, types, defaults, roles). This forces the unification engine to guess parameter bindings using string heuristics, regex matching, and prompt sniffing.

### 2. No Typestate Hierarchy / Subtyping Lattice
Type-to-domain mappings, container expansions, and type compatibility checks are all hardcoded string comparisons instead of being derived from a declared lattice type hierarchy.

### 3. Vector Embedding Distance Undermined by Manual Scoring
The RAG engine's FAISS vector search is systematically overridden by 6+ layers of manual heuristic adjustments (domain weights, action bonuses, container boosts, obscure penalties, dummy distance literals) that were clearly tuned to match benchmark outputs.

### 4. No Formal Placeholder System for Code Templates
Cell code templates inconsistently use `{input_var}`, `{output_var}`, bare variable names, and hardcoded dummy filenames, requiring post-hoc regex/string replacement hacks to patch them up during unification.

### 5. Evaluation Harness Not Independent of System
Benchmark validation scripts contain empty assertions, comment-only checks, existence-only verification, and auto-pass conditions that inflate reported accuracy without testing semantic correctness.
