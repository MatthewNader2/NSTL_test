import re
from pathlib import Path

def patch_file(path: Path, transform_fn):
    if not path.exists():
        print(f"[-] Skipped (not found): {path}")
        return False
    content = path.read_text(encoding="utf-8")
    new_content = transform_fn(content)
    if new_content != content:
        # Create backup
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(content, encoding="utf-8")
        path.write_text(new_content, encoding="utf-8")
        print(f"[✓] Successfully patched: {path}")
        return True
    print(f"[i] No changes needed: {path}")
    return False

# --- 1. Patch src/router.py ---
def patch_router(code: str) -> str:
    # Add fields to QueryIntent if not present
    if "feature_identifiers:" not in code:
        code = re.sub(
            r"(class QueryIntent:.*?\n)(\s+carrier_types:[^\n]+\n)",
            r"\1\2"
            r"    feature_identifiers: list = None\n"
            r"    target_identifier: str = None\n"
            r"    target_identifiers: list = None\n"
            r"    column_identifiers: list = None\n"
            r"    is_supervised: bool = False\n",
            code,
            flags=re.DOTALL
        )

    # Add deduplication and intent extraction to SemanticRouter
    if "def extract_semantic_intent" not in code:
        semantic_extractor = '''
    def extract_semantic_intent(self, prompt: str, literals: list) -> dict:
        """Extract generic feature and target semantic roles from user prompt."""
        features, target, is_supervised = [], None, False
        pred_match = re.search(
            r'(?:train|fit|regress|classify|model)?.*?\bon\s+([A-Za-z0-9_,\s]+?)\s+to\s+predict\s+([A-Za-z0-9_]+)',
            prompt, re.IGNORECASE
        )
        if pred_match:
            raw_feats, raw_tgt = pred_match.group(1), pred_match.group(2)
            target = raw_tgt.strip()
            features = [
                f.strip() for f in re.split(r'[, ]+and\s+|[,\s]+', raw_feats)
                if f.strip() and f.strip().lower() not in ('a', 'the', 'column', 'columns')
            ]
            is_supervised = True
        else:
            inv_match = re.search(
                r'\\bpredict\\s+([A-Za-z0-9_]+)\\s+(?:from|using|on)\\s+([A-Za-z0-9_,\s]+)',
                prompt, re.IGNORECASE
            )
            if inv_match:
                target = inv_match.group(1).strip()
                features = [
                    f.strip() for f in re.split(r'[, ]+and\s+|[,\s]+', inv_match.group(2))
                    if f.strip() and f.strip().lower() not in ('a', 'the', 'column', 'columns')
                ]
                is_supervised = True

        all_ids = [getattr(l, 'value', str(l)) for l in literals if getattr(l, 'kind', '') == 'identifier']
        if not features and is_supervised and all_ids:
            features = [i for i in all_ids if i != target]

        return {
            'features': features,
            'target': target,
            'targets': [target] if target else [],
            'column_identifiers': all_ids,
            'is_supervised': is_supervised
        }
'''
        # Inject method before route or analyze
        if "def route(" in code:
            code = code.replace("    def route(", semantic_extractor + "\n    def route(")
        elif "def extract_universal_literals(" in code:
            code = code.replace("    def extract_universal_literals(", semantic_extractor + "\n    def extract_universal_literals(")

    # Deduplicate literals inside extract_universal_literals
    dedup_code = '''        seen = set()
        deduped = []
        for lit in raw_literals:
            key = (getattr(lit, 'value', str(lit)), getattr(lit, 'kind', 'identifier'))
            if key not in seen:
                seen.add(key)
                deduped.append(lit)
        return deduped'''
    code = re.sub(
        r'(\s+raw_literals\.extend\(self\._extract_clause_literals\(clause\)\)\s+)(return literals|return raw_literals)',
        r'\1' + dedup_code,
        code
    )

    # Populate semantic attributes in analyze_intent / route
    if "semantics = self.extract_semantic_intent" not in code:
        code = re.sub(
            r'(literals = self\.extract_universal_literals\([^)]+\))',
            r"\1\n        semantics = self.extract_semantic_intent(prompt, literals)",
            code
        )
        code = re.sub(
            r'(QueryIntent\([^)]*universal_literals=literals)([^)]*\))',
            r"\1, feature_identifiers=semantics['features'], target_identifier=semantics['target'], target_identifiers=semantics['targets'], column_identifiers=semantics['column_identifiers'], is_supervised=semantics['is_supervised']\2",
            code
        )

    return code

# --- 2. Patch src/planner.py ---
def patch_planner(code: str) -> str:
    # Ensure planner dynamically evaluates supervised target port demands
    if "_is_port_satisfied" in code and "Dual-Port" not in code:
        replacement = '''    def _is_port_satisfied(self, port, path, current_step, intent=None):
        if getattr(port, 'name', '') in ('X', 'df', 'data', 'array', 'self'):
            return True
        # Dual-Port satisfaction for supervised target input
        if getattr(port, 'name', '') == 'y' or getattr(port, 'role', '') == 'target_input':
            if intent and (getattr(intent, 'target_identifier', None) or getattr(intent, 'target_identifiers', None)):
                has_df = any(getattr(s, 'primary_out', '') == 'DataFrame' or 'PD_' in getattr(s, 'id', '') for s in path)
                if has_df:
                    return True
        return False'''
        code = re.sub(r'    def _is_port_satisfied\(self,[^)]+\):.*?(?=\n    def |\Z)', replacement, code, flags=re.DOTALL)
    return code

# --- 3. Patch src/unification.py ---
def patch_unification(code: str) -> str:
    # Upgrade Robinson unifier with Dual Port Typestate Projection
    if "Dual Port Typestate Projection" not in code:
        # 1. Update PortBinding dataclass to support consumed literals
        if "consumed_literals:" not in code:
            code = re.sub(
                r'(class PortBinding:.*?\n)(\s+provenance:[^\n]+\n)',
                r'\1\2    consumed_literals: list = None\n',
                code,
                flags=re.DOTALL
            )

        # 2. Inject Dual Port target resolution
        dual_port_logic = '''
            # Dual Port Typestate Unification for target port 'y'
            if port.name == 'y' or getattr(port, 'role', '') == 'target_input':
                target_var = env.lookup_variable(type_hint=getattr(port, 'port_type', 'ndarray'), role='target')
                if target_var:
                    bindings[port.name] = PortBinding(
                        step=step_num, cell_id=node.id, port_name=port.name,
                        direction="IN", type_and_state=getattr(port, 'port_type', 'ndarray'),
                        required="Yes" if getattr(port, 'required', False) else "No",
                        bound_expr=target_var, provenance="Wired Variable"
                    )
                    continue

                df_var = env.lookup_variable(type_hint='DataFrame')
                target_col = getattr(intent, 'target_identifier', None) if intent else None
                if not target_col and intent and getattr(intent, 'target_identifiers', None):
                    target_col = intent.target_identifiers[0]

                if df_var and target_col:
                    target_expr = f"{df_var}['{target_col}'].to_numpy()" if getattr(port, 'port_type', '') == 'ndarray' else f"{df_var}['{target_col}']"
                    bindings[port.name] = PortBinding(
                        step=step_num, cell_id=node.id, port_name=port.name,
                        direction="IN", type_and_state=getattr(port, 'port_type', 'ndarray'),
                        required="Yes" if getattr(port, 'required', False) else "No",
                        bound_expr=target_expr, provenance="Dual Port Typestate Projection",
                        consumed_literals=[target_col]
                    )
                    continue
'''
        code = re.sub(
            r'(if port\.name == [\'"]y[\'"][^\n]*:)',
            dual_port_logic,
            code
        )

        # 3. Inject Feature Projection on DataFrame-to-Array transforms
        feature_proj_logic = '''
        if 'PANDAS_DATAFRAME_TO_NUMPY' in node.id or (getattr(node, 'primary_in', '') == 'DataFrame' and getattr(node, 'primary_out', '') == 'ndarray'):
            if intent and getattr(intent, 'feature_identifiers', None):
                df_b = bindings.get('df')
                if df_b and '[' not in str(df_b.bound_expr):
                    feats = intent.feature_identifiers
                    df_b.bound_expr = f"{df_b.bound_expr}[{feats!r}]"
                    df_b.provenance = "Dual Port Typestate Projection"
                    if getattr(df_b, 'consumed_literals', None) is not None:
                        df_b.consumed_literals.extend(feats)
'''
        if "return bindings" in code:
            code = code.replace("return bindings", feature_proj_logic + "\n        return bindings")

    return code

# --- 4. Patch src/synthesis.py ---
def patch_synthesis(code: str) -> str:
    # Ensure train_test_split and linear regression emit cleanly with dual-port variables
    if "train_test_split" in code:
        code = re.sub(
            r'lines\.append\(f"\{v_xtr\}, \{v_xte\}, \{v_ytr\}, \{v_yte\} = train_test_split\(\{x_in\}, \{y_in\},',
            r'y_val = getattr(y_in, "bound_expr", str(y_in))\n                lines.append(f"{v_xtr}, {v_xte}, {v_ytr}, {v_yte} = train_test_split({x_in}, {y_val},',
            code
        )
    return code

# --- 5. Patch src/validator.py ---
def patch_validator(code: str) -> str:
    # Validate against unique prompt literals and respect consumed literals from projections
    if "seen_lits" not in code:
        code = re.sub(
            r'(for lit in intent\.universal_literals:)',
            r'seen_lits = set()\n        \1\n            if lit.value in seen_lits: continue\n            seen_lits.add(lit.value)',
            code
        )
    if "getattr(binding, 'consumed_literals'" not in code:
        code = re.sub(
            r'(for port_name, binding in step\.bindings\.items\(\):)',
            r"\1\n                for c_lit in (getattr(binding, 'consumed_literals', None) or []): consumed_literals.add(c_lit)",
            code
        )
    return code

def main():
    root = Path(".")
    patch_file(root / "src/router.py", patch_router)
    patch_file(root / "src/planner.py", patch_planner)
    patch_file(root / "src/unification.py", patch_unification)
    patch_file(root / "src/synthesis.py", patch_synthesis)
    patch_file(root / "src/validator.py", patch_validator)
    print("\n[✓] All patches applied. Run 'git diff' to review changes.")

if __name__ == "__main__":
    main()
