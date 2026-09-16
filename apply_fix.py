import re
from pathlib import Path

def patch_file(path: Path, transform_fn):
    content = path.read_text(encoding="utf-8")
    new_content = transform_fn(content)
    if new_content != content:
        backup = path.with_suffix(path.suffix + ".orig")
        if not backup.exists():
            backup.write_text(content, encoding="utf-8")
        path.write_text(new_content, encoding="utf-8")
        print(f"[✓] Successfully patched: {path}")
        return True
    print(f"[i] No changes needed: {path}")
    return False

# --- 1. Patch src/unification.py ---
def patch_unification(code: str) -> str:
    # 1A. Dual-Port Feature Projection when binding df on DataFrame-to-Array cells
    target_pattern_scoped = """                    if scoped_var is not None:
                        cell_bindings[p_name] = scoped_var
                        continue"""

    replacement_scoped = """                    if scoped_var is not None:
                        # Dual-Port Feature Projection: slice feature columns if bridging to ndarray
                        if p_name == "df" and ("DATAFRAME_TO_NUMPY" in getattr(cell, "cell_id", "") or getattr(cell, "cell_id", "") == "PANDAS_DATAFRAME_TO_NUMPY"):
                            prompt_str = getattr(ctx, "prompt", "")
                            feat_m = re.search(r'\\bon\\s+([A-Za-z0-9_,\\s]+?)\\s+to\\s+predict', prompt_str, re.IGNORECASE)
                            if feat_m:
                                raw_feats = feat_m.group(1).strip()
                                feats = [f.strip() for f in re.split(r'[, ]+and\\s+|[,\\s]+', raw_feats) if f.strip() and f.strip().lower() not in ('a', 'the', 'column', 'columns')]
                                if feats:
                                    cell_bindings[p_name] = f"{scoped_var}[{feats!r}]"
                                    if hasattr(ctx, "consumed_tokens"):
                                        for f in feats:
                                            ctx.consumed_tokens.add(f.lower())
                                    continue
                        cell_bindings[p_name] = scoped_var
                        continue"""

    if target_pattern_scoped in code:
        code = code.replace(target_pattern_scoped, replacement_scoped, 1)

    # 1B. Dual-Port Target Projection for target_input / port y
    target_unconsumed_block = """                    # If no in-scope variable unified, check for unconsumed L0 universal literals
                    unconsumed_lits = [
                        val for _, kind, val in getattr(ctx, "ordered_literals", [])
                        if str(val).lower() not in getattr(ctx, "consumed_tokens", set())
                        and kind in ("identifier", "quoted_str")
                    ]
                    if unconsumed_lits:
                        if hasattr(ctx, "unresolved_ports"):
                            ctx.unresolved_ports.append((cell.cell_id, p_name))
                        ctx.unbindable_count = getattr(ctx, "unbindable_count", 0) + 1
                        continue"""

    replacement_unconsumed = """                    # If no in-scope variable unified, check for unconsumed L0 universal literals
                    unconsumed_lits = [
                        val for _, kind, val in getattr(ctx, "ordered_literals", [])
                        if str(val).lower() not in getattr(ctx, "consumed_tokens", set())
                        and kind in ("identifier", "quoted_str")
                    ]

                    # Dual-Port Typestate Projection:
                    # Project target vector from upstream DataFrame carrier for supervised tasks
                    p_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                    if p_role == "target_input" or p_name in ("y", "target", "y_true"):
                        target_col = None
                        prompt_str = getattr(ctx, "prompt", "")
                        pred_m = re.search(r'(?:to\\s+predict|predict)\\s+([A-Za-z0-9_]+)', prompt_str, re.IGNORECASE)
                        if pred_m:
                            target_col = pred_m.group(1).strip()
                        elif unconsumed_lits:
                            target_col = unconsumed_lits[-1]

                        if target_col:
                            df_candidate = None
                            for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                                tname = str(getattr(v_sig, "signature", None) or getattr(v_sig, "type_name", "")).lower()
                                if "dataframe" in tname or getattr(v_sig, "type_name", "") == "DataFrame":
                                    df_candidate = v_name
                                    break

                            if df_candidate:
                                if str(getattr(concrete_sig, "type_name", "")).lower() in ("ndarray", "tensor"):
                                    cell_bindings[p_name] = f"{df_candidate}['{target_col}'].to_numpy()"
                                else:
                                    cell_bindings[p_name] = f"{df_candidate}['{target_col}']"
                                if hasattr(ctx, "consumed_tokens"):
                                    ctx.consumed_tokens.add(str(target_col).lower())
                                continue

                    if unconsumed_lits:
                        if hasattr(ctx, "unresolved_ports"):
                            ctx.unresolved_ports.append((cell.cell_id, p_name))
                        ctx.unbindable_count = getattr(ctx, "unbindable_count", 0) + 1
                        continue"""

    if target_unconsumed_block in code:
        code = code.replace(target_unconsumed_block, replacement_unconsumed, 1)

    return code

# --- 2. Patch src/preflight.py ---
def patch_preflight(code: str) -> str:
    # Deduplicate universal literal validation checks so repetitions do not duplicate warnings
    old_lit_check = """                # Identifiers and file assets MUST be consumed
                if kind in ("file_asset", "identifier", "quoted_str"):"""

    new_lit_check = """                # Identifiers and file assets MUST be consumed
                if kind in ("file_asset", "identifier", "quoted_str"):
                    if clean_lit in seen_consumed_lits:
                        continue
                    seen_consumed_lits.add(clean_lit)"""

    if "seen_consumed_lits = set()" not in code:
        code = code.replace(
            "for _, clean_lit, kind in literals:",
            "seen_consumed_lits = set()\n            for _, clean_lit, kind in literals:"
        )

    if old_lit_check in code:
        code = code.replace(old_lit_check, new_lit_check, 1)

    # Deduplicate data_ids in estimator check
    code = code.replace(
        "has_multi_data_relation = len(data_ids) >= 2",
        "data_ids = list(dict.fromkeys(data_ids))\n        has_multi_data_relation = len(data_ids) >= 2"
    )

    return code

def main():
    root = Path(".")
    patch_file(root / "src/unification.py", patch_unification)
    patch_file(root / "src/preflight.py", patch_preflight)
    print("\n[✓] Finished applying updates.")

if __name__ == "__main__":
    main()
