from pathlib import Path
import re

# 1. Fix src/preflight.py: initialize seen_consumed_lits
p_preflight = Path("src/preflight.py")
code_pre = p_preflight.read_text(encoding="utf-8")

if "seen_consumed_lits: Set[str] = set()" not in code_pre:
    code_pre = code_pre.replace(
        "bound_values_str: Set[str] = set()",
        "bound_values_str: Set[str] = set()\n            seen_consumed_lits: Set[str] = set()",
        1
    )
    p_preflight.write_text(code_pre, encoding="utf-8")
    print("[✓] Patched src/preflight.py (initialized seen_consumed_lits)")
else:
    print("[i] src/preflight.py already initialized")

# 2. Fix src/unification.py: feature column projection on DATAFRAME_TO_NUMPY
p_unif = Path("src/unification.py")
code_unif = p_unif.read_text(encoding="utf-8")

feat_proj_anchor = """                    for p_name, p_sig in cand_ports:
                        u_sub = unify(prod_sig.signature, p_sig.signature, accumulated_sigma)
                        if u_sub is not None:
                            cell_bindings[p_name] = producer_var
                            accumulated_sigma = u_sub
                            bound_producer = True
                            break"""

feat_proj_replacement = """                    for p_name, p_sig in cand_ports:
                        u_sub = unify(prod_sig.signature, p_sig.signature, accumulated_sigma)
                        if u_sub is not None:
                            cell_bindings[p_name] = producer_var
                            accumulated_sigma = u_sub
                            bound_producer = True
                            break

            # Dual-Port Feature Projection for DataFrame-to-Array transforms
            if bound_producer and ("DATAFRAME_TO_NUMPY" in getattr(cell, "cell_id", "") or getattr(cell, "cell_id", "") == "PANDAS_DATAFRAME_TO_NUMPY"):
                prompt_str = getattr(ctx, "prompt", "")
                feat_m = re.search(r'\\bon\\s+([A-Za-z0-9_,\\s]+?)\\s+to\\s+predict', prompt_str, re.IGNORECASE)
                if feat_m:
                    raw_feats = feat_m.group(1).strip()
                    feats = [f.strip() for f in re.split(r'[, ]+and\\s+|[,\\s]+', raw_feats) if f.strip() and f.strip().lower() not in ('a', 'the', 'column', 'columns')]
                    if feats:
                        for p_k in ("df", getattr(cell.primary_input, "name", "df")):
                            if p_k in cell_bindings and "[" not in str(cell_bindings[p_k]):
                                cell_bindings[p_k] = f"{cell_bindings[p_k]}[{feats!r}]"
                                if hasattr(ctx, "consumed_tokens"):
                                    for f in feats:
                                        ctx.consumed_tokens.add(f.lower())"""

if "Dual-Port Feature Projection for DataFrame-to-Array transforms" not in code_unif:
    code_unif = code_unif.replace(feat_proj_anchor, feat_proj_replacement, 1)
    p_unif.write_text(code_unif, encoding="utf-8")
    print("[✓] Patched src/unification.py (added feature projection)")
else:
    print("[i] src/unification.py already has feature projection")

print("\n[✓] All patches up to date.")
