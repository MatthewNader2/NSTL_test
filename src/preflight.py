"""
src/preflight.py - Neuro-Symbolic Topological Lattice (NSTL)
Static Pre-Flight Linter before GEVR Sandbox Execution.

Structural invariants enforced post-binding:
1. Universal Literal Consumption: Every universal literal extracted at L0 intent
   (identifiers, column names, file paths) is consumed by at least one bound port on
   the final path, or explicitly waived.
2. Data-Bearing Port Integrity: No call position receives a bare None where the bound
   port's role is data-bearing (feature_input, target_input, data_input, model_input,
   source_data, model_sink).
3. Structural Estimator Contract: For any cell whose signature shape matches a
   *supervised* estimator contract (features + target) AND whose prompt carries a
   supervised-learning verb, both feature and target ports must be bound when the
   prompt declares a prediction/training relation (>= 2 data identifiers extracted
   at L0). Unsupervised tasks (clustering, dimensionality reduction, anomaly
   detection) are explicitly exempt.

All role / verb classification is delegated to TypeRegistry. This module contains no
domain-specific identifiers.
"""

from __future__ import annotations
import ast
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass, field

try:
    from .lattice import UNRESOLVED_PORT, TypeRegistry, AlgebraicSignature, PortSignature
    from .tokenizer import CellTokenizer
    from .unification import ExecutionContext, unify, _shape_compatible, _is_sink_cell, resolve_typed_binding_record, TypedBindingRecord
except (ImportError, ValueError):
    from lattice import UNRESOLVED_PORT, TypeRegistry, AlgebraicSignature, PortSignature
    from tokenizer import CellTokenizer
    from unification import ExecutionContext, unify, _shape_compatible, _is_sink_cell, resolve_typed_binding_record, TypedBindingRecord


@dataclass
class VariableSchema:
    """Tracks active and invalidated column schemas per variable (R5-3)."""
    known_columns: Optional[Set[str]] = None
    removed_columns: Set[str] = field(default_factory=set)
    added_columns: Set[str] = field(default_factory=set)
    removed_by: Dict[str, str] = field(default_factory=dict)



class _DynamicDataBearingRoles(frozenset):
    """Dynamic data-bearing roles proxy backed by TypeRegistry."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_data_bearing_roles()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_data_bearing_roles())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_data_bearing_roles())


DATA_BEARING_ROLES = _DynamicDataBearingRoles()


class _DynamicConventionalEstimatorVerbs(frozenset):
    """Dynamic conventional estimator verbs proxy backed by TypeRegistry."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_estimator_verbs()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_estimator_verbs())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_estimator_verbs())


CONVENTIONAL_ESTIMATOR_VERBS = _DynamicConventionalEstimatorVerbs()


# ---------------------------------------------------------------------------
# Unbound-value detection (domain-agnostic)
# ---------------------------------------------------------------------------

# String sentinels that indicate an unbound slot. Case-insensitive.
_UNBOUND_STRING_SENTINELS: Set[str] = {
    "none",
    "null",
    "<unresolved>",
    "<unbound>",
}

try:
    _UNRESOLVED_PORT_SENTINEL = str(UNRESOLVED_PORT).strip().lower()
except Exception:
    _UNRESOLVED_PORT_SENTINEL = ""


def _is_unbound(value: Any) -> bool:
    """Return True when a port binding value represents an unbound/None slot."""
    if value is None or value is UNRESOLVED_PORT:
        return True
    try:
        s = str(value).strip().lower()
    except Exception:
        return False
    if s in _UNBOUND_STRING_SENTINELS:
        return True
    if _UNRESOLVED_PORT_SENTINEL and s == _UNRESOLVED_PORT_SENTINEL:
        return True
    return False


def _check_callable_callees(tree: ast.AST, violations: List[str]) -> None:
    """
    Host-language sanity check (domain-agnostic): a call whose callee is a
    non-callable CONSTANT (a string, number, bool...) can never execute
    successfully -- it is the signature of an unfilled template placeholder
    reaching emission ("var = 'X'(var_1)"). Catching it here keeps the failure
    loud and honest at synthesis time instead of at sandbox time.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Constant):
            violations.append(
                f"Synthesized code calls a non-callable constant "
                f"{node.func.value!r} as a function (unfilled template placeholder "
                f"reached emission)."
            )


# ---------------------------------------------------------------------------
# Errors / results
# ---------------------------------------------------------------------------

class PreflightLintError(Exception):
    """Raised when synthesized code fails static structural pre-flight validation."""
    def __init__(self, message: str, violations: Optional[List[str]] = None, structured_violations: Optional[List[PreflightViolation]] = None):
        super().__init__(message)
        self.violations = violations or [message]
        self.structured_violations = structured_violations or []


@dataclass
class PreflightViolation:
    check_id: str
    message: str
    cell_id: Optional[str] = None
    port_name: Optional[str] = None
    variable_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightLintResult:
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    structured_violations: List[PreflightViolation] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            msg = "Pre-flight lint validation failed:\n  - " + "\n  - ".join(self.violations)
            raise PreflightLintError(msg, violations=self.violations, structured_violations=self.structured_violations)


# ---------------------------------------------------------------------------
# Linter
# ---------------------------------------------------------------------------

class PreflightLinter:
    """
    Domain-agnostic static validator inspecting pipeline bindings and synthesized
    ASTs strictly through structural contracts and port roles supplied by the
    TypeRegistry. No domain-specific identifier, verb, or role name is hardcoded.
    """

    # ------------------------------------------------------------------
    # Registry accessors (all classification flows through here)
    # ------------------------------------------------------------------

    @staticmethod
    def _registry() -> Any:
        return TypeRegistry.get_instance()

    @classmethod
    def _role_set(cls, *accessor_names: str) -> Set[str]:
        """Union of role-name sets returned by any available registry accessor."""
        reg = cls._registry()
        roles: Set[str] = set()
        for name in accessor_names:
            getter = getattr(reg, name, None)
            if callable(getter):
                try:
                    roles.update(str(r).lower() for r in getter())
                except Exception:
                    continue
        return roles

    @classmethod
    def _verb_set(cls, *accessor_names: str) -> Set[str]:
        """Union of verb sets returned by any available registry accessor."""
        reg = cls._registry()
        verbs: Set[str] = set()
        for name in accessor_names:
            getter = getattr(reg, name, None)
            if callable(getter):
                try:
                    verbs.update(str(v).lower() for v in getter())
                except Exception:
                    continue
        return verbs

    @classmethod
    def _feature_roles(cls) -> Set[str]:
        # Declared role vocabulary (trees' verification_semantics); falls back
        # to the legacy accessor names when a registry provides them.
        roles = {
            str(r).lower()
            for r in TypeRegistry.get_instance().get_verification_semantics("feature_roles")
        }
        return roles or cls._role_set("get_feature_roles", "get_feature_bearing_roles")

    @classmethod
    def _target_roles(cls) -> Set[str]:
        roles = {
            str(r).lower()
            for r in TypeRegistry.get_instance().get_verification_semantics("target_roles")
        }
        return roles or cls._role_set("get_target_roles", "get_target_bearing_roles")

    @classmethod
    def _supervised_verbs(cls) -> Set[str]:
        return cls._verb_set("get_supervised_estimator_verbs", "get_supervised_verbs", "get_estimator_verbs")

    @classmethod
    def _unsupervised_verbs(cls) -> Set[str]:
        return cls._verb_set("get_unsupervised_estimator_verbs", "get_unsupervised_verbs")

    # ------------------------------------------------------------------
    # Structural predicates
    # ------------------------------------------------------------------

    @classmethod
    def _prompt_is_supervised(cls, prompt: str) -> bool:
        """
        Decide whether a prompt describes a supervised (features -> target) relation.

        Rules (fail-closed — never infer supervision that isn't attested):
          * If the registry exposes unsupervised verbs and any appear in the prompt,
            the prompt is treated as unsupervised -> returns False.
          * If the registry does not expose supervised verbs, returns False so the
            estimator check is skipped rather than risk a false positive.
          * Otherwise, returns True iff at least one supervised verb appears.
        """
        if not prompt:
            return False
        # Zero-regex tokenization via the shared CellTokenizer (structural
        # identifier splitting only — no pattern scanning, no domain vocab).
        try:
            tokens = {t.lower() for t in CellTokenizer.tokenize_prompt(prompt)}
        except Exception:
            tokens = set()
        if not tokens:
            return False

        unsupervised = cls._unsupervised_verbs()
        if unsupervised and (tokens & unsupervised):
            return False

        supervised = cls._supervised_verbs()
        if not supervised:
            return False
        return bool(tokens & supervised)

    @classmethod
    def _port_role(cls, port_sig: Any) -> str:
        role = getattr(port_sig, "port_role", None) or getattr(port_sig, "derived_role", "") or ""
        return str(role).lower()

    @classmethod
    def _port_is_optional(cls, port_sig: Any) -> bool:
        """A port is optional when its signature explicitly marks it so."""
        if getattr(port_sig, "required", None) is False:
            return True
        if getattr(port_sig, "optional", None) is True:
            return True
        return False

    @classmethod
    def _is_object_receiver_role(cls, role: str) -> bool:
        """True for the declared data-bearing roles that carry a fitted model/estimator
        instance rather than array-like data. The role names are DECLARED tree data
        (verification_semantics.model_roles) -- the same source Check 2 draws on."""
        try:
            model_roles = {
                str(r).lower()
                for r in TypeRegistry.get_instance().get_verification_semantics("model_roles")
            }
        except Exception:
            model_roles = set()
        return role in (model_roles or {"model_input", "model_sink"})

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    @classmethod
    def lint(
        cls,
        pipeline_bindings: List[Tuple[Any, Dict[str, Any]]],
        prompt: str = "",
        code_str: Optional[str] = None,
        waived_literals: Optional[Set[str]] = None,
    ) -> PreflightLintResult:
        violations: List[str] = []
        warnings: List[str] = []
        structured_violations: List[PreflightViolation] = []
        waived = set(waived_literals or set())

        def _add_violation(
            check_id: str,
            message: str,
            cell_id: Optional[str] = None,
            port_name: Optional[str] = None,
            variable_name: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
        ) -> None:
            violations.append(message)
            structured_violations.append(
                PreflightViolation(
                    check_id=check_id,
                    message=message,
                    cell_id=cell_id,
                    port_name=port_name,
                    variable_name=variable_name,
                    details=details or {},
                )
            )

        # Extract universal literals from prompt if ExecutionContext is available.
        extracted_literals: List[Tuple[int, str, str]] = []
        try:
            extracted_literals = ExecutionContext._extract_universal_literals(prompt or "")
        except Exception:
            pass

        # -----------------------------------------------------------------
        # Check 1: Universal Literal Consumption
        # -----------------------------------------------------------------
        if extracted_literals:
            bound_values_str: Set[str] = set()
            for _cell, bindings in pipeline_bindings:
                for _p_name, val in bindings.items():
                    if val is not None:
                        bound_values_str.add(str(val).strip("'\""))

            stopwords = cls._registry().get_stopwords()
            seen_consumed_lits: Set[str] = set()

            for _, kind, lit_val in extracted_literals:
                clean_lit = lit_val.strip("'\"")
                if not clean_lit or clean_lit in waived or clean_lit.lower() in stopwords:
                    continue

                if kind not in ("file_asset", "identifier", "quoted_str"):
                    continue

                if clean_lit in seen_consumed_lits:
                    continue
                seen_consumed_lits.add(clean_lit)

                is_consumed = any(
                    clean_lit == b or clean_lit in b
                    for b in bound_values_str
                )

                if not is_consumed and code_str:
                    try:
                        tree = ast.parse(code_str)
                        clean_lower = clean_lit.lower()
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Constant):
                                val_str = str(node.value).lower()
                                if val_str == clean_lower or clean_lower in val_str:
                                    is_consumed = True
                                    break
                            elif isinstance(node, ast.Name):
                                if node.id.lower() == clean_lower:
                                    is_consumed = True
                                    break
                            elif isinstance(node, ast.keyword):
                                if node.arg and node.arg.lower() == clean_lower:
                                    is_consumed = True
                                    break
                    except SyntaxError:
                        _add_violation("syntax_error", "Synthesized code failed static AST parsing (SyntaxError).")
                        is_consumed = False
                    except Exception:
                        is_consumed = False

                if not is_consumed:
                    msg = (
                        f"Universal literal '{clean_lit}' (kind: {kind}) extracted from prompt "
                        f"was not consumed by any bound port on the path."
                    )
                    _add_violation(
                        "unconsumed_literal",
                        msg,
                        details={"literal": clean_lit, "kind": kind}
                    )

        # -----------------------------------------------------------------
        # Check 1b: Active Column Set Tracking (Column Removal & Schema Tracking, R5-3)
        # -----------------------------------------------------------------
        var_schemas: Dict[str, VariableSchema] = {}
        var_origins: Dict[str, Set[str]] = {}

        for cell, bindings in pipeline_bindings:
            if not isinstance(bindings, dict):
                continue

            # 1. Identify input carrier variable
            carrier_in_var: Optional[str] = None
            prim_in = getattr(cell, "primary_input", None)
            if prim_in and prim_in.name in bindings:
                v = bindings[prim_in.name]
                if isinstance(v, str) and v.startswith("var_"):
                    carrier_in_var = v
            if not carrier_in_var:
                for k in ("df", "data", "self", "port_0"):
                    if k in bindings and isinstance(bindings[k], str) and bindings[k].startswith("var_"):
                        carrier_in_var = bindings[k]
                        break

            # 2. Check if any port binding references a column absent or removed from its target variable
            for p_name, val in bindings.items():
                if val is None or _is_unbound(val):
                    continue

                rec = getattr(cell, "typed_bindings", {}).get(p_name)
                cols_referenced: List[Tuple[str, Optional[str]]] = []  # (col_name, target_var)

                if rec and rec.kind == "projection" and rec.projection:
                    proj_k = rec.projection.get("kind")
                    proj_key = rec.projection.get("key")
                    target_v = rec.root_var or carrier_in_var
                    if proj_k == "column" and isinstance(proj_key, str):
                        cols_referenced.append((proj_key, target_v))
                    elif proj_k == "columns" and isinstance(proj_key, (list, tuple)):
                        for k in proj_key:
                            cols_referenced.append((str(k), target_v))
                else:
                    val_str = str(val)
                    # Check for literal column arguments
                    p_sig = cell.inputs.get(p_name) if getattr(cell, "inputs", None) else None
                    p_role_s = cls._port_role(p_sig) if p_sig else ""
                    if p_name in ("column", "columns", "key", "port_1") or any(
                        r in p_role_s for r in ("column", "target", "feature")
                    ):
                        target_v = carrier_in_var
                        if val_str.startswith("[") and val_str.endswith("]"):
                            try:
                                import ast
                                parsed = ast.literal_eval(val_str)
                                if isinstance(parsed, (list, tuple, set)):
                                    for item in parsed:
                                        cols_referenced.append((str(item).strip("'\""), target_v))
                            except Exception:
                                items = val_str[1:-1].split(",")
                                for item in items:
                                    it = item.strip().strip("'\"")
                                    if it:
                                        cols_referenced.append((it, target_v))
                        else:
                            clean_col = val_str.strip("'\"")
                            if clean_col and not clean_col.startswith("var_"):
                                cols_referenced.append((clean_col, target_v))

                    # Projection / subscript check: var['col'] or var[['col1', 'col2']]
                    for v_name, sch in var_schemas.items():
                        if val_str.startswith(f"{v_name}["):
                            inner = val_str[len(v_name):].strip()
                            if inner.startswith("[") and inner.endswith("]"):
                                inner_c = inner[1:-1].strip()
                                if inner_c.startswith("[") and inner_c.endswith("]"):
                                    inner_c = inner_c[1:-1].strip()
                                try:
                                    import ast
                                    parsed = ast.literal_eval(f"[{inner_c}]")
                                    if isinstance(parsed, (list, tuple, set)):
                                        for item in parsed:
                                            cols_referenced.append((str(item).strip("'\""), v_name))
                                except Exception:
                                    for item in inner_c.split(","):
                                        it = item.strip().strip("'\"")
                                        if it:
                                            cols_referenced.append((it, v_name))
                        elif v_name in val_str:
                            for rem_c in sch.removed_columns:
                                if f"'{rem_c}'" in val_str or f'"{rem_c}"' in val_str:
                                    cols_referenced.append((rem_c, v_name))

                for col, target_v in cols_referenced:
                    if not target_v or target_v not in var_schemas:
                        continue
                    schema = var_schemas[target_v]
                    # Provably removed column
                    if col in schema.removed_columns:
                        remover = schema.removed_by.get(col, "unknown")
                        msg = (
                            f"Cell '{cell.cell_id}' port '{p_name}' references column '{col}' "
                            f"which was provably removed from '{target_v}' by '{remover}'."
                        )
                        _add_violation(
                            "column_absent",
                            msg,
                            cell_id=cell.cell_id,
                            port_name=p_name,
                            variable_name=target_v,
                            details={"column": col, "variable": target_v, "removed_by": remover},
                        )
                    # Provably absent from known schema
                    elif schema.known_columns is not None:
                        if col not in schema.known_columns and col not in schema.added_columns:
                            msg = (
                                f"Cell '{cell.cell_id}' port '{p_name}' references column '{col}' "
                                f"which is absent from known columns of '{target_v}'."
                            )
                            _add_violation(
                                "column_absent",
                                msg,
                                cell_id=cell.cell_id,
                                port_name=p_name,
                                variable_name=target_v,
                                details={"column": col, "variable": target_v, "known_columns": list(schema.known_columns)},
                            )

            # 3. Determine output variable produced by this cell
            out_var: Optional[str] = bindings.get("output_var")
            if not out_var and getattr(cell, "primary_output", None):
                out_var = bindings.get(cell.primary_output.name)
            if not out_var and carrier_in_var and getattr(cell, "mutation_type", "") == "in_place":
                out_var = carrier_in_var

            # 4. Construct or update schema for out_var
            if out_var and isinstance(out_var, str) and out_var.startswith("var_"):
                base_sch = var_schemas.get(carrier_in_var) if carrier_in_var else None
                new_sch = VariableSchema(
                    known_columns=set(base_sch.known_columns) if base_sch and base_sch.known_columns is not None else None,
                    removed_columns=set(base_sch.removed_columns) if base_sch else set(),
                    added_columns=set(base_sch.added_columns) if base_sch else set(),
                    removed_by=dict(base_sch.removed_by) if base_sch else {},
                )

                # Track column additions (e.g. PD_SET_COLUMN)
                c_id_s = str(getattr(cell, "cell_id", "") or "").lower()
                if getattr(cell, "projection", None) == "set_column" or "set_column" in c_id_s:
                    col_k = bindings.get("column") or bindings.get("key") or bindings.get("port_1")
                    if col_k:
                        new_sch.added_columns.add(str(col_k).strip("'\""))

                # Track column removals from effects
                effects = getattr(cell, "effects", []) or []
                if isinstance(effects, (list, tuple, set)):
                    for eff in effects:
                        eff_name = eff.get("name") if isinstance(eff, dict) else str(eff)
                        if eff_name == "removes_columns":
                            from_port = eff.get("from_port") if isinstance(eff, dict) else None
                            cols_to_remove: List[str] = []
                            if from_port and from_port in bindings:
                                c_val = bindings[from_port]
                                if isinstance(c_val, (list, tuple, set)):
                                    cols_to_remove.extend(str(item).strip("'\"") for item in c_val)
                                elif isinstance(c_val, str):
                                    c_str = str(c_val).strip()
                                    if (c_str.startswith('"') and c_str.endswith('"')) or (c_str.startswith("'") and c_str.endswith("'")):
                                        c_str = c_str[1:-1].strip()
                                    if c_str.startswith("[") and c_str.endswith("]"):
                                        try:
                                            import ast
                                            parsed = ast.literal_eval(c_str)
                                            if isinstance(parsed, (list, tuple, set)):
                                                cols_to_remove.extend(str(item).strip("'\"") for item in parsed)
                                        except Exception:
                                            items = c_str[1:-1].split(",")
                                            for item in items:
                                                it = item.strip().strip("'\"")
                                                if it:
                                                    cols_to_remove.append(it)
                                    elif c_str and not _is_unbound(c_str):
                                        cols_to_remove.append(c_str.strip("'\""))
                            else:
                                for col_key in ("columns", "column", "key", "port_1"):
                                    if col_key in bindings and bindings[col_key] is not None:
                                        c_val = bindings[col_key]
                                        if isinstance(c_val, (list, tuple, set)):
                                            cols_to_remove.extend(str(x).strip("'\"") for x in c_val)
                                        elif isinstance(c_val, str) and not _is_unbound(c_val):
                                            c_str = str(c_val).strip()
                                            if (c_str.startswith('"') and c_str.endswith('"')) or (c_str.startswith("'") and c_str.endswith("'")):
                                                c_str = c_str[1:-1].strip()
                                            if c_str.startswith("[") and c_str.endswith("]"):
                                                try:
                                                    import ast
                                                    parsed = ast.literal_eval(c_str)
                                                    if isinstance(parsed, (list, tuple, set)):
                                                        cols_to_remove.extend(str(x).strip("'\"") for x in parsed)
                                                except Exception:
                                                    for item in c_str[1:-1].split(","):
                                                        it = item.strip().strip("'\"")
                                                        if it:
                                                            cols_to_remove.append(it)
                                            else:
                                                cols_to_remove.append(c_str.strip("'\""))
                                c_lits = getattr(cell, "clause_literals", None)
                                if c_lits:
                                    cols_to_remove.extend(str(x).strip("'\"") for x in c_lits)

                            for rem_c in cols_to_remove:
                                if rem_c:
                                    new_sch.removed_columns.add(rem_c)
                                    new_sch.removed_by[rem_c] = getattr(cell, "cell_id", "unknown")
                                    if new_sch.known_columns is not None:
                                        new_sch.known_columns.discard(rem_c)

                var_schemas[out_var] = new_sch
                out_origs: Set[str] = set()
                for inp_name, inp_val in bindings.items():
                    if not isinstance(inp_val, str):
                        continue
                    if inp_val.startswith("var_"):
                        r_v = inp_val.split("[")[0].strip()
                        out_origs.update(var_origins.get(r_v, set()))
                        if "[" in inp_val and "]" in inp_val:
                            try:
                                rec = resolve_typed_binding_record(cell=cell, port_name=inp_name, bound_val=inp_val)
                                if rec and rec.kind == "projection" and rec.projection:
                                    k = rec.projection.get("key")
                                    if isinstance(k, str):
                                        out_origs.add(k)
                                    elif isinstance(k, (list, tuple, set)):
                                        out_origs.update(str(x) for x in k)
                            except Exception:
                                pass
                    elif cell:
                        p_sig = getattr(cell, "inputs", {}).get(inp_name)
                        if p_sig and getattr(p_sig, "binds", None) == "column_key":
                            out_origs.add(inp_val.strip("'\""))
                var_origins[out_var] = out_origs

        # -----------------------------------------------------------------
        # Check 2: No bare None in required data-bearing positions
        # -----------------------------------------------------------------
        for cell, bindings in pipeline_bindings:
            for p_name, port_sig in cell.inputs.items():
                p_role = cls._port_role(port_sig) or "standard"
                if p_role not in DATA_BEARING_ROLES:
                    continue
                if cls._port_is_optional(port_sig):
                    # Optional data-bearing ports may legitimately remain unbound
                    # (e.g. an unsupervised estimator's optional target slot).
                    continue
                bound_val = bindings.get(p_name)
                if _is_unbound(bound_val):
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has data-bearing role '{p_role}' "
                        f"but received bare None or unresolved value."
                    )
                    _add_violation("unbound_data_port", msg, cell_id=cell.cell_id, port_name=p_name, details={"port_role": p_role})

        # -----------------------------------------------------------------
        # Check 2b: Cross-Port Static Type and Carrier Compatibility
        # -----------------------------------------------------------------
        var_signatures: Dict[str, Any] = {}
        var_producers: Dict[str, Tuple[Any, str]] = {}

        for cell, bindings in pipeline_bindings:
            if not isinstance(bindings, dict):
                continue

            cell_outputs = getattr(cell, "outputs", {})
            cell_output_vars = set()
            for out_name, out_sig in cell_outputs.items():
                if out_name in bindings and isinstance(bindings[out_name], str):
                    cell_output_vars.add(bindings[out_name])
            out_v = bindings.get("output_var")
            if isinstance(out_v, str):
                cell_output_vars.add(out_v)

            for p_name, in_port_sig in getattr(cell, "inputs", {}).items():
                # Skip self-declared outputs on the cell
                if p_name in cell_outputs:
                    continue

                bound_val = bindings.get(p_name)
                if bound_val is None or _is_unbound(bound_val) or bound_val is UNRESOLVED_PORT:
                    continue

                if isinstance(bound_val, str) and bound_val in cell_output_vars:
                    continue

                record = getattr(cell, "typed_bindings", {}).get(p_name)
                if record is None:
                    record = resolve_typed_binding_record(
                        cell=cell,
                        port_name=p_name,
                        bound_val=bound_val,
                        port_sig=in_port_sig,
                        var_signatures=var_signatures,
                    )

                if record.kind == "literal" or record.root_var is None:
                    continue

                root_var = record.root_var
                resolved_sig = record.resulting_signature

                if root_var in cell_output_vars:
                    continue

                if resolved_sig is None:
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' references undefined pipeline variable '{root_var}' "
                        f"with no producing cell in DAG history."
                    )
                    _add_violation("undefined_variable", msg, cell_id=cell.cell_id, port_name=p_name, variable_name=root_var)
                    continue

                prod_cell, prod_port = var_producers.get(root_var, (None, None))
                prod_id = prod_cell.cell_id if prod_cell else "unknown"

                # 1. Structural Carrier & Shape Compatibility
                if not _shape_compatible(resolved_sig, in_port_sig):
                    res_quals = getattr(resolved_sig, "qualifiers", frozenset())
                    in_rej = getattr(in_port_sig, "rejected_qualifiers", frozenset())
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has carrier/shape mismatch with "
                        f"bound variable '{bound_val}' (produced by '{prod_id}' port '{prod_port}')."
                    )
                    _add_violation(
                        "shape_carrier_mismatch",
                        msg,
                        cell_id=cell.cell_id,
                        port_name=p_name,
                        variable_name=root_var,
                        details={
                            "bound_val": bound_val,
                            "producer_cell_id": prod_id,
                            "producer_port": prod_port,
                            "resolved_qualifiers": list(res_quals) if isinstance(res_quals, (set, frozenset, list)) else [],
                            "rejected_qualifiers": list(in_rej) if isinstance(in_rej, (set, frozenset, list)) else [],
                        },
                    )
                    continue

                # 2. Algebraic Type Unification
                p_term = getattr(resolved_sig, "signature", resolved_sig)
                c_term = getattr(in_port_sig, "signature", in_port_sig)
                sub = unify(p_term, c_term)
                if sub is None:
                    prod_t = getattr(p_term, "type_name", str(p_term))
                    cons_t = getattr(c_term, "type_name", str(c_term))
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' (expected type '{cons_t}') failed "
                        f"type unification with bound variable '{bound_val}' (producer type '{prod_t}' from '{prod_id}')."
                    )
                    _add_violation(
                        "type_unification_failure",
                        msg,
                        cell_id=cell.cell_id,
                        port_name=p_name,
                        variable_name=root_var,
                        details={
                            "bound_val": bound_val,
                            "producer_cell_id": prod_id,
                            "producer_port": prod_port,
                            "producer_type": prod_t,
                            "consumer_type": cons_t,
                        },
                    )

            if _is_sink_cell(cell):
                continue

            # Register cell outputs for subsequent consumers
            prim_out = getattr(cell, "primary_output", None)
            if prim_out is not None:
                p_name = getattr(prim_out, "name", None)
                if p_name and p_name in bindings:
                    v = bindings[p_name]
                    if isinstance(v, str) and v.isidentifier():
                        var_signatures[v] = prim_out
                        var_producers[v] = (cell, p_name)

            for out_name, out_sig in cell_outputs.items():
                if out_name in bindings:
                    v = bindings[out_name]
                    if isinstance(v, str) and v.isidentifier():
                        var_signatures[v] = out_sig
                        var_producers[v] = (cell, out_name)

            out_var = bindings.get("output_var")
            if isinstance(out_var, str) and out_var.isidentifier() and out_var not in var_signatures:
                first_sig = prim_out or (next(iter(cell_outputs.values())) if cell_outputs else None)
                if first_sig is not None:
                    var_signatures[out_var] = first_sig
                    var_producers[out_var] = (cell, "output_var")

        # -----------------------------------------------------------------
        # Check 3: Dataflow Output Liveness (R5-2)
        # Every non-goal, non-sink cell MUST have its output consumed downstream.
        # -----------------------------------------------------------------
        n_steps = len(pipeline_bindings)
        for i, (cell, bindings) in enumerate(pipeline_bindings):
            if not isinstance(bindings, dict):
                continue
            if _is_sink_cell(cell) or not getattr(cell, "outputs", None):
                continue
            # The final step is the pipeline goal / return value
            if i == n_steps - 1:
                continue

            # Output variables produced by this step
            produced_vars: Set[str] = set()
            for out_name in getattr(cell, "outputs", {}).keys():
                v = bindings.get(out_name)
                if isinstance(v, str) and v.startswith("var_"):
                    produced_vars.add(v)
            out_v = bindings.get("output_var")
            if isinstance(out_v, str) and out_v.startswith("var_"):
                produced_vars.add(out_v)

            if not produced_vars:
                continue

            # Check if any downstream cell reads at least one produced variable
            is_consumed = False
            for downstream_cell, down_bindings in pipeline_bindings[i + 1 :]:
                if not isinstance(down_bindings, dict):
                    continue
                for p_k, down_val in down_bindings.items():
                    if down_val is None:
                        continue
                    # Check typed bindings record first
                    d_rec = getattr(downstream_cell, "typed_bindings", {}).get(p_k)
                    if d_rec and d_rec.root_var and d_rec.root_var in produced_vars:
                        is_consumed = True
                        break
                    # String check
                    down_str = str(down_val)
                    for pv in produced_vars:
                        if pv == down_str or f"{pv}[" in down_str or f"{pv}." in down_str or f"({pv}" in down_str:
                            is_consumed = True
                            break
                    if is_consumed:
                        break
                if is_consumed:
                    break

            if not is_consumed:
                dead_v = next(iter(produced_vars))
                msg = (
                    f"Cell '{cell.cell_id}' produced output variable '{dead_v}' which is never "
                    f"consumed by any downstream cell (dead output)."
                )
                _add_violation(
                    "dead_output",
                    msg,
                    cell_id=cell.cell_id,
                    variable_name=dead_v,
                    details={"cell_id": cell.cell_id, "output_var": dead_v, "produced_vars": list(produced_vars)},
                )

        # -----------------------------------------------------------------
        # Check 4: Monotonic Clause Ordering & Dependency Inversion (R5-2)
        # -----------------------------------------------------------------
        if prompt:
            try:
                from .route_methods.base import RouteMethod
                RouteMethod.tag_cells_with_clause_indices([c for c, _ in pipeline_bindings], prompt)
            except Exception:
                try:
                    from route_methods.base import RouteMethod
                    RouteMethod.tag_cells_with_clause_indices([c for c, _ in pipeline_bindings], prompt)
                except Exception:
                    pass

        # Track which variables each cell produces and consumes
        cell_producers: Dict[str, Tuple[int, Any]] = {}  # var_name -> (step_idx, cell)
        for i, (cell, bindings) in enumerate(pipeline_bindings):
            if not isinstance(bindings, dict):
                continue
            for out_name in getattr(cell, "outputs", {}).keys():
                v = bindings.get(out_name)
                if isinstance(v, str) and v.startswith("var_"):
                    cell_producers[v] = (i, cell)
            out_v = bindings.get("output_var")
            if isinstance(out_v, str) and out_v.startswith("var_"):
                cell_producers[out_v] = (i, cell)

        # Check for order inversions across dependent cell pairs
        for j, (cons_cell, cons_bindings) in enumerate(pipeline_bindings):
            if not isinstance(cons_bindings, dict):
                continue
            cons_clause = getattr(cons_cell, "matched_clause_idx", None)
            if cons_clause is None:
                continue

            for p_k, down_val in cons_bindings.items():
                if down_val is None:
                    continue
                d_rec = getattr(cons_cell, "typed_bindings", {}).get(p_k)
                root_v = d_rec.root_var if d_rec else None
                if not root_v and isinstance(down_val, str) and down_val.startswith("var_"):
                    root_v = down_val

                if root_v and root_v in cell_producers:
                    prod_idx, prod_cell = cell_producers[root_v]
                    prod_clause = getattr(prod_cell, "matched_clause_idx", None)
                    if prod_clause is not None and prod_clause > cons_clause:
                        msg = (
                            f"Order inversion: Cell '{prod_cell.cell_id}' (serving clause {prod_clause}) "
                            f"produces variable '{root_v}' consumed by earlier-clause Cell '{cons_cell.cell_id}' "
                            f"(serving clause {cons_clause})."
                        )
                        _add_violation(
                            "order_inversion",
                            msg,
                            cell_id=prod_cell.cell_id,
                            variable_name=root_v,
                            details={
                                "producer": prod_cell.cell_id,
                                "producer_clause": prod_clause,
                                "consumer": cons_cell.cell_id,
                                "consumer_clause": cons_clause,
                                "variable": root_v,
                            },
                        )

            # Also check if cons_cell is an earlier-clause data cleaner/transformer (e.g. dropna, normalize)
            # placed AFTER a later-clause estimator or consumer that operated on the uncleaned root carrier
            cons_in_v = None
            for k in ("df", "data", "self", "port_0"):
                if k in cons_bindings and isinstance(cons_bindings[k], str) and cons_bindings[k].startswith("var_"):
                    cons_in_v = cons_bindings[k]
                    break
            if cons_in_v:
                for prev_i in range(j):
                    prev_cell, prev_b = pipeline_bindings[prev_i]
                    prev_clause = getattr(prev_cell, "matched_clause_idx", None)
                    if prev_clause is not None and prev_clause > cons_clause:
                        if any(v == cons_in_v or (isinstance(v, str) and cons_in_v in v) for v in prev_b.values()):
                            msg = (
                                f"Order inversion: Cell '{prev_cell.cell_id}' (serving clause {prev_clause}) "
                                f"executed before Cell '{cons_cell.cell_id}' (serving clause {cons_clause}) "
                                f"on shared carrier '{cons_in_v}'."
                            )
                            _add_violation(
                                "order_inversion",
                                msg,
                                cell_id=prev_cell.cell_id,
                                variable_name=cons_in_v,
                                details={
                                    "producer": prev_cell.cell_id,
                                    "producer_clause": prev_clause,
                                    "consumer": cons_cell.cell_id,
                                    "consumer_clause": cons_clause,
                                    "variable": cons_in_v,
                                },
                            )
                            break

        # -----------------------------------------------------------------
        # Check 4c: Clause Adjacency & Provenance Gating (R5-7)
        # A data port bound to the product of a cell covering a different clause
        # than its own without clause adjacency or a matching prompt literal
        # is flagged provenance_mismatch.
        # -----------------------------------------------------------------
        prompt_clauses = []
        try:
            from planner import _segment_prompt_clauses
            prompt_clauses = _segment_prompt_clauses(prompt or "")
        except Exception:
            pass

        for j, (cell, bindings) in enumerate(pipeline_bindings):
            cons_clause = getattr(cell, "matched_clause_idx", None)
            if cons_clause is None:
                continue

            cons_clause_text = prompt_clauses[cons_clause] if cons_clause < len(prompt_clauses) else ""
            cons_clause_toks = {t.lower() for t in cons_clause_text.replace(",", " ").replace('"', " ").replace("'", " ").split()} if cons_clause_text else set()

            for p_name, val_expr in bindings.items():
                if not isinstance(val_expr, str) or not val_expr.startswith("var_"):
                    continue

                p_sig = getattr(cell, "inputs", {}).get(p_name)
                if p_sig is None:
                    continue

                p_binds = getattr(p_sig, "binds", None)
                p_abstract = getattr(p_sig, "abstract_type", "")
                inner_sig = getattr(p_sig, "signature", p_sig)
                p_tn = str(getattr(inner_sig, "type_name", "")).lower()

                # Carrier tables are exempt as they carry throughout the entire pipeline
                if p_binds in ("data_carrier", "carrier") or p_abstract == "table" or cls._registry().is_subtype(p_tn, "table"):
                    continue

                # Literal parameters or column keys are exempt
                if p_binds == "column_key" or getattr(p_sig, "port_role", None) == "literal_parameter":
                    continue

                # Extract root variable if projection
                root_var = val_expr.split("[")[0].strip() if "[" in val_expr else val_expr.strip()
                prod_cell, _ = var_producers.get(root_var, (None, None))
                if prod_cell is None:
                    continue

                prod_clause = getattr(prod_cell, "matched_clause_idx", None)
                if prod_clause is None or prod_clause == cons_clause:
                    continue

                # 1. Matching literal in consumer clause
                var_origs = set(var_origins.get(root_var, set()))
                if "[" in val_expr and "]" in val_expr:
                    try:
                        rec = resolve_typed_binding_record(cell=cell, port_name=p_name, bound_val=val_expr)
                        if rec and rec.kind == "projection" and rec.projection:
                            k = rec.projection.get("key")
                            if isinstance(k, str):
                                var_origs.add(k)
                            elif isinstance(k, (list, tuple, set)):
                                var_origs.update(str(x) for x in k)
                    except Exception:
                        pass
                has_literal_match = False
                if var_origs:
                    for orig in var_origs:
                        if str(orig).lower() in cons_clause_toks:
                            has_literal_match = True
                            break

                if has_literal_match:
                    continue

                # 2. Clause adjacency: immediately preceding clause (cons_clause - 1)
                is_adjacent = (prod_clause == cons_clause - 1)

                if not is_adjacent:
                    msg = (
                        f"Provenance mismatch: Cell '{cell.cell_id}' port '{p_name}' bound variable '{val_expr}' "
                        f"produced by Cell '{prod_cell.cell_id}' (serving clause {prod_clause}) into clause {cons_clause} "
                        f"without clause adjacency or matching prompt literal."
                    )
                    _add_violation(
                        "provenance_mismatch",
                        msg,
                        cell_id=cell.cell_id,
                        port_name=p_name,
                        variable_name=val_expr,
                        details={
                            "consumer": cell.cell_id,
                            "consumer_clause": cons_clause,
                            "producer": prod_cell.cell_id,
                            "producer_clause": prod_clause,
                            "port": p_name,
                            "variable": val_expr,
                        },
                    )

        # -----------------------------------------------------------------
        # Check 5: Structural Estimator Fit/Predict Contract
        #
        # Fire ONLY when:
        #   (a) the prompt carries >= 2 data identifiers (relation declared), AND
        #   (b) the prompt is classified as SUPERVISED by the registry verbs.
        # Unsupervised prompts (clustering / DR / anomaly detection) never trip
        # this check because their target ports are not required.
        # -----------------------------------------------------------------
        data_ids = list(dict.fromkeys(
            val for _, kind, val in extracted_literals
            if kind in ("identifier", "quoted_str")
        ))
        has_multi_data_relation = len(data_ids) >= 2

        if has_multi_data_relation and cls._prompt_is_supervised(prompt):
            feature_roles = cls._feature_roles()
            target_roles = cls._target_roles()

            for cell, bindings in pipeline_bindings:
                required_features: List[str] = []
                required_targets: List[str] = []

                for p_name, port_sig in cell.inputs.items():
                    role = cls._port_role(port_sig)
                    if not role:
                        continue
                    if cls._port_is_optional(port_sig):
                        # Optional ports are outside the estimator contract.
                        continue
                    if role in feature_roles:
                        required_features.append(p_name)
                    elif role in target_roles:
                        required_targets.append(p_name)

                # Only enforce when the cell *structurally* exposes both sides of
                # the estimator contract (feature + target) as required ports.
                if not (required_features and required_targets):
                    continue

                for p_name in required_features:
                    if _is_unbound(bindings.get(p_name)):
                        msg = (
                            f"Estimator-shaped cell '{cell.cell_id}' requires feature port "
                            f"'{p_name}', which was left unbound or None."
                        )
                        _add_violation("estimator_contract", msg, cell_id=cell.cell_id, port_name=p_name, details={"role": "feature"})
                for p_name in required_targets:
                    if _is_unbound(bindings.get(p_name)):
                        msg = (
                            f"Estimator-shaped cell '{cell.cell_id}' requires target port "
                            f"'{p_name}' for the supervised relation implied by data "
                            f"identifiers {data_ids}, but it was left unbound or None."
                        )
                        _add_violation("estimator_contract", msg, cell_id=cell.cell_id, port_name=p_name, details={"role": "target", "data_ids": data_ids})

        # -----------------------------------------------------------------
        # Check 3b: Explicit Unresolved Port Detection
        # -----------------------------------------------------------------
        for cell, bindings in pipeline_bindings:
            for p_name, val in bindings.items():
                if val is UNRESOLVED_PORT:
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has unresolved binding "
                        f"value: '{val}'"
                    )
                    _add_violation("unresolved_port", msg, cell_id=cell.cell_id, port_name=p_name, details={"value": str(val)})
                elif isinstance(val, str) and val in (
                    str(UNRESOLVED_PORT), "<UNRESOLVED>", "<unbound>"
                ):
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has unresolved binding "
                        f"value: '{val}'"
                    )
                    _add_violation("unresolved_port", msg, cell_id=cell.cell_id, port_name=p_name, details={"value": str(val)})

        # -----------------------------------------------------------------
        # Check 3c: Object-Receiver Ports Must Bind a Variable, Not a Literal
        #
        # A `model_input`/`model_sink` port carries a fitted estimator instance, never
        # array-like or scalar data. The emitter's own convention is that a bound
        # reference to a prior cell's output is a bare Python identifier ("var_7");
        # anything else is a literal expression the resolver fell back to (e.g. a
        # tree that misdeclares the port's carrier). This is a syntactic check on the
        # emitted binding, not a check against any particular type name.
        # -----------------------------------------------------------------
        for cell, bindings in pipeline_bindings:
            for p_name, port_sig in cell.inputs.items():
                p_role = cls._port_role(port_sig)
                if not cls._is_object_receiver_role(p_role):
                    continue
                bound_val = bindings.get(p_name)
                if _is_unbound(bound_val) or bound_val is UNRESOLVED_PORT:
                    continue  # already reported by Check 2 / Check 3b
                val_str = str(bound_val).strip()
                if not val_str.isidentifier():
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has data-bearing role "
                        f"'{p_role}' (a fitted model/estimator instance) but was bound to "
                        f"literal expression '{val_str}' instead of a variable reference."
                    )
                    _add_violation("object_receiver_literal", msg, cell_id=cell.cell_id, port_name=p_name, details={"port_role": p_role, "bound_val": val_str})

        # -----------------------------------------------------------------
        # Check 4: AST Syntax Validation (if code_str provided)
        # -----------------------------------------------------------------
        if code_str:
            try:
                tree = ast.parse(code_str)
                pre_cnt = len(violations)
                _check_callable_callees(tree, violations)
                for v in violations[pre_cnt:]:
                    structured_violations.append(
                        PreflightViolation(
                            check_id="callable_callee_constant",
                            message=v,
                            details={"code": code_str},
                        )
                    )
            except SyntaxError as e:
                msg = f"Synthesized code has syntax error: {e}"
                _add_violation("syntax_error", msg, details={"error": str(e)})

        is_valid = len(violations) == 0
        return PreflightLintResult(is_valid=is_valid, violations=violations, warnings=warnings, structured_violations=structured_violations)

    @staticmethod
    def audit_lattice(orchestrator: Any) -> Optional[Any]:
        """Runs static reachability, dead-end, and cross-tree auditing on the loaded lattice."""
        try:
            from lattice_auditor import LatticeAuditor
            return LatticeAuditor(orchestrator).audit()
        except Exception:
            return None
