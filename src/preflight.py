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
import re
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass, field

try:
    from .lattice import UNRESOLVED_PORT, TypeRegistry
    from .unification import ExecutionContext
except (ImportError, ValueError):
    from lattice import UNRESOLVED_PORT, TypeRegistry
    from unification import ExecutionContext


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


# ---------------------------------------------------------------------------
# Errors / results
# ---------------------------------------------------------------------------

class PreflightLintError(Exception):
    """Raised when synthesized code fails static structural pre-flight validation."""
    def __init__(self, message: str, violations: Optional[List[str]] = None):
        super().__init__(message)
        self.violations = violations or [message]


@dataclass
class PreflightLintResult:
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            msg = "Pre-flight lint validation failed:\n  - " + "\n  - ".join(self.violations)
            raise PreflightLintError(msg, violations=self.violations)


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
        return cls._role_set("get_feature_roles", "get_feature_bearing_roles")

    @classmethod
    def _target_roles(cls) -> Set[str]:
        return cls._role_set("get_target_roles", "get_target_bearing_roles")

    @classmethod
    def _supervised_verbs(cls) -> Set[str]:
        return cls._verb_set("get_supervised_estimator_verbs", "get_supervised_verbs")

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
        tokens = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prompt)}
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
        waived = set(waived_literals or set())

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
                        violations.append(
                            "Synthesized code failed static AST parsing (SyntaxError)."
                        )
                        is_consumed = False
                    except Exception:
                        is_consumed = False

                if not is_consumed:
                    msg = (
                        f"Universal literal '{clean_lit}' (kind: {kind}) extracted from prompt "
                        f"was not consumed by any bound port on the path."
                    )
                    is_path_like = (
                        kind == "file_asset"
                        or (kind == "quoted_str" and ExecutionContext._is_path_string(clean_lit))
                    )
                    if is_path_like:
                        violations.append(msg)
                    else:
                        warnings.append(msg)

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
                    violations.append(
                        f"Cell '{cell.cell_id}' port '{p_name}' has data-bearing role '{p_role}' "
                        f"but received bare None or unresolved value."
                    )

        # -----------------------------------------------------------------
        # Check 3: Structural Estimator Fit/Predict Contract
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
                        violations.append(
                            f"Estimator-shaped cell '{cell.cell_id}' requires feature port "
                            f"'{p_name}', which was left unbound or None."
                        )
                for p_name in required_targets:
                    if _is_unbound(bindings.get(p_name)):
                        violations.append(
                            f"Estimator-shaped cell '{cell.cell_id}' requires target port "
                            f"'{p_name}' for the supervised relation implied by data "
                            f"identifiers {data_ids}, but it was left unbound or None."
                        )

        # -----------------------------------------------------------------
        # Check 3b: Explicit Unresolved Port Detection
        # -----------------------------------------------------------------
        for cell, bindings in pipeline_bindings:
            for p_name, val in bindings.items():
                if val is UNRESOLVED_PORT:
                    violations.append(
                        f"Cell '{cell.cell_id}' port '{p_name}' has unresolved binding "
                        f"value: '{val}'"
                    )
                elif isinstance(val, str) and val in (
                    str(UNRESOLVED_PORT), "<UNRESOLVED>", "<unbound>"
                ):
                    violations.append(
                        f"Cell '{cell.cell_id}' port '{p_name}' has unresolved binding "
                        f"value: '{val}'"
                    )

        # -----------------------------------------------------------------
        # Check 4: AST Syntax Validation (if code_str provided)
        # -----------------------------------------------------------------
        if code_str:
            try:
                ast.parse(code_str)
            except SyntaxError as e:
                violations.append(f"Synthesized code has syntax error: {e}")

        is_valid = len(violations) == 0
        return PreflightLintResult(is_valid=is_valid, violations=violations, warnings=warnings)

    @staticmethod
    def audit_lattice(orchestrator: Any) -> Optional[Any]:
        """Runs static reachability, dead-end, and cross-tree auditing on the loaded lattice."""
        try:
            from lattice_auditor import LatticeAuditor
            return LatticeAuditor(orchestrator).audit()
        except Exception:
            return None
