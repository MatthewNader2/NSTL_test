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
3. Structural Estimator Contract: For any cell whose signature shape matches an estimator
   contract (e.g. features + target), both feature and target ports are bound when the
   prompt declares a prediction/training relation (>= 2 data identifiers extracted at L0).
"""

from __future__ import annotations
import ast
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass, field

try:
    from .lattice import UNRESOLVED_PORT
except (ImportError, ValueError):
    from lattice import UNRESOLVED_PORT

DATA_BEARING_ROLES: Set[str] = frozenset({
    "feature_input",
    "target_input",
    "data_input",
    "model_input",
    "source_data",
    "model_sink",
})

CONVENTIONAL_ESTIMATOR_VERBS: Set[str] = frozenset({
    "fit",
    "predict",
    "transform",
    "score",
    "partial_fit",
    "fit_transform",
})


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


class PreflightLinter:
    """
    Domain-agnostic static validator inspecting pipeline bindings and synthesized ASTs
    strictly through structural contracts and port roles.
    """

    @classmethod
    def lint(
        cls,
        pipeline_bindings: List[Tuple[Any, Dict[str, Any]]],
        prompt: str = "",
        code_str: Optional[str] = None,
        waived_literals: Optional[Set[str]] = None
    ) -> PreflightLintResult:
        violations: List[str] = []
        warnings: List[str] = []
        waived = set(waived_literals or set())

        # Extract universal literals from prompt if ExecutionContext is available
        extracted_literals: List[Tuple[int, str, str]] = []
        try:
            from unification import ExecutionContext
            extracted_literals = ExecutionContext._extract_universal_literals(prompt or "")
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # Check 1: Universal Literal Consumption
        # ---------------------------------------------------------------------
        if extracted_literals:
            bound_values_str: Set[str] = set()
            seen_consumed_lits: Set[str] = set()
            for cell, bindings in pipeline_bindings:
                for p_name, val in bindings.items():
                    if val is not None:
                        s_val = str(val).strip("'\"")
                        bound_values_str.add(s_val)

            # Check each extracted literal
            for _, kind, lit_val in extracted_literals:
                clean_lit = lit_val.strip("'\"")
                if not clean_lit or clean_lit in waived:
                    continue

                # Identifiers and file assets MUST be consumed
                if kind in ("file_asset", "identifier", "quoted_str"):
                    if clean_lit in seen_consumed_lits:
                        continue
                    seen_consumed_lits.add(clean_lit)
                    # Check if directly bound or present in bound strings
                    is_consumed = any(
                        clean_lit == b or clean_lit in b
                        for b in bound_values_str
                    )
                    # Check in AST if code_str is provided
                    if not is_consumed and code_str:
                        is_consumed = clean_lit in code_str

                    if not is_consumed:
                        violations.append(
                            f"Universal literal '{clean_lit}' (kind: {kind}) extracted from prompt "
                            f"was not consumed by any bound port on the path."
                        )

        # ---------------------------------------------------------------------
        # Check 2: No bare None in data-bearing positions
        # ---------------------------------------------------------------------
        for cell, bindings in pipeline_bindings:
            for p_name, port_sig in cell.inputs.items():
                p_role = getattr(port_sig, "port_role", None)
                if not p_role:
                    p_role = getattr(port_sig, "derived_role", "standard")

                if p_role in DATA_BEARING_ROLES:
                    bound_val = bindings.get(p_name)
                    # Detect bare None or unresolved port
                    is_bare_none = (
                        bound_val is None
                        or bound_val is UNRESOLVED_PORT
                        or str(bound_val).strip() in ("None", "none", "null", str(UNRESOLVED_PORT), "<UNRESOLVED>", "<unbound>")
                    )
                    if is_bare_none:
                        if True:
                            violations.append(
                                f"Cell '{cell.cell_id}' port '{p_name}' has data-bearing role '{p_role}' "
                                f"but received bare None or unresolved value."
                            )

        # ---------------------------------------------------------------------
        # Check 3: Structural Estimator Fit/Predict Contract
        # ---------------------------------------------------------------------
        # Count unconsumed data identifiers from L0
        data_ids = [
            val for _, kind, val in extracted_literals
            if kind in ("identifier", "quoted_str")
        ]
        data_ids = list(dict.fromkeys(data_ids))
        has_multi_data_relation = len(data_ids) >= 2

        if has_multi_data_relation:
            for cell, bindings in pipeline_bindings:
                # Structural check for estimator signature: has both feature_input and target_input ports
                feature_port = None
                target_port = None
                for p_name, port_sig in cell.inputs.items():
                    role = getattr(port_sig, "port_role", None) or getattr(port_sig, "derived_role", "")
                    if role == "feature_input":
                        feature_port = p_name
                    elif role == "target_input":
                        target_port = p_name

                # If this cell exposes both feature and target contracts:
                if feature_port is not None and target_port is not None:
                    fb = bindings.get(feature_port)
                    tb = bindings.get(target_port)
                    if not fb or fb is UNRESOLVED_PORT or str(fb).strip() in ("None", "none", str(UNRESOLVED_PORT), "<UNRESOLVED>", "<unbound>"):
                        violations.append(
                            f"Estimator-shaped cell '{cell.cell_id}' requires feature port '{feature_port}', "
                            f"which was left unbound or None."
                        )
                    if not tb or tb is UNRESOLVED_PORT or str(tb).strip() in ("None", "none", str(UNRESOLVED_PORT), "<UNRESOLVED>", "<unbound>"):
                        violations.append(
                            f"Estimator-shaped cell '{cell.cell_id}' requires target port '{target_port}' "
                            f"for prompt with multiple data identifiers {data_ids}, but it was left unbound or None."
                        )

        # ---------------------------------------------------------------------
        # Check 3b: Explicit Unresolved Port Detection
        # ---------------------------------------------------------------------
        for cell, bindings in pipeline_bindings:
            for p_name, val in bindings.items():
                if val is UNRESOLVED_PORT or (isinstance(val, str) and val in (str(UNRESOLVED_PORT), "<UNRESOLVED>", "<unbound>")):
                    violations.append(
                        f"Cell '{cell.cell_id}' port '{p_name}' has unresolved binding value: '{val}'"
                    )

        # ---------------------------------------------------------------------
        # Check 4: AST Syntax Validation (if code_str provided)
        # ---------------------------------------------------------------------
        if code_str:
            try:
                ast.parse(code_str)
            except SyntaxError as e:
                violations.append(f"Synthesized code has syntax error: {e}")

        is_valid = len(violations) == 0
        return PreflightLintResult(is_valid=is_valid, violations=violations, warnings=warnings)
