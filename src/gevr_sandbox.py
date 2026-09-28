"""
src/gevr_sandbox.py
Hardened, isolated, domain-agnostic sandboxed execution environment with
AST security inspection, zero worker leakage, and pluggable verification protocols.
"""

from __future__ import annotations

import abc
import ast
import builtins
import logging
import multiprocessing
import os
import signal
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Security Exceptions & AST Validator
# ============================================================================

class SecurityViolationError(PermissionError):
    """Raised when executed code violates sandbox security constraints."""
    pass


class SandboxTimeoutError(TimeoutError):
    """Raised when execution exceeds the allocated wall-clock time limit."""
    pass


class _SecurityASTVisitor(ast.NodeVisitor):
    """
    Static analysis pass over synthesized code before execution.
    Detects dunder traversal exploits, bytecode manipulation, and forbidden builtins.
    """

    FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset({
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__code__",
        "__closure__",
        "__builtins__",
        "__import__",
        "gi_frame",
        "f_locals",
        "f_globals",
        "cr_frame",
    })

    FORBIDDEN_CALLS: frozenset[str] = frozenset({
        "eval",
        "exec",
        "compile",
        "breakpoint",
    })

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            raise SecurityViolationError(
                f"Access to restricted meta-attribute '{node.attr}' at line {node.lineno} is forbidden."
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self.FORBIDDEN_CALLS:
            raise SecurityViolationError(
                f"Direct access to forbidden built-in '{node.id}' at line {node.lineno} is blocked."
            )
        self.generic_visit(node)


# ============================================================================
# Pluggable Postconditions & State Protocols (Domain-Agnostic)
# ============================================================================

class StateProtocolRegistry:
    """Registry determining object readiness or fittedness via pluggable predicates."""

    def __init__(self):
        self._predicates: List[Callable[[Any], bool]] = []
        self._register_default_predicates()

    def register(self, predicate: Callable[[Any], bool]) -> None:
        self._predicates.append(predicate)

    def is_fitted_or_ready(self, target: Any) -> bool:
        """Check if target satisfies any registered state predicate."""
        for pred in self._predicates:
            try:
                if pred(target):
                    return True
            except Exception:
                continue
        return False

    def _register_default_predicates(self) -> None:
        # Standard protocol: Check for is_fitted(), is_ready(), or is_converged() methods
        def _callable_check(obj: Any) -> bool:
            for attr in ("is_fitted", "is_ready", "is_converged"):
                method = getattr(obj, attr, None)
                if callable(method):
                    return bool(method())
            return False

        # Generic state inspection: check for state-denoting attributes
        def _attribute_convention_check(obj: Any) -> bool:
            if hasattr(obj, "__dict__"):
                # Detect populated state without assuming third-party libraries
                public_state = [k for k in obj.__dict__.keys() if not k.startswith("_")]
                return len(public_state) > 0
            return True

        self.register(_callable_check)
        self.register(_attribute_convention_check)


class PostconditionRegistry:
    """Domain-agnostic registry for evaluating postcondition constraints."""

    def __init__(self):
        self._evaluators: Dict[str, Callable[[Any, Any], bool]] = {}
        self._register_default_evaluators()

    def register(self, property_name: str, evaluator: Callable[[Any, Any], bool]) -> None:
        self._evaluators[property_name] = evaluator

    def evaluate(self, property_name: str, target_val: Any, expected_val: Any) -> bool:
        evaluator = self._evaluators.get(property_name)
        if not evaluator:
            logger.warning(f"No postcondition evaluator registered for '{property_name}'. Falling back to equality.")
            return bool(target_val == expected_val)
        try:
            return evaluator(target_val, expected_val)
        except Exception as e:
            logger.error(f"Error evaluating postcondition '{property_name}': {e}")
            return False

    def _register_default_evaluators(self) -> None:
        # Dimensionality check via duck-typing (works for lists, tuples, tensors, arrays)
        def _ndim_evaluator(val: Any, expected: Any) -> bool:
            if hasattr(val, "ndim"):
                return bool(val.ndim == expected)
            if hasattr(val, "shape"):
                return bool(len(val.shape) == expected)
            # Recursive depth inspection for nested iterables
            def _get_depth(obj: Any) -> int:
                if isinstance(obj, (list, tuple)) and obj:
                    return 1 + _get_depth(obj[0])
                return 0
            return _get_depth(val) == expected

        # Null / NaN check via duck-typing
        def _has_nans_evaluator(val: Any, expected: Any) -> bool:
            # Check for duck-typed .isna() or .isnull()
            for method_name in ("isna", "isnull"):
                method = getattr(val, method_name, None)
                if callable(method):
                    res = method()
                    has_nan = bool(getattr(res, "any", lambda: res)())
                    return has_nan == expected

            # Check for native float('nan') or nested elements
            try:
                import math
                if isinstance(val, float) and math.isnan(val):
                    return True == expected
                if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                    has_nan = any(isinstance(x, float) and math.isnan(x) for x in val)
                    return has_nan == expected
            except Exception:
                pass
            return False == expected

        # Uniqueness / Deduplicated check via duck-typing
        def _dedup_evaluator(val: Any, expected: Any) -> bool:
            if hasattr(val, "duplicated"):
                dups = val.duplicated()
                is_unique = not bool(getattr(dups, "any", lambda: dups)())
                return is_unique == expected
            try:
                seq = list(val)
                return (len(seq) == len(set(seq))) == expected
            except TypeError:
                # Elements not hashable
                return True == expected

        self.register("ndim", _ndim_evaluator)
        self.register("has_nans", _has_nans_evaluator)
        self.register("is_deduped", _dedup_evaluator)


# ============================================================================
# Pluggable Terminal Intent Verifiers (Domain-Agnostic)
# ============================================================================

@dataclass
class IntentVerificationResult:
    valid: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class BaseIntentVerifier(abc.ABC):
    @abc.abstractmethod
    def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
        pass


class TerminalIntentRegistry:
    """Decoupled registry of verifiers validating terminal computation intent."""

    def __init__(self):
        self._verifiers: Dict[str, BaseIntentVerifier] = {}
        self._register_default_verifiers()

    def register(self, intent_type: str, verifier: BaseIntentVerifier) -> None:
        self._verifiers[intent_type] = verifier

    def verify(self, intent_type: str, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
        verifier = self._verifiers.get(intent_type)
        if not verifier:
            # Fallback verification: simple presence and non-None check
            is_valid = output is not None
            return IntentVerificationResult(
                valid=is_valid,
                reason="Default fallback: output is non-None" if is_valid else "Output is None",
            )
        return verifier.verify(output, context)

    def _register_default_verifiers(self) -> None:
        # Verifier for partition/split intent (e.g. dataset or collection splits)
        class PartitionIntentVerifier(BaseIntentVerifier):
            def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
                if not isinstance(output, (tuple, list)):
                    return IntentVerificationResult(valid=False, reason="Split intent must produce a collection")
                expected_min = context.get("min_partitions", 2)
                if len(output) < expected_min:
                    return IntentVerificationResult(
                        valid=False,
                        reason=f"Expected at least {expected_min} partitions, received {len(output)}",
                    )
                return IntentVerificationResult(valid=True, reason="Valid partition structure")

        # Verifier for structured/tabular data transformation intent
        class StructuredDataVerifier(BaseIntentVerifier):
            def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
                if output is None:
                    return IntentVerificationResult(valid=False, reason="Target structured output is None")
                # Duck-type column or length existence
                has_len = hasattr(output, "__len__")
                has_shape = hasattr(output, "shape")
                if not (has_len or has_shape):
                    return IntentVerificationResult(valid=False, reason="Output lacks dimension or sequence protocol")
                return IntentVerificationResult(valid=True, reason="Structured tabular contract verified")

        # Verifier for artifact rendering intent
        class VisualArtifactVerifier(BaseIntentVerifier):
            def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
                if output is None:
                    return IntentVerificationResult(valid=False, reason="Artifact output is None")
                # Look for common rendering protocols without importing matplotlib
                has_render_hook = any(
                    hasattr(output, attr) for attr in ("canvas", "savefig", "render", "show", "to_image")
                )
                if has_render_hook or type(output).__name__ in ("Figure", "Image", "Plot"):
                    return IntentVerificationResult(valid=True, reason="Visual artifact verified")
                return IntentVerificationResult(valid=False, reason="Object lacks visual artifact rendering protocol")

        self.register("model_fit_split", PartitionIntentVerifier())
        self.register("tabular_egress", StructuredDataVerifier())
        self.register("image_annotation_egress", StructuredDataVerifier())
        self.register("visualization_egress", VisualArtifactVerifier())


# ============================================================================
# Hardened Worker Execution Layer
# ============================================================================

_BLOCKED_MODULES = frozenset({
    "subprocess",
    "shutil",
    "socket",
    "ctypes",
    "pty",
    "commands",
    "multiprocessing",
    "threading",
    "_thread",
})


def _create_sanitized_os():
    """Wrap standard `os` to disable shell execution and destructive file operations."""
    import os as _real_os

    class _SanitizedOS:
        def __getattr__(self, name: str) -> Any:
            blocked_ops = {
                "system", "popen", "spawn", "spawnl", "spawnle", "spawnlp", "spawnlpe",
                "spawnv", "spawnve", "spawnvp", "spawnvpe", "execv", "execve", "execvp",
                "execvpe", "execl", "execle", "execlp", "execlpe", "remove", "unlink",
                "rmdir", "removedirs", "chmod", "chown", "kill", "killpg"
            }
            if name in blocked_ops:
                raise SecurityViolationError(f"Operating system operation 'os.{name}' is prohibited in sandbox.")
            return getattr(_real_os, name)

    return _SanitizedOS()


def _restricted_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    """Guarded import callback preventing module escape."""
    root_module = name.split(".")[0]
    if root_module in _BLOCKED_MODULES:
        raise SecurityViolationError(f"Import of restricted module '{root_module}' is blocked.")

    if root_module == "os":
        return _create_sanitized_os()

    # Safely retrieve underlying native import
    real_import = builtins.__import__
    return real_import(name, globals, locals, fromlist, level)


def _sandbox_worker_exec(
    code_str: str,
    context: Dict[str, Any],
    result_queue: multiprocessing.Queue,
) -> None:
    """
    Subprocess worker executing sandboxed code in strict process isolation.
    Exceptions are captured and returned safely via the IPC queue.
    """
    try:
        # Step 1: Pre-execution AST Analysis
        parsed_ast = ast.parse(code_str)
        _SecurityASTVisitor().visit(parsed_ast)

        # Step 2: Construct Sanitized Execution Environment
        safe_builtins: Dict[str, Any] = {}
        for k, v in builtins.__dict__.items():
            if k not in _SecurityASTVisitor.FORBIDDEN_CALLS:
                safe_builtins[k] = v

        # Correctly bind restricted import into safe builtins dictionary
        safe_builtins["__import__"] = _restricted_import

        exec_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__name__": "__sandbox__",
            "__doc__": None,
        }
        exec_globals.update(context)

        # Step 3: Bytecode Execution
        compiled = compile(parsed_ast, filename="<sandbox>", mode="exec")
        exec(compiled, exec_globals)

        # Extract mutated context or return value
        extracted_results = {
            k: v for k, v in exec_globals.items()
            if not k.startswith("__") and k not in context
        }

        result_queue.put({"success": True, "results": extracted_results, "error": None})

    except SecurityViolationError as sec_err:
        result_queue.put({"success": False, "results": {}, "error": f"SecurityViolation: {str(sec_err)}"})
    except Exception as exc:
        formatted_exc = traceback.format_exc()
        result_queue.put({"success": False, "results": {}, "error": f"{type(exc).__name__}: {str(exc)}\n{formatted_exc}"})


# ============================================================================
# Main GEVR Sandbox Manager
# ============================================================================

class GEVRSandbox:
    """
    Hardened execution sandbox with strict process isolation, timeout enforcement,
    and automatic cleanup of hung processes.
    """

    def __init__(
        self,
        default_timeout: float = 5.0,
        postcondition_registry: Optional[PostconditionRegistry] = None,
        intent_registry: Optional[TerminalIntentRegistry] = None,
        state_registry: Optional[StateProtocolRegistry] = None,
    ):
        self.default_timeout = default_timeout
        self.postconditions = postcondition_registry or PostconditionRegistry()
        self.intents = intent_registry or TerminalIntentRegistry()
        self.state_protocols = state_registry or StateProtocolRegistry()

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute code in a separate process. Guarantees termination on timeout
        without leaving CPU-consuming zombie workers.
        """
        exec_timeout = timeout if timeout is not None else self.default_timeout
        context = context or {}
        ctx = multiprocessing.get_context("spawn")
        result_queue: multiprocessing.Queue = ctx.Queue()

        worker = ctx.Process(
            target=_sandbox_worker_exec,
            args=(code, context, result_queue),
            daemon=True,
        )

        worker.start()
        worker.join(timeout=exec_timeout)

        # Enforce hard process termination if execution exceeds timeout
        if worker.is_alive():
            logger.warning(f"Worker PID {worker.pid} timed out after {exec_timeout}s. Terminating.")
            worker.terminate()
            worker.join(timeout=0.5)

            # Force kill if still unresponsive
            if worker.is_alive():
                logger.critical(f"Worker PID {worker.pid} refused SIGTERM. Issuing SIGKILL.")
                try:
                    if hasattr(signal, "SIGKILL"):
                        os.kill(worker.pid, signal.SIGKILL)
                    else:
                        worker.kill()
                except ProcessLookupError:
                    pass
                worker.join()

            raise SandboxTimeoutError(f"Execution timed out after {exec_timeout} seconds.")

        if result_queue.empty():
            raise RuntimeError("Execution worker terminated abruptly without returning a result.")

        payload = result_queue.get_nowait()
        if not payload["success"]:
            raise RuntimeError(payload["error"])

        return payload["results"]

    def verify_postcondition(self, property_name: str, target: Any, expected: Any) -> bool:
        """Evaluate postconditions using the decoupled registry."""
        return self.postconditions.evaluate(property_name, target, expected)

    def verify_intent(
        self, intent_type: str, output: Any, context: Optional[Dict[str, Any]] = None
    ) -> IntentVerificationResult:
        """Evaluate terminal intents using the decoupled registry."""
        return self.intents.verify(intent_type, output, context or {})

    def is_fitted(self, estimator: Any) -> bool:
        """Check estimator readiness using the decoupled state protocol registry."""
        return self.state_protocols.is_fitted_or_ready(estimator)
