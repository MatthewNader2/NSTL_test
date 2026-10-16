"""
src/security_policy.py
Non-overridable engine security floor and capability-based access control.
Enforces default-deny for capability-gated modules, with explicit opt-in
via --allow-capabilities. Zero tolerance for low-level OS bypasses.
"""

from __future__ import annotations

import ast
from typing import Dict, FrozenSet, Optional, Set, Tuple


class SecurityViolationError(PermissionError):
    """Raised when synthesized code violates sandbox security constraints."""
    pass


# 1. Minimal, Non-Overridable Security Floor
# No tree, capability, or user flag is permitted to override these restrictions.
NON_OVERRIDABLE_BLOCKED_MODULES: FrozenSet[str] = frozenset({
    "ctypes",
    "_ctypes",
    "pty",
    "subprocess",
    "posix",
    "code",
    "codeop",
})

NON_OVERRIDABLE_BLOCKED_ATTRS: FrozenSet[str] = frozenset({
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__builtins__",
    "gi_frame",
    "f_locals",
    "f_globals",
    "cr_frame",
})

NON_OVERRIDABLE_BLOCKED_CALLS: FrozenSet[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "breakpoint",
    "input",
})

# 2. Standard Default Capability Gating
# Mapping of standard modules to their capability categories.
DEFAULT_CAPABILITY_MODULES: Dict[str, str] = {
    "socket": "net",
    "requests": "net.http",
    "urllib": "net.http",
    "urllib3": "net.http",
    "http": "net.http",
    "ftplib": "net.ftp",
    "smtplib": "net.smtp",
    "telnetlib": "net.telnet",
    "asyncio": "async",
}


class SecurityPolicy:
    """
    Enforces the security floor and evaluates tree-declared capabilities.
    Default-deny: capability-gated modules require explicit authorization.
    """

    def __init__(
        self,
        allowed_capabilities: Optional[Set[str]] = None,
        tree_capabilities: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        self.allowed_capabilities: Set[str] = set(allowed_capabilities or set())
        # Mapping from module name to required capability set
        self.module_to_capabilities: Dict[str, Set[str]] = {}

        # Seed with default capabilities
        for mod, cap in DEFAULT_CAPABILITY_MODULES.items():
            self.module_to_capabilities.setdefault(mod, set()).add(cap)

        # Merge tree-declared capabilities
        if tree_capabilities:
            for cap_name, modules in tree_capabilities.items():
                for mod in modules:
                    self.module_to_capabilities.setdefault(mod, set()).add(cap_name)

    def is_module_allowed(self, module_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if importing module_name is permitted.
        Returns (is_allowed, reason_if_denied).
        """
        root_mod = module_name.split(".")[0]

        # 1. Check non-overridable security floor
        if root_mod in NON_OVERRIDABLE_BLOCKED_MODULES or module_name in NON_OVERRIDABLE_BLOCKED_MODULES:
            return False, f"Module '{module_name}' is blocked by the non-overridable engine security floor."

        # 2. Check capability gating
        req_caps = self.module_to_capabilities.get(root_mod) or self.module_to_capabilities.get(module_name)
        if req_caps:
            # Check if any required capability is allowed
            if not any(cap in self.allowed_capabilities for cap in req_caps):
                caps_str = ", ".join(sorted(req_caps))
                return False, f"Module '{module_name}' requires capability [{caps_str}] which was not granted via --allow-capabilities."

        return True, None

    def validate_ast(self, tree: ast.AST) -> None:
        """
        Perform static AST security inspection on synthesized code.
        Raises SecurityViolationError if any violation is detected.
        """
        visitor = _SecurityVisitor(self)
        visitor.visit(tree)


class _SecurityVisitor(ast.NodeVisitor):
    def __init__(self, policy: SecurityPolicy) -> None:
        self.policy = policy

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            allowed, reason = self.policy.is_module_allowed(alias.name)
            if not allowed:
                raise SecurityViolationError(f"Line {node.lineno}: {reason}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            allowed, reason = self.policy.is_module_allowed(node.module)
            if not allowed:
                raise SecurityViolationError(f"Line {node.lineno}: {reason}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in NON_OVERRIDABLE_BLOCKED_ATTRS:
            raise SecurityViolationError(
                f"Line {node.lineno}: Access to restricted meta-attribute '{node.attr}' is forbidden."
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in NON_OVERRIDABLE_BLOCKED_CALLS:
            raise SecurityViolationError(
                f"Line {node.lineno}: Direct access to forbidden built-in '{node.id}' is blocked."
            )
        self.generic_visit(node)
