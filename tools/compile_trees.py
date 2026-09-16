import importlib.util
from functools import lru_cache

_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


@lru_cache(maxsize=1024)
def _is_importable_root(name: str) -> bool:
    """Dynamically check whether the first dotted segment is importable."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@lru_cache(maxsize=1024)
def _is_resolvable_dotted_name(name: str) -> bool:
    """Dynamically resolve dotted names without using a hardcoded extension list."""
    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
        name,
    ):
        return False

    parts = name.split(".")
    if len(parts) == 1:
        return _is_importable_root(parts[0])

    # If the full dotted path is itself a module path, allow it.
    try:
        if importlib.util.find_spec(name) is not None:
            return True
    except Exception:
        pass

    root = parts[0]
    if not _is_importable_root(root):
        return False

    # If the root module is already loaded, verify the attribute chain dynamically.
    module = sys.modules.get(root)
    if module is not None:
        try:
            obj = module
            for part in parts[1:]:
                if not hasattr(obj, part):
                    return False
                obj = getattr(obj, part)
            return True
        except Exception:
            return False

    # Do not import arbitrary modules during compilation just to check attributes.
    return True


def _looks_like_dynamic_filename(value: object) -> bool:
    """Dynamic filename/path heuristic without a fixed extension whitelist."""
    if not isinstance(value, str):
        return False

    s = value.strip()
    if not s:
        return False

    # Templated strings are allowed.
    if "{" in s or "}" in s:
        return False

    if any(ch in s for ch in "\r\n\t\0"):
        return False

    # Do not treat URLs as local filenames.
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", s):
        return False

    # Path-like strings are suspicious when the basename contains a dot.
    if "/" in s or "\\" in s:
        basename = s.replace("\\", "/").rsplit("/", 1)[-1]
        return "." in basename and not basename.startswith(".")

    # Simple dotted strings: allow module-like/resolvable names, otherwise flag.
    if "." not in s or " " in s:
        return False

    if re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
        s,
    ):
        if _is_resolvable_dotted_name(s):
            return False

        suffix = s.rsplit(".", 1)[-1]
        return bool(suffix) and not suffix.isdigit()

    return False


def _validate_template(
    code: str,
    cell_id: str,
    node_role: str = "function",
) -> Tuple[bool, Optional[str]]:
    """Validate Python template syntax and dynamically reject hardcoded filename strings."""
    if not code or not code.strip():
        return False, "Empty code template"

    dummy_code = _PLACEHOLDER_RE.sub("dummy_var", code)

    try:
        tree = ast.parse(dummy_code)
    except SyntaxError as e:
        # Allow match/case templates when running on older Python versions.
        if re.search(r"^\s*match\s+", dummy_code, flags=re.MULTILINE):
            return True, None
        return False, f"AST SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _looks_like_dynamic_filename(node.value):
                return False, f"Hardcoded string filename '{node.value}'"

    return True, None
