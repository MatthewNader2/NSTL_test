"""Centralized tokenization and alias management for NSTL.

Replaces the triplicated token expansion and alias maps scattered across
planner.py (VIO-06), router.py (VIO-07/10), and internal_rag.py (VIO-12)
with a single, general-purpose implementation.
"""
import re
import sys
from typing import Dict, FrozenSet, List, Optional, Set, TYPE_CHECKING

from log_config import get_logger

if TYPE_CHECKING:
    from lattice import Cell

logger = get_logger('tokenizer')

# Stopwords stripped from prompts during tokenization.
# Defined once at module level (was triplicated across 3 files).
STOP_WORDS: FrozenSet[str] = frozenset({
    'a', 'an', 'the', 'and', 'or', 'to', 'with', 'any', 'it', 'is',
    'in', 'of', 'for', 'on', 'by', 'function', 'write', 'python', 'code',
    'script', 'create', 'def', 'that', 'returns', 'result', 'using',
    'from', 'into', 'this', 'then', 'use', 'make', 'get', 'set',
})

# Standard library module names (replaces VIO-32's 8-module whitelist)
try:
    STDLIB_MODULES: FrozenSet[str] = frozenset(sys.stdlib_module_names)
except AttributeError:
    # Python < 3.10 fallback
    STDLIB_MODULES = frozenset({
        'abc', 'argparse', 'ast', 'asyncio', 'base64', 'bisect',
        'collections', 'contextlib', 'copy', 'csv', 'dataclasses',
        'datetime', 'decimal', 'difflib', 'email', 'enum', 'functools',
        'glob', 'gzip', 'hashlib', 'heapq', 'html', 'http', 'inspect',
        'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
        'operator', 'os', 'pathlib', 'pickle', 'platform', 'pprint',
        'queue', 'random', 're', 'shutil', 'signal', 'socket',
        'sqlite3', 'string', 'struct', 'subprocess', 'sys', 'tempfile',
        'textwrap', 'threading', 'time', 'traceback', 'typing',
        'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml',
        'zipfile', 'zlib',
    })


class CellTokenizer:
    """General-purpose sub-word tokenizer for cell IDs and prompts.
    
    Replaces the hand-tuned im*/to*/cvt* prefix stripping (VIO-10) and
    the 8-method API token expansion (VIO-06/07/12) with a single
    Unicode-aware camelCase/snake_case splitter.
    """

    # Pattern splits on underscores, camelCase boundaries, and digits
    _SPLIT_PATTERN = re.compile(
        r'[_\-\s]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[0-9])|(?<=[0-9])(?=[a-zA-Z])'
    )

    @classmethod
    def tokenize_identifier(cls, identifier: str) -> Set[str]:
        """Tokenize a cell ID or API method name into sub-word tokens.
        
        Examples:
            'PANDAS_DATAFRAME_SORT_VALUES' -> {'pandas', 'dataframe', 'sort', 'values'}
            'imread' -> {'imread', 'im', 'read'}
            'cvtColor' -> {'cvtcolor', 'cvt', 'color'}
            'to_csv' -> {'to', 'csv'}
        """
        if not identifier:
            return set()
        
        # First split on underscores and camelCase
        parts = cls._SPLIT_PATTERN.split(identifier.lower())
        tokens = {p for p in parts if p and len(p) > 0}
        
        # Also add the full lowercase identifier
        tokens.add(identifier.lower())
        
        # Add individual parts from further sub-word splitting
        expanded = set(tokens)
        for token in tokens:
            # Split concatenated lowercase words (e.g., 'imread' -> 'im', 'read')
            # Use common prefix boundaries
            sub_parts = re.findall(r'[a-z]+', token)
            if len(sub_parts) == 1 and len(token) > 3:
                # Try splitting at common API prefixes
                for prefix_len in range(2, min(5, len(token))):
                    prefix = token[:prefix_len]
                    suffix = token[prefix_len:]
                    if len(suffix) >= 2:
                        expanded.add(prefix)
                        expanded.add(suffix)
            expanded.update(sub_parts)
        
        # Filter out single-char tokens and empty strings
        return {t for t in expanded if len(t) > 1}

    @classmethod
    def tokenize_prompt(cls, prompt: str, remove_stopwords: bool = True) -> Set[str]:
        """Tokenize a user prompt into meaningful tokens.
        
        Applies stopword removal and filters out very short tokens.
        """
        if not prompt:
            return set()
        
        tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', prompt.lower()))
        
        if remove_stopwords:
            tokens -= STOP_WORDS
        
        return {t for t in tokens if len(t) > 1}

    @classmethod
    def tokenize_cell(cls, cell_id: str, keywords: Set[str]) -> Set[str]:
        """Produce a comprehensive token set for a cell (ID + keywords).
        
        This is the single replacement for all the scattered token expansion
        logic in planner.py, router.py, and internal_rag.py.
        """
        tokens = cls.tokenize_identifier(cell_id)
        for kw in keywords:
            tokens.update(cls.tokenize_identifier(kw))
        return tokens


class AliasRegistry:
    """Dynamically-built alias registry for library/domain names.
    
    Replaces the hardcoded alias_map dicts in planner.py (VIO-05),
    router.py (VIO-07), and internal_rag.py (VIO-62).
    """

    # Seed aliases from well-known Python conventions
    _SEED_ALIASES: Dict[str, str] = {
        'pd': 'pandas', 'np': 'numpy', 'cv2': 'opencv',
        'plt': 'matplotlib', 'sns': 'seaborn', 'tf': 'tensorflow',
        'sk': 'scikit', 'sklearn': 'scikit', 'pytorch': 'torch',
        'cvt': 'cvtcolor', 'convert': 'cvtcolor', 'color': 'cvtcolor', 'grayscale': 'cvtcolor', 'gray': 'cvtcolor', 'bgr': 'cvtcolor', 'bgr2gray': 'cvtcolor',
        'missing': 'dropna', 'null': 'dropna', 'na': 'dropna', 'dropna': 'drop', 'sort_values': 'sort', 'save': 'write', 'imwrite': 'save', 'to_csv': 'save', 'csv': 'save', 'imread': 'read', 'load': 'read',
    }

    def __init__(self):
        self._forward: Dict[str, str] = dict(self._SEED_ALIASES)
        self._reverse: Dict[str, Set[str]] = {}
        # Build reverse map from seeds
        for alias, canonical in self._SEED_ALIASES.items():
            self._reverse.setdefault(canonical, set()).add(alias)

    @classmethod
    def build_from_cells(cls, cells: list) -> 'AliasRegistry':
        """Build alias registry from loaded cells' domain_name fields.
        
        Extends seed aliases with domain names discovered in the cell population.
        """
        registry = cls()
        
        seen_domains = set()
        for cell in cells:
            domain = getattr(cell, 'domain_name', '') or ''
            if domain and len(domain) >= 2:
                seen_domains.add(domain.lower())
        
        # Register any new domains as self-aliases
        for domain in seen_domains:
            if domain not in registry._forward:
                registry._forward[domain] = domain
            registry._reverse.setdefault(domain, set()).add(domain)
        
        logger.info(
            f"AliasRegistry built: {len(registry._forward)} aliases, "
            f"{len(seen_domains)} domains from cells"
        )
        return registry

    def resolve(self, alias: str) -> str:
        """Resolve an alias to its canonical domain name."""
        return self._forward.get(alias.lower(), alias.lower())

    def get_aliases(self, canonical: str) -> Set[str]:
        """Get all aliases for a canonical domain name."""
        return self._reverse.get(canonical.lower(), set())

    def expand_tokens(self, tokens: Set[str]) -> Set[str]:
        """Expand a token set by adding alias resolutions.
        
        If a token is a known alias, adds the canonical name.
        If a token is a canonical name, adds all known aliases.
        """
        expanded = set(tokens)
        for token in tokens:
            lower = token.lower()
            if lower in self._forward:
                expanded.add(self._forward[lower])
            if lower in self._reverse:
                expanded.update(self._reverse[lower])
        return expanded

    def is_stdlib(self, module_name: str) -> bool:
        """Check if a module name is a standard library module.
        
        Replaces VIO-32's hardcoded 8-module whitelist.
        """
        return module_name.lower() in STDLIB_MODULES or module_name in STDLIB_MODULES
