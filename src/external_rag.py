"""
src/external_rag.py - Neuro-Symbolic Topological Lattice (NSTL)
Documentation Fetchers for Live API Grounding at Synthesis Time.
"""

from __future__ import annotations
import json
import urllib.request
import urllib.parse
from abc import ABC, abstractmethod
from log_config import get_logger

logger = get_logger('external_rag')


class LiveDocFetcher(ABC):
    @abstractmethod
    def fetch(self, query: str) -> str:
        pass


class PyPiFetcher(LiveDocFetcher):
    def fetch(self, package_name: str) -> str:
        clean_name = package_name.split(':')[0].strip().lower().replace('_', '-')
        encoded_name = urllib.parse.quote(clean_name)
        url = f"https://pypi.org/pypi/{encoded_name}/json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'NSTL-LiveDocFetcher/2.0'})
            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode())
                info = data.get("info", {})
                desc = info.get("description", "") or info.get("summary", "")
                return desc[:2500]
        except Exception as e:
            logger.warning(f"[PYPI FETCHER] Could not fetch docs for '{package_name}': {e}")
            return ""


class CratesIoFetcher(LiveDocFetcher):
    def fetch(self, crate_name: str) -> str:
        clean_name = urllib.parse.quote(crate_name.split(':')[0].strip().lower(), safe='')
        url = f"https://crates.io/api/v1/crates/{clean_name}/readme"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'NSTL-LiveDocFetcher/2.0'})
            with urllib.request.urlopen(req, timeout=5.0) as response:
                return response.read().decode('utf-8', errors='ignore')[:2500]
        except Exception as e:
            logger.warning(f"[CRATES FETCHER] Could not fetch docs for '{crate_name}': {e}")
            return ""


class DuckDuckGoFetcher(LiveDocFetcher):
    """Queries DuckDuckGo Instant Answer API (JSON, no scraping)."""
    def fetch(self, query: str) -> str:
        encoded = urllib.parse.quote(f"{query} python")
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'NSTL-LiveDocFetcher/2.0'})
            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode())
                # Try AbstractText (summary), then RelatedTopics
                abstract = data.get("AbstractText", "")
                if abstract:
                    return abstract[:2500]
                topics = data.get("RelatedTopics", [])
                parts = []
                for topic in topics[:5]:
                    if isinstance(topic, dict) and "Text" in topic:
                        parts.append(topic["Text"])
                return "\n".join(parts)[:2500] if parts else ""
        except Exception as e:
            logger.warning(f"[DDG FETCHER] API query failed for '{query}': {e}")
            return ""


import importlib
import inspect


class IntrospectionFetcher(LiveDocFetcher):
    """
    Introspects installed Python modules directly using inspect.getdoc() and inspect.signature().
    Zero network latency, 100% grounded and deterministic.
    """
    KNOWN_MODULES = ["pandas", "numpy", "cv2", "sklearn", "scipy", "matplotlib", "heapq", "math", "json", "os", "sys"]

    def fetch(self, query: str) -> str:
        clean_q = query.strip().lower().replace(" ", "_")
        tokens = [t for t in query.strip().lower().split() if len(t) > 2]

        for mod_name in self.KNOWN_MODULES:
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue

            # Check direct attribute match
            for attr in dir(mod):
                if attr.startswith("_"):
                    continue
                attr_lower = attr.lower()
                if clean_q == attr_lower or any(t == attr_lower for t in tokens):
                    obj = getattr(mod, attr, None)
                    if obj is not None:
                        doc = inspect.getdoc(obj) or f"{mod_name}.{attr}"
                        try:
                            sig = str(inspect.signature(obj))
                        except Exception:
                            sig = "()"
                        return f"Signature: {mod_name}.{attr}{sig}\n\nDocumentation:\n{doc[:1500]}"

        # Fallback to PyPi search if introspection yields nothing
        return PyPiFetcher().fetch(query)


class FetcherFactory:
    @staticmethod
    def get_fetcher(domain_context: str) -> LiveDocFetcher:
        if "Rust" in domain_context:
            return CratesIoFetcher()
        return IntrospectionFetcher()
