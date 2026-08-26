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
    def fetch(self, query: str) -> str:
        encoded = urllib.parse.quote(f"{query} python official documentation")
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=5.0) as response:
                html = response.read().decode('utf-8', errors='ignore')
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                snippet = soup.find('a', class_='result__snippet')
                return snippet.get_text(strip=True)[:2500] if snippet else ""
        except Exception as e:
            logger.warning(f"[DDG FETCHER] Search failed for '{query}': {e}")
            return ""


class FetcherFactory:
    @staticmethod
    def get_fetcher(domain_context: str) -> LiveDocFetcher:
        if "Rust" in domain_context:
            return CratesIoFetcher()
        elif "Python" in domain_context:
            return PyPiFetcher()
        return DuckDuckGoFetcher()
