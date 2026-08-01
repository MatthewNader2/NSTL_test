import json
from log_config import get_logger
import urllib.request
import urllib.parse
from abc import ABC, abstractmethod

logger = get_logger('external_rag')

class LiveDocFetcher(ABC):
    @abstractmethod
    def fetch(self, query: str) -> str:
        """Fetches raw live documentation for a concept/package."""
        pass

class PyPiFetcher(LiveDocFetcher):
    def fetch(self, package_name: str) -> str:
        """Fetches the official README/description from PyPI."""
        # Sanitize concept string (e.g., "PYTHON_ADD: convert..." -> "python-add")
        clean_name = package_name.split(':')[0].strip().lower().replace('_', '-')
        clean_name = urllib.parse.quote(clean_name)
        url = f"https://pypi.org/pypi/{clean_name}/json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'NSTL-LiveFetcher/1.0'})
            with urllib.request.urlopen(req, timeout=10.0) as response:
                data = json.loads(response.read().decode())
                info = data.get("info", {})
                desc = info.get("description", "")
                if len(desc) < 50:
                    desc = info.get("summary", "")
                
                # Truncate to save tokens for the LLM
                return desc[:4000]
        except Exception as e:
            logger.error(f"PyPiFetcher failed for {package_name}: {e}")
            return ""

class CratesIoFetcher(LiveDocFetcher):
    def fetch(self, crate_name: str) -> str:
        """Fetches the official README from Crates.io."""
        # Sanitize to prevent SSRF/path traversal via crafted crate names
        clean_name = crate_name.split(':')[0].strip().lower()
        clean_name = urllib.parse.quote(clean_name, safe='')
        url = f"https://crates.io/api/v1/crates/{clean_name}/readme"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'NSTL-LiveFetcher/1.0'})
            with urllib.request.urlopen(req, timeout=10.0) as response:
                # BUG 10 FIX: The crates.io readme endpoint returns raw HTML or Markdown,
                # NOT JSON. The previous json.loads() call always raised JSONDecodeError.
                # Simply read and return the raw text content.
                content = response.read().decode('utf-8', errors='ignore')
                return content[:4000]
        except Exception:
            # Fallback to general crate info if readme missing
            try:
                info_url = f"https://crates.io/api/v1/crates/{crate_name}"
                req = urllib.request.Request(info_url, headers={'User-Agent': 'NSTL-LiveFetcher/1.0'})
                with urllib.request.urlopen(req, timeout=10.0) as response:
                    data = json.loads(response.read().decode())
                    return data.get("crate", {}).get("description", "")[:4000]
            except Exception as e:
                logger.error(f"CratesIoFetcher failed for {crate_name}: {e}")
                return ""

class DuckDuckGoFetcher(LiveDocFetcher):
    def fetch(self, query: str) -> str:
        """Fallback generic web scraper using DDG HTML."""
        search_query = f"{query} python official documentation"
        encoded_query = urllib.parse.quote(search_query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=10.0) as response:
                html = response.read().decode('utf-8', errors='ignore')
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract text from the first result snippet
                result = soup.find('a', class_='result__snippet')
                if result:
                    text = result.get_text(strip=True)
                else:
                    # Fallback to extracting all text if snippet not found
                    text = soup.get_text(separator=' ', strip=True)
                    
                return text[:4000]
        except Exception as e:
            logger.error(f"DuckDuckGoFetcher failed for query '{query}': {e}")
            return ""
            # BUG 22 FIX: Removed duplicate dead-code lines that were placed after
            # the return statement (copy-paste error). Original lines 79-80 deleted.

class FetcherFactory:
    @staticmethod
    def get_fetcher(domain_context: str) -> LiveDocFetcher:
        if "Python" in domain_context:
            return PyPiFetcher()
        elif "Rust" in domain_context:
            return CratesIoFetcher()
        return DuckDuckGoFetcher()
