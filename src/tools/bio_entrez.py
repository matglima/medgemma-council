"""
PubMed Tool: Bio.Entrez wrapper for the Research Agent.

Wraps NCBI E-utilities (esearch, efetch) via Biopython to search PubMed
and retrieve article abstracts. All network calls are isolated in
_esearch/_efetch methods for easy mocking in tests.

Per CLAUDE.md: Research Agent must provide PMIDs for all claims.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PubMedTool:
    """
    PubMed search and retrieval tool using NCBI E-utilities.

    Args:
        email: Required by NCBI policy for API access.
        api_key: Optional NCBI API key (increases rate limit to 10 req/s).
    """

    def __init__(self, email: str, api_key: Optional[str] = None) -> None:
        self.email = email
        self.api_key = api_key or os.getenv("NCBI_API_KEY")

    def search_articles(
        self,
        query: str,
        max_results: int = 5,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> List[str]:
        """
        Search PubMed for articles matching the query.

        Args:
            query: Search query (natural language or MeSH terms).
            max_results: Maximum number of PMIDs to return.
            min_date: Minimum publication date (e.g., '2020').
            max_date: Maximum publication date (e.g., '2025').

        Returns:
            List of PMID strings.
        """
        try:
            record = self._esearch(
                query=query,
                max_results=max_results,
                min_date=min_date,
                max_date=max_date,
            )
            return record.get("IdList", [])
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, str]]:
        """
        Fetch article details (title, abstract) for given PMIDs.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of dicts with keys: pmid, title, abstract.
        """
        if not pmids:
            return []

        try:
            return self._efetch(pmids)
        except Exception as e:
            logger.error(f"PubMed fetch error: {e}")
            return []

    def format_results(self, articles: List[Dict[str, str]]) -> str:
        """
        Format retrieved articles into a human-readable citation string.

        Args:
            articles: List of article dicts from fetch_abstracts.

        Returns:
            Formatted string with PMID citations.
        """
        if not articles:
            return "No articles found."

        parts = []
        for article in articles:
            parts.append(
                f"PMID: {article['pmid']}\n"
                f"Title: {article['title']}\n"
                f"Abstract: {article['abstract']}\n"
            )
        return "\n---\n".join(parts)

    def _esearch(
        self,
        query: str,
        max_results: int = 5,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Internal: Execute an ESearch query against PubMed.
        Isolated for mocking in tests.
        """
        from Bio import Entrez  # type: ignore

        Entrez.email = self.email

        kwargs: Dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if min_date:
            kwargs["mindate"] = min_date
            kwargs["datetype"] = "pdat"
        if max_date:
            kwargs["maxdate"] = max_date

        handle = Entrez.esearch(**kwargs)
        record = Entrez.read(handle)
        handle.close()
        return record

    def _efetch(self, pmids: List[str]) -> List[Dict[str, str]]:
        """
        Internal: Fetch article details via EFetch.
        Isolated for mocking in tests.
        """
        from Bio import Entrez  # type: ignore

        Entrez.email = self.email

        kwargs: Dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key

        handle = Entrez.efetch(**kwargs)
        records = Entrez.read(handle)
        handle.close()

        results = []
        for article in records.get("PubmedArticle", []):
            citation = article.get("MedlineCitation", {})
            art_data = citation.get("Article", {})

            title = art_data.get("ArticleTitle", "No title available")

            abstract_data = art_data.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_data, list):
                abstract_text = " ".join(str(s) for s in abstract_data)
            else:
                abstract_text = str(abstract_data)

            if not abstract_text:
                abstract_text = "No abstract available."

            pmid = str(citation.get("PMID", "Unknown"))

            results.append({
                "pmid": pmid,
                "title": str(title),
                "abstract": abstract_text,
            })

        return results
