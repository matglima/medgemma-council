"""
ResearchAgent: PubMed-based literature retrieval agent.

Uses Bio.Entrez to search PubMed when:
- Specialist agents provide contradicting recommendations (conflict_detected)
- A specialist requests external evidence (<RESEARCH_NEEDED> tag)
- The case involves a rare condition where guidelines are sparse

Per CLAUDE.md: Must provide PMIDs for all claims.
Per RESEARCH_REPORT: Acts as a medical librarian using PICO search.
"""

import logging
from typing import Any, Dict, List

from agents.base import BaseAgent
from tools.bio_entrez import PubMedTool

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    PubMed research agent that retrieves primary literature
    to resolve clinical gray zones and specialist conflicts.
    """

    def __init__(
        self,
        llm: Any,
        pubmed_email: str,
        api_key: str = None,
        system_prompt: str = "",
    ) -> None:
        default_prompt = (
            "You are a medical research librarian. Your role is to: "
            "1) Convert clinical controversies into PICO search strings, "
            "2) Search PubMed favoring Meta-Analyses and RCTs from the last 5 years, "
            "3) Summarize the top results with PMID citations, "
            "4) Provide evidence-based resolution to clinical conflicts."
        )
        super().__init__(llm=llm, system_prompt=system_prompt or default_prompt)
        self.pubmed_tool = PubMedTool(email=pubmed_email, api_key=api_key)

    @property
    def name(self) -> str:
        return "ResearchAgent"

    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform literature retrieval based on current state.

        Builds a search query from patient context and agent conflicts,
        retrieves relevant articles, and summarizes findings.
        """
        query = self._build_search_query(state)
        articles = self._search_and_fetch(query)
        summary = self._summarize(articles)

        return {"research_findings": summary}

    def _build_search_query(self, state: Dict[str, Any]) -> str:
        """
        Build a PubMed search query from patient context and conflicts.
        Isolated for mocking in tests.

        In production, uses the LLM to convert the clinical controversy
        into a structured PICO search string with MeSH terms.
        """
        patient_context = state.get("patient_context", {})
        agent_outputs = state.get("agent_outputs", {})

        # Build query components from available information
        complaint = patient_context.get("chief_complaint", "")
        conflict_texts = " ".join(agent_outputs.values())

        # In production, the LLM would refine this into MeSH terms
        query_parts = []
        if complaint:
            query_parts.append(complaint)
        if conflict_texts:
            query_parts.append(conflict_texts[:200])  # Truncate for query

        return " ".join(query_parts) if query_parts else "clinical decision support"

    def _search_and_fetch(self, query: str) -> List[Dict[str, str]]:
        """
        Search PubMed and fetch article abstracts.
        Isolated for mocking in tests.
        """
        pmids = self.pubmed_tool.search_articles(
            query=query,
            max_results=5,
            min_date="2020",
            max_date="2025",
        )
        if not pmids:
            return []
        return self.pubmed_tool.fetch_abstracts(pmids)

    def _summarize(self, articles: List[Dict[str, str]]) -> str:
        """
        Summarize retrieved articles using the LLM.
        Isolated for mocking in tests.

        Feeds the articles to the LLM with instructions to summarize
        key findings and provide PMID citations.
        """
        if not articles:
            return "No relevant articles found in PubMed."

        formatted = self.pubmed_tool.format_results(articles)

        # Use the LLM to generate a clinical summary
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Summarize the following PubMed articles for clinical decision-making. "
            f"Focus on: study design, key findings, and clinical implications. "
            f"Always include PMID citations.\n\n"
            f"{formatted}\n\n"
            f"Provide a concise evidence summary (3-5 sentences) with PMID references."
        )

        try:
            result = self.llm(prompt, max_tokens=1024)
            if isinstance(result, dict):
                return result["choices"][0]["text"]
            return str(result)
        except Exception:
            # Fallback to raw formatted results if LLM fails
            return formatted
