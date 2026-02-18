"""
Tests for ResearchAgent (PubMed-based literature retrieval).

TDD: Written BEFORE src/agents/researcher.py.
Per CLAUDE.md: Research Agent must provide PMIDs for all claims.
Per MASTER_PROMPT: Workflow is query -> MeSH terms -> esearch -> efetch -> summarize.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestResearchAgent:
    """Tests for the ResearchAgent PubMed workflow."""

    def test_inherits_base_agent(self):
        """ResearchAgent must inherit from BaseAgent."""
        from agents.researcher import ResearchAgent
        from agents.base import BaseAgent

        assert issubclass(ResearchAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """ResearchAgent.name must return 'ResearchAgent'."""
        from agents.researcher import ResearchAgent

        agent = ResearchAgent(llm=mock_llm, pubmed_email="test@example.com")
        assert agent.name == "ResearchAgent"

    def test_analyze_returns_research_findings(self, mock_llm):
        """analyze() must return research_findings with PMID citations."""
        from agents.researcher import ResearchAgent

        agent = ResearchAgent(llm=mock_llm, pubmed_email="test@example.com")
        state = {
            "messages": [],
            "patient_context": {"chief_complaint": "anthracycline cardiotoxicity"},
            "medical_images": [],
            "agent_outputs": {
                "CardiologyAgent": "Stop chemo due to LVEF < 40%.",
                "OncologyAgent": "Continue chemo to prevent relapse.",
            },
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": True,
            "iteration_count": 0,
            "final_plan": "",
        }

        mock_articles = [
            {
                "pmid": "99999999",
                "title": "Management of Anthracycline Cardiotoxicity",
                "abstract": "Recent evidence suggests dose modification...",
            }
        ]

        with patch.object(agent, "_search_and_fetch", return_value=mock_articles):
            with patch.object(agent, "_summarize", return_value="PMID: 99999999 - Dose modification recommended."):
                result = agent.analyze(state)

        assert "research_findings" in result
        assert "PMID" in result["research_findings"]

    def test_analyze_builds_query_from_conflict(self, mock_llm):
        """analyze() should build a search query from conflicting agent outputs."""
        from agents.researcher import ResearchAgent

        agent = ResearchAgent(llm=mock_llm, pubmed_email="test@example.com")
        state = {
            "messages": [],
            "patient_context": {"chief_complaint": "chest pain"},
            "medical_images": [],
            "agent_outputs": {
                "CardiologyAgent": "Possible acute coronary syndrome.",
                "OncologyAgent": "Consider cardiac involvement from tumor.",
            },
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": True,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_build_search_query", return_value="cardiac tumor involvement") as mock_query:
            with patch.object(agent, "_search_and_fetch", return_value=[]):
                with patch.object(agent, "_summarize", return_value="No relevant articles found."):
                    agent.analyze(state)
                    mock_query.assert_called_once()

    def test_analyze_no_conflict_still_works(self, mock_llm):
        """analyze() should still produce findings even without conflict."""
        from agents.researcher import ResearchAgent

        agent = ResearchAgent(llm=mock_llm, pubmed_email="test@example.com")
        state = {
            "messages": [],
            "patient_context": {"chief_complaint": "rare genetic disorder"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_search_and_fetch", return_value=[]):
            with patch.object(agent, "_summarize", return_value="No articles found."):
                result = agent.analyze(state)

        assert "research_findings" in result

    def test_has_pubmed_tool(self, mock_llm):
        """ResearchAgent must have a PubMedTool instance."""
        from agents.researcher import ResearchAgent

        agent = ResearchAgent(llm=mock_llm, pubmed_email="test@example.com")
        assert hasattr(agent, "pubmed_tool")
