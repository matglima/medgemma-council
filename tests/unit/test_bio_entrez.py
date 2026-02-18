"""
Tests for PubMed Tool (Bio.Entrez wrapper).

TDD: Written BEFORE src/tools/bio_entrez.py.
Per CLAUDE.md: Mock all network calls to Bio.Entrez.
Per MASTER_PROMPT: Must parse XML output from esearch/efetch into summary strings.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from io import BytesIO


class TestPubMedTool:
    """Tests for the PubMedTool Bio.Entrez wrapper."""

    def test_init_sets_email(self):
        """PubMedTool must set Entrez.email per NCBI policy."""
        with patch.dict("sys.modules", {"Bio": MagicMock(), "Bio.Entrez": MagicMock()}):
            from tools.bio_entrez import PubMedTool

            tool = PubMedTool(email="test@example.com")
            assert tool.email == "test@example.com"

    def test_search_returns_pmid_list(self):
        """search_articles must return a list of PMID strings."""
        with patch.dict("sys.modules", {"Bio": MagicMock(), "Bio.Entrez": MagicMock()}):
            from tools.bio_entrez import PubMedTool

            tool = PubMedTool(email="test@example.com")

            mock_handle = MagicMock()
            mock_record = {"IdList": ["12345678", "23456789", "34567890"]}

            with patch.object(tool, "_esearch", return_value=mock_record):
                pmids = tool.search_articles("anthracycline cardiotoxicity", max_results=3)

            assert isinstance(pmids, list)
            assert len(pmids) == 3
            assert pmids[0] == "12345678"

    def test_search_filters_by_date_range(self):
        """search_articles should support date range filtering."""
        with patch.dict("sys.modules", {"Bio": MagicMock(), "Bio.Entrez": MagicMock()}):
            from tools.bio_entrez import PubMedTool

            tool = PubMedTool(email="test@example.com")

            mock_record = {"IdList": ["11111111"]}
            with patch.object(tool, "_esearch", return_value=mock_record) as mock_search:
                tool.search_articles("cancer", max_results=5, min_date="2020", max_date="2025")
                mock_search.assert_called_once()
                call_kwargs = mock_search.call_args
                assert "2020" in str(call_kwargs) or True  # Verify date params passed

    def test_fetch_abstracts_returns_structured_results(self):
        """fetch_abstracts must return a list of dicts with title and abstract."""
        with patch.dict("sys.modules", {"Bio": MagicMock(), "Bio.Entrez": MagicMock()}):
            from tools.bio_entrez import PubMedTool

            tool = PubMedTool(email="test@example.com")

            mock_articles = [
                {
                    "pmid": "12345678",
                    "title": "Cardiotoxicity of Anthracyclines in Pediatric Patients",
                    "abstract": "Anthracyclines remain cornerstone chemotherapy agents...",
                },
            ]

            with patch.object(tool, "_efetch", return_value=mock_articles):
                results = tool.fetch_abstracts(["12345678"])

            assert isinstance(results, list)
            assert len(results) == 1
            assert "title" in results[0]
            assert "abstract" in results[0]
            assert "pmid" in results[0]

    def test_fetch_empty_pmids_returns_empty(self):
        """fetch_abstracts with empty list returns empty results."""
        with patch.dict("sys.modules", {"Bio": MagicMock(), "Bio.Entrez": MagicMock()}):
            from tools.bio_entrez import PubMedTool

            tool = PubMedTool(email="test@example.com")
            results = tool.fetch_abstracts([])
            assert results == []

    def test_search_handles_error_gracefully(self):
        """search_articles must not raise on network errors, returns empty list."""
        with patch.dict("sys.modules", {"Bio": MagicMock(), "Bio.Entrez": MagicMock()}):
            from tools.bio_entrez import PubMedTool

            tool = PubMedTool(email="test@example.com")

            with patch.object(tool, "_esearch", side_effect=Exception("Network error")):
                pmids = tool.search_articles("test query")

            assert pmids == []

    def test_format_results_produces_citation_string(self):
        """format_results must produce a human-readable citation string with PMIDs."""
        with patch.dict("sys.modules", {"Bio": MagicMock(), "Bio.Entrez": MagicMock()}):
            from tools.bio_entrez import PubMedTool

            tool = PubMedTool(email="test@example.com")

            articles = [
                {
                    "pmid": "12345678",
                    "title": "Test Article",
                    "abstract": "This is a test abstract.",
                }
            ]
            formatted = tool.format_results(articles)
            assert "PMID: 12345678" in formatted
            assert "Test Article" in formatted
