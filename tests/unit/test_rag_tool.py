"""
Tests for RAG Tool (LlamaIndex + ChromaDB retriever).

TDD: Written BEFORE src/tools/rag_tool.py.
Per MASTER_PROMPT: Uses LlamaIndex + ChromaDB for local persistence.
All external dependencies are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestRAGTool:
    """Tests for the RAGTool retriever wrapper."""

    def test_init_creates_tool(self):
        """RAGTool must initialize with a persist directory."""
        from tools.rag_tool import RAGTool

        tool = RAGTool(persist_dir="/fake/vector_store")
        assert tool.persist_dir == "/fake/vector_store"

    def test_init_sets_default_collection_and_reference_dir(self):
        """RAGTool should default to guidelines collection and docs dir."""
        from tools.rag_tool import RAGTool

        tool = RAGTool(persist_dir="/fake/vector_store")
        assert tool.collection_name == "guidelines"
        assert "reference_docs" in tool.reference_docs_dir

    def test_query_returns_results(self):
        """query() must return a list of retrieved text chunks."""
        from tools.rag_tool import RAGTool

        tool = RAGTool(persist_dir="/fake/vector_store")

        mock_results = [
            {"text": "ACC/AHA Class I: Initiate GDMT for HFrEF.", "score": 0.92},
            {"text": "NCCN recommends platinum-based chemo for stage III NSCLC.", "score": 0.87},
        ]
        with patch.object(tool, "_retrieve", return_value=mock_results):
            results = tool.query("heart failure treatment guidelines", top_k=2)

        assert isinstance(results, list)
        assert len(results) == 2
        assert "text" in results[0]

    def test_query_respects_top_k(self):
        """query() must respect the top_k parameter."""
        from tools.rag_tool import RAGTool

        tool = RAGTool(persist_dir="/fake/vector_store")

        mock_results = [{"text": "Result 1", "score": 0.9}]
        with patch.object(tool, "_retrieve", return_value=mock_results) as mock_ret:
            tool.query("test", top_k=1)
            mock_ret.assert_called_once_with("test", 1)

    def test_query_empty_returns_empty(self):
        """query() with no matches returns empty list."""
        from tools.rag_tool import RAGTool

        tool = RAGTool(persist_dir="/fake/vector_store")

        with patch.object(tool, "_retrieve", return_value=[]):
            results = tool.query("obscure topic no matches")

        assert results == []

    def test_format_context_produces_string(self):
        """format_context() must join retrieved chunks into a single string."""
        from tools.rag_tool import RAGTool

        tool = RAGTool(persist_dir="/fake/vector_store")

        chunks = [
            {"text": "Guideline A content.", "score": 0.9},
            {"text": "Guideline B content.", "score": 0.8},
        ]
        context = tool.format_context(chunks)
        assert "Guideline A content." in context
        assert "Guideline B content." in context
        assert isinstance(context, str)

    def test_ingest_documents_accepts_file_paths(self):
        """ingest() should accept a list of document file paths."""
        from tools.rag_tool import RAGTool

        tool = RAGTool(persist_dir="/fake/vector_store")

        with patch.object(tool, "_ingest_files") as mock_ingest:
            tool.ingest(["/fake/doc1.pdf", "/fake/doc2.pdf"])
            mock_ingest.assert_called_once_with(["/fake/doc1.pdf", "/fake/doc2.pdf"])

    def test_query_bootstraps_ingestion_when_store_empty(self):
        """query() should ingest reference docs when collection is empty."""
        from tools.rag_tool import RAGTool

        tool = RAGTool(
            persist_dir="/fake/vector_store",
            reference_docs_dir="/fake/reference_docs",
        )

        with patch.object(tool, "_collection_count", return_value=0), \
             patch.object(tool, "_list_reference_docs", return_value=["/fake/reference_docs/cardio.md"]), \
             patch.object(tool, "_ingest_files") as mock_ingest, \
             patch.object(tool, "_retrieve", return_value=[]):
            tool.query("acute coronary syndrome", top_k=3)

        mock_ingest.assert_called_once_with(["/fake/reference_docs/cardio.md"])

    def test_query_does_not_bootstrap_when_store_has_documents(self):
        """query() should skip auto-ingestion when collection already populated."""
        from tools.rag_tool import RAGTool

        tool = RAGTool(
            persist_dir="/fake/vector_store",
            reference_docs_dir="/fake/reference_docs",
        )

        with patch.object(tool, "_collection_count", return_value=5), \
             patch.object(tool, "_list_reference_docs", return_value=["/fake/reference_docs/cardio.md"]), \
             patch.object(tool, "_ingest_files") as mock_ingest, \
             patch.object(tool, "_retrieve", return_value=[]):
            tool.query("acute coronary syndrome", top_k=3)

        mock_ingest.assert_not_called()
