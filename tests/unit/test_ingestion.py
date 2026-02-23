"""
Tests for Phase 14: RAG ingestion pipeline.

Tests the ingestion module (src/tools/ingestion.py) and the CLI script
(scripts/ingest_guidelines.py) that chunk, embed, and index clinical
guideline PDFs into ChromaDB.

TDD: Written BEFORE implementation.
Per CLAUDE.md: "Write failing tests BEFORE implementation code for every module."

Key behaviors:
1. GuidelineChunker splits text into overlapping chunks with metadata
2. IngestionPipeline orchestrates chunking -> embedding -> indexing
3. CLI script accepts directory or file paths, chunk_size, chunk_overlap
4. All heavy deps (LlamaIndex, ChromaDB, sentence-transformers) mocked
5. Progress logging emitted during ingestion
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# GuidelineChunker tests
# ---------------------------------------------------------------------------


class TestGuidelineChunker:
    """Tests for text chunking with overlap and metadata."""

    def test_import(self):
        """GuidelineChunker should be importable from tools.ingestion."""
        from tools.ingestion import GuidelineChunker

    def test_init_stores_params(self):
        """GuidelineChunker should store chunk_size and chunk_overlap."""
        from tools.ingestion import GuidelineChunker

        chunker = GuidelineChunker(chunk_size=512, chunk_overlap=64)
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 64

    def test_default_params(self):
        """GuidelineChunker should have sensible defaults."""
        from tools.ingestion import GuidelineChunker

        chunker = GuidelineChunker()
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 128

    def test_chunk_text_returns_list_of_dicts(self):
        """chunk_text() should return a list of dicts with 'text' and 'metadata' keys."""
        from tools.ingestion import GuidelineChunker

        chunker = GuidelineChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 100  # 100 chars, should produce multiple chunks
        chunks = chunker.chunk_text(text, source="test.pdf")
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk

    def test_chunk_text_includes_source_metadata(self):
        """Each chunk's metadata should include the source filename."""
        from tools.ingestion import GuidelineChunker

        chunker = GuidelineChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 100
        chunks = chunker.chunk_text(text, source="nccn_lung.pdf")
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "nccn_lung.pdf"

    def test_chunk_text_includes_chunk_index(self):
        """Each chunk's metadata should include its sequential index."""
        from tools.ingestion import GuidelineChunker

        chunker = GuidelineChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 100
        chunks = chunker.chunk_text(text, source="test.pdf")
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_chunk_text_exposes_legacy_top_level_metadata_fields(self):
        """Chunks should keep legacy top-level keys for notebook compatibility."""
        from tools.ingestion import GuidelineChunker

        chunker = GuidelineChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 100
        chunks = chunker.chunk_text(text, source="legacy.pdf")

        for i, chunk in enumerate(chunks):
            assert chunk["source"] == "legacy.pdf"
            assert chunk["chunk_index"] == i

    def test_chunk_overlap_creates_overlap(self):
        """Adjacent chunks should share overlapping text."""
        from tools.ingestion import GuidelineChunker

        chunker = GuidelineChunker(chunk_size=20, chunk_overlap=5)
        # Create text with distinct characters so we can verify overlap
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdef"
        chunks = chunker.chunk_text(text, source="test.pdf")

        if len(chunks) >= 2:
            # The end of chunk[0] should overlap with start of chunk[1]
            chunk0_text = chunks[0]["text"]
            chunk1_text = chunks[1]["text"]
            overlap_region = chunk0_text[-5:]  # last 5 chars of chunk 0
            assert chunk1_text.startswith(overlap_region)

    def test_chunk_empty_text_returns_empty(self):
        """Empty text should return an empty chunk list."""
        from tools.ingestion import GuidelineChunker

        chunker = GuidelineChunker()
        chunks = chunker.chunk_text("", source="empty.pdf")
        assert chunks == []

    def test_chunk_small_text_returns_single_chunk(self):
        """Text smaller than chunk_size should produce exactly one chunk."""
        from tools.ingestion import GuidelineChunker

        chunker = GuidelineChunker(chunk_size=1024, chunk_overlap=128)
        text = "Short clinical guideline excerpt."
        chunks = chunker.chunk_text(text, source="short.pdf")
        assert len(chunks) == 1
        assert chunks[0]["text"] == text


# ---------------------------------------------------------------------------
# IngestionPipeline tests
# ---------------------------------------------------------------------------


class TestIngestionPipeline:
    """Tests for the orchestrated ingestion pipeline."""

    def test_import(self):
        """IngestionPipeline should be importable from tools.ingestion."""
        from tools.ingestion import IngestionPipeline

    def test_init_stores_config(self):
        """IngestionPipeline should store persist_dir and collection name."""
        from tools.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(
            persist_dir="/fake/vector_store",
            collection_name="guidelines",
        )
        assert pipeline.persist_dir == "/fake/vector_store"
        assert pipeline.collection_name == "guidelines"

    def test_default_collection_name(self):
        """Default collection_name should be 'guidelines'."""
        from tools.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(persist_dir="/fake")
        assert pipeline.collection_name == "guidelines"

    def test_init_accepts_legacy_persist_directory_alias(self):
        """Constructor should accept persist_directory for older notebooks."""
        from tools.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(persist_directory="/legacy/path")
        assert pipeline.persist_dir == "/legacy/path"

    def test_ingest_file_calls_chunker(self):
        """ingest_file() should read the file and call the chunker."""
        from tools.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(persist_dir="/fake")

        with patch.object(pipeline, "_read_file", return_value="Some text content") as mock_read, \
             patch.object(pipeline, "_store_chunks") as mock_store:
            pipeline.ingest_file("/fake/guideline.pdf")
            mock_read.assert_called_once_with("/fake/guideline.pdf")

    def test_ingest_file_stores_chunks(self):
        """ingest_file() should pass chunks to _store_chunks."""
        from tools.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(persist_dir="/fake")

        with patch.object(pipeline, "_read_file", return_value="Some text " * 200) as mock_read, \
             patch.object(pipeline, "_store_chunks") as mock_store:
            pipeline.ingest_file("/fake/guideline.pdf")
            mock_store.assert_called_once()
            # First arg should be a list of chunk dicts
            stored_chunks = mock_store.call_args[0][0]
            assert isinstance(stored_chunks, list)
            assert len(stored_chunks) > 0

    def test_ingest_directory_processes_all_files(self):
        """ingest_directory() should process all PDF, TXT, and MD files."""
        from tools.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(persist_dir="/fake")

        with patch("os.listdir", return_value=["acc_hf.pdf", "nccn_lung.pdf", "who_child.txt", "readme.md"]), \
             patch("os.path.isfile", return_value=True), \
             patch.object(pipeline, "ingest_file") as mock_ingest:
            pipeline.ingest_directory("guidelines/")
            # Should ingest pdf, txt, and md files (all 4 are supported)
            assert mock_ingest.call_count == 4

    def test_ingest_directory_skips_non_document_files(self):
        """ingest_directory() should skip files that aren't PDF/TXT."""
        from tools.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(persist_dir="/fake")

        with patch("os.listdir", return_value=["data.csv", "image.png", ".gitkeep"]), \
             patch("os.path.isfile", return_value=True), \
             patch.object(pipeline, "ingest_file") as mock_ingest:
            pipeline.ingest_directory("guidelines/")
            mock_ingest.assert_not_called()

    def test_get_stats_returns_dict(self):
        """get_stats() should return a dict with collection info."""
        from tools.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(persist_dir="/fake")

        with patch.object(pipeline, "_get_collection_stats", return_value={
            "collection_name": "guidelines",
            "document_count": 42,
        }):
            stats = pipeline.get_stats()
            assert isinstance(stats, dict)
            assert "collection_name" in stats
            assert "document_count" in stats


# ---------------------------------------------------------------------------
# CLI script tests
# ---------------------------------------------------------------------------


class TestIngestGuidelinesCLI:
    """Tests for the scripts/ingest_guidelines.py CLI entry point."""

    def test_parse_args_default_dir(self):
        """Default input_dir should be data/reference_docs/."""
        from scripts.ingest_guidelines import parse_args

        args = parse_args([])
        assert "reference_docs" in args.input_dir

    def test_parse_args_custom_input(self):
        """--input-dir flag should override default."""
        from scripts.ingest_guidelines import parse_args

        args = parse_args(["--input-dir", "/custom/path"])
        assert args.input_dir == "/custom/path"

    def test_parse_args_output_dir(self):
        """--output-dir flag should set the vector store path."""
        from scripts.ingest_guidelines import parse_args

        args = parse_args(["--output-dir", "/custom/vector_store"])
        assert args.output_dir == "/custom/vector_store"

    def test_parse_args_chunk_size(self):
        """--chunk-size flag should set chunk size."""
        from scripts.ingest_guidelines import parse_args

        args = parse_args(["--chunk-size", "512"])
        assert args.chunk_size == 512

    def test_parse_args_chunk_overlap(self):
        """--chunk-overlap flag should set chunk overlap."""
        from scripts.ingest_guidelines import parse_args

        args = parse_args(["--chunk-overlap", "64"])
        assert args.chunk_overlap == 64

    def test_parse_args_collection_name(self):
        """--collection flag should set the collection name."""
        from scripts.ingest_guidelines import parse_args

        args = parse_args(["--collection", "cardiology_guidelines"])
        assert args.collection == "cardiology_guidelines"

    def test_run_ingestion_calls_pipeline(self):
        """run_ingestion() should create an IngestionPipeline and call ingest_directory."""
        from scripts.ingest_guidelines import run_ingestion

        with patch("scripts.ingest_guidelines.IngestionPipeline") as MockPipeline, \
             patch("os.makedirs"):
            mock_pipeline = MockPipeline.return_value
            mock_pipeline.get_stats.return_value = {
                "collection_name": "guidelines",
                "document_count": 10,
            }
            run_ingestion(
                input_dir="/fake/docs",
                output_dir="/fake/store",
                chunk_size=1024,
                chunk_overlap=128,
                collection="guidelines",
            )
            MockPipeline.assert_called_once_with(
                persist_dir="/fake/store",
                collection_name="guidelines",
                chunk_size=1024,
                chunk_overlap=128,
            )
            mock_pipeline.ingest_directory.assert_called_once_with("/fake/docs")
