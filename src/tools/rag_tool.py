"""RAG Tool: ChromaDB-backed guideline retrieval.

This module provides retrieval for specialist agents and includes automatic
bootstrap behavior: if the vector store is empty and local reference docs are
available, it ingests those docs before the first query.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SUPPORTED_DOC_EXTENSIONS = {".pdf", ".txt", ".md"}


def _module_repo_root() -> str:
    """Return repository root path for this source tree."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_runtime_path(path: str) -> str:
    """Resolve relative paths robustly across notebook working directories.

    Resolution order for relative inputs:
    1. Current working directory (if path exists there)
    2. Repository root (if path exists there)
    3. Repository-root-relative fallback (for paths that may be created later)
    """
    if os.path.isabs(path):
        return path

    cwd_candidate = os.path.abspath(path)
    repo_candidate = os.path.abspath(os.path.join(_module_repo_root(), path))

    if os.path.exists(cwd_candidate):
        return cwd_candidate
    if os.path.exists(repo_candidate):
        return repo_candidate
    return repo_candidate


class RAGTool:
    """
    Retrieval-Augmented Generation tool backed by ChromaDB.

    Args:
        persist_dir: Path to the ChromaDB persistence directory.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "guidelines",
        reference_docs_dir: str = "data/reference_docs",
        auto_bootstrap: bool = True,
    ) -> None:
        self.persist_dir = _resolve_runtime_path(persist_dir)
        self.collection_name = collection_name
        self.reference_docs_dir = _resolve_runtime_path(reference_docs_dir)
        self.auto_bootstrap = auto_bootstrap
        self._bootstrap_checked = False

        logger.debug(
            "RAGTool init: persist_dir=%s, reference_docs_dir=%s",
            self.persist_dir,
            self.reference_docs_dir,
        )

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant guideline chunks.

        Args:
            query_text: Natural language clinical query.
            top_k: Number of top results to return.

        Returns:
            List of dicts with 'text' and 'score' keys.
        """
        try:
            if self.auto_bootstrap:
                self._ensure_bootstrapped()
            return self._retrieve(query_text, top_k)
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return []

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Join retrieved chunks into a single context string for LLM consumption.

        Args:
            chunks: List of result dicts from query().

        Returns:
            Concatenated string of all chunk texts.
        """
        if not chunks:
            return ""
        return "\n\n".join(
            f"[Source {i + 1} | {c.get('source', 'unknown')} | score: {c.get('score', 'N/A')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )

    def ingest(self, file_paths: List[str]) -> None:
        """
        Ingest documents into the vector store.

        Args:
            file_paths: List of paths to PDF/text documents.
        """
        self._ingest_files(file_paths)

    def _retrieve(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Internal: Execute retrieval directly against ChromaDB.
        Isolated for mocking in tests.
        """
        try:
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=self.persist_dir)
            collection = client.get_or_create_collection(self.collection_name)

            query = collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "distances", "metadatas"],
            )

            documents = (query.get("documents") or [[]])[0]
            distances = (query.get("distances") or [[]])[0]
            metadatas = (query.get("metadatas") or [[]])[0]

            results: List[Dict[str, Any]] = []
            for i, text in enumerate(documents):
                distance = distances[i] if i < len(distances) else None
                metadata = metadatas[i] if i < len(metadatas) else {}

                if isinstance(distance, (int, float)):
                    score = 1.0 / (1.0 + float(distance))
                else:
                    score = 0.0

                source = "unknown"
                if isinstance(metadata, dict):
                    source = str(metadata.get("source", "unknown"))

                results.append(
                    {
                        "text": str(text),
                        "score": round(score, 6),
                        "source": source,
                    }
                )

            return results

        except ImportError as e:
            logger.warning(
                "RAG dependencies missing (%s). Install with: pip install chromadb",
                e,
            )
            return []

    def _ingest_files(self, file_paths: List[str]) -> None:
        """
        Internal: Ingest documents into ChromaDB via IngestionPipeline.
        Isolated for mocking in tests.
        """
        try:
            from tools.ingestion import IngestionPipeline

            pipeline = IngestionPipeline(
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
            )

            total_chunks = 0
            for path in file_paths:
                total_chunks += pipeline.ingest_file(path)

            logger.info(
                "Ingested %s files into collection '%s' (%s chunks)",
                len(file_paths),
                self.collection_name,
                total_chunks,
            )

        except ImportError as e:
            logger.warning(
                "Ingestion dependencies missing (%s). Install with: "
                "pip install chromadb",
                e,
            )

    def _ensure_bootstrapped(self) -> None:
        """Auto-ingest local reference docs when the collection is empty."""
        if self._bootstrap_checked:
            return

        self._bootstrap_checked = True

        count = self._collection_count()
        if count > 0:
            logger.debug(
                "RAG collection '%s' already populated (%s chunks)",
                self.collection_name,
                count,
            )
            return

        files = self._list_reference_docs()
        if not files:
            logger.warning(
                "RAG collection '%s' is empty and no reference docs were found at '%s'. "
                "Run scripts/scrape_guidelines.py then scripts/ingest_guidelines.py.",
                self.collection_name,
                self.reference_docs_dir,
            )
            return

        logger.info(
            "RAG collection '%s' is empty. Auto-ingesting %s reference docs...",
            self.collection_name,
            len(files),
        )
        self._ingest_files(files)

    def _list_reference_docs(self) -> List[str]:
        """List supported reference documents from the configured docs folder."""
        if not os.path.isdir(self.reference_docs_dir):
            return []

        files: List[str] = []
        for filename in sorted(os.listdir(self.reference_docs_dir)):
            path = os.path.join(self.reference_docs_dir, filename)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in _SUPPORTED_DOC_EXTENSIONS:
                files.append(path)
        return files

    def _collection_count(self) -> int:
        """Return number of documents currently stored in the collection."""
        try:
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=self.persist_dir)
            collection = client.get_or_create_collection(self.collection_name)
            return int(collection.count())
        except Exception:
            return 0
