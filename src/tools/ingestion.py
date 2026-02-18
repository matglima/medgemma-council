"""
Ingestion module for clinical guideline documents.

Provides:
- GuidelineChunker: Splits document text into overlapping chunks with metadata.
- IngestionPipeline: Orchestrates reading, chunking, and indexing documents
  into ChromaDB via LlamaIndex.

All heavy dependencies (LlamaIndex, ChromaDB, PyPDF) are isolated in
internal _methods for easy mocking in tests.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Supported file extensions for ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class GuidelineChunker:
    """
    Splits document text into overlapping chunks with metadata.

    Uses a simple sliding-window approach: each chunk is chunk_size
    characters long, and consecutive chunks overlap by chunk_overlap
    characters. This ensures that no sentence is split without context
    on both sides.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.

        Args:
            text: The full document text to chunk.
            source: Source filename for metadata tagging.

        Returns:
            List of dicts, each with 'text' and 'metadata' keys.
            metadata contains 'source' and 'chunk_index'.
        """
        if not text or not text.strip():
            return []

        chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index,
                },
            })

            chunk_index += 1

            # Advance by (chunk_size - chunk_overlap) to create overlap
            step = self.chunk_size - self.chunk_overlap
            if step <= 0:
                step = 1  # Safety: prevent infinite loop
            start += step

            # If we've consumed all the text, stop
            if end >= len(text):
                break

        return chunks


class IngestionPipeline:
    """
    Orchestrates the ingestion of clinical guideline documents into
    a ChromaDB vector store.

    Pipeline: read file -> chunk text -> embed & store in ChromaDB.

    Args:
        persist_dir: Path to ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.
        chunk_size: Characters per chunk (passed to GuidelineChunker).
        chunk_overlap: Overlap between chunks (passed to GuidelineChunker).
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "guidelines",
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chunker = GuidelineChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single document file into the vector store.

        Args:
            file_path: Path to a PDF, TXT, or MD file.

        Returns:
            Number of chunks stored.
        """
        logger.info(f"Ingesting file: {file_path}")

        text = self._read_file(file_path)
        if not text:
            logger.warning(f"No text extracted from {file_path}")
            return 0

        source = os.path.basename(file_path)
        chunks = self.chunker.chunk_text(text, source=source)

        if chunks:
            self._store_chunks(chunks)
            logger.info(f"Stored {len(chunks)} chunks from {source}")

        return len(chunks)

    def ingest_directory(self, directory: str) -> int:
        """
        Ingest all supported documents from a directory.

        Args:
            directory: Path to directory containing document files.

        Returns:
            Total number of chunks stored across all files.
        """
        logger.info(f"Ingesting directory: {directory}")

        total_chunks = 0
        for filename in sorted(os.listdir(directory)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                logger.debug(f"Skipping unsupported file: {filename}")
                continue

            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue

            total_chunks += self.ingest_file(file_path)

        logger.info(f"Directory ingestion complete: {total_chunks} total chunks")
        return total_chunks

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current vector store collection.

        Returns:
            Dict with collection_name, document_count, etc.
        """
        return self._get_collection_stats()

    # ------------------------------------------------------------------
    # Internal methods (isolated for mocking in tests)
    # ------------------------------------------------------------------

    def _read_file(self, file_path: str) -> str:
        """
        Read text content from a file.

        Supports:
        - .txt / .md: Direct text read
        - .pdf: Extracts text via PyPDF2 or falls back to basic read

        Isolated for mocking in tests.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in (".txt", ".md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                return ""

        if ext == ".pdf":
            return self._read_pdf(file_path)

        logger.warning(f"Unsupported file type: {ext}")
        return ""

    def _read_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Tries PyPDF2 first, falls back to basic binary read.
        Isolated for mocking in tests.
        """
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            logger.warning(
                "PyPDF2 not installed. Install with: pip install PyPDF2"
            )
            # Fallback: try reading as text
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading PDF {file_path}: {e}")
                return ""
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            return ""

    def _store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Store chunks into ChromaDB.

        Isolated for mocking in tests. In production, uses ChromaDB's
        Python client to upsert document chunks with embeddings.
        """
        try:
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=self.persist_dir)
            collection = client.get_or_create_collection(
                name=self.collection_name
            )

            # Prepare batch for ChromaDB
            ids = []
            documents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                source = chunk["metadata"].get("source", "unknown")
                chunk_idx = chunk["metadata"].get("chunk_index", i)
                doc_id = f"{source}__chunk_{chunk_idx}"

                ids.append(doc_id)
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])

            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                f"Upserted {len(ids)} chunks to collection "
                f"'{self.collection_name}'"
            )

        except ImportError:
            logger.warning(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

    def _get_collection_stats(self) -> Dict[str, Any]:
        """
        Get stats from the ChromaDB collection.

        Isolated for mocking in tests.
        """
        try:
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=self.persist_dir)
            collection = client.get_or_create_collection(
                name=self.collection_name
            )
            count = collection.count()

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_dir": self.persist_dir,
            }

        except ImportError:
            logger.warning("ChromaDB not installed.")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_dir": self.persist_dir,
                "error": "ChromaDB not installed",
            }
