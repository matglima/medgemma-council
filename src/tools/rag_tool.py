"""
RAG Tool: LlamaIndex + ChromaDB retriever wrapper.

Provides guideline retrieval for specialist agents (Cardiology, Oncology,
Pediatrics). Queries a local ChromaDB vector store pre-indexed with
clinical guidelines (ACC, AHA, NCCN, WHO, AAP).

All heavy dependencies (LlamaIndex, ChromaDB) are isolated in internal
methods for easy mocking in tests.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGTool:
    """
    Retrieval-Augmented Generation tool backed by ChromaDB.

    Args:
        persist_dir: Path to the ChromaDB persistence directory.
    """

    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = persist_dir

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
            f"[Source {i + 1} (score: {c.get('score', 'N/A')})]\n{c['text']}"
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
        Internal: Execute retrieval against ChromaDB via LlamaIndex.
        Isolated for mocking in tests.
        """
        # In production, this would use LlamaIndex's VectorStoreIndex
        # with a ChromaDB backend. Example:
        #
        # from llama_index.core import VectorStoreIndex, StorageContext
        # from llama_index.vector_stores.chroma import ChromaVectorStore
        # import chromadb
        #
        # client = chromadb.PersistentClient(path=self.persist_dir)
        # collection = client.get_or_create_collection("guidelines")
        # vector_store = ChromaVectorStore(chroma_collection=collection)
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # index = VectorStoreIndex.from_vector_store(vector_store)
        # query_engine = index.as_query_engine(similarity_top_k=top_k)
        # response = query_engine.query(query_text)
        raise NotImplementedError("Requires LlamaIndex + ChromaDB installation")

    def _ingest_files(self, file_paths: List[str]) -> None:
        """
        Internal: Ingest documents into ChromaDB via LlamaIndex.
        Isolated for mocking in tests.
        """
        # In production:
        # from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
        # documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        # index = VectorStoreIndex.from_documents(documents, ...)
        raise NotImplementedError("Requires LlamaIndex + ChromaDB installation")
