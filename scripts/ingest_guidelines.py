#!/usr/bin/env python3
"""
CLI script to ingest clinical guideline documents into ChromaDB.

Usage:
    python scripts/ingest_guidelines.py
    python scripts/ingest_guidelines.py --input-dir /path/to/guidelines
    python scripts/ingest_guidelines.py --chunk-size 512 --chunk-overlap 64
    python scripts/ingest_guidelines.py --collection cardiology_guidelines

This script reads PDF, TXT, and MD files from the input directory,
chunks them with configurable overlap, and indexes them into a ChromaDB
vector store for retrieval by specialist agents.
"""

import argparse
import logging
import os
import sys

# Add src/ to path so we can import tools.ingestion
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.ingestion import IngestionPipeline  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    """
    Parse command-line arguments for the ingestion script.

    Args:
        argv: Optional list of argument strings. Defaults to sys.argv[1:].

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Ingest clinical guideline documents into ChromaDB",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "data", "reference_docs"
        ),
        help="Directory containing guideline documents (default: data/reference_docs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "data", "vector_store"
        ),
        help="Directory for ChromaDB persistence (default: data/vector_store/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Maximum characters per chunk (default: 1024)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Overlap characters between adjacent chunks (default: 128)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="guidelines",
        help="ChromaDB collection name (default: guidelines)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


def run_ingestion(
    input_dir: str,
    output_dir: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    collection: str = "guidelines",
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        input_dir: Path to directory containing guideline documents.
        output_dir: Path to ChromaDB persistence directory.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        collection: ChromaDB collection name.

    Returns:
        Dict with ingestion stats.
    """
    logger.info(f"Starting ingestion pipeline")
    logger.info(f"  Input:      {input_dir}")
    logger.info(f"  Output:     {output_dir}")
    logger.info(f"  Chunk size: {chunk_size}")
    logger.info(f"  Overlap:    {chunk_overlap}")
    logger.info(f"  Collection: {collection}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    pipeline = IngestionPipeline(
        persist_dir=output_dir,
        collection_name=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    total_chunks = pipeline.ingest_directory(input_dir)
    stats = pipeline.get_stats()

    logger.info(f"Ingestion complete: {total_chunks} chunks indexed")
    logger.info(f"Collection stats: {stats}")

    return stats


def main():
    """CLI entry point."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = run_ingestion(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        collection=args.collection,
    )

    print(f"\nIngestion complete!")
    print(f"  Collection: {stats.get('collection_name', 'unknown')}")
    print(f"  Documents:  {stats.get('document_count', 0)}")
    print(f"  Store:      {stats.get('persist_dir', 'unknown')}")


if __name__ == "__main__":
    main()
