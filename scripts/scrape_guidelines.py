#!/usr/bin/env python3
"""Download guideline reference pages and optionally ingest them into ChromaDB.

This script bootstraps `data/reference_docs/` with specialty-specific sources,
then (unless `--skip-ingest` is used) runs the ingestion pipeline so the RAG
retriever can immediately return chunks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import logging
import os
import re
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


DEFAULT_SOURCES_BY_SPECIALIST: Dict[str, List[Dict[str, str]]] = {
    "CardiologyAgent": [
        {
            "title": "NICE Acute Coronary Syndromes (NG185)",
            "url": "https://www.nice.org.uk/guidance/ng185",
        },
        {
            "title": "AHA Heart Attack Warning Signs",
            "url": "https://www.heart.org/en/health-topics/heart-attack/warning-signs-of-a-heart-attack",
        },
    ],
    "OncologyAgent": [
        {
            "title": "WHO Cancer Fact Sheet",
            "url": "https://www.who.int/news-room/fact-sheets/detail/cancer",
        },
        {
            "title": "NICE Suspected Cancer Recognition (NG12)",
            "url": "https://www.nice.org.uk/guidance/ng12",
        },
    ],
    "PediatricsAgent": [
        {
            "title": "WHO Pocket Book of Hospital Care for Children",
            "url": "https://www.who.int/publications/i/item/9789241548373",
        },
        {
            "title": "NICE Fever in Under 5s (NG143)",
            "url": "https://www.nice.org.uk/guidance/ng143",
        },
    ],
    "PsychiatryAgent": [
        {
            "title": "NICE Depression in Adults (NG222)",
            "url": "https://www.nice.org.uk/guidance/ng222",
        },
        {
            "title": "NICE Generalised Anxiety Disorder (CG113)",
            "url": "https://www.nice.org.uk/guidance/cg113",
        },
    ],
    "EmergencyMedicineAgent": [
        {
            "title": "NICE Sepsis (NG51)",
            "url": "https://www.nice.org.uk/guidance/ng51",
        },
        {
            "title": "Surviving Sepsis Campaign 2021",
            "url": "https://www.sccm.org/clinical-resources/guidelines/guidelines/surviving-sepsis-guidelines-2021",
        },
    ],
    "DermatologyAgent": [
        {
            "title": "NICE Melanoma Assessment and Management (NG14)",
            "url": "https://www.nice.org.uk/guidance/ng14",
        },
        {
            "title": "NICE Atopic Eczema in Under 12s (CG57)",
            "url": "https://www.nice.org.uk/guidance/cg57",
        },
    ],
    "NeurologyAgent": [
        {
            "title": "NICE Stroke and TIA in Over 16s (NG128)",
            "url": "https://www.nice.org.uk/guidance/ng128",
        },
        {
            "title": "AHA Stroke Warning Signs",
            "url": "https://www.stroke.org/en/about-stroke/stroke-symptoms",
        },
    ],
    "EndocrinologyAgent": [
        {
            "title": "NICE Type 2 Diabetes in Adults (NG28)",
            "url": "https://www.nice.org.uk/guidance/ng28",
        },
        {
            "title": "Endocrine Society Clinical Practice Guidelines",
            "url": "https://www.endocrine.org/clinical-practice-guidelines",
        },
    ],
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Scrape guideline pages and bootstrap RAG reference docs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "reference_docs"),
        help="Directory to write scraped markdown docs",
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "vector_store"),
        help="Directory for ChromaDB persistence",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="guidelines",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size for ingestion",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Chunk overlap for ingestion",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Only scrape docs; do not run ingestion",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args(argv)


def slugify(text: str) -> str:
    """Normalize title text into a safe filename slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "source"


def strip_html(raw_html: str) -> str:
    """Extract rough text content from an HTML payload."""
    text = re.sub(r"<script.*?>.*?</script>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_guidelines(
    output_dir: str,
    sources_by_specialist: Optional[Dict[str, List[Dict[str, str]]]] = None,
    timeout: int = 20,
) -> Dict[str, int]:
    """Fetch source pages and save one markdown file per source URL."""
    sources = sources_by_specialist or DEFAULT_SOURCES_BY_SPECIALIST
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    failed = 0

    for specialty, items in sources.items():
        specialty_slug = slugify(specialty.replace("Agent", ""))
        for item in items:
            title = item["title"]
            url = item["url"]

            filename = f"{specialty_slug}__{slugify(title)}.md"
            path = os.path.join(output_dir, filename)

            body = ""
            try:
                response = requests.get(
                    url,
                    timeout=timeout,
                    headers={"User-Agent": "medgemma-council-rag-bootstrap/1.0"},
                )
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "").lower()
                if "html" in content_type:
                    body = strip_html(response.text)
                else:
                    body = response.text

                if not body:
                    raise ValueError("Fetched response had no text body")

            except Exception as e:
                failed += 1
                logger.warning("Failed to fetch %s (%s): %s", title, url, e)
                body = f"Fetch failed for {url}: {e}"

            fetched_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            content = (
                f"# {title}\n\n"
                f"Specialty: {specialty}\n"
                f"Source URL: {url}\n"
                f"Fetched At: {fetched_at}\n\n"
                f"---\n\n"
                f"{body[:50000]}\n"
            )

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            written += 1
            logger.info("Wrote %s", path)

    return {
        "written_files": written,
        "failed_sources": failed,
    }


def ingest_guidelines(
    input_dir: str,
    vector_dir: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, int]:
    """Ingest scraped docs into ChromaDB."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from tools.ingestion import IngestionPipeline  # noqa: E402

    os.makedirs(vector_dir, exist_ok=True)
    pipeline = IngestionPipeline(
        persist_dir=vector_dir,
        collection_name=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    total_chunks = pipeline.ingest_directory(input_dir)
    stats = pipeline.get_stats()
    stats["total_chunks_ingested"] = total_chunks
    return stats


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    scrape_stats = fetch_guidelines(
        output_dir=args.output_dir,
        timeout=args.timeout,
    )
    print(
        f"Scraped guideline pages -> files: {scrape_stats['written_files']}, "
        f"failed: {scrape_stats['failed_sources']}"
    )

    if args.skip_ingest:
        return

    ingest_stats = ingest_guidelines(
        input_dir=args.output_dir,
        vector_dir=args.vector_dir,
        collection=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        f"Ingestion complete -> collection={ingest_stats.get('collection_name')}, "
        f"documents={ingest_stats.get('document_count')}, "
        f"chunks={ingest_stats.get('total_chunks_ingested')}"
    )


if __name__ == "__main__":
    main()
