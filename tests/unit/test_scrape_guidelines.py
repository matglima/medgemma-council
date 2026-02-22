"""Tests for guideline scraping bootstrap script.

TDD: written before scripts/scrape_guidelines.py implementation.
All network calls are mocked.
"""

from unittest.mock import MagicMock, patch


class TestScrapeGuidelinesScript:
    """Tests for scripts/scrape_guidelines.py."""

    def test_parse_args_defaults(self):
        """CLI parser should provide sensible defaults."""
        from scripts.scrape_guidelines import parse_args

        args = parse_args([])
        assert "reference_docs" in args.output_dir
        assert args.timeout == 20

    def test_slugify_produces_safe_filename(self):
        """slugify should normalize punctuation and spaces."""
        from scripts.scrape_guidelines import slugify

        assert slugify("AHA/ACC: Acute Coronary Syndrome") == "aha-acc-acute-coronary-syndrome"

    def test_scrape_sources_writes_markdown_files(self):
        """fetch_guidelines should write one markdown per source."""
        from scripts.scrape_guidelines import fetch_guidelines

        fake_sources = {
            "CardiologyAgent": [
                {
                    "title": "Acute Coronary Syndromes",
                    "url": "https://example.com/acs",
                }
            ]
        }

        fake_response = MagicMock()
        fake_response.text = "<html><body><h1>ACS Guideline</h1><p>Use ECG urgently.</p></body></html>"
        fake_response.raise_for_status.return_value = None
        fake_response.headers = {"Content-Type": "text/html"}

        with patch("scripts.scrape_guidelines.requests.get", return_value=fake_response):
            with patch("scripts.scrape_guidelines.open", create=True) as mock_open:
                with patch("scripts.scrape_guidelines.os.makedirs"):
                    stats = fetch_guidelines(
                        output_dir="/fake/reference_docs",
                        sources_by_specialist=fake_sources,
                        timeout=10,
                    )

        assert stats["written_files"] == 1
        assert stats["failed_sources"] == 0
        assert mock_open.called
