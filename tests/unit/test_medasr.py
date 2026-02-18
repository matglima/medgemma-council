"""
Tests for MedASR Tool (audio transcription wrapper).

TDD: Written BEFORE src/tools/medasr.py.
Per MASTER_PROMPT: MedASR provides speech-to-text for clinical audio input.
All heavy compute (model loading) must be mocked.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestMedASRTool:
    """Tests for the MedASR speech-to-text wrapper."""

    def test_init_creates_tool(self):
        """MedASRTool must initialize without loading a model."""
        from tools.medasr import MedASRTool

        tool = MedASRTool()
        assert tool is not None

    def test_transcribe_returns_text(self, tmp_path):
        """transcribe() must return a string transcription."""
        from tools.medasr import MedASRTool

        tool = MedASRTool()

        # Create a real temp file so os.path.exists passes
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"\x00" * 100)

        with patch.object(
            tool, "_run_pipeline", return_value="Patient reports chest pain since yesterday."
        ):
            result = tool.transcribe(str(audio_file))

        assert isinstance(result, str)
        assert "chest pain" in result

    def test_transcribe_handles_missing_file(self):
        """transcribe() must handle missing audio files gracefully."""
        from tools.medasr import MedASRTool

        tool = MedASRTool()
        result = tool.transcribe("/nonexistent/file.wav")
        assert isinstance(result, str)
        assert "error" in result.lower() or result == ""

    def test_transcribe_returns_empty_for_silence(self, tmp_path):
        """transcribe() returns empty string for silent/empty audio."""
        from tools.medasr import MedASRTool

        tool = MedASRTool()

        # Create a real temp file so os.path.exists passes
        audio_file = tmp_path / "silence.wav"
        audio_file.write_bytes(b"\x00" * 100)

        with patch.object(tool, "_run_pipeline", return_value=""):
            result = tool.transcribe(str(audio_file))

        assert result == ""

    def test_supported_formats(self):
        """MedASRTool should declare supported audio formats."""
        from tools.medasr import MedASRTool

        tool = MedASRTool()
        assert hasattr(tool, "supported_formats")
        assert "wav" in tool.supported_formats
