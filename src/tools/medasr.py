"""
MedASR Tool: Speech-to-text wrapper for clinical audio input.

Wraps the MedASR model (or a general ASR pipeline via transformers)
for transcribing clinical dictations and patient audio into text.

The heavy model loading is isolated in _run_pipeline for mocking.
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class MedASRTool:
    """
    Audio transcription tool for clinical speech input.

    Supported formats: wav, mp3, flac, ogg.
    """

    def __init__(self) -> None:
        self.supported_formats: List[str] = ["wav", "mp3", "flac", "ogg"]

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Transcribed text string, or error message if file not found.
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return f"Error: Audio file not found at {audio_path}"

        ext = audio_path.rsplit(".", 1)[-1].lower() if "." in audio_path else ""
        if ext not in self.supported_formats:
            logger.warning(f"Unsupported audio format: {ext}")
            return f"Error: Unsupported audio format '{ext}'"

        try:
            return self._run_pipeline(audio_path)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Error: Transcription failed - {e}"

    def _run_pipeline(self, audio_path: str) -> str:
        """
        Internal: Run the ASR pipeline on an audio file.
        Isolated for mocking in tests.

        In production, this would use:
            from transformers import pipeline
            asr = pipeline("automatic-speech-recognition", model="google/medasr")
            result = asr(audio_path)
            return result["text"]
        """
        raise NotImplementedError("Requires transformers + MedASR model installation")
