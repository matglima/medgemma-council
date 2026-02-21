"""
Global pytest fixtures for MedGemma-Council test suite.

CRITICAL: These fixtures mock all heavy compute (LLM inference, vision models)
so that tests never load real 27B/4B parameter models. Tests must run in <10s.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="session", autouse=True)
def force_mock_models():
    """
    Ensure tests NEVER load real models, even if MEDGEMMA_USE_REAL_MODELS=true
    is set in the environment (e.g., on Kaggle).

    This fixture runs automatically before all tests and forces mock mode.
    """
    # Save original value
    original = os.environ.get("MEDGEMMA_USE_REAL_MODELS", None)
    # Force mock mode for tests
    os.environ["MEDGEMMA_USE_REAL_MODELS"] = "false"
    yield
    # Restore original value after tests
    if original is not None:
        os.environ["MEDGEMMA_USE_REAL_MODELS"] = original
    else:
        os.environ.pop("MEDGEMMA_USE_REAL_MODELS", None)


@pytest.fixture
def mock_llm():
    """
    Simulates llama_cpp.Llama for MedGemma-27B text inference.
    Returns a deterministic string without loading GPU weights.
    """
    llm = MagicMock()
    llm.return_value = {
        "choices": [
            {
                "text": "Mocked clinical reasoning: Based on the patient presentation, "
                "the differential diagnosis includes acute coronary syndrome. "
                "Recommend ECG and troponin levels. (ACC/AHA Guidelines 2023)"
            }
        ]
    }
    # Also support the __call__ pattern used by llama-cpp-python
    llm.create_completion = MagicMock(
        return_value={
            "choices": [
                {
                    "text": "Mocked clinical reasoning: Based on the patient presentation, "
                    "the differential diagnosis includes acute coronary syndrome. "
                    "Recommend ECG and troponin levels. (ACC/AHA Guidelines 2023)"
                }
            ]
        }
    )
    return llm


@pytest.fixture
def mock_vision_model():
    """
    Simulates the MedGemma 1.5 4B multimodal pipeline (transformers).
    Accepts a dummy image tensor and returns a structured radiology report
    without importing transformers or loading any model weights.
    """
    pipeline = MagicMock()
    pipeline.return_value = [
        {
            "generated_text": (
                "Findings: Opacity detected in right lower lobe consistent with "
                "pneumonia. No pleural effusion. Cardiac silhouette is normal. "
                "Impression: Right lower lobe pneumonia."
            )
        }
    ]
    return pipeline


@pytest.fixture
def sample_patient_context():
    """Provides a reusable sample patient context dictionary for tests."""
    return {
        "age": 65,
        "sex": "Male",
        "chief_complaint": "Chest pain radiating to left arm",
        "history": "Hypertension, Type 2 Diabetes, former smoker",
        "vitals": {
            "bp": "160/95",
            "hr": 92,
            "temp": 98.6,
            "spo2": 96,
        },
        "labs": {
            "troponin": 0.08,
            "bnp": 450,
            "creatinine": 1.4,
        },
        "medications": ["Metformin", "Lisinopril", "Aspirin"],
    }


@pytest.fixture
def sample_medical_images(tmp_path):
    """Creates temporary dummy image file paths for testing image pipelines."""
    image_paths = []
    for i in range(3):
        img_file = tmp_path / f"ct_slice_{i}.png"
        # Write minimal bytes so the file exists (not a real image)
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        image_paths.append(str(img_file))
    return image_paths
