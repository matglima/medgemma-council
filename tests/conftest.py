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
def mock_text_tokenizer():
    """
    Creates a properly configured mock tokenizer for TextModelWrapper tests.
    
    The mock simulates apply_chat_template returning a tensor (not BatchEncoding),
    with correct shape and device handling.
    """
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    
    # Create a mock tensor that behaves like a real tensor
    # Key: __contains__ should return False for "input_ids" (it's a tensor, not dict)
    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 10)
    mock_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
    mock_input_ids.to.return_value = mock_input_ids  # .to() returns self
    # Make "input_ids" in mock_input_ids return False
    mock_input_ids.__contains__ = MagicMock(return_value=False)
    
    mock_tokenizer.apply_chat_template.return_value = mock_input_ids
    mock_tokenizer.decode.return_value = "response text"
    
    return mock_tokenizer


@pytest.fixture
def mock_text_model():
    """
    Creates a properly configured mock model for TextModelWrapper tests.
    
    The mock simulates generate() returning output with correct shape.
    """
    mock_model = MagicMock()
    
    # Create mock output that has shape and slicing behavior
    mock_output = MagicMock()
    mock_output.shape = (1, 20)  # 10 input + 10 generated
    # output[0] returns something sliceable
    mock_full_seq = MagicMock()
    mock_full_seq.__getitem__ = MagicMock(return_value=MagicMock())
    mock_output.__getitem__ = MagicMock(return_value=mock_full_seq)
    
    mock_model.generate.return_value = mock_output
    
    return mock_model


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
