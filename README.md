# MedGemma-Council

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-115%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)

**A multi-agent clinical decision support system** for the Kaggle MedGemma Impact Challenge. A "Council of Experts" debates clinical cases via a LangGraph state machine, powered by MedGemma 1.5 models (4B multimodal + 27B text).

> **Disclaimer:** This is an AI research project. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## Architecture

```
                         ┌─────────────┐
                         │  Ingestion  │
                         │   (Validate │
                         │    State)   │
                         └──────┬──────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │ Supervisor Route │
                      │  (Select Agents) │
                      └────────┬─────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │   Specialists    │
                      │ Cardiology │ Onc │
                      │ Peds │ Rad │ EM  │
                      │ Psych │ Derm    │
                      └────────┬─────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │ Conflict Check   │◄─────────────────┐
                      └────────┬─────────┘                  │
                       ┌───────┴───────┐                    │
                  No Conflict     Conflict &                │
                       │          iter < 3                   │
                       ▼               │                    │
                ┌────────────┐         ▼                    │
                │  Synthesis │   ┌───────────┐              │
                │ (Final Plan│   │ Research  │              │
                │   + Judge) │   │ (PubMed)  │              │
                └────────────┘   └─────┬─────┘              │
                       │               ▼                    │
                       ▼         ┌───────────┐              │
                      END        │  Debate   │──────────────┘
                                 └───────────┘
```

### Council Topology

The system implements a **7-node LangGraph StateGraph** with a conditional debate loop:

1. **Ingestion** — Validates patient state, resets counters
2. **Supervisor Route** — Analyzes case and selects relevant specialists
3. **Specialist** — Runs activated specialist agents in parallel
4. **Conflict Check** — Detects contradictions between specialist outputs
5. **Research** — Fetches PubMed literature to resolve conflicts (conditional)
6. **Debate** — Specialists critique each other using evidence (conditional)
7. **Synthesis** — Produces final clinical management plan

The debate loop (steps 5-6) runs up to `MAX_DEBATE_ROUNDS=3` times before forcing synthesis.

---

## Specialist Agents

| Agent | Specialty | Knowledge Base | Key Competencies |
|-------|-----------|---------------|------------------|
| **CardiologyAgent** | Cardiology | ACC/AHA, ESC | ECG interpretation, ASCVD risk, CHA2DS2-VASc |
| **OncologyAgent** | Oncology | NCCN, ASCO | AJCC staging, RECIST, molecular profiling |
| **PediatricsAgent** | Pediatrics | AAP, WHO Pocket Book | Age-weight dosing, developmental milestones |
| **RadiologyAgent** | Radiology | ACR Appropriateness | Volumetric analysis, longitudinal comparison |
| **ResearchAgent** | Literature | PubMed/MEDLINE | PICO search, meta-analysis synthesis |
| **PsychiatryAgent** | Psychiatry | APA, DSM-5-TR | PHQ-9/GAD-7, suicide risk (C-SSRS) |
| **EmergencyMedicineAgent** | Emergency Medicine | ACLS, ATLS, EMTALA | Triage (ESI), trauma, sepsis bundles |
| **DermatologyAgent** | Dermatology | AAD, BAD | Lesion morphology, dermoscopy, ABCDE criteria |

---

## Safety Guardrails

Every agent output passes through three safety layers:

1. **Red Flag Scanner** — Detects 7 emergency patterns (suicide risk, sepsis, cardiac arrest, stroke, anaphylaxis, tension pneumothorax, status epilepticus). Triggers immediate emergency override.
2. **PII Redaction** — Strips SSNs, phone numbers, emails, and MRNs before display.
3. **Clinical Disclaimer** — Appended to all outputs.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph) |
| Text Inference | `llama-cpp-python` (MedGemma-27B Q4_K_M) |
| Vision Inference | `transformers` (MedGemma 1.5 4B) |
| RAG | LlamaIndex + ChromaDB |
| Literature Search | BioPython (`Bio.Entrez`) |
| UI (Primary) | Gradio |
| UI (Alternative) | Streamlit |
| State Schema | Python TypedDict |
| Testing | pytest + unittest.mock |

---

## Project Structure

```
medgemma-council/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAgent ABC
│   │   ├── radiology.py         # RadiologyAgent (vision)
│   │   ├── researcher.py        # ResearchAgent (PubMed)
│   │   ├── specialists.py       # Cardiology, Oncology, Pediatrics,
│   │   │                        # Psychiatry, EmergencyMedicine, Dermatology
│   │   └── supervisor.py        # SupervisorAgent (router + judge)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── bio_entrez.py        # PubMedTool
│   │   ├── rag_tool.py          # RAGTool (ChromaDB)
│   │   └── medasr.py            # MedASRTool (speech-to-text)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── safety.py            # Red flags, PII redaction, disclaimers
│   │   └── model_loader.py      # VRAM management
│   └── graph.py                 # LangGraph state machine + CouncilState
├── tests/
│   ├── conftest.py              # Global mock fixtures
│   ├── unit/                    # Unit tests for all modules
│   └── integration/             # End-to-end graph flow tests
├── app.py                       # Streamlit UI
├── app_gradio.py                # Gradio UI
├── council_cli.py               # CLI interface for Kaggle notebooks
├── example_kaggle_notebook.ipynb # Example Kaggle notebook
├── setup.py                     # Package configuration
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container build
└── README.md
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/medgemma-council.git
cd medgemma-council

# Install in development mode
pip install -e ".[dev]"

# Or install dependencies directly
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v
```

### CLI Usage (Recommended for Kaggle)

```bash
# Basic case analysis
python council_cli.py \
  --age 65 \
  --sex Male \
  --complaint "Chest pain radiating to left arm" \
  --history "Hypertension, Type 2 Diabetes" \
  --medications "Aspirin, Lisinopril"

# With medical images
python council_cli.py \
  --age 65 \
  --sex Male \
  --complaint "Chest pain radiating to left arm" \
  --images /path/to/xray.png /path/to/ct_scan.dcm

# JSON output
python council_cli.py \
  --age 65 \
  --sex Male \
  --complaint "Chest pain" \
  --output json
```

### Gradio UI

```bash
python app_gradio.py
# Opens at http://localhost:7860
```

### Streamlit UI

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Docker

```bash
# Build
docker build -t medgemma-council .

# Run tests
docker run --rm medgemma-council pytest tests/ -v

# Run Streamlit app
docker run -p 8501:8501 medgemma-council streamlit run app.py

# Run Gradio app
docker run -p 7860:7860 medgemma-council python app_gradio.py
```

---

## Kaggle Notebook Usage

```python
# Install the package
!pip install -e /kaggle/working/medgemma-council

# CLI-first workflow
from council_cli import run_council_cli

result = run_council_cli(
    age=65,
    sex="Male",
    chief_complaint="Chest pain radiating to left arm",
    history="Hypertension, Type 2 Diabetes",
    medications=["Aspirin", "Lisinopril"],
)

print(result["final_plan"])
```

See `example_kaggle_notebook.ipynb` for a complete walkthrough.

---

## API Reference

### CouncilState (TypedDict)

The shared state flowing through all graph nodes:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `List[Dict]` | LangChain-style message history |
| `patient_context` | `Dict` | Patient demographics, symptoms, vitals, labs |
| `medical_images` | `List[str]` | File paths to medical images |
| `agent_outputs` | `Dict[str, str]` | Agent name -> latest finding |
| `debate_history` | `List[str]` | Argument rounds log |
| `consensus_reached` | `bool` | Whether debate terminated with agreement |
| `research_findings` | `str` | PubMed literature summaries |
| `conflict_detected` | `bool` | Whether specialists disagree |
| `iteration_count` | `int` | Current debate round number |
| `final_plan` | `str` | Synthesized clinical management plan |

### Key Functions

```python
from graph import build_council_graph, CouncilState

# Build and run the council
graph = build_council_graph()
result = graph.invoke(initial_state)

# Safety utilities
from utils.safety import scan_for_red_flags, redact_pii, add_disclaimer

flags = scan_for_red_flags("Patient in cardiac arrest")
clean_text = redact_pii("SSN: 123-45-6789")
output = add_disclaimer("Clinical findings...")

# Model management
from utils.model_loader import ModelLoader

loader = ModelLoader()
loader.load_text_model("medgemma-27b", "/path/to/model.gguf", n_gpu_layers=40)
loader.swap_model("medgemma-27b", "medgemma-4b", "/path/to/vision", model_type="vision")
```

---

## Hardware Requirements

### Kaggle (Dual T4 GPUs)

| Model | VRAM | GPU |
|-------|------|-----|
| MedGemma-27B (Q4_K_M) | ~14 GB | T4 #1 |
| MedGemma 1.5 4B | ~8 GB | T4 #2 |

The `ModelLoader` class manages dynamic model swapping to stay within 16 GB VRAM per GPU. Models are loaded on-demand and unloaded after use.

### Local Development

Tests run without any GPU — all model calls are mocked. The full test suite completes in < 1 second.

---

## Contributing

1. **TDD Required:** Write failing tests before implementation code.
2. **Mock Heavy Compute:** Never load real models in tests.
3. **Atomic Commits:** Each commit should have a single, clear purpose.
4. **Safety First:** All outputs must pass through the safety pipeline.
5. **Citations Required:** Research agent must cite PMIDs. Clinical agents must cite specific guidelines.

---

## License

MIT
