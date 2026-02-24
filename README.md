# MedGemma-Council

![MedGemma Council Demo](assets/Gemini_Generated_Image_we5i2bwe5i2bwe5i.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-483%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)

**A multi-agent clinical decision support system** for the Kaggle MedGemma Impact Challenge. A "Council of Experts" debates clinical cases via a LangGraph state machine, powered by MedGemma 1.5 4B by default (with optional 27B override for text).

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
                       │  (Sequential by  │
                       │  default; config │
                       │  via env var)    │
                       └────────┬─────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │  Safety Check    │
                      │ (Red Flag Scan)  │
                      └────────┬─────────┘
                       ┌───────┴────────┐
                  Red Flag           No Flag
                       │                │
                       ▼                ▼
              ┌────────────────┐ ┌──────────────────┐
              │   Emergency    │ │ Conflict Check   │◄────────────┐
              │  Synthesis     │ └────────┬─────────┘             │
              │ (Override Plan)│  ┌───────┴───────┐               │
              └───────┬────────┘ No Conflict  Conflict &          │
                      │              │        iter < 3            │
                      ▼              ▼             │              │
                     END      ┌────────────┐      ▼              │
                              │  Synthesis │ ┌───────────┐       │
                              │ (Final Plan│ │ Research  │       │
                              │   + Judge) │ │ (PubMed)  │       │
                              └─────┬──────┘ └─────┬─────┘       │
                                    │              ▼              │
                                    ▼        ┌───────────┐       │
                                   END       │  Debate   │───────┘
                                             └───────────┘
```

### Council Topology

The system implements a **9-node LangGraph StateGraph** with a conditional debate loop and safety override:

1. **Ingestion** -- Validates patient state, resets counters
2. **Supervisor Route** -- Analyzes case and selects relevant specialists
3. **Specialist** -- Runs activated specialist agents sequentially by default (configurable via `COUNCIL_MAX_WORKERS`)
4. **Safety Check** -- Scans all specialist outputs for red flags (suicide risk, sepsis, cardiac arrest, etc.)
5. **Emergency Synthesis** -- Produces emergency override plan when red flags detected (terminates graph)
6. **Conflict Check** -- Detects contradictions between specialist outputs
7. **Research** -- Fetches PubMed literature to resolve conflicts (conditional)
8. **Debate** -- Specialists critique each other using evidence (conditional)
9. **Synthesis** -- Produces final clinical management plan

The debate loop (steps 7-8) runs up to `MAX_DEBATE_ROUNDS=3` times before forcing synthesis. The safety override (step 5) short-circuits the entire pipeline when life-threatening emergencies are detected.

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
| **NeurologyAgent** | Neurology | AAN, AHA/ASA | Stroke scales (NIHSS), seizure management |
| **EndocrinologyAgent** | Endocrinology | ADA, Endocrine Society | Diabetes management, thyroid disorders |

---

## Safety Guardrails

Every agent output passes through multiple safety layers:

### Output-level Safety
1. **Red Flag Scanner** -- Detects 7 emergency patterns (suicide risk, sepsis, cardiac arrest, stroke, anaphylaxis, tension pneumothorax, status epilepticus). Triggers immediate emergency override.
2. **PII Redaction** -- Strips SSNs, phone numbers, emails, and MRNs before display.
3. **Clinical Disclaimer** -- Appended to all outputs.

### Graph-level Safety Override
The `safety_check` node scans all specialist outputs after they run. If any red flag is detected, the graph immediately routes to `emergency_synthesis`, bypassing the normal debate/synthesis flow. This ensures life-threatening emergencies are never delayed by the deliberation process.

---

## Evaluation Harness

### Clinical Benchmarks

The evaluation harness supports three standard medical QA benchmarks:

| Benchmark | Dataset | Size | Metric |
|-----------|---------|------|--------|
| **MedQA** | `GBaker/MedQA-USMLE-4-options` | 1.27k test | Accuracy |
| **PubMedQA** | `qiaojin/PubMedQA` (pqa_labeled) | 1k | Accuracy |
| **MedMCQA** | `openlifescienceai/medmcqa` | 194k (21 subjects) | Accuracy + per-specialty |

MedGemma published baselines: MedQA 89.8%, MedMCQA 74.2%, PubMedQA 76.8%, MMLU Med 87.0%.

```bash
# Run evaluation via CLI
python -m evaluation.runner --benchmark medqa --limit 100
python -m evaluation.runner --benchmark medmcqa --specialty Cardiology --limit 50
python -m evaluation.runner --benchmark pubmedqa --output results.json
```

### PMC-Patients Evaluation

Patient case evaluation using the PMC-Patients dataset with retrieval metrics and LLM-as-Judge scoring:

- **Dataset:** `zhengyun21/PMC-Patients` on HuggingFace
- **Retrieval Metrics:** Mean Reciprocal Rank (MRR), NDCG@k
- **LLM-as-Judge:** Automated clinical plan scoring (1-5 scale) for accuracy, completeness, safety, and evidence-based reasoning

```python
from evaluation.pmc_patients import load_pmc_patients, format_pmc_patient_prompt
from evaluation.retrieval_metrics import compute_mrr, compute_ndcg
from evaluation.llm_judge import LLMJudge, generate_judging_prompt

# Load patient cases
patients = load_pmc_patients(limit=100)
prompt = format_pmc_patient_prompt(patients[0])

# Evaluate retrieval quality
mrr = compute_mrr(retrieval_results)
ndcg = compute_ndcg(ranking_results, k=10)

# LLM-as-Judge evaluation
judge = LLMJudge(llm=my_llm)
score = judge.evaluate_plan(patient_context, clinical_plan)
batch_scores = judge.evaluate_batch(cases)
```

---

## RAG Ingestion Pipeline

Bootstrap specialty guideline documents from the web and ingest them into a ChromaDB vector store for retrieval-augmented generation:

```bash
# Scrape curated sources for each specialist and ingest in one step
python scripts/scrape_guidelines.py \
  --output-dir data/reference_docs/ \
  --vector-dir data/vector_store/ \
  --collection guidelines \
  --chunk-size 512 \
  --chunk-overlap 64
```

Manual ingestion (if documents already exist in `data/reference_docs/`):

```bash
# Ingest guidelines from a directory
python scripts/ingest_guidelines.py \
  --input-dir data/reference_docs/ \
  --output-dir data/vector_store/ \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --collection guidelines

# Custom collection name
python scripts/ingest_guidelines.py \
  --input-dir /path/to/guidelines/ \
  --collection cardiology_guidelines
```

`RAGTool` auto-bootstraps on first query: if the vector store is empty and reference docs exist, ingestion runs automatically.

The pipeline uses sliding-window chunking with configurable overlap and attaches source metadata (filename, chunk index) to each chunk for citation tracing.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph) |
| Text Inference | `transformers` + `bitsandbytes` (4-bit NF4 quantization) |
| Vision Inference | `transformers` (MedGemma 1.5 4B, bfloat16) |
| Model Parallelism | `device_map="auto"` across 2xT4 GPUs via `accelerate` |
| RAG | ChromaDB (auto-bootstrap + local ingestion) |
| Guideline Ingestion | Custom chunker + ChromaDB upsert |
| Literature Search | BioPython (`Bio.Entrez`) |
| Evaluation | MedQA, PubMedQA, MedMCQA, PMC-Patients, LLM-as-Judge |
| UI (Primary) | Gradio |
| UI (Alternative) | Streamlit |
| State Schema | Python TypedDict |
| Testing | pytest + unittest.mock (483 tests) |

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
│   │   │                        # Psychiatry, EmergencyMedicine,
│   │   │                        # Dermatology, Neurology, Endocrinology
│   │   └── supervisor.py        # SupervisorAgent (router + judge)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── bio_entrez.py        # PubMedTool
│   │   ├── rag_tool.py          # RAGTool (ChromaDB)
│   │   ├── medasr.py            # MedASRTool (speech-to-text)
│   │   └── ingestion.py         # GuidelineChunker + IngestionPipeline
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── safety.py            # Red flags, PII redaction, disclaimers
│   │   ├── model_loader.py      # VRAM management + model registry
│   │   ├── model_factory.py     # ModelFactory (mock/real mode switching)
│   │   ├── model_wrappers.py    # TextModelWrapper, VisionModelWrapper,
│   │   │                        # MockModelWrapper
│   │   └── quantization.py      # BitsAndBytes 4-bit config, GPU detection
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmarks.py        # MedQA, PubMedQA, MedMCQA loaders
│   │   ├── evaluator.py         # CouncilEvaluator (single + batch)
│   │   ├── metrics.py           # Accuracy, per-specialty, CI, reports
│   │   ├── runner.py            # CLI evaluation runner
│   │   ├── pmc_patients.py      # PMC-Patients dataset loader
│   │   ├── retrieval_metrics.py # MRR, NDCG@k
│   │   └── llm_judge.py         # LLM-as-Judge evaluator
│   └── graph.py                 # LangGraph 9-node state machine
├── scripts/
│   ├── __init__.py
│   ├── ingest_guidelines.py     # CLI for guideline ingestion
│   └── scrape_guidelines.py     # Scrape + ingest curated guideline sources
├── tests/
│   ├── conftest.py              # Global mock fixtures
│   ├── unit/                    # 26 unit test modules
│   └── integration/             # Graph flow + safety override tests
├── data/
│   ├── reference_docs/          # Scraped guideline docs (PDF/TXT/MD)
│   ├── vector_store/            # ChromaDB index (gitignored)
│   └── uploads/                 # Uploaded medical images
├── app.py                       # Streamlit UI
├── app_gradio.py                # Gradio UI
├── council_cli.py               # CLI interface
├── example_kaggle_notebook.ipynb # Complete Kaggle walkthrough
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
git clone https://github.com/matglima/medgemma-council.git
cd medgemma-council

# Install in development mode
pip install -e ".[dev]"

# Or install dependencies directly
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all 483 tests (< 2 seconds, no GPU needed)
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

# Optional: override to larger 27B text model (slower, higher VRAM)
python council_cli.py \
  --age 65 \
  --sex Male \
  --complaint "Chest pain" \
  --model-id google/medgemma-27b-text-it

# Quiet mode (suppress debug logging; verbose is default)
python council_cli.py \
  --age 65 \
  --sex Male \
  --complaint "Chest pain" \
  --quiet

# JSON output
python council_cli.py \
  --age 65 \
  --sex Male \
  --complaint "Chest pain" \
  --output json
```

#### Programmatic Usage

```python
from council_cli import run_council_cli

# Default: verbose=True (DEBUG-level logging), uses google/medgemma-1.5-4b-it
result = run_council_cli(
    age=65,
    sex="Male",
    chief_complaint="Chest pain radiating to left arm",
    history="Hypertension, Type 2 Diabetes",
    medications=["Aspirin", "Lisinopril"],
)

# Optional: override to 27B model on larger hardware
result = run_council_cli(
    age=65,
    sex="Male",
    chief_complaint="Chest pain",
    text_model_id="google/medgemma-27b-text-it",
)

# Quiet mode
result = run_council_cli(
    age=65,
    sex="Male",
    chief_complaint="Chest pain",
    verbose=False,
)
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

# CLI-first workflow (verbose debug logging on by default)
from council_cli import run_council_cli

result = run_council_cli(
    age=65,
    sex="Male",
    chief_complaint="Chest pain radiating to left arm",
    history="Hypertension, Type 2 Diabetes",
    medications=["Aspirin", "Lisinopril"],
)

print(result["final_plan"])

# Optional: override to 27B model (default already uses google/medgemma-1.5-4b-it)
result = run_council_cli(
    age=65,
    sex="Male",
    chief_complaint="Chest pain radiating to left arm",
    text_model_id="google/medgemma-27b-text-it",
)
```

You can also select the text model via environment variable:

```python
import os
os.environ["MEDGEMMA_TEXT_MODEL_ID"] = "google/medgemma-27b-text-it"
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
| `red_flag_detected` | `bool` | Whether a safety red flag was found |
| `emergency_override` | `str` | Emergency plan text (when red flag triggers) |

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

# Model management (with 4-bit quantization)
from utils.model_factory import ModelFactory
from utils.quantization import get_model_kwargs

factory = ModelFactory()  # defaults to mock mode
text_model = factory.create_text_model()
vision_model = factory.create_vision_model()

# For real models on Kaggle (set MEDGEMMA_USE_REAL_MODELS=true)
kwargs = get_model_kwargs()  # auto-detects GPUs, applies 4-bit NF4 config

# Evaluation
from evaluation.evaluator import CouncilEvaluator
from evaluation.llm_judge import LLMJudge

evaluator = CouncilEvaluator(graph=graph)
result = evaluator.evaluate_single(question, answer, prompt)
judge = LLMJudge(llm=text_model)
score = judge.evaluate_plan(patient_context, clinical_plan)

# Guideline ingestion
from tools.ingestion import GuidelineChunker, IngestionPipeline

pipeline = IngestionPipeline(persist_dir="data/vector_store/")
pipeline.ingest_directory("data/reference_docs/")

# Or bootstrap docs + ingest in one command
# !python scripts/scrape_guidelines.py --output-dir data/reference_docs/ --vector-dir data/vector_store/
```

---

## Hardware Requirements

### Kaggle (Dual T4 GPUs)

| Model | Format | VRAM | Placement |
|-------|--------|------|-----------|
| MedGemma 1.5 4B (default text/vision) | fp16/bfloat16 | ~8 GB | Single T4, loaded by default |
| MedGemma-27B (optional text override) | 4-bit NF4 via `bitsandbytes` | ~13.5 GB | Split across T4 #1 + T4 #2 (`device_map="auto"`) |

Quantization config:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # auto-selected: float16 for T4, bfloat16 for Ampere+
)
# Memory budget: max_memory={0: "14GiB", 1: "14GiB"}
```

> **Note:** The compute dtype is automatically overridden based on GPU compute capability. T4 GPUs (CC 7.5) use `float16` because bfloat16 4-bit dequantization produces inf/nan logits on hardware without native bfloat16 tensor cores. Ampere+ GPUs (CC >= 8.0) use `bfloat16`.

The `ModelFactory` class manages model creation with a feature flag (`MEDGEMMA_USE_REAL_MODELS` env var). In mock mode (default), no GPU is needed. In real mode, models are loaded with automatic tensor parallelism via `accelerate`.

**GPU Memory Management:** `ModelFactory` uses class-level model caching to prevent loading the same model multiple times during graph execution. Without caching, each graph node (supervisor, specialist, conflict_check, synthesis) would load a fresh model, exhausting VRAM. For `google/medgemma-1.5-4b-it`, text inference follows the official `pipeline("image-text-to-text")` path using chat-formatted messages. Legacy/larger text models still use `TextModelWrapper` with explicit tokenization and `generate()`.

**NaN Logits Stability Guard:** For quantized text inference, model loading now sets both `dtype` and `torch_dtype` in `from_pretrained()` kwargs for cross-version transformers compatibility, and forces `attn_implementation="eager"` for text models. This avoids silent dtype fallback and unstable fused attention kernel selection that can yield `NaN` logits and blank outputs on some Kaggle CUDA/transformers builds.

**Chat Template Formatting:** `TextModelWrapper` applies `tokenizer.apply_chat_template(tokenize=True, truncation=True)` to wrap prompts in the instruction-tuned model's expected format (e.g., `<start_of_turn>user`/`<start_of_turn>model` markers for Gemma 2 IT). Using `tokenize=True` in a single step ensures truncation preserves structural markers; the older two-step approach (tokenize=False then separate truncation) would cut off model-turn markers, causing empty outputs. Falls back gracefully to string-based tokenization, and then to raw prompt for tokenizers without chat template support.

**Vision Model Routing:** When `RadiologyAgent` is activated, `_run_specialists()` creates a separate `VisionModelWrapper` via `factory.create_vision_model()` and passes it specifically to RadiologyAgent. All other specialists receive the text model. The vision model is only loaded when RadiologyAgent is among the activated specialists, avoiding unnecessary model loading. The vision pipeline formats prompts as chat messages with `{"type": "image"}` entries (one per image) to satisfy MedGemma 4B IT's processor requirements.

**Verbose Logging:** All model wrappers, specialist agents, and supervisor methods include `logger.debug()` / `logger.info()` calls that trace: prompt lengths (chars), input token counts, template path used, RAG chunk counts, routing decisions, conflict results, and output previews (first 200 chars). Enable with `verbose=True` (the default) in `run_council_cli()`, or set `--quiet` on the CLI to suppress.

### Local Development

Tests run without any GPU -- all model calls are mocked. The full test suite (483 tests) completes in < 2 seconds.

---

## Specialist Execution

Specialist agents run via `ThreadPoolExecutor` with fault isolation -- if one specialist fails, the others still complete.

Configuration:
- Default `max_workers` = 1 (sequential execution to prevent CUDA OOM on limited-VRAM GPUs)
- Override via `COUNCIL_MAX_WORKERS` environment variable (e.g., set to 3 for parallelism on larger GPUs)
- Individual specialist timeouts produce error entries (not crashes)

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
