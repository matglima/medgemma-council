# MedGemma-Council

Multi-Agent Clinical Decision Support System using MedGemma 1.5.

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Docker

```bash
docker build -t medgemma-council .
docker run --rm medgemma-council pytest
```
