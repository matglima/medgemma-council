"""
Root-level conftest: adds src/ and project root to the Python path so that
test imports like `from graph import CouncilState` resolve to `src/graph.py`
and `from scripts.ingest_guidelines import parse_args` resolves correctly.
"""

import sys
import os

# Add src/ so that `from graph import ...` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Add project root so that `from scripts.ingest_guidelines import ...` works
sys.path.insert(0, os.path.dirname(__file__))
