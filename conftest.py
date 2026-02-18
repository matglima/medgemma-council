"""
Root-level conftest: adds src/ to the Python path so that test imports
like `from graph import CouncilState` resolve to `src/graph.py`.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
