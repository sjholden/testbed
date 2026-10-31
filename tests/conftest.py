import sys
from pathlib import Path

# Make the repository root (containing test.py) importable as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
