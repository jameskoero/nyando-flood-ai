"""
conftest.py  (place at repo ROOT, not inside tests/)
Adds src/ to sys.path so pytest can collect tests without ImportError.
"""
import sys
from pathlib import Path

# Make src/ importable from any test file
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
