import sys
from pathlib import Path

# Ensure src directory is in sys.path
_SRC_DIR = str(Path(__file__).parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from src.evaluator import settings, get_settings

__all__ = ["settings", "get_settings"]
