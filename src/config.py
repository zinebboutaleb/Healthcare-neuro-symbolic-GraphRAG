"""Central configuration. Everything is overridable through environment variables / .env."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # optional dependency; .env is read from the project root regardless of the current directory
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)  # override=True: .env wins over stale shell variables
except ImportError:  # pragma: no cover
    print("WARNING: python-dotenv is not installed, so .env is ignored. Run: pip install python-dotenv")
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DOC_DIR = DATA_DIR / "documents"
CYPHER_DIR = ROOT / "cypher"
EVAL_DIR = ROOT / "evaluation"


def env(name: str, default=None):
    """Read an env var at call time (so tests can override it); empty string == unset."""
    value = os.getenv(name)
    return value if value not in (None, "") else default