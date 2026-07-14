"""Environment loading for local experiment entry points."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_env() -> bool:
    """Load this repository's .env, overriding inherited shell variables.

    Local experiment commands treat the repository .env as their explicit
    credential source.  This prevents an unrelated globally exported provider
    key from silently taking precedence.  Cloud code continues to use its
    managed environment because it does not call this helper.
    """
    return load_dotenv(PROJECT_ROOT / ".env", override=True)
