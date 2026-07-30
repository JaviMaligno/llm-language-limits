"""Every live entrypoint must load THIS repo's .env explicitly.

The README states the invariant: repo .env overrides inherited shell variables, so a key
exported for another Azure subscription is never used by accident. The cipher entrypoints
were relying on ambient shell variables, which both breaks that guarantee and fails
outright in a clean shell (KeyError: AZURE_OPENAI_ENDPOINT).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = sorted(
    p for p in ROOT.glob("experiments/*/*.py")
    if p.name.startswith("run_") or p.name.endswith("_probe.py")
)


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


def test_entrypoints_were_discovered():
    assert len(ENTRYPOINTS) >= 7


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_entrypoint_loads_project_env(path: Path):
    assert "load_project_env" in _calls(path), (
        f"{path.relative_to(ROOT)} must call load_project_env() before building clients")
