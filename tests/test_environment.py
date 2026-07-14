from __future__ import annotations

import os

from llm_language_limits import environment


def test_project_env_overrides_inherited_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("AZURE_OPENAI_API_KEY=repository-key\n")
    monkeypatch.setattr(environment, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "inherited-key")

    assert environment.load_project_env() is True
    assert os.environ["AZURE_OPENAI_API_KEY"] == "repository-key"
