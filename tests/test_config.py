from llm_language_limits.config import (
    MODEL_REGISTRY, DEFAULT_N_GRID, models_for, Provider,
)

def test_registry_has_expected_families():
    labels = set(MODEL_REGISTRY)
    assert {"claude-sonnet", "gpt-5", "qwen7b-instruct", "qwen7b-base"} <= labels

def test_base_model_flag_set():
    assert MODEL_REGISTRY["qwen7b-base"].is_base is True
    assert MODEL_REGISTRY["qwen7b-instruct"].is_base is False

def test_smoke_tier_is_cheap_and_small():
    smoke = models_for("smoke")
    assert len(smoke) == 1
    assert smoke[0].provider in {Provider.MODAL, Provider.AZURE_OPENAI}

def test_n_grid_is_ascending():
    assert DEFAULT_N_GRID == sorted(DEFAULT_N_GRID)
    assert DEFAULT_N_GRID[0] == 1
