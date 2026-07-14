from llm_language_limits.config import (
    DEFAULT_N_GRID, MODEL_REGISTRY, MULTITURN_DEFAULT_CAP,
    MULTITURN_MAX_TURNS, MULTI_TURN_N_GRID, Provider,
    SINGLE_TURN_N_GRID, models_for,
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

def test_full_mode_grids_keep_expensive_multiturn_cells_out():
    assert SINGLE_TURN_N_GRID == [1, 3, 10, 30, 100, 300, 1000]
    assert MULTI_TURN_N_GRID == [1, 3, 10, 30, 100]
    assert MULTITURN_DEFAULT_CAP == MULTITURN_MAX_TURNS == 100
