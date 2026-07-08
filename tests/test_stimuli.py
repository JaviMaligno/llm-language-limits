from pathlib import Path
from llm_language_limits.stimuli import load_stimuli, EXPECTED_CATEGORIES

MANIFEST = Path(__file__).parent.parent / "experiments/repetition/stimuli.yaml"

def test_loads_all_nine_categories():
    stimuli = load_stimuli(MANIFEST)
    cats = {s.category for s in stimuli}
    assert cats == EXPECTED_CATEGORIES
    assert len(EXPECTED_CATEGORIES) == 9

def test_every_stimulus_has_nonempty_text():
    for s in load_stimuli(MANIFEST):
        assert s.text.strip()

def test_rejects_unknown_category(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nstimuli:\n  - category: nope\n    text: x\n    note: y\n")
    import pytest
    with pytest.raises(ValueError, match="unknown category"):
        load_stimuli(bad)
