"""Freeze what a publication needs to pin: model identities, sampling settings, data digests.

The article drafts both say the same thing before publication: freeze exact model ids, API
dates, sampling settings, the cipher seed, and archive the JSONL with the commit that
generated it. The data files are gitignored (they contain raw model output, and for the
appendix they are derived from harmful prompts), so "archive" here means recording a
content digest and record count next to the commit — enough to prove later that a given
figure came from a given file, without committing the file.

Everything is read from the modules that actually produced the numbers rather than retyped,
so the frozen record cannot drift away from the code.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASETS = [
    "data/full.jsonl",              # part 1, repetition sweep
    "data/pilot.jsonl",
    "data/ciphers_pilot.jsonl",     # part 2, pilot (Claude refusal rates)
    "data/ciphers_full.jsonl",      # part 2, latency/difficulty matrix
    "data/jailbreak_results.jsonl", # appendix, labels only
    "data/jailbreak_rejudged.jsonl",
]

# Azure model versions as served at generation time (read from the portal/CLI, not derivable
# from code: the deployment name hides the underlying version).
DEPLOYMENT_VERSIONS = {
    "gpt-5.4": "2026-03-05",
    "gpt-4o-2 (judge 2)": "2024-08-06",
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_dataset(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"exists": False, "records": 0, "bytes": 0, "sha256": "",
                "last_modified_utc": ""}
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    out = {
        "exists": True,
        "records": len(lines),
        "bytes": path.stat().st_size,
        "sha256": file_digest(path),
        # when the generating calls actually ran — the "API date" a publication must pin
        "last_modified_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    versions: dict[str, int] = {}
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "label" in record:
            key = str(record.get("probe_version", 1))
            versions[key] = versions.get(key, 0) + 1
    if len(versions) > 1:      # only interesting when one file mixes probe versions
        out["by_probe_version"] = dict(sorted(versions.items()))
    return out


def _literal_from(module_path: Path, name: str):
    """Read a module-level literal without importing (keeps this script import-light)."""
    tree = ast.parse(Path(module_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {module_path}")


def _default_of(module_path: Path, function: str, argument: str):
    tree = ast.parse(Path(module_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            args = node.args
            names = [a.arg for a in args.args + args.kwonlyargs]
            defaults = list(args.defaults) + list(args.kw_defaults)
            pairs = dict(zip(names[-len(defaults):], defaults)) if defaults else {}
            if argument in pairs and pairs[argument] is not None:
                return ast.literal_eval(pairs[argument])
    raise KeyError(f"default for {argument} not found in {function}")


def sampling_parameters() -> dict:
    cipher_config = ROOT / "experiments" / "ciphers" / "config.py"
    runner = ROOT / "src" / "llm_language_limits" / "cipher_runner.py"
    ciphers = ROOT / "src" / "llm_language_limits" / "ciphers.py"
    probe = ROOT / "experiments" / "ciphers" / "jailbreak_probe.py"
    return {
        "temperature": _default_of(runner, "run_conversation", "temperature"),
        "max_tokens_cipher_turn": _default_of(runner, "run_conversation", "max_tokens"),
        "cipher_turn_cap": _literal_from(cipher_config, "TURN_CAP"),
        "cipher_replicates": _literal_from(cipher_config, "REPLICATES"),
        "cipher_explicit_every": _default_of(runner, "run_conversation", "explicit_every"),
        "cipher_fewshot_k": _default_of(runner, "run_conversation", "fewshot_k"),
        "cipher_seed": _default_of(ciphers, "make_ciphers", "seed"),
        "cipher_set": _literal_from(cipher_config, "CIPHER_SET"),
        "protocols": _literal_from(cipher_config, "PROTOCOLS"),
        "probe_ciphers": _literal_from(probe, "PROBE_CIPHERS"),
        "probe_version": _literal_from(probe, "PROBE_VERSION"),
    }


def model_identities() -> dict:
    from llm_language_limits.config import MODEL_REGISTRY
    return {label: {"id": spec.id, "provider": spec.provider.value,
                    "is_base": spec.is_base}
            for label, spec in MODEL_REGISTRY.items()}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout.strip()


def render(payload: dict) -> str:
    lines = [
        "# Publication freeze — provenance record",
        "",
        "Generated by `scripts/freeze_provenance.py`. Regenerate rather than editing by hand.",
        "",
        f"- **Commit**: `{payload['commit']}`",
        f"- **Generated (UTC)**: {payload['generated_utc']}",
        "",
        "## Datasets",
        "",
        "Data files are gitignored (raw model output; for the appendix, derived from harmful "
        "prompts). The digest below is what ties a published figure to a specific file.",
        "",
        "| File | Records | Last written (UTC) | sha256 |",
        "|---|---:|---|---|",
    ]
    for name, info in payload["datasets"].items():
        if not info["exists"]:
            lines.append(f"| `{name}` | — | — | _absent at freeze time_ |")
            continue
        count = str(info["records"])
        if info.get("by_probe_version"):
            breakdown = ", ".join(f"v{version}: {n}"
                                  for version, n in info["by_probe_version"].items())
            count += f" ({breakdown})"
        lines.append(f"| `{name}` | {count} | {info.get('last_modified_utc', '')} | "
                     f"`{info['sha256'][:32]}…` |")
    lines += ["", "## Model identities", "",
              "| Label | Provider id | Provider | Base model |", "|---|---|---|---|"]
    for label, info in sorted(payload["models"].items()):
        lines.append(f"| `{label}` | `{info['id']}` | {info['provider']} | "
                     f"{'yes' if info['is_base'] else 'no'} |")
    lines += ["", "### Served model versions (not derivable from the deployment name)", ""]
    for deployment, version in sorted(payload["deployments"].items()):
        lines.append(f"- `{deployment}` → {version}")
    lines += ["", "## Sampling", "", "```json",
              json.dumps(payload["sampling"], indent=2, sort_keys=True), "```", ""]
    if payload.get("notes"):
        lines += ["## Notes", ""] + [f"- {note}" for note in payload["notes"]] + [""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "PUBLICATION_FREEZE.md")
    args = ap.parse_args()

    payload = {
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets": {name: describe_dataset(ROOT / name) for name in DATASETS},
        "models": model_identities(),
        "sampling": sampling_parameters(),
        "deployments": DEPLOYMENT_VERSIONS,
        "notes": [
            "Part 1 (repetition) judge: `claude-sonnet-5`; the 286 records temporarily "
            "annotated by an Azure fallback were re-annotated with it before publication.",
            "Part 2 core metrics use no LLM judge: comprehension is oracle-checked and "
            "production is verified by inverse cipher.",
            "Appendix judges: primary `gpt-5` (also a subject), second `gpt-4o`. Every "
            "appendix row carries `judge_model`; labels from different judges must not be "
            "pooled without re-annotation.",
            "Appendix `probe_version` 1 records are an invalid measurement and are excluded "
            "by both the resume logic and the analysis.",
            "Keyed ciphers (random substitution, block permutation) derive their key from "
            "`cipher_seed`; a different seed yields different mappings and is a different "
            "experiment.",
            "Bootstrap/inference seed: 20260729 (`experiments/ciphers/analyze.py`), 20260714 "
            "(`experiments/repetition/infer.py`).",
            "base64 x gpt-5 is absent by gateway rejection (self-harm verdict on encoded "
            "text), not by sampling choice.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(payload))
    print(f"[freeze] {args.output.relative_to(ROOT)} written at commit {payload['commit'][:12]}")
    for name, info in payload["datasets"].items():
        state = f"{info['records']} records" if info["exists"] else "ABSENT"
        print(f"  {name:34} {state}")


if __name__ == "__main__":
    main()
