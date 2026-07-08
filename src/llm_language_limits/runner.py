# src/llm_language_limits/runner.py
from __future__ import annotations
import concurrent.futures
import time
from typing import Callable
from .config import DEFAULT_MAX_WORKERS, MULTITURN_DEFAULT_CAP, ModelSpec, SYSTEM_PROMPT
from .stimuli import Stimulus
from .prompts import build_single_turn, build_multi_turn
from .clients.base import ModelClient
from . import metrics
from .judge import judge_response
from .storage import append_record, read_records, record_key


def run_cell(client: ModelClient, judge_client: ModelClient, spec: ModelSpec,
             stimulus: Stimulus, n: int, mode: str, replicate: int, *,
             temperature: float = 0.0, max_tokens: int = 256,
             multiturn_cap: int = MULTITURN_DEFAULT_CAP) -> dict:
    self_sim_last = None
    turns_run = None
    if mode == "single":
        msgs = build_single_turn(stimulus.text, n)
        res = client.chat(msgs, SYSTEM_PROMPT, temperature, max_tokens)
        final_text = res.text
    elif mode == "multi":
        turns = min(n, multiturn_cap)
        turns_run = turns
        replies: list[str] = []
        res = None
        for t in range(turns):
            msgs = build_multi_turn(stimulus.text, t + 1, prior_assistant=replies)
            res = client.chat(msgs, SYSTEM_PROMPT, temperature, max_tokens)
            replies.append(res.text)
        final_text = replies[-1]
        if len(replies) >= 2:
            self_sim_last = metrics.self_similarity(replies[-2], replies[-1])
    else:
        raise ValueError(f"unknown mode: {mode}")

    verdict = judge_response(judge_client, final_text)
    rec = {
        "model": spec.label, "category": stimulus.category, "n": n,
        "mode": mode, "replicate": replicate,
        "length": metrics.response_length_chars(final_text),
        "repetition_ratio": metrics.repetition_ratio(final_text),
        "entropy": metrics.token_entropy(final_text),
        "is_refusal": metrics.is_refusal(final_text),
        "judge_labels": verdict.labels,
        "judge_confidence": verdict.confidence,
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        "text": final_text,
    }
    if turns_run is not None:
        rec["turns_run"] = turns_run
    if self_sim_last is not None:
        rec["self_similarity_last"] = self_sim_last
    return rec


def _run_cell_with_retry(client, judge_client, spec, stimulus, n, mode, replicate,
                         *, max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        try:
            return run_cell(client, judge_client, spec, stimulus, n, mode, replicate)
        except Exception as e:  # noqa: BLE001 — keep the sweep alive on any provider error
            label = f"{spec.label}/{stimulus.category}/N={n}/{mode}/rep{replicate}"
            if attempt == max_attempts:
                print(f"[skip] {label} failed after {max_attempts} attempts: "
                      f"{type(e).__name__}: {e}")
                return None
            backoff = min(30, 2 ** attempt)
            print(f"[retry] {label} attempt {attempt} failed "
                  f"({type(e).__name__}); backoff {backoff}s")
            time.sleep(backoff)


def run_matrix(client_factory: Callable[[ModelSpec], ModelClient],
               judge_client: ModelClient, specs: list[ModelSpec],
               stimuli: list[Stimulus], n_grid: list[int], modes: list[str],
               replicates: int, out_path, *, resume: bool = True,
               max_workers: int = DEFAULT_MAX_WORKERS) -> None:
    done = {record_key(r) for r in read_records(out_path)} if resume else set()

    # One client per spec, reused across all of that spec's cells (SDK clients
    # are thread-safe, so sharing across worker threads is safe).
    clients = {spec.label: client_factory(spec) for spec in specs}

    pending: list[tuple] = []
    for spec in specs:
        for stim in stimuli:
            for n in n_grid:
                for mode in modes:
                    for rep in range(replicates):
                        key = (spec.label, stim.category, n, mode, rep)
                        if key in done:
                            continue
                        pending.append((spec, stim, n, mode, rep))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_run_cell_with_retry, clients[spec.label], judge_client,
                       spec, stim, n, mode, rep)
            for spec, stim, n, mode, rep in pending
        ]
        # Results are only ever written from this (main) thread, so appends to
        # the JSONL file stay serialized and the file never gets corrupted by
        # interleaved writes from worker threads.
        for future in concurrent.futures.as_completed(futures):
            rec = future.result()
            if rec is not None:
                append_record(out_path, rec)
