# experiments/ciphers/sweep.py
from __future__ import annotations
import concurrent.futures
import time
from llm_language_limits.ciphers import CIPHERS
from llm_language_limits.oracle import TASK_BANK
from llm_language_limits.cipher_runner import run_conversation
from llm_language_limits.storage import append_record, read_records


def cell_key(rec: dict) -> tuple:
    return (rec["model"], rec["cipher"], rec["protocol"], rec["replicate"])


def _run_cell(client, spec, cipher_name, protocol, replicate, turn_cap):
    rec = run_conversation(client, CIPHERS[cipher_name], TASK_BANK, protocol,
                           turn_cap=turn_cap)
    rec["model"] = spec.label
    rec["replicate"] = replicate
    return rec


def _run_cell_with_retry(client, spec, cipher_name, protocol, replicate, turn_cap,
                         *, max_attempts=5):
    # Mirror part-1 runner._run_cell_with_retry: a whole multi-turn conversation is
    # expensive, so retry transient provider errors (429/5xx) with exponential backoff
    # rather than discarding the cell on the first hiccup. Returns None if it never
    # succeeds; the caller skips None so the sweep stays alive and resumable.
    for attempt in range(1, max_attempts + 1):
        try:
            return _run_cell(client, spec, cipher_name, protocol, replicate, turn_cap)
        except Exception as e:  # noqa: BLE001 — keep the sweep alive on any provider error
            label = f"{spec.label}/{cipher_name}/{protocol}/rep{replicate}"
            if attempt == max_attempts:
                print(f"[skip] {label} failed after {max_attempts} attempts: "
                      f"{type(e).__name__}: {e}")
                return None
            backoff = min(30, 2 ** attempt)
            print(f"[retry] {label} attempt {attempt} failed "
                  f"({type(e).__name__}: {e}); backoff {backoff}s")
            time.sleep(backoff)


def run_sweep(client_factory, specs, cipher_names, protocols, replicates, out_path,
              *, turn_cap, resume=True, max_workers=4):
    done = {cell_key(r) for r in read_records(out_path)} if resume else set()
    clients = {s.label: client_factory(s) for s in specs}
    pending = []
    for spec in specs:
        for cn in cipher_names:
            for proto in protocols:
                for rep in range(replicates):
                    if (spec.label, cn, proto, rep) in done:
                        continue
                    pending.append((spec, cn, proto, rep))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_run_cell_with_retry, clients[s.label], s, cn, proto, rep, turn_cap)
                for (s, cn, proto, rep) in pending]
        for fut in concurrent.futures.as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:  # keep the sweep alive (defensive; retry already swallows)
                print(f"[skip] cell failed: {type(e).__name__}: {e}")
                continue
            if rec is None:  # cell exhausted its retries
                continue
            append_record(out_path, rec)
