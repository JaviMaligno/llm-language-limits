# experiments/ciphers/sweep.py
from __future__ import annotations
import concurrent.futures
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
        futs = [pool.submit(_run_cell, clients[s.label], s, cn, proto, rep, turn_cap)
                for (s, cn, proto, rep) in pending]
        for fut in concurrent.futures.as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:  # keep the sweep alive
                print(f"[skip] cell failed: {type(e).__name__}: {e}")
                continue
            append_record(out_path, rec)
