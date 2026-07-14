# modal_app/run_sweep_cloud.py
# Option C: run the whole sweep ORCHESTRATOR inside Modal (cloud-side), so a run
# survives the local machine being closed/off. The orchestrator reuses the exact
# same run_matrix + clients as local; Qwen inference runs in a GPU class in the
# same Modal App, while Anthropic/Azure use their normal API clients.
# Data is written to a persistent Modal Volume and committed every 60s so
# progress is durable even if the run is interrupted.
#
# Secrets required in the active Modal account:
#   `sweep-env` — ANTHROPIC_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
#                 AZURE_OPENAI_API_VERSION, MODAL_CHAT_URL
#
# Launch (detached, survives Mac off):
#   modal run --detach modal_app/run_sweep_cloud.py --models qwen7b-base
# Download results afterwards:
#   modal volume get llm-lang-limits-data /full.jsonl ./data/full.jsonl --force
import sys
import threading

import modal

app = modal.App("llm-language-limits-sweep")
vol = modal.Volume.from_name("llm-lang-limits-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("llm-lang-limits-hf-cache", create_if_missing=True)


def _download_qwen_base():
    import os
    from huggingface_hub import snapshot_download

    snapshot_download(
        "Qwen/Qwen2.5-7B",
        token=os.environ["HF_TOKEN"],
        cache_dir="/root/.cache/huggingface",
    )

image = (
    modal.Image.debian_slim()
    .pip_install(
        "anthropic>=0.34", "openai>=1.40", "pydantic>=2.7", "pyyaml>=6.0",
        "numpy>=1.26", "pandas>=2.2", "pyarrow>=16.0",
    )
    .add_local_dir("src/llm_language_limits", remote_path="/root/llm_language_limits")
    .add_local_file("experiments/repetition/stimuli.yaml", remote_path="/root/stimuli.yaml")
)

gpu_image = (
    modal.Image.debian_slim()
    .pip_install("transformers>=4.44", "torch>=2.3", "accelerate>=0.33")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(
        _download_qwen_base,
        secrets=[modal.Secret.from_name("huggingface")],
        volumes={"/root/.cache/huggingface": hf_cache},
        timeout=3600,
    )
)

OUT = "/data/full.jsonl"


def _load_qwen(model_id):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", token=token,
    )
    return tokenizer, model


def _generate_qwen(tokenizer, model, messages, system, temperature, max_tokens):
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, *messages],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = system + "\n" + "\n".join(m["content"] for m in messages) + "\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = int(inputs.input_ids.shape[1])
    kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "return_dict_in_generate": True,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    output = model.generate(**inputs, **kwargs)
    sequence = output.sequences[0][input_tokens:]
    return {
        "text": tokenizer.decode(sequence, skip_special_tokens=True),
        "input_tokens": input_tokens,
        "output_tokens": int(sequence.shape[0]),
    }


@app.cls(
    image=gpu_image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/root/.cache/huggingface": hf_cache},
    gpu="A10G",
    scaledown_window=300,
    startup_timeout=900,
)
class SweepGenerator7B:
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        self.tokenizer, self.model = _load_qwen(self.model_id)

    @modal.method()
    def chat(self, messages, system, temperature, max_tokens):
        return _generate_qwen(
            self.tokenizer, self.model, messages, system, temperature, max_tokens,
        )


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("sweep-env")],
    timeout=86400,  # 24h
)
def sweep(models: list[str], single_n_grid: list[int], multi_n_grid: list[int],
          replicates: int, out_path: str = OUT, max_workers: int = 8,
          skip_judge: bool = False):
    sys.path.insert(0, "/root")
    import llm_language_limits.runner as runner_mod
    from llm_language_limits.config import MODEL_REGISTRY, Provider
    from llm_language_limits.stimuli import load_stimuli
    from llm_language_limits.clients import get_client
    from llm_language_limits.clients.base import ChatResult
    from llm_language_limits.storage import read_records, append_record as _append

    class _InClusterQwen:
        def __init__(self, spec):
            self.spec = spec
            self._g = SweepGenerator7B(model_id=spec.id)

        def chat(self, messages, system, temperature, max_tokens):
            d = self._g.chat.remote(messages, system, temperature, max_tokens)
            return ChatResult(text=d["text"], input_tokens=d["input_tokens"],
                              output_tokens=d["output_tokens"])

    def client_factory(spec):
        if spec.provider is Provider.MODAL:
            return _InClusterQwen(spec)
        return get_client(spec)

    done0 = len(read_records(out_path))
    print(f"[cloud-sweep] resume baseline: {done0} records at {out_path}", flush=True)

    # Commit the volume after EVERY new record so progress is durable and
    # externally visible, and LOG commit failures instead of swallowing them.
    # (A silent background committer previously left the volume looking empty.)
    state = {"n": 0}

    def _append_commit(path, rec):
        _append(path, rec)
        state["n"] += 1
        try:
            vol.commit()
        except Exception as e:  # surface, don't swallow
            print(f"[cloud-sweep] VOL COMMIT ERROR: {type(e).__name__}: {e}", flush=True)
        if state["n"] <= 3 or state["n"] % 10 == 0:
            print(f"[cloud-sweep] +{state['n']} new records committed "
                  f"(last: {rec.get('model')}/{rec.get('category')}/N{rec.get('n')}/{rec.get('mode')})",
                  flush=True)

    runner_mod.append_record = _append_commit  # run_matrix resolves this at call time

    specs = [MODEL_REGISTRY[m] for m in models]
    stimuli = load_stimuli("/root/stimuli.yaml")
    judge = None if skip_judge else get_client(MODEL_REGISTRY["claude-sonnet"])
    mode_grids = (("single", single_n_grid), ("multi", multi_n_grid))
    print(f"[cloud-sweep] models={models} mode_grids={mode_grids} reps={replicates}",
          flush=True)
    for mode, n_grid in mode_grids:
        if not n_grid:
            continue
        print(f"[cloud-sweep] starting {mode}-turn grid {n_grid}", flush=True)
        runner_mod.run_matrix(
            client_factory, judge, specs, stimuli, n_grid, [mode],
            replicates, out_path, resume=True, max_workers=max_workers,
        )

    vol.commit()
    n = len(read_records(out_path))
    print(f"[cloud-sweep] DONE — {n} total records ({n - done0} new)", flush=True)
    return n


@app.local_entrypoint()
def main(models: str = "qwen7b-base", quick: bool = False,
         skip_judge: bool = False):
    # models: comma-separated labels, e.g. "qwen7b-base" or "qwen7b-instruct,qwen7b-base"
    # quick: tiny grid for plumbing validation (single-turn, N in {1,3}, 1 rep).
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    single_n_grid = [1, 3] if quick else [1, 3, 10, 30, 100, 300, 1000]
    multi_n_grid = [] if quick else [1, 3, 10, 30, 100]
    reps = 1 if quick else 3
    if quick:
        # blocking, for local plumbing validation
        print(f"[cloud-sweep] returned "
              f"{sweep.remote(model_list, single_n_grid, multi_n_grid, reps, '/data/quick.jsonl', 1, skip_judge)} "
              "quick records")
    else:
        # fire-and-forget: the function runs on Modal fully independent of this
        # client, so closing/killing the local machine does NOT stop it.
        call = sweep.spawn(
            model_list, single_n_grid, multi_n_grid, reps, OUT, 8, skip_judge,
        )
        print(f"[cloud-sweep] spawned detached call id={call.object_id} — runs server-side; "
              f"safe to close the machine. Poll the volume for progress.")
