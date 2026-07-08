import os

import modal

image = (
    modal.Image.debian_slim()
    .pip_install("transformers>=4.44", "torch>=2.3", "accelerate>=0.33",
                 "fastapi[standard]")  # fastapi required for @modal.fastapi_endpoint
)
app = modal.App("llm-language-limits-open")

MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": "A10G",
    "Qwen/Qwen2.5-7B": "A10G",
}
# The 72B needs 2xA100-80GB, which requires a Modal payment method. Enable it
# (both the registry entry and the Generator72B class below) only once that is
# set up, by deploying with MODAL_ENABLE_72B=1.
_ENABLE_72B = os.environ.get("MODAL_ENABLE_72B") == "1"
if _ENABLE_72B:
    MODELS["Qwen/Qwen2.5-72B-Instruct"] = "A100-80GB:2"


def _load(model_id):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_kw = {"token": os.environ["HF_TOKEN"]}
    tok = AutoTokenizer.from_pretrained(model_id, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", **tok_kw)
    return tok, model


def _chat(tok, model, messages, system, temperature, max_tokens, return_hidden_states=False):
    # Base models have no chat template; fall back to concatenation.
    if tok.chat_template:
        full = [{"role": "system", "content": system}, *messages]
        prompt = tok.apply_chat_template(full, tokenize=False,
                                          add_generation_prompt=True)
    else:
        prompt = system + "\n" + "\n".join(m["content"] for m in messages) + "\n"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    in_toks = int(inputs.input_ids.shape[1])
    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "output_hidden_states": return_hidden_states,
        "return_dict_in_generate": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    out = model.generate(**inputs, **generate_kwargs)
    seq = out.sequences[0][in_toks:]
    text = tok.decode(seq, skip_special_tokens=True)
    hidden = None
    if return_hidden_states:
        # last layer, last prompt token — proxy for the model's representation
        last = out.hidden_states[0][-1][0, -1, :]
        hidden = last.float().cpu().tolist()
    return {"text": text, "input_tokens": in_toks,
            "output_tokens": int(seq.shape[0]), "hidden_state_last": hidden}


@app.cls(image=image, secrets=[modal.Secret.from_name("huggingface")],
         gpu="A10G", scaledown_window=300)
class Generator7B:
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        self.tok, self.model = _load(self.model_id)

    @modal.method()
    def chat(self, messages, system, temperature, max_tokens, return_hidden_states=False):
        return _chat(self.tok, self.model, messages, system, temperature,
                     max_tokens, return_hidden_states)


if _ENABLE_72B:
    @app.cls(image=image, secrets=[modal.Secret.from_name("huggingface")],
             gpu="A100-80GB:2", scaledown_window=300)
    class Generator72B:
        model_id: str = modal.parameter()

        @modal.enter()
        def load(self):
            self.tok, self.model = _load(self.model_id)

        @modal.method()
        def chat(self, messages, system, temperature, max_tokens, return_hidden_states=False):
            return _chat(self.tok, self.model, messages, system, temperature,
                         max_tokens, return_hidden_states)


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def chat_endpoint(payload: dict):
    model_id = payload["model_id"]
    if model_id not in MODELS:
        raise ValueError(f"unknown model_id {model_id!r}; known: {list(MODELS)}")
    gpu = MODELS[model_id]
    cls = Generator7B if gpu == "A10G" else Generator72B
    gen = cls(model_id=model_id)
    return gen.chat.remote(
        payload["messages"], payload["system"],
        payload.get("temperature", 0.0), payload.get("max_tokens", 256),
        payload.get("return_hidden_states", False))
