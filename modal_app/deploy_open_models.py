import modal

image = (
    modal.Image.debian_slim()
    .pip_install("transformers>=4.44", "torch>=2.3", "accelerate>=0.33")
)
app = modal.App("llm-language-limits-open")

MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": "A10G",
    "Qwen/Qwen2.5-7B": "A10G",
    "Qwen/Qwen2.5-72B-Instruct": "A100-80GB:2",
}


@app.cls(image=image, secrets=[modal.Secret.from_name("huggingface")],
         gpu="A10G", scaledown_window=300)
class Generator:
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        import os, torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok_kw = {"token": os.environ["HF_TOKEN"]}
        self.tok = AutoTokenizer.from_pretrained(self.model_id, **tok_kw)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto", **tok_kw)

    @modal.method()
    def chat(self, messages, system, temperature, max_tokens, return_hidden_states=False):
        import torch
        # Base models have no chat template; fall back to concatenation.
        if self.tok.chat_template:
            full = [{"role": "system", "content": system}, *messages]
            prompt = self.tok.apply_chat_template(full, tokenize=False,
                                                  add_generation_prompt=True)
        else:
            prompt = system + "\n" + "\n".join(m["content"] for m in messages) + "\n"
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        in_toks = int(inputs.input_ids.shape[1])
        do_sample = temperature > 0
        out = self.model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            output_hidden_states=return_hidden_states, return_dict_in_generate=True)
        seq = out.sequences[0][in_toks:]
        text = self.tok.decode(seq, skip_special_tokens=True)
        hidden = None
        if return_hidden_states:
            # last layer, last prompt token — proxy for the model's representation
            last = out.hidden_states[0][-1][0, -1, :]
            hidden = last.float().cpu().tolist()
        return {"text": text, "input_tokens": in_toks,
                "output_tokens": int(seq.shape[0]), "hidden_state_last": hidden}


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def chat_endpoint(payload: dict):
    gen = Generator(model_id=payload["model_id"])
    return gen.chat.remote(
        payload["messages"], payload["system"],
        payload.get("temperature", 0.0), payload.get("max_tokens", 256),
        payload.get("return_hidden_states", False))
