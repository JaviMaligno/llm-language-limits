from __future__ import annotations
import os
import urllib.request
import json
from .base import ChatResult
from ..config import ModelSpec


class ModalClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.url = os.environ["MODAL_CHAT_URL"]  # printed by `modal deploy`

    def _post(self, messages, system, temperature, max_tokens, hidden):
        body = json.dumps({
            "model_id": self.spec.id, "messages": messages, "system": system,
            "temperature": temperature, "max_tokens": max_tokens,
            "return_hidden_states": hidden}).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def chat(self, messages, system, temperature, max_tokens) -> ChatResult:
        d = self._post(messages, system, temperature, max_tokens, False)
        return ChatResult(text=d["text"], input_tokens=d["input_tokens"],
                          output_tokens=d["output_tokens"])

    def chat_with_hidden(self, messages, system, temperature, max_tokens):
        d = self._post(messages, system, temperature, max_tokens, True)
        res = ChatResult(text=d["text"], input_tokens=d["input_tokens"],
                         output_tokens=d["output_tokens"])
        return res, d.get("hidden_state_last")
