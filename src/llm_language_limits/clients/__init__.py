from __future__ import annotations
from ..config import ModelSpec, Provider
from .base import ModelClient, ChatResult, FakeClient


def get_client(spec: ModelSpec) -> ModelClient:
    if spec.label.startswith("fake"):
        return FakeClient()
    if spec.provider is Provider.ANTHROPIC:
        from .anthropic_client import AnthropicClient
        return AnthropicClient(spec)
    if spec.provider is Provider.AZURE_OPENAI:
        from .azure_openai_client import AzureOpenAIClient
        return AzureOpenAIClient(spec)
    if spec.provider is Provider.MODAL:
        from .modal_client import ModalClient
        return ModalClient(spec)
    raise NotImplementedError(spec.provider)
