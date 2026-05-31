from __future__ import annotations

"""
Fabrique de client LLM : choisit la bonne implémentation selon la config.

C'est le seul endroit du code qui décide "llm.lab" vs "vllm". Le reste du
pipeline appelle get_llm_client(config.llm) et reçoit un objet qui respecte
l'interface LLMClient, sans se soucier du backend.
"""

from ..config import LLMConfig, get_secret
from .base import LLMClient
from .openai_client import OpenAICompatibleClient


def get_llm_client(llm_cfg: LLMConfig) -> LLMClient:
    backend = (llm_cfg.backend or "llm_lab").lower()

    if backend in ("llm_lab", "openai", "openai_compatible"):
        # La clé n'est jamais dans le YAML : on la lit dans la variable d'env.
        api_key = get_secret(llm_cfg.api_key_env, required=True)
        return OpenAICompatibleClient(
            base_url=llm_cfg.base_url,
            api_key=api_key,
            timeout_sec=llm_cfg.timeout_sec,
            max_concurrency=llm_cfg.max_concurrency,
        )

    if backend == "vllm":
        # Mode optionnel (modèles en local sur GPU). Sera ajouté plus tard via
        # l'extra [gpu]. En attendant, on guide l'utilisateur vers llm.lab.
        raise NotImplementedError(
            "Le backend 'vllm' (modèles en local sur GPU) est optionnel et n'est "
            "pas encore implémenté. Utilisez backend: llm_lab en attendant."
        )

    raise ValueError(
        f"Backend LLM inconnu : {backend!r} (attendu 'llm_lab' ou 'vllm')."
    )
