from __future__ import annotations

"""
Couche LLM du pipeline.

Exposé public :
  - LLMClient            : l'interface commune
  - get_llm_client(cfg)  : fabrique un client à partir de la config
  - list_models(...)     : liste pratique des modèles, SANS avoir besoin d'un
                           projet (utilisée par l'interface Streamlit pour
                           remplir les menus déroulants).
"""

from typing import Dict, List

from .base import LLMClient, Message
from .factory import get_llm_client
from .openai_client import OpenAICompatibleClient

__all__ = [
    "LLMClient",
    "Message",
    "get_llm_client",
    "OpenAICompatibleClient",
    "list_models",
]


def list_models(
    base_url: str = "https://llm.lab.sspcloud.fr/api",
    api_key_env: str = "LLM_LAB_API_KEY",
) -> Dict[str, List[str]]:
    """
    Renvoie les modèles disponibles sur l'API, triés en 'generation' /
    'embedding'. Ne nécessite pas de projet : utile pour l'écran d'accueil de
    l'interface.
    """
    from ..config import get_secret

    api_key = get_secret(api_key_env, required=True)
    client = OpenAICompatibleClient(base_url=base_url, api_key=api_key)
    return client.list_models()
