from __future__ import annotations

"""
Package champslibres_pipeline (socle).

Pour l'instant on n'expose que le socle (config + couche LLM). Les ré-exports
complets (Project, init, hist...) seront rebranchés au fil du refactoring des
étapes, quand on touchera à api.py.
"""

from .config import ProjectConfig, load_config
from .llm import get_llm_client, list_models

__all__ = [
    "ProjectConfig",
    "load_config",
    "get_llm_client",
    "list_models",
]

__version__ = "0.3.0"
