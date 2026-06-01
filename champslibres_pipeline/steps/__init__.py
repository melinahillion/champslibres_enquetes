from __future__ import annotations

"""
Sous-package des étapes du pipeline.

On importe les étapes au fur et à mesure de leur migration vers la nouvelle
architecture (config unifiée + couche LLM). Pour l'instant : étape 01.
"""

from . import step01_gec
from . import step02_dedup
from . import step03_embed
from . import step04_cluster
from . import step05_label

__all__ = [
    "step01_gec",
    "step02_dedup",
    "step03_embed",
    "step04_cluster",
    "step05_label",
]
