from __future__ import annotations

"""
Gestion des prompts du pipeline.

Principe : chaque étape qui utilise un LLM a un prompt PAR DÉFAUT (défini dans
le code, en français). L'utilisateur peut le remplacer par le sien sans toucher
au code : il suffit de pointer un fichier texte dans la config (champs
*_file), ou d'éditer le texte dans l'interface Streamlit.

load_text() applique cette règle : si un fichier de remplacement existe, on
l'utilise ; sinon, on garde le défaut.
"""

from pathlib import Path
from typing import Optional


def load_text(default: str, override_path: Optional[str] = None) -> str:
    """
    Renvoie le prompt à utiliser :
      - le contenu du fichier `override_path` s'il est fourni et existe,
      - sinon le prompt `default` fourni par le package.
    (Pour l'instant, override_path est un chemin local ; la prise en charge des
    chemins S3 sera ajoutée quand on câblera l'écriture/lecture S3 des étapes.)
    """
    if override_path:
        p = Path(override_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
    return default
