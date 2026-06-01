from __future__ import annotations

"""
Liste les modèles disponibles sur llm.lab (génération et embeddings).
À lancer :  uv run python scripts/list_models.py

Sert à remplir exemple.yaml : llm.default_model, label.model, classf.model,
eval.models, embed.model_name.
Nécessite LLM_LAB_API_KEY (dans Vault, injectée au lancement du service).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champslibres_pipeline.llm import list_models


def main():
    if not os.environ.get("LLM_LAB_API_KEY"):
        print("[!] LLM_LAB_API_KEY absente. Ajoute-la dans Vault puis relance.")
        return 1
    m = list_models()
    print("Modèles de GÉNÉRATION (pour label.model, classf.model, eval.models) :")
    for name in m.get("generation", []):
        print(f"  - {name}")
    print("\nModèles d'EMBEDDINGS (pour embed.model_name) :")
    for name in m.get("embedding", []):
        print(f"  - {name}")
    print("\nÀ recopier dans configs/exemple.yaml, par exemple :")
    gen = m.get("generation", [])
    if gen:
        print(f'  label:  {{ model: "{gen[0]}" }}')
        print(f'  eval:   {{ models: {gen[:2]} }}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
