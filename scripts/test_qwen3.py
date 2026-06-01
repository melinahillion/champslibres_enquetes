# scripts/test_qwen3.py
import os, sys
sys.path.insert(0,".")
from champslibres_pipeline.llm import get_llm_client, list_models
from champslibres_pipeline.config import load_config

cfg = load_config("configs/auteur_isi2026.yaml")
client = get_llm_client(cfg.llm)

raw = client.complete(
    [{"role": "system", "content": "Réponds en JSON uniquement, sans explication."},
     {"role": "user",   "content":
      'Classe : "mon mari m\'a frappée". Codes possibles : R01, C31, C61, OTHER.\n'
      'Réponds UNIQUEMENT : {"assignments":[{"id":"1","code":"<code>"}]}'}],
    model="qwen3-6-35b-moe", temperature=0.0, max_tokens=2000)

print("=== RÉPONSE BRUTE ===")
print(repr(raw[:3000]))