from __future__ import annotations

"""
Étape 00 — Statistiques descriptives des champs libres (reprise de l'existant).
Compte de mots par réponse non vide : moyenne, min, 1er/9e déciles, max, effectif.
Pur pandas, aucun appel réseau.
"""

from typing import Optional

import numpy as np
import pandas as pd


def describe_text(df: pd.DataFrame, text_col: str, *, decimals: int = 1,
                  q_low: float = 0.10, q_high: float = 0.90) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Colonne '{text_col}' absente.")

    def _norm(x) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return " ".join(str(x).split())

    s = df[text_col].map(_norm)
    s = s[s != ""]
    wc = s.apply(lambda t: len(t.split()))
    if len(wc):
        row = {"nb_mots_moyen": round(float(wc.mean()), decimals),
               "nb_mots_min": int(wc.min()),
               "nb_mots_premier_decile": int(np.floor(wc.quantile(q_low))),
               "nb_mots_dernier_decile": int(np.ceil(wc.quantile(q_high))),
               "nb_mots_max": int(wc.max()),
               "nb_sequences_non_vides": int(len(wc))}
    else:
        row = {k: 0 for k in ["nb_mots_moyen", "nb_mots_min", "nb_mots_premier_decile",
                              "nb_mots_dernier_decile", "nb_mots_max", "nb_sequences_non_vides"]}
    return pd.DataFrame([row])
