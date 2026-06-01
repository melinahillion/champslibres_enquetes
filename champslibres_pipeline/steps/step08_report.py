from __future__ import annotations

"""
Étape 08 — Tables et figures de restitution (proches de results.ipynb).

  - distribution_par_classe : % de séquences attribuées à chaque classe, par codeur
    (humains + modèles), chaque colonne sommant à 100 ;
  - distribution_figure      : diagramme en barres groupées (PNG) ;
  - format_accords           : table d'accords mise en forme "Po [IC95%]" / "κ [IC95%]".
"""

import re
from typing import Dict, List, Optional, Sequence

import pandas as pd


def _sort_key(code: str):
    if code == "OTHER":
        return (9, "", 9999)
    m = re.match(r"([A-Za-z]+)(\d+)", code)
    prefix = m.group(1) if m else code
    num = int(m.group(2)) if m else 0
    order = {"R": 0, "F": 1, "C": 2}.get(prefix[:1].upper(), 5)
    return (order, prefix, num)


def distribution_par_classe(df, rater_cols: Sequence[str],
                            display_names: Optional[Dict[str, str]] = None, decimals: int = 1) -> pd.DataFrame:
    series, classes = {}, set()
    for col in rater_cols:
        s = df[col].astype(str).str.strip()
        s = s[(s != "") & (s.str.lower() != "nan")]
        vc = s.value_counts()
        pct = (vc / vc.sum() * 100).round(decimals) if vc.sum() else vc
        series[col] = pct
        classes |= set(pct.index)
    classes = sorted(classes, key=_sort_key)
    out = pd.DataFrame(index=classes)
    out.index.name = "Classe attribuée"
    for col in rater_cols:
        out[(display_names or {}).get(col, col)] = series[col].reindex(classes).fillna(0.0)
    return out


def distribution_figure(table: pd.DataFrame, path: str, title: str = "Répartition par classe") -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("[report] matplotlib absent : figure non générée (table CSV seule).")
        return None
    classes = list(table.index)
    coders = list(table.columns)
    x = np.arange(len(classes))
    w = 0.8 / max(1, len(coders))
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.6), 5))
    for i, c in enumerate(coders):
        ax.bar(x + i * w, table[c].values, w, label=str(c))
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_ylabel("% des séquences")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _fr(x, decimals=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{decimals}f}".replace(".", ",")


def _ci(est, lo, hi, decimals=3):
    if est is None or (isinstance(est, float) and pd.isna(est)):
        return "—"
    s = _fr(est, decimals)
    if pd.isna(lo) or pd.isna(hi):
        return s
    return f"{s} [{_fr(lo, decimals)}–{_fr(hi, decimals)}]"


def format_accords(pairwise: pd.DataFrame, human_names: Sequence[str]) -> pd.DataFrame:
    human_set = set(human_names)
    rows = []
    for _, r in pairwise.iterrows():
        if r["type"] == "humain-modèle":
            modele = r["juge_b"] if r["juge_a"] in human_set else r["juge_a"]
        elif r["type"] == "modèle-modèle":
            modele = f"{r['juge_a']} ↔ {r['juge_b']}"
        else:
            modele = ""
        rows.append({
            "Type de binôme": r["type"], "Modèle": modele, "n": int(r["n"]),
            "Accord brut (Po) [IC95%]": _ci(r["Po"], r["Po_IC95_bas"], r["Po_IC95_haut"]),
            "Cohen's κ [IC95%]": _ci(r["kappa"], r["kappa_IC95_bas"], r["kappa_IC95_haut"]),
        })
    return pd.DataFrame(rows)
