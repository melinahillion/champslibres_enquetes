from __future__ import annotations

"""
Étape 05 — Construction du label book.

Sous-tâches LLM :
  (i)   labelliser chaque cluster Cxx -> intitulé + description
  (ii)  proposer des super-catégories Fxx        (si des Rxx sont fournies)
  (iii) rattacher chaque Cxx à {Rxx, Fxx, OTHER}  (si des Rxx sont fournies)

Regroupement des Cxx en Fxx SANS Rxx (label.grouping.method) :
  - "none"        : pas de regroupement ;
  - "dendrogram"  : on coupe le DENDROGRAMME NATIF de BERTopic
                    (topic_model.hierarchical_topics) par seuil de distance
                    (grouping.threshold) ou pour <= grouping.n_max groupes,
                    et on exporte la visualisation interactive (HTML) ;
  - "llm"         : regroupement par un LLM, avec un nb max de Fxx.

Le contexte de l'enquête (question, contexte, Rxx) est lu depuis cfg.survey.
"""

import ast
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import ProjectConfig
from ..llm.base import LLMClient
from ..prompts import load_text
from ..prompts.label import (
    GROUP_LLM_SYSTEM,
    GROUP_LLM_USER,
    LABEL_CLUSTER_SYSTEM,
    LABEL_CLUSTER_USER,
    MAPPING_SYSTEM,
    MAPPING_USER,
    SUPERCAT_SYSTEM,
    SUPERCAT_USER,
)


# ===========================================================================
# Utilitaires
# ===========================================================================
def _loads_loose(raw: str) -> Any:
    """Lecture JSON tolérante (texte/``` autour du JSON)."""
    text = (raw or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    raise ValueError("Aucun JSON valide trouvé dans la réponse du modèle.")


def cluster_code(cluster_id: int) -> str:
    return f"C{int(cluster_id):02d}"


def _code_to_int(code: str) -> int:
    return int(code[1:])


def _format_examples(examples: Sequence[Tuple[str, str]], max_n: int = 10) -> str:
    lines = [f"- {text}" for _id, text in list(examples)[:max_n]]
    return "\n".join(lines) if lines else "(aucun exemple)"


def _format_modalities(mods: Sequence[Dict[str, str]]) -> str:
    out = [f"- {m.get('code', '?')} {m.get('label', '')}: {m.get('description', '')}" for m in mods]
    return "\n".join(out) if out else "(aucune)"


def _format_cxx(labelled: Sequence[Dict[str, str]]) -> str:
    out = [f"- {c['category_id']} {c.get('title', '')}: {c.get('description', '')}" for c in labelled]
    return "\n".join(out) if out else "(aucune)"


def load_modalities(cfg: ProjectConfig) -> List[Dict[str, str]]:
    s = cfg.survey
    if s.modalities:
        return list(s.modalities)
    if s.modalities_file:
        p = Path(s.modalities_file)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return []


# ===========================================================================
# (i) Labellisation des clusters
# ===========================================================================
def _label_schema() -> Dict[str, Any]:
    return {
        "title": "label_cluster", "type": "object",
        "properties": {"category_id": {"type": "string"}, "title": {"type": "string"},
                       "description": {"type": "string"}},
        "required": ["category_id", "title", "description"],
    }


def label_clusters(keywords, representative_docs, cfg, *, llm_client, question="", context=""):
    if not cfg.label.model:
        raise ValueError("label.model doit être renseigné (modèle llm.lab) pour l'étape 05.")
    system = LABEL_CLUSTER_SYSTEM
    user_template = load_text(LABEL_CLUSTER_USER, cfg.label.prompt_subs_file)
    cluster_ids = sorted(keywords.keys())
    cids = [cluster_code(c) for c in cluster_ids]
    batch = []
    for c, cid in zip(cluster_ids, cids):
        user = user_template.format(
            context=context or "(non précisé)", question=question or "(non précisée)", cid=cid,
            examples=_format_examples(representative_docs.get(c, [])),
            keywords=", ".join(keywords.get(c, [])) or "(aucun)")
        batch.append([{"role": "system", "content": system}, {"role": "user", "content": user}])
    raws = llm_client.complete_batch(batch, model=cfg.label.model, temperature=cfg.label.temperature,
                                     max_tokens=512, json_schema=_label_schema())
    labels = []
    for raw, cid in zip(raws, cids):
        try:
            data = _loads_loose(raw)
            labels.append({"category_id": cid, "title": str(data.get("title", "")).strip(),
                           "description": str(data.get("description", "")).strip()})
        except Exception:
            labels.append({"category_id": cid, "title": "(à préciser)", "description": "Thèmes variés."})
    return labels


# ===========================================================================
# (ii) Super-catégories Fxx via Rxx
# ===========================================================================
def _supercat_schema():
    return {"title": "supercategories", "type": "object",
            "properties": {"supercategories": {"type": "array", "items": {"type": "object",
                "properties": {"super_cat": {"type": "string"}, "super_label": {"type": "string"},
                               "super_description": {"type": "string"}},
                "required": ["super_cat", "super_label", "super_description"]}}},
            "required": ["supercategories"]}


def propose_supercategories(existing_modalities, labelled_cxx, cfg, *, llm_client, question="", context=""):
    user = load_text(SUPERCAT_USER, cfg.label.prompt_supers_file).format(
        context=context or "(non précisé)", question=question or "(non précisée)",
        rxx=_format_modalities(existing_modalities), cxx=_format_cxx(labelled_cxx))
    raw = llm_client.complete([{"role": "system", "content": SUPERCAT_SYSTEM}, {"role": "user", "content": user}],
                              model=cfg.label.model, temperature=cfg.label.temperature, max_tokens=800,
                              json_schema=_supercat_schema())
    try:
        data = _loads_loose(raw)
        items = data.get("supercategories", []) if isinstance(data, dict) else data
        return [{"super_cat": str(s.get("super_cat", "")).strip(),
                 "super_label": str(s.get("super_label", "")).strip(),
                 "super_description": str(s.get("super_description", "")).strip()}
                for s in items if s.get("super_cat")]
    except Exception:
        return []


# ===========================================================================
# (iii) Rattachement Cxx -> {Rxx, Fxx, OTHER}
# ===========================================================================
def _mapping_schema():
    return {"title": "mappings", "type": "object",
            "properties": {"mappings": {"type": "array", "items": {"type": "object",
                "properties": {"sub_cat": {"type": "string"}, "super_cat": {"type": "string"}},
                "required": ["sub_cat", "super_cat"]}}},
            "required": ["mappings"]}


def map_subcategories(labelled_cxx, existing_modalities, supercats, cfg, *, llm_client, question="", context=""):
    fxx_as_mods = [{"code": f["super_cat"], "label": f.get("super_label", ""),
                    "description": f.get("super_description", "")} for f in supercats]
    user = load_text(MAPPING_USER, cfg.label.prompt_mapping_file).format(
        context=context or "(non précisé)", question=question or "(non précisée)",
        rxx=_format_modalities(existing_modalities), fxx=_format_modalities(fxx_as_mods),
        cxx=_format_cxx(labelled_cxx))
    raw = llm_client.complete([{"role": "system", "content": MAPPING_SYSTEM}, {"role": "user", "content": user}],
                              model=cfg.label.model, temperature=cfg.label.temperature,
                              max_tokens=256 + 32 * len(labelled_cxx), json_schema=_mapping_schema())
    valid = {m["code"] for m in existing_modalities} | {f["super_cat"] for f in supercats} | {"OTHER"}
    result: Dict[str, str] = {}
    try:
        data = _loads_loose(raw)
        items = data.get("mappings", []) if isinstance(data, dict) else data
        for it in items:
            sub = str(it.get("sub_cat", "")).strip()
            parent = str(it.get("super_cat", "")).strip()
            if sub:
                result[sub] = parent if parent in valid else "OTHER"
    except Exception:
        pass
    for c in labelled_cxx:
        result.setdefault(c["category_id"], "OTHER")
    return result


# ===========================================================================
# Regroupement SANS Rxx : par LLM
# ===========================================================================
def _group_llm_schema():
    return {"title": "supercategories", "type": "object",
            "properties": {"supercategories": {"type": "array", "items": {"type": "object",
                "properties": {"super_cat": {"type": "string"}, "super_label": {"type": "string"},
                               "super_description": {"type": "string"},
                               "members": {"type": "array", "items": {"type": "string"}}},
                "required": ["super_cat", "super_label", "members"]}}},
            "required": ["supercategories"]}


def group_by_llm(labelled_cxx, cfg, *, llm_client, question="", context=""):
    n_max = cfg.label.grouping.n_max
    system = GROUP_LLM_SYSTEM.format(n_max=n_max)
    user = load_text(GROUP_LLM_USER, cfg.label.prompt_group_file).format(
        context=context or "(non précisé)", question=question or "(non précisée)",
        cxx=_format_cxx(labelled_cxx), n_max=n_max)
    raw = llm_client.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                              model=cfg.label.model, temperature=cfg.label.temperature, max_tokens=1500,
                              json_schema=_group_llm_schema())
    supercats: List[Dict[str, str]] = []
    mapping: Dict[str, str] = {}
    try:
        data = _loads_loose(raw)
        items = data.get("supercategories", []) if isinstance(data, dict) else data
        for i, s in enumerate(items, start=1):
            fcode = str(s.get("super_cat") or f"F{i:02d}").strip()
            supercats.append({"super_cat": fcode, "super_label": str(s.get("super_label", "")).strip(),
                              "super_description": str(s.get("super_description", "")).strip()})
            for m in s.get("members", []):
                mapping[str(m).strip()] = fcode
    except Exception:
        pass
    for c in labelled_cxx:
        mapping.setdefault(c["category_id"], "OTHER")
    return supercats, mapping


# ===========================================================================
# Regroupement SANS Rxx : par le DENDROGRAMME NATIF de BERTopic
# (coupe par union-find, d'après l'approche éprouvée du script de référence)
# ===========================================================================
def _as_int_list(x) -> List[int]:
    if isinstance(x, list):
        return [int(i) for i in x]
    try:
        v = ast.literal_eval(str(x))
        return [int(i) for i in v] if isinstance(v, list) else []
    except Exception:
        return []


def hierarchy_to_merges(hier_df):
    """À partir du dataframe hierarchical_topics de BERTopic, renvoie
    (leaves, merges) où merges = [(distance, feuilles_gauche, feuilles_droite)]."""
    dfh = hier_df.copy()
    for c in ["Parent_ID", "Child_Left_ID", "Child_Right_ID"]:
        dfh[c] = dfh[c].astype(int)
    dfh["Distance"] = dfh["Distance"].astype(float)
    dfh["Topics"] = dfh["Topics"].apply(_as_int_list)
    node_leaves = {int(pid): {int(t) for t in ts if int(t) != -1}
                   for pid, ts in zip(dfh["Parent_ID"], dfh["Topics"])}
    leaves = sorted({t for s in node_leaves.values() for t in s})
    merges = []
    for r in dfh.itertuples(index=False):
        L = node_leaves.get(int(r.Child_Left_ID), {int(r.Child_Left_ID)})
        R = node_leaves.get(int(r.Child_Right_ID), {int(r.Child_Right_ID)})
        merges.append((float(r.Distance), L, R))
    merges.sort(key=lambda x: x[0])
    return leaves, merges


class _DSU:
    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}
        self.n_comp = len(items)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.n_comp -= 1
        return True


def cut_hierarchy(leaves, merges, *, threshold=None, n_max=None) -> Dict[int, int]:
    """Coupe l'arbre -> {topic_id: index_de_super_catégorie}.
    Par seuil de distance si threshold, sinon pour obtenir <= n_max groupes."""
    dsu = _DSU(leaves)
    if threshold is not None:
        for dist, L, R in merges:
            if dist > threshold:
                break
            if L and R:
                dsu.union(min(L), min(R))
    else:
        K = max(1, int(n_max or 1))
        for _dist, L, R in merges:
            if dsu.n_comp <= K:
                break
            if L and R:
                dsu.union(min(L), min(R))
    buckets = defaultdict(list)
    for t in leaves:
        buckets[dsu.find(t)].append(t)
    ordered = sorted(buckets, key=lambda r: (-len(buckets[r]), min(buckets[r])))
    root_to_super = {r: i for i, r in enumerate(ordered)}
    return {t: root_to_super[dsu.find(t)] for t in leaves}


def _save_dendrogram(topic_model, hier_df, base_path: str, labelled_cxx=None) -> Optional[str]:
    """Exporte le dendrogramme : PNG si possible (Chrome/kaleido), sinon HTML.
    Les feuilles sont étiquetées 'Cxx : titre' (au lieu des mots-clés TF-IDF)."""
    stem = base_path.rsplit(".", 1)[0] if base_path.endswith((".png", ".html")) else base_path
    use_custom = False
    if labelled_cxx:
        labels = {}
        for c in labelled_cxx:
            title = (c.get("title") or "").strip()
            text = f"{c['category_id']} : {title}" if title else c["category_id"]
            labels[_code_to_int(c["category_id"])] = text[:55]
        try:
            topic_model.set_topic_labels(labels)
            use_custom = True
        except Exception:  # noqa: BLE001
            use_custom = False
    try:
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hier_df, custom_labels=use_custom,
                                              width=1000, height=600)
    except TypeError:
        fig = topic_model.visualize_hierarchy(width=1000, height=600)
    try:
        fig.write_image(stem + ".png", width=1000, height=600, scale=2)  # nécessite Chrome
        print(f"[dendro] PNG -> {stem}.png")
        return stem + ".png"
    except Exception as e:  # noqa: BLE001
        try:
            fig.write_html(stem + ".html", include_plotlyjs="cdn")
            print(f"[dendro] PNG indisponible ({type(e).__name__}) -> HTML : {stem}.html")
            return stem + ".html"
        except Exception as e2:  # noqa: BLE001
            print(f"[dendro] visualisation non générée ({e2!r}).")
            return None


def _supertopic_desc(members: Sequence[str], keywords: Dict[int, List[str]], top_n: int = 8) -> str:
    """Description d'un Fxx à partir des mots-clés de ses Cxx enfants (sans doublon)."""
    terms: List[str] = []
    for m in members:
        for w in keywords.get(_code_to_int(m), []):
            if w not in terms:
                terms.append(w)
    top = terms[:top_n]
    return "Thème regroupé autour de : " + ", ".join(top) if top else ""


def group_by_dendrogram(topic_model, docs, labelled_cxx, keywords, cfg, *, dendro_path: Optional[str] = None):
    """Regroupe via le dendrogramme natif de BERTopic. Renvoie (supercats, mapping)
    et exporte la visualisation (PNG sinon HTML) si dendro_path est fourni."""
    if len(labelled_cxx) < 2:
        print(f"[dendro] {len(labelled_cxx)} cluster(s) : hiérarchie impossible, pas de regroupement.")
        return [], {}
    hier_df = topic_model.hierarchical_topics(list(docs))
    leaves, merges = hierarchy_to_merges(hier_df)
    g = cfg.label.grouping
    assign = cut_hierarchy(leaves, merges, threshold=g.threshold, n_max=g.n_max)

    if dendro_path:
        _save_dendrogram(topic_model, hier_df, dendro_path, labelled_cxx=labelled_cxx)

    by_super: "OrderedDict[int, List[str]]" = OrderedDict()
    for c in labelled_cxx:
        sidx = assign.get(_code_to_int(c["category_id"]), -1)
        by_super.setdefault(sidx, []).append(c["category_id"])

    supercats: List[Dict[str, str]] = []
    mapping: Dict[str, str] = {}
    for i, (_sidx, members) in enumerate(by_super.items(), start=1):
        fcode = f"F{i:02d}"
        supercats.append({"super_cat": fcode, "super_label": "",
                          "super_description": _supertopic_desc(members, keywords)})
        for m in members:
            mapping[m] = fcode
    return supercats, mapping


# ===========================================================================
# Assemblage + orchestrateur
# ===========================================================================
def build_label_book(question, labelled_cxx, existing_modalities=None, supercats=None,
                     mapping=None, context=""):
    rxx = list(existing_modalities or [])
    fxx = list(supercats or [])
    cxx_by_code = {c["category_id"]: c for c in labelled_cxx}
    categories: List[Dict[str, Any]] = []
    if rxx or fxx:
        children: Dict[str, List[Dict[str, str]]] = {}
        for sub, parent in (mapping or {}).items():
            c = cxx_by_code.get(sub)
            if c:
                children.setdefault(parent, []).append(c)
        for r in rxx:
            categories.append({"code": r["code"], "label": r.get("label", ""),
                               "description": r.get("description", ""), "source": "existing",
                               "children": children.get(r["code"], [])})
        for f in fxx:
            categories.append({"code": f["super_cat"], "label": f.get("super_label", ""),
                               "description": f.get("super_description", ""), "source": "generated",
                               "children": children.get(f["super_cat"], [])})
        categories.append({"code": "OTHER", "label": "Autre", "description": "Catégorie résiduelle.",
                           "source": "residual", "children": children.get("OTHER", [])})
    else:
        for c in labelled_cxx:
            categories.append({"code": c["category_id"], "label": c.get("title", ""),
                               "description": c.get("description", ""), "source": "cluster", "children": []})
        categories.append({"code": "OTHER", "label": "Autre", "description": "Catégorie résiduelle.",
                           "source": "residual", "children": []})
    return {"question": question, "context": context,
            "has_existing_modalities": bool(rxx), "categories": categories}


def run_label_book(keywords, representative_docs, cfg, *, llm_client,
                   topic_model=None, docs=None, dendro_path=None):
    """
    (i) labellisation, puis :
      - si des Rxx existent : (ii) Fxx + (iii) rattachement ;
      - sinon : selon cfg.label.grouping.method (none | dendrogram | llm).
    La méthode 'dendrogram' exige topic_model (BERTopic) + docs.
    """
    question = cfg.survey.question
    context = cfg.survey.context
    rxx = load_modalities(cfg)

    labelled = label_clusters(keywords, representative_docs, cfg, llm_client=llm_client,
                              question=question, context=context)
    supercats: Optional[List[Dict[str, str]]] = None
    mapping: Optional[Dict[str, str]] = None

    if rxx:
        supercats = propose_supercategories(rxx, labelled, cfg, llm_client=llm_client,
                                            question=question, context=context)
        mapping = map_subcategories(labelled, rxx, supercats, cfg, llm_client=llm_client,
                                    question=question, context=context)
    else:
        method = cfg.label.grouping.method
        if method == "dendrogram":
            if topic_model is None or docs is None:
                raise ValueError("grouping.method='dendrogram' : topic_model + docs requis.")
            supercats, mapping = group_by_dendrogram(topic_model, docs, labelled, keywords, cfg, dendro_path=dendro_path)
        elif method == "llm":
            supercats, mapping = group_by_llm(labelled, cfg, llm_client=llm_client,
                                              question=question, context=context)

    return build_label_book(question, labelled, existing_modalities=(rxx or None),
                            supercats=supercats, mapping=mapping, context=context)
