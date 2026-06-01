from __future__ import annotations

"""
API Projet — point d'entrée unique pour lancer le pipeline sur S3.

Flux avec validation humaine du label book :
    p = Project("configs/auteur_isi2026.yaml")
    p.build_label_book()    # 00->05 : décrit, clusterise, propose le label book (+ dendrogramme)
    #   ... un humain révise le label book et le dépose sur S3 (label_book_human.json) ...
    p.classify()            # 06 : classe avec le label book (humain si configuré)
    p.evaluate()            # 07-08 : accords + tables/figures, loggués dans MLflow

p.run_all() enchaîne build_label_book() + classify() (label book automatique).
"""

import os
import tempfile
from typing import Optional, Union

import pandas as pd

from . import s3io
from .config import ProjectConfig, load_config
from .paths import ProjectPaths


class Project:
    def __init__(self, config: Union[str, ProjectConfig], *, fs=None, llm_client=None):
        self.cfg = config if isinstance(config, ProjectConfig) else load_config(config)
        self.paths = ProjectPaths(self.cfg)
        self._fs = fs
        self._client = llm_client
        if self.cfg.cluster.mlflow.experiment in (None, "", "champslibres-clustering"):
            self.cfg.cluster.mlflow.experiment = self.cfg.project

    # ------------------------------------------------------------------ accès
    @property
    def fs(self):
        if self._fs is None:
            self._fs = s3io.make_s3_filesystem(self.cfg)
        return self._fs

    @property
    def client(self):
        if self._client is None:
            from .llm import get_llm_client
            self._client = get_llm_client(self.cfg.llm)
        return self._client

    def read_source(self) -> pd.DataFrame:
        return s3io.read_csv(self.fs, self.cfg.io.source_csv, dtype=str)

    def save_config_copy(self) -> str:
        import yaml
        return s3io.write_text(self.fs, yaml.safe_dump(self.cfg.model_dump(), allow_unicode=True,
                                                       sort_keys=False), self.paths.config_copy)

    def scaffold(self):
        """Crée l'arborescence du projet sur S3 (dossiers visibles + note de contenu),
        pour pouvoir y déposer le label book révisé avant même de lancer le pipeline."""
        notes = {
            "00_describe": "Statistiques descriptives des champs libres (nb de mots).",
            "01_gec": "Texte corrigé / normalisé.",
            "02_dedup": "Réponses dédupliquées + table de correspondance.",
            "03_embed": "Embeddings (cache) + identifiants.",
            "04_cluster": "clusters.csv, keywords.csv, representative.csv.",
            "05_label": ("label_book.json (généré). DÉPOSER ICI 'label_book_human.json' "
                         "(version révisée par un humain). labels.csv, dendrogramme.html."),
            "06_classify": "classifications.csv (sortie de la classification LLM).",
            "07_eval": "pairwise.csv, accords.csv, distribution_par_classe.csv/.png, fleiss.json.",
        }
        self.save_config_copy()
        for folder, note in notes.items():
            s3io.write_text(self.fs, note + "\n", f"{self.paths.root}/{folder}/_contenu.txt")
        print(f"Arborescence créée sous : {self.paths.root}/")
        print(f"-> Dépose le label book révisé ici : {self.paths.label_book_human}")
        return self.paths.root

    def _load_book(self):
        """Charge le label book à utiliser (humain si classf.label_book est défini,
        sinon celui généré automatiquement) et le normalise au format interne."""
        from .steps.step05_label import normalize_label_book
        path = self.cfg.classf.label_book or self.paths.label_book
        obj = s3io.read_json(self.fs, path)
        return normalize_label_book(obj, question=self.cfg.survey.question, context=self.cfg.survey.context)

    # ------------------------------------------------------------------ étape 00
    def describe(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        from .steps.step00_describe import describe_text
        df = self.read_source() if df is None else df
        desc = describe_text(df, self.cfg.columns.text)
        stem = self.cfg.io.source_csv.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        s3io.write_csv(self.fs, desc, self.paths.describe(stem))
        return desc

    # --------------------------------------------------- étapes 01-03 (+ cache)
    def preprocess_and_embed(self):
        from .steps.step01_gec import apply_gec
        from .steps.step02_dedup import dedup
        from .steps.step03_embed import embed_texts

        print("01  Correction + déduplication...")
        df = apply_gec(self.read_source(), self.cfg,
                       llm_client=(self.client if self.cfg.gec.mode == "llm" else None))
        s3io.write_csv(self.fs, df, self.paths.gec)
        dedup_df, mapping = dedup(df, self.cfg)
        s3io.write_csv(self.fs, dedup_df, self.paths.dedup)
        s3io.write_csv(self.fs, mapping, self.paths.dedup_mapping)
        print(f"    {len(df)} -> {len(dedup_df)} uniques")

        print("03  Embeddings...")
        emb = embed_texts(dedup_df[self.cfg.columns.clean].tolist(), self.cfg, llm_client=self.client)
        s3io.save_npy(self.fs, emb, self.paths.embeddings)
        s3io.write_csv(self.fs, dedup_df[[self.cfg.columns.id]], self.paths.embed_ids)
        print(f"    matrice {emb.shape}")
        return dedup_df, emb

    def load_cached(self):
        dedup_df = s3io.read_csv(self.fs, self.paths.dedup, dtype=str)
        emb = s3io.load_npy(self.fs, self.paths.embeddings)
        return dedup_df, emb

    # --------------------------------------------------- étapes 04-05
    def cluster_and_label(self, dedup_df: pd.DataFrame, emb):
        from .steps.step04_cluster import cluster_embeddings, log_to_mlflow
        from .steps.step05_label import cluster_code, run_label_book

        ids = dedup_df[self.cfg.columns.id].tolist()
        texts = dedup_df[self.cfg.columns.clean].tolist()

        print("04  Clustering...")
        res = cluster_embeddings(texts, emb, ids, self.cfg)
        print(f"    {res['n_clusters']} clusters, {res['n_noise']} 'bruit'")
        run_id = log_to_mlflow(self.cfg, res, emb.shape[1])
        if run_id:
            print(f"    run MLflow : {run_id}")
        self._write_cluster_tables(res, ids, texts, cluster_code)

        print("05  Label book...")
        with tempfile.TemporaryDirectory() as tmp:
            stem = f"{tmp}/dendrogramme"
            book = run_label_book(res["keywords"], res["representative_docs"], self.cfg,
                                  llm_client=self.client, topic_model=res["topic_model"],
                                  docs=texts, dendro_path=stem)
            self._upload_dendrogram(stem)
        s3io.write_json(self.fs, book, self.paths.label_book)
        s3io.write_csv(self.fs, pd.DataFrame(self._collect_cxx(book)), self.paths.labels)
        return res, book

    def build_label_book(self):
        """Étapes 00->05 : prépare les données et produit le label book automatique."""
        self.save_config_copy()
        self.describe()
        dedup_df, emb = self.preprocess_and_embed()
        res, book = self.cluster_and_label(dedup_df, emb)
        print(f"\nLabel book : {self.paths.label_book}")
        print("-> Pour la suite avec un label book révisé : dépose-le sur "
              f"{self.paths.label_book_human} et renseigne classf.label_book dans la config.")
        return res, book

    # --------------------------------------------------- étape 06
    def classify(self):
        from .steps.step06_classify import run_classification

        dedup_df, _ = self.load_cached() if s3io.exists(self.fs, self.paths.embeddings) else (None, None)
        if dedup_df is None:
            dedup_df = s3io.read_csv(self.fs, self.paths.dedup, dtype=str)
        book = self._load_book()
        if not self.cfg.classf.model:
            self.cfg.classf.model = self.cfg.label.model
        mode = self.cfg.classf.mode

        print(f"06  Classification (mode '{mode}', label book : "
              f"{'humain' if self.cfg.classf.label_book else 'auto'})...")
        clusters = self._clusters_from_csv(dedup_df) if mode == "sans" else None
        classif = run_classification(dedup_df, book, self.cfg,
                                     llm_client=(self.client if mode == "llm" else None),
                                     clusters=clusters)
        s3io.write_csv(self.fs, classif, self.paths.classifications)
        print(f"    -> {self.paths.classifications}")
        return classif

    def run_all(self):
        self.build_label_book()
        return self.classify()

    def rerun_from_embeddings(self):
        dedup_df, emb = self.load_cached()
        res, book = self.cluster_and_label(dedup_df, emb)
        return self.classify()

    # --------------------------------------------------- étape 07-08
    def evaluate(self):
        from .steps.step07_eval import classify_models, run_evaluation
        from .steps.step08_report import distribution_figure, distribution_par_classe, format_accords

        id_col = self.cfg.eval.id_column or self.cfg.columns.id
        text_col = self.cfg.eval.text_column or self.cfg.columns.text
        df = s3io.read_csv(self.fs, self.cfg.eval.annotated_csv, dtype=str)
        book = self._load_book()

        models = list(self.cfg.eval.models)
        if models:
            print(f"Classification du jeu annoté par {len(models)} modèle(s)...")
            preds = classify_models(df[text_col].tolist(), df[id_col].tolist(), book, self.cfg,
                                    llm_client=self.client, models=models)
            df = df.merge(preds, on=id_col, how="left")

        res = run_evaluation(df, book, self.cfg)

        # Tables + figure de restitution
        table = distribution_par_classe(res["table"], res["rater_cols"], res["display"])
        accords = format_accords(res["pairwise"], [res["display"][c] for c in res["human_cols"]])

        s3io.write_csv(self.fs, res["pairwise"], self.paths.eval_pairwise)
        s3io.write_csv(self.fs, res["by_type"], self.paths.eval_by_type)
        s3io.write_json(self.fs, res["fleiss"], self.paths.eval_fleiss)
        s3io.write_csv(self.fs, table.reset_index(), self.paths.eval_distribution)
        s3io.write_csv(self.fs, accords, self.paths.eval_accords)
        with tempfile.TemporaryDirectory() as tmp:
            fig = distribution_figure(table, f"{tmp}/dist.png")
            if fig:
                s3io.upload_file(self.fs, fig, self.paths.eval_distribution_fig)

        self._mlflow_log_eval(res, accords, table)
        print(f"Évaluation + tables/figure écrites sous : {self.paths.root}/07_eval/")
        return {**res, "distribution": table, "accords": accords}

    # ------------------------------------------------------------------ helpers
    def _clusters_from_csv(self, dedup_df):
        cl = s3io.read_csv(self.fs, self.paths.clusters, dtype=str)
        code2int = {str(i): (int(c[1:]) if isinstance(c, str) and c.startswith("C") else -1)
                    for i, c in zip(cl["id"], cl["cluster"])}
        return [code2int.get(str(i), -1) for i in dedup_df[self.cfg.columns.id]]

    def _write_cluster_tables(self, res, ids, texts, cluster_code):
        kw = pd.DataFrame([{"cluster": cluster_code(c), "n_docs": int((res["labels"] == c).sum()),
                            "keywords": ", ".join(w)} for c, w in res["keywords"].items()])
        s3io.write_csv(self.fs, kw, self.paths.keywords)
        cl = pd.DataFrame({"id": list(map(str, ids)), "text": texts,
                           "cluster": [cluster_code(c) if c != -1 else "bruit" for c in res["labels"]]})
        s3io.write_csv(self.fs, cl, self.paths.clusters)
        rep = pd.DataFrame([{"cluster": cluster_code(c), "id": i, "text": t}
                            for c, docs in res["representative_docs"].items() for (i, t) in docs])
        s3io.write_csv(self.fs, rep, self.paths.representative)

    def _upload_dendrogram(self, stem):
        for ext in (".html", ".png"):
            if os.path.exists(stem + ext):
                dest = self.paths.dendrogram.rsplit(".", 1)[0] + ext
                s3io.upload_file(self.fs, stem + ext, dest)
                print(f"    dendrogramme -> {dest}")
                return

    def _mlflow_log_eval(self, res, accords, table):
        if not self.cfg.mlflow.enabled:
            return
        try:
            import mlflow
        except ImportError:
            return
        uri = os.environ.get(self.cfg.mlflow.tracking_uri_env)
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.cfg.project)
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_params({"models": ",".join(self.cfg.eval.models) or "(aucun)",
                               "n_humains": len(res["human_cols"]),
                               "label_book": self.cfg.classf.label_book or "auto"})
            for _, r in res["pairwise"].iterrows():
                tag = f"{r['juge_a']}_{r['juge_b']}".replace(" ", "")[:80]
                if pd.notna(r["kappa"]):
                    mlflow.log_metric(f"kappa__{tag}", float(r["kappa"]))
                if pd.notna(r["Po"]):
                    mlflow.log_metric(f"Po__{tag}", float(r["Po"]))
            for k, v in res["fleiss"].items():
                if pd.notna(v):
                    mlflow.log_metric(f"fleiss_{k}", float(v))
            with tempfile.TemporaryDirectory() as tmp:
                accords.to_csv(f"{tmp}/accords.csv", index=False)
                table.reset_index().to_csv(f"{tmp}/distribution_par_classe.csv", index=False)
                from .steps.step08_report import distribution_figure
                distribution_figure(table, f"{tmp}/distribution_par_classe.png")
                for f in os.listdir(tmp):
                    mlflow.log_artifact(f"{tmp}/{f}")

    @staticmethod
    def _collect_cxx(book):
        rows = []
        for cat in book["categories"]:
            if cat["code"].startswith("C"):
                rows.append({"category_id": cat["code"], "title": cat["label"], "description": cat["description"]})
            for ch in cat.get("children", []):
                rows.append({"category_id": ch["category_id"], "title": ch.get("title", ""),
                             "description": ch.get("description", "")})
        return rows