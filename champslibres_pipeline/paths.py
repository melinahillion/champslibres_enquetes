from __future__ import annotations

"""
Plan de nommage des artefacts d'un projet sur S3.

Tout est rangé sous une racine par projet :

    {bucket}/{outputs_prefix}/{project}/
        config.yaml                      copie de la config utilisée (traçabilité)
        00_describe/    <input>_description.csv
        01_gec/         gec.csv
        02_dedup/       dedup.csv, mapping.csv
        03_embed/       embeddings.npy, embed_ids.csv
        04_cluster/     clusters.csv, keywords.csv, representative.csv
        05_label/       label_book.json, labels.csv, dendrogramme.html
        06_classify/    classifications.csv
        07_eval/        pairwise.csv, by_type.csv, fleiss.json

L'entrée brute (io.source_csv) et le jeu annoté (eval.annotated_csv) restent où
ils sont (chemins complets dans la config) ; on n'écrit que les artefacts ici.
"""

from .config import ProjectConfig


class ProjectPaths:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.root = f"{cfg.bucket}/{cfg.s3.outputs_prefix}/{cfg.project}"

    def _p(self, *parts: str) -> str:
        return "/".join([self.root, *parts])

    # traçabilité
    @property
    def config_copy(self) -> str: return self._p("config.yaml")

    # par étape
    def describe(self, stem: str) -> str: return self._p("00_describe", f"{stem}_description.csv")
    @property
    def gec(self) -> str: return self._p("01_gec", "gec.csv")
    @property
    def dedup(self) -> str: return self._p("02_dedup", "dedup.csv")
    @property
    def dedup_mapping(self) -> str: return self._p("02_dedup", "mapping.csv")
    @property
    def embeddings(self) -> str: return self._p("03_embed", "embeddings.npy")
    @property
    def embed_ids(self) -> str: return self._p("03_embed", "embed_ids.csv")
    @property
    def clusters(self) -> str: return self._p("04_cluster", "clusters.csv")
    @property
    def keywords(self) -> str: return self._p("04_cluster", "keywords.csv")
    @property
    def representative(self) -> str: return self._p("04_cluster", "representative.csv")
    @property
    def label_book(self) -> str: return self._p("05_label", "label_book.json")
    @property
    def label_book_human(self) -> str: return self._p("05_label", "label_book_human.json")
    @property
    def labels(self) -> str: return self._p("05_label", "labels.csv")
    @property
    def dendrogram(self) -> str: return self._p("05_label", "dendrogramme.html")
    @property
    def classifications(self) -> str: return self._p("06_classify", "classifications.csv")
    @property
    def eval_pairwise(self) -> str: return self._p("07_eval", "pairwise.csv")
    @property
    def eval_by_type(self) -> str: return self._p("07_eval", "by_type.csv")
    @property
    def eval_fleiss(self) -> str: return self._p("07_eval", "fleiss.json")
    @property
    def eval_distribution(self) -> str: return self._p("07_eval", "distribution_par_classe.csv")
    @property
    def eval_distribution_fig(self) -> str: return self._p("07_eval", "distribution_par_classe.png")
    @property
    def eval_accords(self) -> str: return self._p("07_eval", "accords.csv")
