from __future__ import annotations

"""
Chargement et validation de la configuration unifiée d'un projet.

Idée centrale : UN seul fichier YAML par projet décrit tout le pipeline.
Ce module le lit, vérifie qu'il est bien formé, remplit les valeurs par défaut
manquantes, et renvoie un objet Python pratique à manipuler (ProjectConfig).

Pourquoi Pydantic ? C'est une librairie qui valide les données. Si une valeur
est absente ou d'un mauvais type (ex. un texte là où on attend un nombre),
l'erreur est signalée TOUT DE SUITE, avec un message clair, plutôt que de
planter beaucoup plus tard au milieu du pipeline. Pour un débutant, c'est un
vrai filet de sécurité.

Règle de sécurité : ce fichier YAML ne contient JAMAIS de secret. Pour les
clés (S3, LLM), on n'y met que le NOM d'une variable d'environnement
(ex. "LLM_LAB_API_KEY"). La vraie valeur est lue à l'exécution via get_secret(),
à partir des variables injectées par le SSP Cloud.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Lecture des secrets (jamais stockés dans le YAML)
# ---------------------------------------------------------------------------
def get_secret(env_var_name: str, *, required: bool = True) -> Optional[str]:
    """
    Lit un secret depuis les variables d'environnement.
    On ne met que le NOM de la variable dans le YAML ; la vraie valeur est lue ici.
    """
    value = os.environ.get(env_var_name)
    if required and not value:
        raise EnvironmentError(
            f"Variable d'environnement manquante : {env_var_name!r}. "
            f"Sur le SSP Cloud, vérifiez qu'elle est bien injectée dans le service "
            f"(onglet 'Vault'/'Variables d'environnement' à la création du service)."
        )
    return value


# ---------------------------------------------------------------------------
# Sections de la configuration
# Chaque classe correspond à une section du YAML. Les valeurs par défaut
# permettent d'écrire un YAML minimal : tout ce qui n'est pas précisé prend
# la valeur par défaut indiquée ici.
# ---------------------------------------------------------------------------
class IOConfig(BaseModel):
    source_csv: str


class ColumnsConfig(BaseModel):
    id: str = "SEQ_ID"
    text: str = "text_brut"
    clean: str = "text_clean"


class S3Config(BaseModel):
    endpoint_env: str = "AWS_S3_ENDPOINT"


class LLMConfig(BaseModel):
    backend: str = "llm_lab"          # "llm_lab" (défaut) | "vllm" (GPU local, option)
    base_url: str = "https://llm.lab.sspcloud.fr/api"
    api_key_env: str = "LLM_LAB_API_KEY"   # NOM de la variable, pas la clé
    default_model: Optional[str] = None    # null => l'utilisateur choisit dans la liste
    max_concurrency: int = 8               # appels simultanés (plateforme mutualisée)
    timeout_sec: int = 120


class GECConfig(BaseModel):
    # Un seul réglage : le mode de traitement du texte.
    mode: str = "light"               # "none" (aucune modif) | "light" (normalisation) | "llm"
    model: Optional[str] = None       # modèle llm.lab si mode = "llm"
    temperature: float = 0.0
    batch_size: int = 32
    prompt_file: Optional[str] = None


class EmbedConfig(BaseModel):
    # 'model_name' commence par 'model_' : on désactive l'avertissement Pydantic.
    model_config = ConfigDict(protected_namespaces=())

    backend: str = "llm_lab"          # "llm_lab" (défaut) | "huggingface"
    model_name: Optional[str] = None  # null + llm_lab => 1er modèle d'embedding dispo
    batch_size: int = 32
    normalize: bool = True            # vecteurs L2-normalisés (recommandé pour le clustering)
    device: str = "cpu"               # backend huggingface : "cpu" | "cuda"


class UMAPConfig(BaseModel):
    n_neighbors: int = 15
    n_components: int = 10
    min_dist: float = 0.0
    metric: str = "cosine"
    random_state: int = 42


class HDBSCANConfig(BaseModel):
    # null -> auto : min_cluster_size = max(2, round(nb réponses uniques / 100))
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None   # null -> auto = min_cluster_size
    metric: str = "euclidean"


class ClusterMLflowConfig(BaseModel):
    enabled: bool = True
    experiment: str = "champslibres-clustering"


class ClusterConfig(BaseModel):
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = Field(default_factory=HDBSCANConfig)
    top_n_words: int = 10
    nr_representative_docs: int = 10
    language: str = "french"
    mlflow: ClusterMLflowConfig = Field(default_factory=ClusterMLflowConfig)


class GroupingConfig(BaseModel):
    # Regroupement des Cxx en super-catégories Fxx, quand AUCUNE modalité Rxx
    # n'est fournie. Trois méthodes au choix.
    method: str = "none"            # "none" | "dendrogram" | "llm"
    n_max: int = 10                 # nb max de super-catégories Fxx
    threshold: Optional[float] = None  # (dendrogram) seuil de distance ; si null -> n_max


class LabelConfig(BaseModel):
    mode: str = "llm"                 # "llm" (auto) | "human"
    model: Optional[str] = None
    temperature: float = 0.3
    prompt_subs_file: Optional[str] = None
    prompt_supers_file: Optional[str] = None
    prompt_mapping_file: Optional[str] = None
    prompt_group_file: Optional[str] = None
    grouping: GroupingConfig = Field(default_factory=GroupingConfig)
    human_label_book: Optional[str] = None


class ClassfConfig(BaseModel):
    mode: str = "llm"                 # "sans" | "llm" | "humain"
    model: Optional[str] = None
    n_iterations: int = 3             # nb de passages -> vote majoritaire
    temperature: float = 0.0
    batch_size: int = 20              # nb de réponses classées par appel LLM
    enforce_json_schema: bool = True
    prompt_file: Optional[str] = None
    human_column: str = "code_humain" # (mode "humain") colonne contenant les codes
    label_book: Optional[str] = None


class EvalConfig(BaseModel):
    # Jeu de données annoté : id, texte, + une colonne par annotateur humain.
    annotated_csv: Optional[str] = None
    id_column: Optional[str] = None        # défaut : columns.id
    text_column: Optional[str] = None      # défaut : columns.text
    # Noms des colonnes d'annotateurs humains (1..N), paramétrables.
    # Les humains annotent directement en super-catégories (Rxx et/ou Fxx).
    human_columns: List[str] = Field(default_factory=list)
    # Modèles llm.lab à comparer (1..M). Chaque modèle classe en Cxx, puis on
    # mappe vers Rxx/Fxx via le label book avant de comparer.
    models: List[str] = Field(default_factory=list)
    bootstrap_B: int = 2000
    random_state: int = 42


class MLflowConfig(BaseModel):
    enabled: bool = True
    tracking_uri_env: str = "MLFLOW_TRACKING_URI"


class SurveyConfig(BaseModel):
    # Contexte de l'enquête, utilisé pour guider le LLM (étape 05).
    question: str = ""                # la question à laquelle répondent les champs libres
    context: str = ""                 # contexte global de l'enquête (quelques phrases)
    # Modalités existantes (Rxx), OPTIONNELLES. Soit en clair ici, soit via un fichier JSON.
    # Format attendu d'un élément : {"code": "R01", "label": "...", "description": "..."}
    modalities: List[Dict[str, str]] = Field(default_factory=list)
    modalities_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Configuration complète d'un projet
# ---------------------------------------------------------------------------
class ProjectConfig(BaseModel):
    # extra="forbid" au niveau racine : si une SECTION est mal orthographiée
    # (ex. "clasf" au lieu de "classf"), on prévient l'utilisateur au lieu de
    # l'ignorer silencieusement.
    model_config = ConfigDict(extra="forbid")

    project: str
    bucket: str

    io: IOConfig
    survey: SurveyConfig = Field(default_factory=SurveyConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    s3: S3Config = Field(default_factory=S3Config)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    gec: GECConfig = Field(default_factory=GECConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    classf: ClassfConfig = Field(default_factory=ClassfConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


# ---------------------------------------------------------------------------
# Chargement / sauvegarde
# ---------------------------------------------------------------------------
def load_config(path: Union[str, Path]) -> ProjectConfig:
    """
    Lit un fichier YAML et le transforme en ProjectConfig validé.
    Lève une erreur claire si le fichier est introuvable ou mal formé.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {p}")
    with p.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    return ProjectConfig(**raw)


def dump_config(config: ProjectConfig, path: Union[str, Path]) -> None:
    """Écrit une ProjectConfig dans un fichier YAML lisible."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config.model_dump(), f, sort_keys=False, allow_unicode=True
        )
