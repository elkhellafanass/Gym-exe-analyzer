"""Classifieurs good/bad et validation leave-one-subject-out (LOSO).

Pourquoi leave-one-subject-out ? Chaque sujet a 6 vidéos (good/bad x 3 vues).
Si une vidéo d'un sujet est dans l'entraînement et une autre dans le test, le
modèle peut reconnaître la PERSONNE (morphologie, cadrage, caméra) au lieu de
la technique : le score serait trop optimiste (fuite). On retire donc, à
chaque pli, les 6 vidéos d'un sujet ; on entraîne sur les 24 autres ; on
prédit le sujet retiré. 25 plis, chaque vidéo est prédite une seule fois, par
un modèle qui n'a jamais vu cette personne.

Pourquoi un Pipeline scikit-learn ? L'imputation (valeurs manquantes) et la
normalisation (moyenne / écart-type) sont APPRISES : elles font partie du
modèle. Dans un Pipeline, elles sont recalculées sur l'entraînement de chaque
pli, jamais sur le sujet testé.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_LABELS = {"logreg": "Régression logistique", "random_forest": "Forêt aléatoire"}


def make_models(config: dict) -> dict[str, Callable[[], Pipeline]]:
    """Fabriques de modèles, hyperparamètres lus dans config.yaml (fixés d'avance).

    On renvoie des fonctions (et non des modèles) : chaque pli doit partir
    d'un modèle neuf.
    """
    ev = config["evaluation"]
    return {
        "logreg": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(**ev["logreg"])),
        "random_forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(**ev["random_forest"])),
    }


def loso_predict(table: pd.DataFrame, features: list[str], factory: Callable[[], Pipeline],
                 label_col: str = "label") -> tuple[np.ndarray, list[Pipeline]]:
    """Probabilité "bad" de chaque ligne, prédite par un modèle qui n'a jamais vu son sujet.

    table : une vue, une ligne par vidéo, colonnes subject + features + label.
    Renvoie (probabilités dans l'ordre de table, modèles des plis).
    """
    proba = np.full(len(table), np.nan)
    models = []
    subjects = table["subject"].to_numpy()
    X = table[features].to_numpy(dtype=float)
    y = table[label_col].to_numpy()
    for subject in sorted(set(subjects)):
        test = subjects == subject
        train = ~test
        # Garde-fou anti-fuite : le sujet testé n'est jamais dans l'entraînement
        assert subject not in set(subjects[train])
        model = factory().fit(X[train], y[train])
        proba[test] = model.predict_proba(X[test])[:, 1]
        models.append(model)
    return proba, models


def verdict(proba: float, uncertain_zone: tuple[float, float]) -> str:
    """'bad', 'good' ou 'incertain' : jamais une certitude au milieu."""
    low, high = uncertain_zone
    if np.isnan(proba):
        return "non analysable"
    if low <= proba <= high:
        return "incertain"
    return "bad" if proba > high else "good"
