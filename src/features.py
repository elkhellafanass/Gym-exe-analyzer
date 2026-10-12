"""Vecteur d'indicateurs d'une vidéo (entrée du classifieur).

Pourquoi un module unique ? Le classifieur doit recevoir EXACTEMENT les mêmes
calculs à l'entraînement (scripts/train.py) et dans l'app. Si les deux
calculaient chacun de leur côté, une petite différence suffirait à fausser
toutes les prédictions sans que personne ne s'en rende compte.

Une vidéo -> une ligne : pour chaque règle applicable à la vue filmée, la
MÉDIANE sur les répétitions de la valeur mesurée (la même valeur que celle
comparée au seuil par rules.py), plus le rapport de tempo descente/remontée.
Pourquoi la médiane ? Une répétition mal détectée ne doit pas tout faire
basculer ; la médiane l'ignore, la moyenne non.

Les seuils ne sont PAS utilisés ici : le classifieur voit les valeurs
continues, pas les verdicts des règles (option A).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import Analysis, analyze_pose, load_or_extract
from src.rules import rule_reliability


def feature_names(view: str, rules: dict) -> list[str]:
    """Indicateurs utilisés pour une vue : règles applicables + tempo.

    Dérivé de config.yaml (champ views des règles) : profil -> profondeur,
    talons, tronc-tibia, tête ; face et diagonale -> valgus, bassin.
    """
    names = [name for name, rule in rules.items() if rule_reliability(rule, view)]
    return names + ["tempo_ratio"]


def video_features(analysis: Analysis, rules: dict) -> dict:
    """Médianes par vidéo + verdict des règles (pour le détecteur à seuils).

    n_reps = 0 -> indicateurs NaN : la vidéo est "non analysable".
    """
    table = analysis.rep_table()
    row = {"n_reps": len(analysis.reps)}
    for name in feature_names(analysis.view, rules):
        row[name] = float(np.nanmedian(table[name])) if (
            name in table and table[name].notna().any()) else np.nan
    for s in analysis.summary.itertuples():
        row[f"{s.rule}_flag"] = bool(s.video_error)
    return row


def build_feature_table(index: pd.DataFrame, config: dict, cache_dir: Path,
                        progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse toutes les vidéos de l'index.

    Renvoie (une ligne par vidéo, une ligne par répétition). Le tableau par
    répétition sert à l'analyse secondaire (option B : seuils recalibrés).
    """
    rules = config["exercises"]["squat"]["rules"]
    videos, reps = [], []
    for n, r in enumerate(index.itertuples(), 1):
        raw = load_or_extract(Path(r.path), config, cache_dir)
        analysis = analyze_pose(raw, r.view, config)
        meta = {"subject": r.subject, "quality": r.quality, "view": r.view,
                "label": int(r.label), "filename": r.filename}
        videos.append({**meta, **video_features(analysis, rules)})
        table = analysis.rep_table()
        keep = ["rep"] + [c for c in feature_names(r.view, rules) if c in table]
        for rep in table[keep].to_dict("records"):
            reps.append({**meta, **rep})
        if progress:
            progress(n, len(index), r.filename)
    return pd.DataFrame(videos), pd.DataFrame(reps)
