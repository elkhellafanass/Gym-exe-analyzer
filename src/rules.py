"""Étape 4 du pipeline : indicateurs + répétitions -> erreurs signalées.

Pourquoi des règles plutôt qu'un modèle appris ? Une règle est EXPLICABLE :
"le talon s'est soulevé de 22 % de la longueur du pied, la tolérance est de
15 %". Chaque seuil vit dans config.yaml avec sa source et son statut
("sourcé" ou "à valider par un expert"). Ce module ne contient AUCUN seuil.

Une règle n'est appliquée que dans les vues où elle a un sens (un angle
sagittal ne se mesure pas de face). Hors de ces vues, elle est "non évaluée",
ce qui est différent de "pas d'erreur".
"""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd

OPS = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}
AGGS = {"max": np.nanmax, "min": np.nanmin}


def rule_reliability(rule: dict, view: str) -> str | None:
    """'valid', 'reduced' (approximation, signalée) ou None (non applicable)."""
    if view in rule["views"].get("valid", []):
        return "valid"
    if view in rule["views"].get("reduced", []):
        return "reduced"
    return None


def window_frames(indicators: pd.DataFrame, rep: pd.Series, window: str) -> np.ndarray:
    """Indices des images de la fenêtre d'analyse d'une répétition.

    'bottom_half' = images où la hanche est dans la moitié basse de sa course.
    Pourquoi ? Les défauts comme le valgus ou le tronc penché s'évaluent
    quand le genou est fléchi ; debout, ces mesures n'ont pas de sens.
    """
    frames = np.arange(int(rep["start"]), int(rep["end"]) + 1)
    if window == "rep":
        return frames
    hip_y = indicators["hip_y"].to_numpy()
    top = (hip_y[int(rep["start"])] + hip_y[int(rep["end"])]) / 2
    half = top + 0.5 * (hip_y[int(rep["bottom"])] - top)
    return frames[hip_y[frames] >= half]


def evaluate_reps(indicators: pd.DataFrame, reps: pd.DataFrame, view: str,
                  rules: dict) -> pd.DataFrame:
    """Une ligne par (répétition, règle applicable) : valeur mesurée et verdict.

    error vaut NaN si la mesure est impossible (points non visibles) :
    on n'invente pas un verdict sans mesure.
    """
    rows = []
    for _, rep in reps.iterrows():
        for name, rule in rules.items():
            reliability = rule_reliability(rule, view)
            if reliability is None:
                continue
            frames = window_frames(indicators, rep, rule["window"])
            series = indicators[rule["indicator"]].to_numpy()
            values = series[frames]
            if rule.get("reference") == "start":
                # Mesure RELATIVE à la posture debout du début de la répétition
                # (médiane de 3 images, pour ne pas dépendre d'une image bruitée)
                s0 = int(rep["start"])
                ref = series[max(0, s0 - 1):s0 + 2]
                # Référence non mesurable -> toute la répétition est "non mesurée"
                ref_value = np.nanmedian(ref) if np.any(~np.isnan(ref)) else np.nan
                values = values - ref_value
            if len(values) == 0 or np.all(np.isnan(values)):
                value, error = np.nan, np.nan
            else:
                value = float(AGGS[rule["agg"]](values))
                error = bool(OPS[rule["op"]](value, rule["value"]))
            rows.append({
                "rep": int(rep["rep"]), "rule": name, "label": rule["label"],
                "value": value, "threshold": rule["value"], "op": rule["op"],
                "error": error, "reliability": reliability, "status": rule["status"],
            })
    return pd.DataFrame(rows, columns=["rep", "rule", "label", "value", "threshold", "op",
                                       "error", "reliability", "status"])


def summarize_video(rep_results: pd.DataFrame, rules: dict,
                    min_reps_for_error: int) -> pd.DataFrame:
    """Synthèse par règle sur toute la vidéo.

    Un défaut est retenu au niveau de la vidéo s'il apparaît sur au moins
    `min_reps_for_error` répétitions (Myer et al., 2014 : 2 répétitions ou
    plus). Pourquoi pas 1 ? Une seule répétition fautive peut venir d'une
    erreur de mesure ponctuelle.
    """
    rows = []
    for name, group in rep_results.groupby("rule", sort=False):
        measured = group["error"].dropna()
        n_err = int(measured.sum())
        rows.append({
            "rule": name, "label": rules[name]["label"],
            "reps_measured": len(measured), "reps_with_error": n_err,
            "video_error": n_err >= min_reps_for_error,
            "reliability": group["reliability"].iloc[0],
            "status": rules[name]["status"], "message": rules[name]["message"],
        })
    return pd.DataFrame(rows, columns=["rule", "label", "reps_measured", "reps_with_error",
                                       "video_error", "reliability", "status", "message"])
