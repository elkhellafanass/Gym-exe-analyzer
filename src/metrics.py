"""Métriques d'évaluation et intervalles de confiance par bootstrap sur les sujets.

Pourquoi bootstrap SUR LES SUJETS et pas sur les vidéos ? Les vidéos d'un même
sujet ne sont pas indépendantes (même personne, même caméra). Rééchantillonner
les vidéos ferait comme si on avait 150 observations indépendantes et donnerait
des intervalles trop étroits. L'unité indépendante est le sujet : on tire 25
sujets avec remise et on garde toutes leurs vidéos.

Conventions : classe positive = "bad" (1). Une métrique non définie (ex.
précision quand rien n'est prédit "bad") vaut NaN, jamais 0 ou 1 par défaut.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, cohen_kappa_score, roc_auc_score


def _ratio(num: float, den: float) -> float:
    return num / den if den else np.nan


def binary_metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None) -> dict:
    """Matrice de confusion + précision, rappel, spécificité, F1, exactitude (+ AUC, Brier)."""
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    out = {
        "n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": _ratio(tp + tn, len(y)),
        "precision": precision, "recall": recall,
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }
    out["balanced_accuracy"] = np.nanmean([out["recall"], out["specificity"]])
    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        out["auc"] = roc_auc_score(y, proba) if len(set(y)) == 2 else np.nan
        out["brier"] = brier_score_loss(y, proba)
    return out


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Kappa de Cohen : accord au-delà du hasard (0 = hasard, 1 = parfait)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or (len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]):
        return np.nan   # accord attendu = 1 : kappa non défini
    return float(cohen_kappa_score(a, b))


def bootstrap_by_subject(df: pd.DataFrame, statistic: Callable[[pd.DataFrame], dict],
                         n: int, seed: int, level: float = 0.95) -> pd.DataFrame:
    """IC percentile de chaque statistique, en rééchantillonnant les sujets.

    statistic(df) -> dict {nom: valeur}. Renvoie un tableau
    nom | estimate | ci_low | ci_high | n_boot_valid.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    rows_of = {s: idx.to_numpy() for s, idx in df.groupby("subject").groups.items()}
    subjects = list(rows_of)
    estimate = statistic(df)
    samples = {k: [] for k in estimate}
    for _ in range(n):
        drawn = rng.choice(subjects, size=len(subjects), replace=True)
        boot = df.iloc[np.concatenate([rows_of[s] for s in drawn])]
        for k, v in statistic(boot).items():
            samples[k].append(v)
    alpha = (1 - level) / 2
    out = []
    for k, v in estimate.items():
        values = np.asarray(samples[k], dtype=float)
        valid = values[~np.isnan(values)]
        low, high = (np.quantile(valid, [alpha, 1 - alpha]) if len(valid)
                     else (np.nan, np.nan))
        out.append({"metric": k, "estimate": v, "ci_low": low, "ci_high": high,
                    "n_boot_valid": len(valid)})
    return pd.DataFrame(out)
