"""PHASE B - ANALYSE SECONDAIRE (option B) : seuils recalibrés sur le dataset.

À NE PAS CONFONDRE avec l'analyse principale (scripts/evaluate.py), où les
seuils restent ceux de la littérature. Ici, on se demande : "si on choisissait
les seuils à partir des données, les règles sépareraient-elles good et bad ?"

Pour éviter toute fuite, le seuil est choisi en leave-one-subject-out : à
chaque pli, on cherche le seuil sur les 24 sujets d'entraînement, puis on
l'applique au sujet retiré, qui n'a pas servi au choix.

Fixé AVANT les résultats :
  - sens de la règle inchangé (ex. "profondeur < seuil" reste "<") ;
  - même logique qu'en option A : défaut retenu si >= 2 répétitions concernées ;
  - critère de choix : indice de Youden (sensibilité + spécificité - 1) sur
    les vidéos d'entraînement ; égalité -> seuil le plus proche de la littérature ;
  - détecteur combiné : vidéo "bad" si au moins une règle recalibrée signale.
Ces seuils sont "calibrés sur le dataset", JAMAIS "sourcés" : ils ne vont pas
dans config.yaml.

Étapes :
  1) lecture des valeurs par répétition (outputs/rep_values.csv, train.py)
  2) choix du seuil dans chaque pli et prédiction du sujet retiré
  3) métriques et IC bootstrap sur les sujets
  4) stabilité des seuils d'un pli à l'autre
  5) sauvegarde

Usage : python scripts/evaluate_option_b.py   (après scripts/train.py)
"""

import operator
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.metrics import binary_metrics, bootstrap_by_subject  # noqa: E402
from src.rules import rule_reliability  # noqa: E402

OPS = {"<": operator.lt, ">": operator.gt}
VIEWS = ["side", "front", "diagonal"]
RESULTS = ROOT / "docs/results"
config = load_config()
squat = config["exercises"]["squat"]
rules, MIN_REPS = squat["rules"], squat["min_reps_for_video_error"]
NB, SEED = config["evaluation"]["bootstrap"]["n"], config["evaluation"]["bootstrap"]["seed"]


def video_flags(reps: pd.DataFrame, videos: pd.DataFrame, rule: str, op: str,
                threshold: float) -> np.ndarray:
    """Vidéo signalée si au moins MIN_REPS répétitions dépassent le seuil (comme rules.py)."""
    errors = OPS[op](reps[rule], threshold) & reps[rule].notna()
    count = errors.groupby(reps["filename"]).sum()
    return (videos["filename"].map(count).fillna(0) >= MIN_REPS).to_numpy()


def choose_threshold(reps: pd.DataFrame, videos: pd.DataFrame, rule: str) -> float:
    """Seuil qui maximise l'indice de Youden sur les vidéos d'entraînement."""
    op, literature = rules[rule]["op"], rules[rule]["value"]
    candidates = np.unique(np.r_[reps[rule].dropna().to_numpy(), literature])
    y = videos["label"].to_numpy()
    best, best_j = literature, -np.inf
    for t in candidates:
        flag = video_flags(reps, videos, rule, op, t)
        j = flag[y == 1].mean() - flag[y == 0].mean()
        if j > best_j + 1e-12 or (abs(j - best_j) <= 1e-12
                                  and abs(t - literature) < abs(best - literature)):
            best, best_j = t, j
    return float(best)


print("=== 1) Lecture des valeurs par répétition ===")
reps_all = pd.read_csv(ROOT / "outputs/rep_values.csv", dtype={"subject": str})
videos_all = pd.read_csv(ROOT / "outputs/features.csv", dtype={"subject": str})
videos_all = videos_all[videos_all["n_reps"] > 0].reset_index(drop=True)
print(f"{len(reps_all)} répétitions, {len(videos_all)} vidéos analysables")

print("\n=== 2) Seuil choisi dans chaque pli, appliqué au sujet retiré ===")
pred_rows, threshold_rows = [], []
for view in VIEWS:
    view_rules = [r for r in rules if rule_reliability(rules[r], view)]
    videos = videos_all[videos_all["view"] == view].reset_index(drop=True)
    reps = reps_all[reps_all["view"] == view]
    out = videos[["subject", "view", "label", "filename"]].copy()
    for subject in sorted(videos["subject"].unique()):
        test = (videos["subject"] == subject).to_numpy()
        train_videos = videos[~test]
        train_reps = reps[reps["subject"] != subject]
        assert subject not in set(train_videos["subject"])     # garde-fou anti-fuite
        for rule in view_rules:
            t = choose_threshold(train_reps, train_videos, rule)
            threshold_rows.append({"view": view, "rule": rule, "held_out": subject,
                                   "threshold": t, "literature": rules[rule]["value"]})
            out.loc[test, f"{rule}_flag"] = video_flags(
                reps[reps["subject"] == subject], videos[test], rule, rules[rule]["op"], t)
    out["detector"] = out[[f"{r}_flag" for r in view_rules]].astype(bool).any(axis=1)
    pred_rows.append(out)
    print(f"  {view:8s} : {len(view_rules)} règles x {videos['subject'].nunique()} plis")
pred = pd.concat(pred_rows, ignore_index=True)

print("\n=== 3) Métriques (label du dataset), IC bootstrap sur les sujets ===")
rows = []
for view in ["toutes"] + VIEWS:
    d = pred if view == "toutes" else pred[pred["view"] == view]
    cols = ["detector"] + ([] if view == "toutes" else
                           [c for c in d if c.endswith("_flag") and d[c].notna().all()])
    for col in cols:
        def stat(x, col=col):
            m = binary_metrics(x["label"], x[col].astype(bool))
            return {k: m[k] for k in ["precision", "recall", "specificity", "f1",
                                      "balanced_accuracy"]}
        ci = bootstrap_by_subject(d, stat, NB, SEED)
        m = binary_metrics(d["label"], d[col].astype(bool))
        for r in ci.itertuples():
            rows.append({"analysis": "option B (seuils calibrés sur le dataset, LOSO)",
                         "view": view, "method": "≥ 1 règle" if col == "detector" else col,
                         "n": m["n"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                         "tn": m["tn"], "metric": r.metric, "estimate": r.estimate,
                         "ci_low": r.ci_low, "ci_high": r.ci_high})
perf = pd.DataFrame(rows)
wide = perf.pivot_table(index=["view", "method"], columns="metric", values="estimate")
print(wide[["precision", "recall", "specificity", "f1", "balanced_accuracy"]].round(2))

print("\n=== 4) Stabilité des seuils d'un pli à l'autre ===")
thresholds = pd.DataFrame(threshold_rows)
stability = (thresholds.groupby(["view", "rule"])
             .agg(literature=("literature", "first"), median=("threshold", "median"),
                  min=("threshold", "min"), max=("threshold", "max"))
             .reset_index())
print(stability.round(3).to_string(index=False))

print("\n=== 5) Sauvegarde ===")
perf.to_csv(RESULTS / "option_b_performance.csv", index=False, float_format="%.3f")
stability.to_csv(RESULTS / "option_b_thresholds.csv", index=False, float_format="%.4f")
print("  docs/results/option_b_performance.csv, docs/results/option_b_thresholds.csv")
