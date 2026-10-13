"""PHASE B - Évaluation (option A, analyse principale : seuils figés).

Lit les prédictions leave-one-subject-out de scripts/train.py et l'annotation
en aveugle de l'auteur (data/annotations/side_blind.csv). Rien n'est réglé
ici : ni seuil, ni hyperparamètre. On mesure seulement.

Étapes :
  1) lecture des prédictions LOSO, des features et de l'annotation
  2) accord entre l'annotateur et le label du dataset (kappa de Cohen)
  3) règles de profil contre l'annotation : détection et fausses alertes
  4) performances du classifieur et du détecteur à seuils, comparées au hasard
     et à des classifieurs triviaux (IC bootstrap sur les sujets)
  5) test de permutation (le classifieur fait-il mieux que le hasard ?)
  6) fiabilité des probabilités : score de Brier, courbe de fiabilité
  7) table de fiabilité des règles lue par l'app
  8) sauvegarde des figures

Conventions FIXÉES AVANT les résultats (docs/PIPELINE.md, "Protocole") :
  - classe positive = "bad" ; seuil de décision 0,5 pour les classifieurs ;
  - détecteur à seuils : vidéo "bad" si au moins une règle valide (ou de
    fiabilité réduite) dans la vue retient un défaut (>= 2 répétitions) ;
  - vidéo sans répétition détectée : "non analysable", exclue et comptée ;
  - annotation : défaut "présent" s'il est noté sur quelques répétitions OU
    la plupart (définition principale) ; "la plupart" seulement (secondaire).

Usage : python scripts/evaluate.py   (après scripts/train.py)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from src.classifier import loso_predict, make_models  # noqa: E402
from src.config import load_config  # noqa: E402
from src.features import feature_names  # noqa: E402
from src.metrics import binary_metrics, bootstrap_by_subject, kappa  # noqa: E402
from src.rules import rule_reliability  # noqa: E402

VIEWS = ["side", "front", "diagonal"]
SUBSETS = ["toutes"] + VIEWS
OUT, RESULTS, FIGURES = ROOT / "outputs", ROOT / "docs/results", ROOT / "docs/figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

config = load_config()
ev = config["evaluation"]
rules = config["exercises"]["squat"]["rules"]
NB, SEED = ev["bootstrap"]["n"], ev["bootstrap"]["seed"]
PRESENT_ANY = {"quelques répétitions", "la plupart des répétitions"}
PRESENT_MOST = {"la plupart des répétitions"}
# Règle de profil -> colonne de l'annotation qui décrit le même défaut
RULE_TO_ANNOT = {"depth": "depth", "heel_rise": "heel_rise",
                 "trunk_vs_tibia": "trunk_lean", "head_position": "head"}
CI_METRICS = ["accuracy", "precision", "recall", "specificity", "f1",
              "balanced_accuracy", "auc", "brier"]


def fmt(row) -> str:
    return f"{row.estimate:.2f} [{row.ci_low:.2f} ; {row.ci_high:.2f}]"


print("=== 1) Lecture ===")
features = pd.read_csv(OUT / "features.csv", dtype={"subject": str})
pred = pd.read_csv(OUT / "loso_predictions.csv", dtype={"subject": str})
annot = pd.read_csv(ROOT / "data/annotations/side_blind.csv")
pred = pred.merge(features[["filename"] + [c for c in features if c.endswith("_flag")]],
                  on="filename")
# Détecteur à seuils (option A) : au moins une règle applicable retient un défaut
flag_cols = [c for c in pred if c.endswith("_flag")]
pred["detector"] = pred[flag_cols].fillna(False).astype(bool).any(axis=1).astype(int)
analysable = pred["n_reps"] > 0
print(f"{len(pred)} vidéos, dont {(~analysable).sum()} non analysable(s) exclue(s) : "
      f"{pred.loc[~analysable, 'filename'].tolist()}")
pred = pred[analysable].reset_index(drop=True)
side = pred[pred["view"] == "side"].merge(annot, on="filename", how="inner")
print(f"Annotation : {len(annot)} vidéos annotées, {len(side)} appariées aux vidéos de profil")
assert len(side) == len(annot) == 50

print("\n=== 2) Accord annotateur / label du dataset (vidéos de profil) ===")
print(pd.crosstab(side["overall"], side["quality"], margins=True))
OVERALL = {"bonne": 0, "mauvaise": 1}
conventions = {
    # Convention principale : on compare là où l'annotateur s'est prononcé
    "incertain exclus": lambda d: d[d["overall"] != "incertain"].assign(
        a=lambda x: x["overall"].map(OVERALL)),
    # Pire cas : chaque "incertain" compté comme un désaccord
    "incertain = désaccord": lambda d: d.assign(a=np.where(
        d["overall"] == "incertain", 1 - d["label"], d["overall"].map(OVERALL))),
    # Meilleur cas : chaque "incertain" compté comme un accord
    "incertain = accord": lambda d: d.assign(a=np.where(
        d["overall"] == "incertain", d["label"], d["overall"].map(OVERALL))),
}
agreement_rows = []
for name, convert in conventions.items():
    def stat(d, convert=convert):
        c = convert(d)
        return {"kappa": kappa(c["a"], c["label"]),
                "accord_brut": float(np.mean(c["a"] == c["label"])) if len(c) else np.nan,
                "n": len(c)}
    ci = bootstrap_by_subject(side, stat, NB, SEED)
    for r in ci.itertuples():
        agreement_rows.append({"convention": name, **r._asdict()})
    k = ci.set_index("metric").loc["kappa"]
    print(f"  {name:22s} n={int(stat(side)['n']):2d}  kappa = {k.estimate:.2f} "
          f"[{k.ci_low:.2f} ; {k.ci_high:.2f}]")
agreement = pd.DataFrame(agreement_rows).drop(columns="Index")
agreement.to_csv(RESULTS / "annotator_vs_label.csv", index=False, float_format="%.3f")
pd.crosstab(side["overall"], side["quality"]).to_csv(RESULTS / "annotator_vs_label_table.csv")

print("\n=== 3) Règles de profil contre l'annotation en aveugle ===")
# 3a. Ce que l'annotateur voit dans les vidéos good et bad
by_label = []
for col in ["depth", "heel_rise", "trunk_lean", "head", "knees_forward", "other"]:
    for q in ("good", "bad"):
        s = side[side["quality"] == q]
        by_label.append({"annotation": col, "quality": q, "n": len(s),
                         "present_any": int(s[col].isin(PRESENT_ANY).sum()),
                         "present_most": int(s[col].isin(PRESENT_MOST).sum())})
by_label = pd.DataFrame(by_label)
by_label.to_csv(RESULTS / "annotation_by_label.csv", index=False)
print(by_label.pivot(index="annotation", columns="quality",
                     values="present_any").rename(columns=lambda c: f"présent ({c}, /25)"))

# 3b. Détection (la règle signale quand l'annotateur voit le défaut) et
#     fausses alertes (la règle signale alors que l'annotateur ne voit rien)
rule_rows = []
for rule, col in RULE_TO_ANNOT.items():
    for definition, present in [("quelques ou la plupart", PRESENT_ANY),
                                ("la plupart", PRESENT_MOST)]:
        d = side.assign(truth=side[col].isin(present).astype(int),
                        flag=side[f"{rule}_flag"].astype(int))

        def stat(x):
            m = binary_metrics(x["truth"], x["flag"])
            return {"detection": m["recall"], "false_alert": 1 - m["specificity"],
                    "kappa": kappa(x["flag"], x["truth"])}
        ci = bootstrap_by_subject(d, stat, NB, SEED).set_index("metric")
        m = binary_metrics(d["truth"], d["flag"])
        rule_rows.append({
            "rule": rule, "annotation": col, "definition": definition,
            "n_present": m["tp"] + m["fn"], "n_absent": m["tn"] + m["fp"],
            "flagged_when_present": m["tp"], "flagged_when_absent": m["fp"],
            **{f"{k}{suffix}": ci.loc[k, c] for k in ["detection", "false_alert", "kappa"]
               for suffix, c in [("", "estimate"), ("_ci_low", "ci_low"),
                                 ("_ci_high", "ci_high")]},
        })
rule_vs_annot = pd.DataFrame(rule_rows)
rule_vs_annot.to_csv(RESULTS / "rule_vs_annotation.csv", index=False, float_format="%.3f")
for r in rule_vs_annot.itertuples():
    print(f"  {r.rule:15s} ({r.definition:22s}) présent {r.n_present:2d}/50 | détection "
          f"{r.detection:.2f} [{r.detection_ci_low:.2f};{r.detection_ci_high:.2f}] | "
          f"fausses alertes {r.false_alert:.2f} "
          f"[{r.false_alert_ci_low:.2f};{r.false_alert_ci_high:.2f}] | kappa {r.kappa:.2f}")

print("\n=== 4) Performances (LOSO) contre le label du dataset ===")
methods = {
    "Régression logistique": lambda d: (d["proba_logreg"] > 0.5, d["proba_logreg"]),
    "Forêt aléatoire": lambda d: (d["proba_random_forest"] > 0.5, d["proba_random_forest"]),
    "Détecteur à seuils (règles)": lambda d: (d["detector"], None),
    "Trivial : toujours bad": lambda d: (np.ones(len(d)), None),
    "Trivial : toujours good": lambda d: (np.zeros(len(d)), None),
}
perf_rows, confusions = [], {}
for subset in SUBSETS:
    d = pred if subset == "toutes" else pred[pred["view"] == subset]
    for method, predict in methods.items():
        def stat(x, predict=predict):
            p, proba = predict(x)
            return {k: v for k, v in binary_metrics(x["label"], p, proba).items()
                    if k in CI_METRICS}
        ci = bootstrap_by_subject(d, stat, NB, SEED)
        p, proba = predict(d)
        m = binary_metrics(d["label"], p, proba)
        confusions[(method, subset)] = m
        for r in ci.itertuples():
            perf_rows.append({"method": method, "view": subset, "n": m["n"],
                              "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                              "metric": r.metric, "estimate": r.estimate,
                              "ci_low": r.ci_low, "ci_high": r.ci_high})
    # Hasard : pile ou face (probabilité 0,5) pour chaque vidéo, 10 000 tirages
    rng = np.random.default_rng(SEED)
    sims = pd.DataFrame([binary_metrics(d["label"], rng.integers(0, 2, len(d)))
                         for _ in range(10_000)])
    for metric in ["accuracy", "precision", "recall", "specificity", "f1",
                   "balanced_accuracy"]:
        lo, med, hi = np.nanquantile(sims[metric], [0.025, 0.5, 0.975])
        perf_rows.append({"method": "Hasard (pile ou face)", "view": subset, "n": len(d),
                          "metric": metric, "estimate": med, "ci_low": lo, "ci_high": hi})
    perf_rows.append({"method": "Hasard (pile ou face)", "view": subset, "n": len(d),
                      "metric": "auc", "estimate": 0.5, "ci_low": np.nan, "ci_high": np.nan})
    perf_rows.append({"method": "Probabilité constante 0,5", "view": subset, "n": len(d),
                      "metric": "brier", "estimate": 0.25, "ci_low": np.nan,
                      "ci_high": np.nan})
perf = pd.DataFrame(perf_rows)
perf.to_csv(RESULTS / "performance.csv", index=False, float_format="%.3f")
for subset in SUBSETS:
    print(f"\n  -- vue : {subset} --")
    for method in list(methods) + ["Hasard (pile ou face)"]:
        p = perf[(perf["view"] == subset) & (perf["method"] == method)].set_index("metric")
        line = "  ".join(f"{k} {fmt(p.loc[k])}" for k in ["precision", "recall", "f1"]
                         if k in p.index)
        auc = f"  AUC {fmt(p.loc['auc'])}" if "auc" in p.index and not np.isnan(
            p.loc["auc", "ci_low"]) else ""
        print(f"  {method:30s} {line}{auc}")

print("\n=== 5) Test de permutation (régression logistique, toutes vues) ===")
# Hypothèse nulle : les indicateurs ne contiennent aucune information sur le
# label. On échange au hasard good/bad À L'INTÉRIEUR de chaque paire
# (sujet, vue), ce qui garde la structure du dataset (1 good + 1 bad par paire),
# puis on refait toute la validation LOSO.
factory = make_models(config)["logreg"]
view_tables = {}
for view in VIEWS:
    rows = np.flatnonzero(pred["view"].to_numpy() == view)
    feats = feature_names(view, rules)
    table = pred.iloc[rows][["subject", "filename"]].merge(features[["filename"] + feats],
                                                           on="filename")
    view_tables[view] = (rows, feats, table)


def loso_auc(labels: np.ndarray) -> tuple[float, float]:
    proba = np.full(len(pred), np.nan)
    for rows, feats, table in view_tables.values():
        table = table.assign(label=labels[rows])
        proba[rows], _ = loso_predict(table, feats, factory)
    m = binary_metrics(labels, proba > 0.5, proba)
    return m["auc"], m["balanced_accuracy"]


observed = loso_auc(pred["label"].to_numpy())
rng = np.random.default_rng(ev["permutation"]["seed"])
pairs = pred.groupby(["subject", "view"]).indices
null = []
for i in range(ev["permutation"]["n"]):
    labels = pred["label"].to_numpy().copy()
    for idx in pairs.values():
        if rng.random() < 0.5:
            labels[idx] = 1 - labels[idx]
    null.append(loso_auc(labels))
    print(f"  permutation {i + 1}/{ev['permutation']['n']}", end="\r")
null = np.array(null)
p_auc = (1 + np.sum(null[:, 0] >= observed[0])) / (1 + len(null))
p_bacc = (1 + np.sum(null[:, 1] >= observed[1])) / (1 + len(null))
permutation = pd.DataFrame([
    {"metric": "auc", "observed": observed[0], "null_median": np.median(null[:, 0]),
     "null_q95": np.quantile(null[:, 0], 0.95), "p_value": p_auc},
    {"metric": "balanced_accuracy", "observed": observed[1],
     "null_median": np.median(null[:, 1]), "null_q95": np.quantile(null[:, 1], 0.95),
     "p_value": p_bacc}])
permutation.to_csv(RESULTS / "permutation_test.csv", index=False, float_format="%.4f")
print("\n", permutation.to_string(index=False))

print("\n=== 6) Fiabilité des probabilités ===")
low, high = ev["uncertain_zone"]
calib_rows, curves = [], {}
for name, col in [("Régression logistique", "proba_logreg"),
                  ("Forêt aléatoire", "proba_random_forest")]:
    for subset in SUBSETS:
        d = pred if subset == "toutes" else pred[pred["view"] == subset]
        zone = d[col].between(low, high)
        outside = d[~zone]
        m_out = binary_metrics(outside["label"], outside[col] > 0.5) if len(outside) else {}
        calib_rows.append({
            "model": name, "view": subset, "n": len(d),
            "brier": binary_metrics(d["label"], d[col] > 0.5, d[col])["brier"],
            "brier_reference_constant_0_5": 0.25,
            "n_incertain": int(zone.sum()), "share_incertain": float(zone.mean()),
            "n_outside_zone": len(outside),
            "accuracy_outside_zone": m_out.get("accuracy", np.nan),
            "proba_min": d[col].min(), "proba_max": d[col].max(),
        })
    curves[name] = calibration_curve(pred["label"], pred[col],
                                     n_bins=ev["calibration_bins"], strategy="quantile")
calibration = pd.DataFrame(calib_rows)
calibration.to_csv(RESULTS / "calibration.csv", index=False, float_format="%.3f")
print(calibration[["model", "view", "brier", "n_incertain", "n_outside_zone",
                   "accuracy_outside_zone"]].to_string(index=False))

print("\n=== 7) Table de fiabilité des règles (lue par l'app) ===")
reliability_rows = []
for rule_name, rule in rules.items():
    for view in VIEWS:
        status = rule_reliability(rule, view)
        if status is None:
            continue
        d = pred[pred["view"] == view]
        flags = d[f"{rule_name}_flag"].astype(bool)
        row = {"rule": rule_name, "view": view, "view_status": status,
               "flag_rate_good": float(flags[d["label"] == 0].mean()),
               "flag_rate_bad": float(flags[d["label"] == 1].mean()),
               "n_good": int((d["label"] == 0).sum()), "n_bad": int((d["label"] == 1).sum())}
        if view == "side" and rule_name in RULE_TO_ANNOT:
            r = rule_vs_annot[(rule_vs_annot["rule"] == rule_name) & (
                rule_vs_annot["definition"] == "quelques ou la plupart")].iloc[0]
            row.update({k: r[k] for k in [
                "n_present", "n_absent", "detection", "detection_ci_low", "detection_ci_high",
                "false_alert", "false_alert_ci_low", "false_alert_ci_high"]})
            row["evaluated_against"] = "annotation en aveugle de l'auteur (non-expert, 50 vidéos)"
        else:
            row["evaluated_against"] = "fiabilité non évaluée contre annotation"
        reliability_rows.append(row)
reliability = pd.DataFrame(reliability_rows)
reliability.insert(3, "provenance",
                   "Mendeley Data (label vidéo) ; annotation en aveugle de l'auteur (profil)")
# Les lignes des jeux externes (scripts/evaluate_external.py) sont conservées
rel_path = RESULTS / "rule_reliability.csv"
if rel_path.exists():
    old = pd.read_csv(rel_path)
    if "provenance" in old:
        external = old[old["provenance"] != reliability["provenance"].iloc[0]]
        reliability = pd.concat([reliability, external], ignore_index=True)
reliability.to_csv(rel_path, index=False, float_format="%.3f")
print(reliability[["rule", "view", "flag_rate_good", "flag_rate_bad",
                   "evaluated_against"]].to_string(index=False))

print("\n=== 8) Figures ===")
# 8a. Matrices de confusion
shown = ["Régression logistique", "Forêt aléatoire", "Détecteur à seuils (règles)"]
fig, axes = plt.subplots(len(shown), len(SUBSETS), figsize=(13, 9.5))
for i, method in enumerate(shown):
    for j, subset in enumerate(SUBSETS):
        m = confusions[(method, subset)]
        mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax = axes[i, j]
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, mat.sum() / 2))
        for (a, b), v in np.ndenumerate(mat):
            ax.text(b, a, str(v), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["prédit good", "prédit bad"])
        ax.set_yticks([0, 1], ["label good", "label bad"])
        ax.set_title(f"{method}\nvue : {subset} (n={m['n']})", fontsize=9)
fig.suptitle("Matrices de confusion, leave-one-subject-out (label du dataset)")
fig.tight_layout()
fig.savefig(FIGURES / "confusion_matrices.png", dpi=120)
plt.close(fig)

# 8b. F1 et AUC avec IC, comparés aux références
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric in zip(axes, ["f1", "auc"], strict=True):
    sub = perf[perf["metric"] == metric]
    labels_y, ticks, y = [], [], 0
    for subset in SUBSETS:
        for method in sub["method"].unique():
            r = sub[(sub["view"] == subset) & (sub["method"] == method)]
            if r.empty or np.isnan(r["estimate"].iloc[0]):
                continue
            r = r.iloc[0]
            err = (np.array([[r.estimate - r.ci_low], [r.ci_high - r.estimate]])
                   if not np.isnan(r.ci_low) else None)
            color = "#999999" if method.startswith(("Hasard", "Trivial")) else "#4C72B0"
            ax.errorbar(r.estimate, y, xerr=err, fmt="o", color=color, capsize=3)
            labels_y.append(f"{subset} | {method}")
            ticks.append(y)
            y += 1
        y += 0.5
    ax.set_yticks(ticks, labels_y, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0.5, color="red", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_title(f"{metric.upper()} (IC 95 % bootstrap sur les sujets ; gris = références)")
fig.tight_layout()
fig.savefig(FIGURES / "performance.png", dpi=120)
plt.close(fig)

# 8c. Courbe de fiabilité
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="parfaitement fiable")
for name, (frac, mean_p) in curves.items():
    ax.plot(mean_p, frac, "o-", label=name)
ax.axvspan(low, high, color="orange", alpha=0.15, label="zone « incertain »")
ax.set_xlabel("probabilité 'bad' prédite (moyenne du groupe)")
ax.set_ylabel("proportion réelle de 'bad' dans le groupe")
ax.set_title(f"Courbe de fiabilité (LOSO, {len(pred)} vidéos, "
             f"{ev['calibration_bins']} groupes)", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "reliability_curve.png", dpi=120)
plt.close(fig)

# 8d. Règles de profil contre l'annotation
d = rule_vs_annot[rule_vs_annot["definition"] == "quelques ou la plupart"]
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(d))
for k, (col, name, color) in enumerate([("detection", "détection", "#4C72B0"),
                                        ("false_alert", "fausses alertes", "#C44E52")]):
    ax.bar(x + (k - 0.5) * 0.38, d[col], 0.38, label=name, color=color,
           yerr=[d[col] - d[f"{col}_ci_low"], d[f"{col}_ci_high"] - d[col]], capsize=3)
ax.set_xticks(x, [f"{r}\n(présent {n}/50)" for r, n in zip(d["rule"], d["n_present"],
                                                            strict=True)])
ax.set_ylim(0, 1.05)
ax.set_ylabel("proportion de vidéos")
ax.set_title("Règles de profil contre l'annotation en aveugle (IC 95 %)", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "rule_vs_annotation.png", dpi=120)
plt.close(fig)
print("  docs/figures/ : confusion_matrices.png, performance.png, reliability_curve.png, "
      "rule_vs_annotation.png")
print("  docs/results/ : annotator_vs_label*.csv, annotation_by_label.csv, "
      "rule_vs_annotation.csv, performance.csv, permutation_test.csv, calibration.csv, "
      "rule_reliability.csv")
