"""VALIDATION EXTERNE : les règles figées sur deux jeux de données jamais vus.

Applique le protocole committé AVANT exécution (commit 65bc4c1,
docs/PIPELINE.md, "Étape 3"). Rien n'est réglé ici : ni seuil, ni
regroupement, ni hypothèse. Toute déviation est signalée dans les résultats.

  A) Teng 2025 : images de profil ; trunk_vs_tibia contre Bad Back (référence
     Good). Adaptation déclarée : règle appliquée à UNE image.
  B) Dai 2026 v2 : une répétition par fichier ; depth contre Shallow
     (référence Standard), sous 3 hypothèses de format (16:9, 4:3, 1:1).

Étapes :
  1) lecture des inspections (scènes, blocs, points MediaPipe, fichiers B)
  2) A : indicateurs et verdicts image par image
  3) A : métriques par scène (principal) + sensibilités + tranches d'angle du genou
  4) B : pipeline de l'app sur chaque fichier, sous les 3 formats
  5) B : métriques (a) par répétition, (b) par (sujet, classe), sensibilités
  6) B : confusion côté filmé / classe
  7) table de fiabilité des règles (avec provenance) lue par l'app
  8) figures et tableaux

Usage : python scripts/evaluate_external.py   (après scripts/inspect_external.py)
"""

import operator
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis import analyze_pose  # noqa: E402
from src.angles import compute_indicators  # noqa: E402
from src.config import load_config  # noqa: E402
from src.external import read_normalized, to_pose_series  # noqa: E402
from src.metrics import bootstrap_by_subject, kappa  # noqa: E402
from src.pose import PoseSeries  # noqa: E402
from src.rules import evaluate_reps  # noqa: E402

# Les images ou fichiers sans point mesurable produisent des médianes de NaN :
# attendu (ils sont comptés "non mesurables"), on masque l'avertissement répétitif.
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
OPS = {"<": operator.lt, ">": operator.gt}
OUT, RESULTS, FIGURES = ROOT / "outputs/external", ROOT / "docs/results", ROOT / "docs/figures"
config = load_config()
squat = config["exercises"]["squat"]
RULES, MIN_REPS = squat["rules"], squat["min_reps_for_video_error"]
MIN_VIS = config["pose"]["min_visibility"]
NB, SEED = config["evaluation"]["bootstrap"]["n"], config["evaluation"]["bootstrap"]["seed"]
ASPECTS = {"16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0}   # 16:9 = hypothèse principale
FPS_CONVENTION = 30.0   # inconnue ; n'entre dans aucune règle (protocole, point 6)
DEVIATIONS = []         # toute déviation au protocole est notée ici et publiée


def status(n_clusters: int) -> str:
    """Critère d'effectif décidé d'avance (protocole, point 3)."""
    return "concluant" if n_clusters >= 10 else "indicatif" if n_clusters >= 5 else "descriptif"


WEAKEST = {"descriptif": 0, "indicatif": 1, "concluant": 2}


def pair_metrics(units: pd.DataFrame, target: str, reference: str, label: dict) -> list[dict]:
    """Détection, fausses alertes, exactitude équilibrée, kappa pour un couple (règle, classe).

    units : une ligne par unité, colonnes subject (= cluster), cls, flag (bool ou NaN si
    non mesurable). Deux variantes : unités mesurables seulement (principal) et
    "non mesurable = non signalé" (sensibilité, ce que verrait l'utilisateur).
    """
    rows = []
    for variant in ("mesurables", "non mesurable = non signalé"):
        d = units[units["cls"].isin([target, reference])].copy()
        if variant == "mesurables":
            d = d[d["flag"].notna()]
        d["flag"] = d["flag"].fillna(False).astype(bool)
        d["is_target"] = d["cls"] == target
        n_cl_t = d.loc[d["is_target"], "subject"].nunique()
        n_cl_r = d.loc[~d["is_target"], "subject"].nunique()

        def stat(x):
            t, r = x[x["is_target"]], x[~x["is_target"]]
            det = t["flag"].mean() if len(t) else np.nan
            fa = r["flag"].mean() if len(r) else np.nan
            return {"detection": det, "false_alert": fa,
                    "balanced_accuracy": (det + 1 - fa) / 2,
                    "kappa": kappa(x["flag"].astype(int), x["is_target"].astype(int))}
        st = {"detection": status(n_cl_t), "false_alert": status(n_cl_r)}
        st["balanced_accuracy"] = st["kappa"] = min(st.values(), key=WEAKEST.get)
        ci = bootstrap_by_subject(d, stat, NB, SEED).set_index("metric")
        for metric in ["detection", "false_alert", "balanced_accuracy", "kappa"]:
            s = st[metric]
            est, lo, hi = ci.loc[metric, ["estimate", "ci_low", "ci_high"]]
            if s == "descriptif":
                lo = hi = np.nan                        # comptes seulement, pas d'IC
            if s != "concluant":
                conclusion = f"{s}, pas de conclusion"
            elif metric == "balanced_accuracy":
                conclusion = ("sépare mieux que le hasard" if lo > 0.5 else
                              "sens inverse" if hi < 0.5 else "aucune séparation démontrée")
            else:
                conclusion = "taux mesuré"
            t, r = d[d["is_target"]], d[~d["is_target"]]
            rows.append({**label, "target": target, "reference": reference, "variant": variant,
                         "metric": metric, "estimate": est, "ci_low": lo, "ci_high": hi,
                         "status": s, "conclusion": conclusion,
                         "n_clusters_target": n_cl_t, "n_clusters_reference": n_cl_r,
                         "n_target": len(t), "flagged_target": int(t["flag"].sum()),
                         "n_reference": len(r), "flagged_reference": int(r["flag"].sum())})
    return rows


def flag_counts(units: pd.DataFrame, label: dict) -> pd.DataFrame:
    """Comptes par classe : mesurables, non mesurables, signalés (descriptif)."""
    g = units.groupby("cls")
    return pd.DataFrame({
        "n_units": g.size(), "n_measurable": g["flag"].apply(lambda s: s.notna().sum()),
        "n_non_measurable": g["flag"].apply(lambda s: s.isna().sum()),
        "n_flagged": g["flag"].apply(lambda s: int((s == True).sum())),  # noqa: E712
        "n_clusters": g["subject"].nunique(),
    }).assign(flag_rate=lambda x: x["n_flagged"] / x["n_measurable"]).reset_index().assign(
        **label)


metrics_rows, count_tables = [], []

print("=== 1) Lecture des inspections (outputs/external, scripts/inspect_external.py) ===")
A = pd.read_csv(OUT / "teng_images.csv")
lm = np.load(OUT / "teng_landmarks.npz")
assert (A["split"] + "/" + A["folder"] + "/" + A["file"]).tolist() == list(lm["files"])
B_files = pd.read_csv(OUT / "dai_files_v2.csv", dtype={"subject": str})
B_geo = pd.read_csv(OUT / "dai_v2_geometry.csv", dtype={"subject": str})
B_files = B_files.merge(B_geo[["cls", "subject", "file", "visible_side"]],
                        on=["cls", "subject", "file"], validate="one_to_one")
print(f"A : {len(A)} images ; B v2 : {len(B_files)} fichiers uniques, "
      f"{B_files['subject'].nunique()} sujets")

print("\n=== 2) A : indicateurs et verdicts image par image ===")
# Même chaîne que l'app, sauf le temps : visibilité < seuil -> point absent,
# puis compute_indicators sur une série d'UNE image (adaptation déclarée).
values = {"trunk_vs_tibia": [], "depth": []}
for i in range(len(A)):
    xy, vis = lm["xy"][i].copy(), lm["vis"][i]
    xy[vis < MIN_VIS] = np.nan
    one = PoseSeries(xy[None], vis[None], 1.0, int(A.at[i, "width"]), int(A.at[i, "height"]))
    with np.errstate(all="ignore"):
        ind, _ = compute_indicators(one)
    for rule in values:
        values[rule].append(float(ind[RULES[rule]["indicator"]].iloc[0]))
for rule, v in values.items():
    v = np.asarray(v)
    A[f"{rule}_value"] = v
    A[f"{rule}_flag"] = pd.Series(OPS[RULES[rule]["op"]](v, RULES[rule]["value"])).where(
        ~np.isnan(v))
A[["class", "scene_10", "phase", "knee_angle", "trunk_vs_tibia_value", "depth_value"]].to_csv(
    OUT / "teng_rule_values.csv", index=False)
CLS_A = {"bad back": "Bad Back", "bad heel": "Bad Heel", "good": "Good"}
A["cls"] = A["class"].map(CLS_A)
for rule in values:
    nm = A.groupby("cls")[f"{rule}_flag"].apply(lambda s: s.isna().sum()).to_dict()
    print(f"  {rule:15s} non mesurables par classe : {nm}")

print("\n=== 3) A : métriques ===")
CLUSTERINGS = ["scene_10", "scene_6", "scene_8", "scene_12", "scene_15", "block_0.05"]
for clustering in CLUSTERINGS:
    units = A.assign(subject=A[clustering], flag=A["trunk_vs_tibia_flag"])
    label = {"dataset": "A Teng 2025", "rule": "trunk_vs_tibia", "analysis":
             "principal" if clustering == "scene_10" else "sensibilité regroupement",
             "clusters": clustering, "format": "connu (pixels)", "stratum": "toutes images"}
    metrics_rows += pair_metrics(units, "Bad Back", "Good", label)
# Facteur de confusion : la profondeur -> mêmes métriques par tranche d'angle du genou
for stratum in ["< 90° (bas)", "90-120°"]:
    units = A[A["phase"] == stratum].assign(subject=lambda x: x["scene_10"],
                                            flag=lambda x: x["trunk_vs_tibia_flag"])
    label = {"dataset": "A Teng 2025", "rule": "trunk_vs_tibia",
             "analysis": "confusion profondeur (tranche d'angle du genou)",
             "clusters": "scene_10", "format": "connu (pixels)", "stratum": stratum}
    metrics_rows += pair_metrics(units, "Bad Back", "Good", label)
# Descriptif : taux de signalement par classe (Bad Heel inclus, non évaluable)
for rule in values:
    units = A.assign(subject=A["scene_10"], flag=A[f"{rule}_flag"])
    count_tables.append(flag_counts(units, {"dataset": "A Teng 2025", "rule": rule,
                                            "analysis": "descriptif", "format": "connu"}))
m = pd.DataFrame(metrics_rows)
show = m[(m["variant"] == "mesurables") & (m["metric"].isin(["detection", "false_alert",
                                                             "balanced_accuracy"]))]
print(show.pivot_table(index=["clusters", "stratum"], columns="metric", values="estimate",
                       aggfunc="first").round(3).to_string())
print(show[["clusters", "stratum", "metric", "status", "conclusion"]]
      .drop_duplicates().to_string(index=False))

print("\n=== 4) B : pipeline de l'app sur chaque fichier, sous 3 formats ===")
B_RULES = ["depth", "trunk_vs_tibia", "heel_rise", "head_position"]
rep_rows, file_rows = [], []
for fmt, aspect in ASPECTS.items():
    for r in B_files.itertuples():
        xn, vis, _ = read_normalized(Path(r.path))
        with np.errstate(all="ignore"):
            a = analyze_pose(to_pose_series(xn, vis, aspect, FPS_CONVENTION), "side", config)
            # Sensibilité "fichier = répétition entière" : début, hanche la plus basse, fin
            hip_y = a.indicators["hip_y"].to_numpy()
            whole = pd.DataFrame([{"rep": 1, "start": 0, "end": len(hip_y) - 1,
                                   "bottom": int(np.nanargmax(hip_y))
                                   if np.any(~np.isnan(hip_y)) else 0}])
            whole_res = evaluate_reps(a.indicators, whole, "side", RULES)
        meta = {"format": fmt, "cls": r.cls, "subject": r.subject, "file": r.file,
                "visible_side": r.visible_side}
        for x in a.rep_results.itertuples():
            rep_rows.append({**meta, "rep": x.rep, "rule": x.rule, "value": x.value,
                             "error": x.error})
        row = {**meta, "n_reps": len(a.reps)}
        for rule in B_RULES:
            res = a.rep_results[a.rep_results["rule"] == rule]["error"].dropna()
            row[f"{rule}_flag"] = bool(res.any()) if len(res) else np.nan
            w = whole_res[whole_res["rule"] == rule]["error"].dropna()
            row[f"{rule}_flag_whole"] = bool(w.iloc[0]) if len(w) else np.nan
            row[f"{rule}_value_whole"] = (whole_res[whole_res["rule"] == rule]["value"].iloc[0]
                                          if len(whole_res) else np.nan)
        file_rows.append(row)
F = pd.DataFrame(file_rows)
R = pd.DataFrame(rep_rows)
F.to_csv(OUT / "dai_v2_file_verdicts.csv", index=False)
print("Répétitions détectées par fichier, selon le format :")
print(pd.crosstab(F["format"], F["n_reps"]).to_string())
print("Fichiers sans répétition détectée (non mesurables), par classe et format :")
print(pd.crosstab(F["format"], F["cls"], values=(F["n_reps"] == 0), aggfunc="sum").to_string())
# Vérification annoncée : depth ne dépend pas du format (sauf via la segmentation)
p = F.pivot_table(index=["cls", "subject", "file"], columns="format", values="depth_flag",
                  aggfunc="first")
both = p.dropna()
diff = (both.nunique(axis=1) > 1).sum()
print(f"depth : fichiers mesurables sous les 3 formats {len(both)}/{len(p)} ; verdict différent "
      f"selon le format : {diff}")
pw = F.pivot_table(index=["cls", "subject", "file"], columns="format",
                   values="depth_value_whole", aggfunc="first")
same_sign = (np.sign(pw).nunique(axis=1) <= 1).mean()
print(f"depth (fichier = répétition entière) : même signe sous les 3 formats pour "
      f"{same_sign:.1%} des fichiers")

print("\n=== 5) B : métriques ===")
B_PAIRS = [("depth", "Shallow", "Standard"), ("trunk_vs_tibia", "GM", "Standard")]
for fmt in ASPECTS:
    f = F[F["format"] == fmt]
    for rule, target, ref in B_PAIRS:
        for col, analysis in [(f"{rule}_flag", "(a) par répétition (principal)"),
                              (f"{rule}_flag_whole",
                               "(a) sensibilité : fichier = répétition entière")]:
            units = f.assign(flag=f[col])
            metrics_rows += pair_metrics(units, target, ref, {
                "dataset": "B Dai 2026 v2", "rule": rule, "analysis": analysis,
                "clusters": "sujets", "format": fmt, "stratum": "tous"})
        # (b) regroupement (sujet, classe) : signalé si >= MIN_REPS répétitions signalées
        reps = R[(R["format"] == fmt) & (R["rule"] == rule)]
        g = reps.groupby(["subject", "cls"])["error"]
        groups = pd.DataFrame({"measured": g.apply(lambda s: s.notna().sum()),
                               "n_err": g.apply(lambda s: int((s == True).sum()))})  # noqa: E712
        groups = groups.reindex(f.groupby(["subject", "cls"]).size().index, fill_value=0)
        groups["flag"] = (groups["n_err"] >= MIN_REPS).where(groups["measured"] >= MIN_REPS)
        metrics_rows += pair_metrics(groups.reset_index(), target, ref, {
            "dataset": "B Dai 2026 v2", "rule": rule,
            "analysis": "(b) par (sujet, classe), >= 2 répétitions",
            "clusters": "sujets", "format": fmt, "stratum": "tous"})
    # Règles sans classe cible : taux de signalement (fausses alertes sur Standard)
    for rule in B_RULES:
        count_tables.append(flag_counts(f.assign(flag=f[f"{rule}_flag"]), {
            "dataset": "B Dai 2026 v2", "rule": rule, "analysis": "(a) par répétition",
            "format": fmt}))
        std = f[f["cls"] == "Standard"]
        std = std[std[f"{rule}_flag"].notna()].assign(flag=std[f"{rule}_flag"].dropna()
                                                      .astype(bool))
        ci = bootstrap_by_subject(std, lambda x: {"rate": x["flag"].mean()}, NB, SEED).iloc[0]
        n_cl = std["subject"].nunique()
        metrics_rows.append({
            "dataset": "B Dai 2026 v2", "rule": rule, "analysis": "(a) par répétition",
            "clusters": "sujets", "format": fmt, "stratum": "tous", "target": "",
            "reference": "Standard", "variant": "mesurables", "metric": "false_alert",
            "estimate": ci["estimate"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "status": status(n_cl), "conclusion": "taux mesuré" if status(n_cl) == "concluant"
            else f"{status(n_cl)}, pas de conclusion", "n_clusters_reference": n_cl,
            "n_reference": len(std), "flagged_reference": int(std["flag"].sum())})
m = pd.DataFrame(metrics_rows)
bm = m[(m["dataset"] == "B Dai 2026 v2") & (m["variant"] == "mesurables")
       & m["metric"].isin(["detection", "false_alert", "balanced_accuracy"])]
print(bm.pivot_table(index=["rule", "analysis", "metric"], columns="format", values="estimate",
                     aggfunc="first").round(3).to_string())
# Une conclusion n'est retenue que si elle est la même sous les 3 formats.
# Pour un TAUX (détection, fausses alertes), "même conclusion" = même valeur
# estimée sous les 3 formats (précision ajoutée après exécution, voir DEVIATIONS).
DEVIATIONS.append(
    "Précision du point 6 (ajoutée APRÈS avoir vu les résultats) : le protocole ne disait pas "
    "comment juger qu'un TAUX est 'le même sous les trois formats'. Règle appliquée : la "
    "valeur estimée doit être identique (à 0,001 près) ; sinon 'dépend du format' avec "
    "l'étendue des valeurs. Raison : les fausses alertes de trunk_vs_tibia, heel_rise et "
    "head_position varient selon le format alors que leur libellé 'taux mesuré' était identique.")


def robust_conclusion(g: pd.DataFrame) -> str:
    values = g["estimate"].round(3)
    if g["conclusion"].nunique() > 1 or values.nunique() > 1:
        return (f"non retenue : dépend du format ({values.min():.3f} à {values.max():.3f})")
    return g["conclusion"].iloc[0]


robust = (bm.groupby(["rule", "analysis", "metric", "target"])[["conclusion", "estimate"]]
          .apply(robust_conclusion).rename("conclusion").reset_index())
print(robust.to_string(index=False))

print("\n=== 6) B : confusion côté filmé / classe ===")
f = F[F["format"] == "16:9"]
print(pd.crosstab(f["cls"], f["visible_side"]).to_string())
paired_subjects = sorted(f.loc[f["cls"] == "Shallow", "subject"].unique())
same_side = [s for s in paired_subjects
             if f[(f["subject"] == s) & (f["cls"].isin(["Shallow", "Standard"]))]
             ["visible_side"].nunique() == 1]
print(f"Sujets avec Shallow : {paired_subjects} ; même côté dans les deux classes : {same_side}")
for subset, subjects, analysis in [
        ("mêmes sujets", paired_subjects, "confusion : mêmes sujets (01-06)"),
        ("même côté", same_side, "confusion : même côté filmé"),
        ("côté opposé", [s for s in paired_subjects if s not in same_side],
         "confusion : côté opposé")]:
    for fmt in ASPECTS:
        ff = F[(F["format"] == fmt) & F["subject"].isin(subjects)]
        metrics_rows += pair_metrics(ff.assign(flag=ff["depth_flag"]), "Shallow", "Standard", {
            "dataset": "B Dai 2026 v2", "rule": "depth", "analysis": analysis,
            "clusters": "sujets", "format": fmt, "stratum": subset})
m = pd.DataFrame(metrics_rows)
conf = m[m["analysis"].str.startswith("confusion :") & (m["variant"] == "mesurables")
         & (m["format"] == "16:9")]
print(conf.pivot_table(index="analysis", columns="metric", values="estimate",
                       aggfunc="first").round(3).to_string())

print("\n=== 7) Table de fiabilité des règles (avec provenance) ===")
rel_path = RESULTS / "rule_reliability.csv"
rel = pd.read_csv(rel_path)
MENDELEY = "Mendeley Data (label vidéo) ; annotation en aveugle de l'auteur (profil)"
if "provenance" not in rel:
    rel["provenance"] = MENDELEY
rel = rel[rel["provenance"] == MENDELEY]           # on remplace les lignes externes
ext_rows = []
pick = m[m["variant"] == "mesurables"]
for dataset, rule, analysis, fmt, target, _ref, prov, against in [
        ("A Teng 2025", "trunk_vs_tibia", "principal", "connu (pixels)", "Bad Back", "Good",
         "Teng 2025 (images, adaptation : règle sur une image, 10 scènes)",
         "classe Bad Back (dos) contre Good"),
        ("B Dai 2026 v2", "depth", "(a) par répétition (principal)", "16:9", "Shallow",
         "Standard", "Dai 2026 v2 (répétitions, format 16:9 ; 4:3 et 1:1 en sensibilité)",
         "classe Shallow contre Standard"),
        ("B Dai 2026 v2", "trunk_vs_tibia", "(a) par répétition (principal)", "16:9", "GM",
         "Standard", "Dai 2026 v2 (répétitions, format 16:9)",
         "classe GoodMorning (1 sujet) contre Standard")]:
    s = pick[(pick["dataset"] == dataset) & (pick["rule"] == rule)
             & (pick["analysis"] == analysis) & (pick["format"] == fmt)
             & (pick["target"] == target)].set_index("metric")
    if dataset.startswith("A"):
        s = s[s["clusters"] == "scene_10"]
    rob = robust[(robust["rule"] == rule) & (robust["analysis"] == analysis)
                 & (robust["target"] == target)].set_index("metric")["conclusion"] \
        if dataset.startswith("B") else None
    row = {"rule": rule, "view": "side", "view_status": "valid", "provenance": prov,
           "evaluated_against": against, "n_present": s.loc["detection", "n_target"],
           "n_absent": s.loc["false_alert", "n_reference"]}
    for metric, name in [("detection", "detection"), ("false_alert", "false_alert"),
                         ("balanced_accuracy", "balanced_accuracy")]:
        row[name] = s.loc[metric, "estimate"]
        row[f"{name}_ci_low"] = s.loc[metric, "ci_low"]
        row[f"{name}_ci_high"] = s.loc[metric, "ci_high"]
        row[f"{name}_status"] = s.loc[metric, "status"]
        row[f"{name}_conclusion"] = (rob.loc[metric] if rob is not None else
                                     s.loc[metric, "conclusion"])
    ext_rows.append(row)
for rule in ["heel_rise", "head_position"]:
    s = pick[(pick["dataset"] == "B Dai 2026 v2") & (pick["rule"] == rule)
             & (pick["format"] == "16:9") & (pick["metric"] == "false_alert")].iloc[0]
    # conclusion robuste au format (même règle que pour les autres lignes de B)
    rob = robust[(robust["rule"] == rule) & (robust["metric"] == "false_alert")
                 & (robust["analysis"] == "(a) par répétition")]["conclusion"].iloc[0]
    ext_rows.append({"rule": rule, "view": "side", "view_status": "valid",
                     "provenance": "Dai 2026 v2 (répétitions, format 16:9)",
                     "evaluated_against": "taux de signalement sur Standard (pas de classe cible)",
                     "n_absent": s["n_reference"], "false_alert": s["estimate"],
                     "false_alert_ci_low": s["ci_low"], "false_alert_ci_high": s["ci_high"],
                     "false_alert_status": s["status"], "false_alert_conclusion": rob})
rel = pd.concat([rel, pd.DataFrame(ext_rows)], ignore_index=True)
rel.to_csv(rel_path, index=False, float_format="%.3f")
print(rel[["rule", "provenance", "detection", "false_alert", "detection_status",
           "false_alert_status"]].to_string(index=False))

print("\n=== 8) Sauvegarde des tableaux et figures ===")
m.to_csv(RESULTS / "external_metrics.csv", index=False, float_format="%.3f")
robust.to_csv(RESULTS / "external_B_robustness_formats.csv", index=False)
pd.concat(count_tables, ignore_index=True).to_csv(RESULTS / "external_flag_counts.csv",
                                                  index=False, float_format="%.3f")
pd.crosstab(f["cls"], f["visible_side"]).to_csv(RESULTS / "external_B_visible_side.csv")
pd.DataFrame({"deviation": DEVIATIONS or ["aucune"]}).to_csv(
    RESULTS / "external_deviations.csv", index=False)

# Figure 1 : détection / fausses alertes avec IC et statut
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
COLORS = {"concluant": "#4C72B0", "indicatif": "#DD8452", "descriptif": "#999999"}
for ax, sel, title in [
        (axes[0], (m["dataset"] == "A Teng 2025") & (m["analysis"] != "descriptif"),
         "A (Teng) : trunk_vs_tibia, Bad Back / Good"),
        (axes[1], (m["dataset"] == "B Dai 2026 v2") & (m["rule"] == "depth")
         & (m["target"] == "Shallow"), "B (Dai v2) : depth, Shallow / Standard")]:
    d = m[sel & (m["variant"] == "mesurables") & m["metric"].isin(["detection", "false_alert"])]
    labels = []
    for k, (key, g) in enumerate(d.groupby(["analysis", "clusters", "format", "stratum"],
                                           sort=False)):
        for metric, dy, marker in [("detection", -0.15, "o"), ("false_alert", 0.15, "s")]:
            r = g[g["metric"] == metric]
            if r.empty:
                continue
            r = r.iloc[0]
            err = None if np.isnan(r["ci_low"]) else [[r["estimate"] - r["ci_low"]],
                                                      [r["ci_high"] - r["estimate"]]]
            ax.errorbar(r["estimate"], k + dy, xerr=err, fmt=marker, capsize=3,
                        color=COLORS[r["status"]])
        labels.append(" | ".join(str(x) for x in key if x not in ("tous", "toutes images")))
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("● détection (classe cible)   ■ fausses alertes (référence)")
handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=s) for s, c in COLORS.items()]
fig.legend(handles=handles, title="statut (protocole)", fontsize=8, loc="lower center", ncol=3)
fig.suptitle("Validation externe : règles figées, IC 95 % bootstrap par scène (A) ou sujet (B)")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIGURES / "external_detection.png", dpi=120)
plt.close(fig)

# Figure 2 : distributions des valeurs mesurées par classe, avec le seuil figé
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
for c in ["Good", "Bad Back", "Bad Heel"]:
    v = A.loc[A["cls"] == c, "trunk_vs_tibia_value"].dropna()
    ax.hist(v, bins=40, range=(-40, 60), histtype="step", lw=1.5, label=f"{c} (n={len(v)})")
ax.axvline(RULES["trunk_vs_tibia"]["value"], color="red", ls="--", label="seuil figé (15°)")
ax.set_xlabel("tronc − tibia (°), une image")
ax.set_title("A (Teng)", fontsize=10)
ax.legend(fontsize=8)
ax = axes[1]
f = F[F["format"] == "16:9"]
for c in ["Standard", "Shallow", "GM"]:
    v = f.loc[f["cls"] == c, "depth_value_whole"].dropna()
    ax.hist(v, bins=40, range=(-1.5, 0.8), histtype="step", lw=1.5, density=True,
            label=f"{c} (n={len(v)})")
ax.axvline(RULES["depth"]["value"], color="red", ls="--", label="seuil figé (0)")
ax.set_xlabel("profondeur max (longueurs de cuisse, > 0 = hanche sous le genou)")
ax.set_title("B (Dai v2), format 16:9, fichier = répétition entière", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "external_distributions.png", dpi=120)
plt.close(fig)
print("  docs/results/external_*.csv, docs/results/rule_reliability.csv (provenance)")
print("  docs/figures/external_detection.png, docs/figures/external_distributions.png")
print(f"Déviations au protocole : {DEVIATIONS or 'aucune'}")
