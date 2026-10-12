"""PHASE B - Entraînement du classifieur good/bad (option A, analyse principale).

Les seuils des règles restent FIGÉS (littérature) : ils ne sont pas touchés
ici. Le classifieur apprend le label du dataset à partir des VALEURS CONTINUES
des indicateurs (une ligne par vidéo, voir src/features.py).

Un modèle par vue (profil, face, diagonale) : les indicateurs valides ne sont
pas les mêmes selon la vue, et l'app demande la vue à l'utilisateur.

Étapes :
  1) lecture de data/index.csv
  2) extraction des landmarks et des indicateurs, vidéo par vidéo
  3) tableau de features (une ligne par vidéo)
  4) découpage leave-one-subject-out (25 plis, 6 vidéos par sujet)
  5) entraînement : régression logistique et forêt aléatoire, dans chaque pli
  6) métriques rapides (l'évaluation complète est dans scripts/evaluate.py)
  7) sauvegarde du modèle final, des prédictions et des figures

Hyperparamètres : config.yaml, section evaluation, fixés avant la première
exécution et jamais réglés ensuite (aucune recherche d'hyperparamètres).

Usage : python scripts/train.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from src.classifier import MODEL_LABELS, loso_predict, make_models  # noqa: E402
from src.config import load_config  # noqa: E402
from src.features import build_feature_table, feature_names  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402

VIEWS = ["side", "front", "diagonal"]
OUT, RESULTS, FIGURES = ROOT / "outputs", ROOT / "docs/results", ROOT / "docs/figures"
for d in (OUT, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

config = load_config()
rules = config["exercises"]["squat"]["rules"]
models = make_models(config)

print("=== 1) Lecture de data/index.csv ===")
index = pd.read_csv(ROOT / config["dataset"]["index_csv"], dtype={"subject": str})
print(f"{len(index)} vidéos, {index['subject'].nunique()} sujets, "
      f"{(index['label'] == 1).sum()} bad / {(index['label'] == 0).sum()} good")

print("\n=== 2) Extraction des landmarks et des indicateurs (cache outputs/landmarks) ===")
videos, reps = build_feature_table(
    index, config, OUT / "landmarks",
    progress=lambda n, total, name: print(f"  [{n}/{total}] {name}", end="\r"))
print(f"\n{len(reps)} répétitions détectées au total")

print("\n=== 3) Tableau de features (une ligne par vidéo) ===")
videos.to_csv(OUT / "features.csv", index=False)
reps.to_csv(OUT / "rep_values.csv", index=False)
analysable = videos["n_reps"] > 0
print(f"Vidéos non analysables (0 répétition) : {(~analysable).sum()} "
      f"{videos.loc[~analysable, 'filename'].tolist()}")
for view in VIEWS:
    print(f"  {view:8s} : {feature_names(view, rules)}")

print("\n=== 4) Découpage leave-one-subject-out ===")
subjects = sorted(videos["subject"].unique())
print(f"{len(subjects)} plis : à chaque pli, les 6 vidéos d'un sujet sont mises de côté, "
      f"le modèle est entraîné sur les {len(subjects) - 1} autres sujets.")

print("\n=== 5) Entraînement dans chaque pli ===")
predictions = videos[["subject", "quality", "view", "label", "filename", "n_reps"]].copy()
importance_rows = []
final_models = {}
for view in VIEWS:
    feats = feature_names(view, rules)
    rows = videos.index[(videos["view"] == view) & analysable]
    table = videos.loc[rows].reset_index(drop=True)
    for name, factory in models.items():
        proba, fold_models = loso_predict(table, feats, factory)
        predictions.loc[rows, f"proba_{name}"] = proba
        # Ce que chaque modèle de pli a appris (stabilité d'un pli à l'autre)
        for k, m in enumerate(fold_models):
            est = m[-1]
            values = (est.coef_[0] if name == "logreg" else est.feature_importances_)
            for f, v in zip(feats, values, strict=True):
                importance_rows.append({"view": view, "model": name, "fold": k,
                                        "feature": f, "value": float(v)})
        print(f"  {view:8s} {MODEL_LABELS[name]:22s} : {len(fold_models)} plis, "
              f"{len(table)} vidéos")
    # Modèle final de l'app : entraîné sur les 25 sujets (après l'évaluation LOSO)
    app_model = config["evaluation"]["app_model"]
    final_models[view] = {
        "features": feats,
        "model": models[app_model]().fit(table[feats].to_numpy(dtype=float),
                                         table["label"].to_numpy()),
        "n_videos": len(table), "n_subjects": table["subject"].nunique(),
    }

print("\n=== 6) Métriques rapides (LOSO, seuil 0,5) ===")
for view in VIEWS + ["toutes"]:
    sel = predictions[analysable] if view == "toutes" else predictions[
        analysable & (predictions["view"] == view)]
    for name in models:
        m = binary_metrics(sel["label"], sel[f"proba_{name}"] > 0.5, sel[f"proba_{name}"])
        print(f"  {view:8s} {MODEL_LABELS[name]:22s} exactitude {m['accuracy']:.2f}  "
              f"F1 {m['f1']:.2f}  AUC {m['auc']:.2f}  (n={m['n']})")

print("\n=== 7) Sauvegarde ===")
predictions.to_csv(OUT / "loso_predictions.csv", index=False)
bundle = {
    "views": final_models,
    "model_type": config["evaluation"]["app_model"],
    "uncertain_zone": config["evaluation"]["uncertain_zone"],
    "label_meaning": "probabilité que le label du dataset soit 'bad'",
    "sklearn_version": sklearn.__version__,
    "dataset": config["dataset"]["citation"],
}
joblib.dump(bundle, ROOT / "models/squat_classifier.joblib")
print("  models/squat_classifier.joblib (régression logistique, un modèle par vue)")

importance = pd.DataFrame(importance_rows)
summary = (importance.groupby(["view", "model", "feature"])["value"]
           .agg(mean="mean", sd="std", min="min", max="max").reset_index())
summary.to_csv(RESULTS / "feature_importance.csv", index=False, float_format="%.4f")
print("  docs/results/feature_importance.csv")

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for i, name in enumerate(models):
    for j, view in enumerate(VIEWS):
        ax = axes[i, j]
        s = summary[(summary["model"] == name) & (summary["view"] == view)]
        ax.barh(s["feature"], s["mean"], xerr=s["sd"], color="#4C72B0" if i == 0 else "#55A868")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{MODEL_LABELS[name]} - {view}", fontsize=10)
axes[0, 0].set_xlabel("coefficient standardisé (> 0 : pousse vers 'bad')")
axes[1, 0].set_xlabel("importance (impureté), moyenne des 25 plis")
fig.suptitle("Ce que les modèles ont appris (moyenne ± écart-type sur les 25 plis LOSO)")
fig.tight_layout()
fig.savefig(FIGURES / "feature_importance.png", dpi=120)
print("  docs/figures/feature_importance.png")
print("\nÉtape suivante : python scripts/evaluate.py")
