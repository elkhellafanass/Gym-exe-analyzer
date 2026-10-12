"""PHASE A : les règles distinguent-elles les vidéos good et bad ? (exploratoire)

Pour chaque vidéo (points en cache, voir extract_landmarks.py), on applique le
pipeline, puis on compare, POUR UN MÊME SUJET ET UNE MÊME VUE, la valeur médiane
de chaque règle entre la vidéo good et la vidéo bad (test de Wilcoxon apparié).
On affiche aussi le % de vidéos où chaque règle retient un défaut.

Ce n'est PAS l'évaluation finale (scripts/evaluate.py, phase B) : c'est une
vérification que les mesures se comportent comme prévu.
Sortie : outputs/phaseA_check.csv
Usage : python scripts/explore_rules.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from src.analysis import analyze_pose  # noqa: E402
from src.config import load_config  # noqa: E402
from src.pose import PoseSeries  # noqa: E402

config = load_config()
index = pd.read_csv(ROOT / "data/index.csv", dtype={"subject": str})
cache = ROOT / "outputs/landmarks"
rows, reprows = [], []
for r in index.itertuples():
    f = cache / f"{Path(r.filename).stem}.npz"
    if not f.exists():
        continue
    a = analyze_pose(PoseSeries.load(f), r.view, config)
    t = a.rep_table()
    base = {"subject": r.subject, "quality": r.quality, "view": r.view, "n_reps": len(a.reps),
            "detect": float(np.mean(PoseSeries.load(f).detected()))}
    for c in ["depth", "heel_rise", "trunk_vs_tibia", "head_position", "knee_valgus", "hip_shift"]:
        if c in t:
            base[c] = t[c].median()
    for s in a.summary.itertuples():
        base[f"{s.rule}_err"] = s.video_error
    base["tempo"] = (t["descent_s"] / t["ascent_s"]).median() if len(t) else np.nan
    rows.append(base)
d = pd.DataFrame(rows)
d.to_csv(ROOT / "outputs/phaseA_check.csv", index=False)
print("videos:", len(d), "| reps median good/bad:",
      d.groupby("quality").n_reps.median().to_dict())
print("videos <3 reps:\n", d[d.n_reps < 3][["subject", "quality", "view", "n_reps"]])
for view, cols in [("side", ["depth", "heel_rise", "trunk_vs_tibia", "head_position", "tempo"]),
                   ("front", ["knee_valgus", "hip_shift", "tempo"]),
                   ("diagonal", ["knee_valgus", "hip_shift"])]:
    v = d[d.view == view].set_index(["subject", "quality"])
    print(f"\n== {view} : médiane good -> bad | bad>good | p Wilcoxon "
          "| % vidéos en erreur good / bad")
    for c in cols:
        w = v[c].unstack("quality").dropna()
        if len(w) < 5:
            continue
        diff = w["bad"] - w["good"]
        p = wilcoxon(diff).pvalue
        err = ""
        if f"{c}_err" in v:
            e = v[f"{c}_err"].unstack("quality")
            err = f"| {e['good'].mean():.0%} / {e['bad'].mean():.0%}"
        print(f"{c:15s} {w['good'].median():7.3f} -> {w['bad'].median():7.3f} | "
              f"{int((diff > 0).sum())}/{len(w)} | p={p:.3f} {err}")
