"""Page « Entraînement » : méthode, paramètres, résultats réels et code de scripts/train.py."""

import pandas as pd
import streamlit as st

from app.common import FIGURES, disclaimer, fmt_ci, read_csv
from src.config import ROOT, load_config
from src.features import feature_names

config = load_config()
ev = config["evaluation"]
rules = config["exercises"]["squat"]["rules"]

st.title("Entraînement du modèle")
disclaimer()

st.markdown(
    """
**Ce que le modèle apprend.** Il reproduit le label *good / bad* du dataset Mendeley (un label par
vidéo, critère non documenté) à partir des **valeurs mesurées** des indicateurs (médiane sur les
répétitions). Les seuils des règles ne sont pas utilisés par le modèle.

**Étapes de `scripts/train.py`** : 1) lecture de `data/index.csv` ; 2) extraction des points et des
indicateurs par vidéo ; 3) tableau de features ; 4) découpage *leave-one-subject-out* ;
5) entraînement ; 6) métriques ; 7) sauvegarde du modèle et des figures.

**Leave-one-subject-out.** Chaque sujet a 6 vidéos. À chaque pli, on retire les 6 vidéos d'un
sujet, on entraîne sur les 24 autres, et on prédit le sujet retiré. Le modèle ne voit donc jamais la
personne qu'il évalue. L'imputation et la normalisation sont apprises dans chaque pli.
"""
)

st.subheader("Paramètres (fixés avant tout résultat, `config.yaml`)")
c1, c2 = st.columns(2)
c1.markdown(f"- Régression logistique : `{ev['logreg']}` (**modèle de l'app**, choisi "
            "d'avance pour son explicabilité)\n"
            f"- Forêt aléatoire : `{ev['random_forest']}` (comparaison)\n"
            f"- Zone « aucune tendance » : probabilité dans `{ev['uncertain_zone']}`\n"
            f"- IC : bootstrap de {ev['bootstrap']['n']} tirages sur les sujets")
features = pd.DataFrame({"vue": ["profil", "face", "diagonale"],
                         "indicateurs": [", ".join(feature_names(v, rules))
                                         for v in ("side", "front", "diagonal")]})
c2.dataframe(features, hide_index=True)

st.subheader("Résultats réels (leave-one-subject-out, 149 vidéos)")
perf = read_csv("performance.csv")
rows = []
for method in ["Régression logistique", "Forêt aléatoire"]:
    for view in ["toutes", "side", "front", "diagonal"]:
        p = perf[(perf["method"] == method) & (perf["view"] == view)].set_index("metric")
        rows.append({"modèle": method, "vue": view,
                     **{name: fmt_ci(*p.loc[m, ["estimate", "ci_low", "ci_high"]])
                        for m, name in [("balanced_accuracy", "exactitude équilibrée"),
                                        ("auc", "AUC"), ("brier", "Brier")]}})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.markdown(
    "Lecture : à peine mieux que le hasard (0,5) pour la régression logistique (test de "
    "permutation p = 0,002) ; la forêt aléatoire n'est pas meilleure que le hasard. Score de "
    "Brier ≈ 0,25 = celui d'une réponse « 50 % » constante : **les probabilités ne sont pas "
    "fiables**, l'app n'en affiche aucune."
)
st.image(str(FIGURES / "feature_importance.png"),
         caption="Ce que les modèles ont appris (25 plis). Le tempo pèse dans les trois "
                 "vues : il est mesuré à 15 images/s dans ces données.")

st.subheader("Code de `scripts/train.py`")
st.code((ROOT / "scripts" / "train.py").read_text(encoding="utf-8"), language="python")
