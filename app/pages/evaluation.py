"""Page « Évaluation » : Mendeley (phase B), validation externe, fiabilité des règles.

Chaque métrique externe affiche son statut fixé d'avance par le protocole :
concluant (≥ 10 groupes), indicatif (5 à 9), descriptif (< 5).
"""

import pandas as pd
import streamlit as st

from app.common import FIGURES, disclaimer, fmt_ci, read_csv, status_badge

st.title("Évaluation")
disclaimer()
st.markdown(
    "Tous les chiffres viennent de `docs/results/` (scripts `evaluate.py` et "
    "`evaluate_external.py`). Les protocoles ont été committés **avant** les exécutions."
)
tab_m, tab_x, tab_r = st.tabs(["Dataset Mendeley (phase B)", "Validation externe",
                               "Fiabilité des règles"])

with tab_m:
    st.markdown("**Classifieur et détecteur à seuils, contre le label du dataset** "
                "(leave-one-subject-out, IC 95 % par bootstrap sur les 25 sujets)")
    perf = read_csv("performance.csv")
    view = st.selectbox("Vue", ["toutes", "side", "front", "diagonal"])
    rows = []
    for method in perf["method"].unique():
        p = perf[(perf["method"] == method) & (perf["view"] == view)].set_index("metric")
        if p.empty:
            continue
        rows.append({"méthode": method, **{
            name: fmt_ci(*p.loc[m, ["estimate", "ci_low", "ci_high"]]) if m in p.index else "–"
            for m, name in [("precision", "précision"), ("recall", "rappel"), ("f1", "F1"),
                            ("balanced_accuracy", "exact. équilibrée"), ("auc", "AUC"),
                            ("brier", "Brier")]}})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Le F1 de « toujours bad » (0,67) dépasse celui des modèles : avec 50 % de bad, "
               "c'est l'exactitude équilibrée et l'AUC qu'il faut comparer au hasard (0,5).")
    c1, c2 = st.columns(2)
    c1.image(str(FIGURES / "confusion_matrices.png"), caption="Matrices de confusion (LOSO)")
    c2.image(str(FIGURES / "reliability_curve.png"),
             caption="Courbe de fiabilité : les probabilités ne sont pas calibrées")
    perm = read_csv("permutation_test.csv")
    st.markdown("**Test de permutation (régression logistique)** : " + " ; ".join(
        f"{r.metric} observé {r.observed:.2f}, p = {r.p_value:.3f}" for r in perm.itertuples()))

    st.markdown("**Accord de l'annotateur (auteur, en aveugle) avec le label** : "
                "50 vidéos de profil")
    agr = read_csv("annotator_vs_label.csv")
    k = agr[agr["metric"] == "kappa"]
    st.dataframe(k.assign(kappa=[fmt_ci(r.estimate, r.ci_low, r.ci_high) for r in k.itertuples()])
                 [["convention", "kappa"]], hide_index=True)
    st.markdown("**Règles de profil contre l'annotation en aveugle**")
    rva = read_csv("rule_vs_annotation.csv")
    rva = rva[rva["definition"] == "quelques ou la plupart"]
    st.dataframe(pd.DataFrame({
        "règle": rva["rule"], "défaut présent (annotation)": rva["n_present"].astype(str) + "/50",
        "détection": [fmt_ci(r.detection, r.detection_ci_low, r.detection_ci_high)
                      for r in rva.itertuples()],
        "fausses alertes": [fmt_ci(r.false_alert, r.false_alert_ci_low, r.false_alert_ci_high)
                            for r in rva.itertuples()],
        "kappa": rva["kappa"].round(2)}), hide_index=True)
    st.image(str(FIGURES / "rule_vs_annotation.png"))

with tab_x:
    st.markdown(
        "Règles **figées**, aucune recalibration. Statut fixé d'avance selon le nombre de groupes "
        "(scènes pour Teng, sujets pour Dai) : :green-badge[concluant] ≥ 10, "
        ":orange-badge[indicatif] 5 à 9, :gray-badge[descriptif] < 5."
    )
    m = read_csv("external_metrics.csv")
    m = m[m["variant"] == "mesurables"]
    principal = m[((m["dataset"] == "A Teng 2025") & (m["analysis"] == "principal"))
                  | ((m["dataset"] == "B Dai 2026 v2")
                     & (m["analysis"] == "(a) par répétition (principal)")
                     & (m["format"] == "16:9"))]
    principal = principal[principal["metric"].isin(["detection", "false_alert",
                                                    "balanced_accuracy"])]
    robust = read_csv("external_B_robustness_formats.csv")
    for (dataset, rule, target), g in principal.groupby(["dataset", "rule", "target"], sort=False):
        st.markdown(f"**{dataset} · `{rule}` · cible : {target}**")
        for r in g.itertuples():
            status = r.status
            rb = robust[(robust["rule"] == rule) & (robust["metric"] == r.metric)
                        & (robust["analysis"] == r.analysis) & (robust["target"] == target)]
            if len(rb) and str(rb["conclusion"].iloc[0]).startswith("non retenue"):
                status = "non retenu (dépend du format)"
            name = {"detection": "détection", "false_alert": "fausses alertes",
                    "balanced_accuracy": "exactitude équilibrée"}[r.metric]
            st.markdown(f"- {status_badge(status)} {name} : "
                        f"{fmt_ci(r.estimate, r.ci_low, r.ci_high)}")
    st.image(str(FIGURES / "external_detection.png"))
    st.image(str(FIGURES / "external_distributions.png"))
    dev = read_csv("external_deviations.csv")
    st.markdown("**Déviation déclarée au protocole** : " + " ".join(dev["deviation"]))
    st.info("Lecture : sur Teng et Shallow, les règles signalent, à titre indicatif, des "
            "erreurs marquées et mises en scène. Les 4,7 % de fausses alertes de la profondeur "
            "sur Dai sont compatibles avec l'hypothèse que le label Mendeley est peu fiable, sans "
            "la prouver (laboratoires, caméras et sujets différents).")

with tab_r:
    st.markdown("Table lue par le rapport de correction (`docs/results/rule_reliability.csv`), "
                "avec la **provenance** de chaque mesure et son **statut**.")
    rel = read_csv("rule_reliability.csv")
    cols = [c for c in ["rule", "view", "provenance", "evaluated_against", "detection",
                        "detection_status", "false_alert", "false_alert_status",
                        "false_alert_conclusion", "flag_rate_good", "flag_rate_bad"] if c in rel]
    st.dataframe(rel[cols], hide_index=True)
