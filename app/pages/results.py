"""Page « Résultats par répétition » : mesures, règles, fiabilité, corrections."""

import streamlit as st

from app.common import disclaimer, page_link, status_badge
from src.report import VIEW_LABELS, rep_figure, rule_figure

st.title("Résultats par répétition")
disclaimer()

if "report" not in st.session_state:
    st.info("Aucune analyse pour l'instant. Commencez par envoyer une vidéo.")
    page_link("app/pages/analyze.py", label="Analyser une vidéo", icon="▶️")
    st.stop()

report, analysis = st.session_state["report"], st.session_state["analysis"]
st.caption(f"Vidéo : {report.video_name} · vue de {VIEW_LABELS[report.view]} · "
           f"{len(analysis.reps)} répétitions détectées")
st.plotly_chart(rep_figure(analysis), width="stretch")

st.subheader("Règle par règle")
st.markdown(
    "Pour chaque règle : la valeur mesurée à chaque répétition, le seuil (figé, issu de la "
    "littérature), et la **fiabilité réelle** de la règle mesurée lors des évaluations. "
    "Seuls les statuts :green-badge[concluant] sont des résultats ; :orange-badge[indicatif] et "
    ":gray-badge[descriptif] ne permettent pas de conclure."
)
for r in report.rules.itertuples():
    flag = "🔴 défaut retenu" if r.video_error else "🟢 non retenu"
    title = (f"{r.label} : {flag} ({r.reps_with_error}/{r.reps_measured} répétitions) · "
             f"fiabilité : {r.reliability_level}")
    with st.expander(title, expanded=bool(r.video_error)):
        if r.reliability == "reduced":
            st.warning("Fiabilité réduite dans cette vue (approximation déformée par la "
                       "perspective).")
        st.plotly_chart(rule_figure(analysis, r.rule), width="stretch")
        st.markdown(f"**Fiabilité de la règle ({r.reliability_level})** : {r.reliability_text}")
        for line in report.reliability[r.rule]:
            st.markdown(f"- {status_badge(line['status'])} *{line['provenance']}* : {line['text']}")
        correction = next((c for c in report.corrections if c["label"] == r.label), None)
        if correction:
            st.markdown(f"**Correction** : {correction['message']}")
            st.caption(f"Source : {correction['source']} · statut du seuil : "
                       f"{correction['status']}. {correction['expert']}")

st.subheader("Tableau des répétitions")
st.caption("Durées à ±1 image près ; tempo = durée de descente / durée de remontée.")
st.dataframe(report.reps.round(3), hide_index=True)
st.download_button("Télécharger le tableau (CSV)", report.reps.to_csv(index=False),
                   file_name="repetitions.csv", mime="text/csv")
st.download_button("Télécharger le rapport complet (HTML)", st.session_state["report_html"],
                   file_name="rapport_squat.html", mime="text/html")
st.caption("Corrections : critères de Myer et al. (2014). Aucun conseil médical. Les seuils "
           "marqués « à valider par un expert » sont des choix du projet, non sourcés.")
