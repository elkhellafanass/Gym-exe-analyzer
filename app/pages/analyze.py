"""Page « Analyse d'une vidéo » : upload, analyse, verdict, vidéo annotée, export HTML.

Confidentialité : la vidéo envoyée est écrite dans un dossier temporaire,
analysée, puis le dossier est supprimé. Seuls restent en mémoire de la
session : les points du squelette, les résultats et la vidéo annotée.
Aucune vidéo du dataset n'est utilisée ni affichée ici.
"""

import tempfile
from pathlib import Path

import streamlit as st

from app.common import disclaimer, page_link
from src.analysis import analyze_video
from src.config import load_config
from src.report import VIEW_LABELS, build_report, to_html
from src.video import render_annotated

config = load_config()
st.title("Analyse d'une vidéo")
disclaimer()

st.markdown(
    "Envoyez une vidéo de squat (**mp4** ou **mov**) filmée avec un téléphone posé. "
    "Déclarez la vue : l'outil ne la devine pas, et chaque règle n'est valable que dans "
    "certaines vues."
)
view = st.radio("Vue filmée", list(VIEW_LABELS), format_func=lambda v: VIEW_LABELS[v].capitalize(),
                horizontal=True,
                help="Profil recommandé : profondeur, talons, tronc et tête. "
                     "Face : genoux et bassin. Diagonale : approximations.")
hide_image = st.checkbox("Vidéo annotée : squelette seul (masquer le visage et le décor)")
consent = st.checkbox("Je confirme avoir le consentement de chaque personne filmée.")
upload = st.file_uploader("Vidéo du squat", type=["mp4", "mov"])

if upload is not None and not consent:
    st.info("Cochez la case de consentement pour lancer l'analyse.")

if upload is not None and consent and st.button("Analyser la vidéo", type="primary"):
    suffix = Path(upload.name).suffix.lower() or ".mp4"
    bar = st.progress(0.0, text="Détection du squelette (environ 10 s par 30 s de vidéo)…")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f"video{suffix}"
            src.write_bytes(upload.getbuffer())
            analysis = analyze_video(
                src, view, config,
                progress=lambda f: bar.progress(0.7 * f, text="Détection du squelette…"))
            out = render_annotated(
                src, analysis, Path(tmp) / "annotee.mp4", hide_image=hide_image,
                progress=lambda f: bar.progress(0.7 + 0.3 * f, text="Vidéo annotée…"))
            annotated = out.read_bytes()
        # Le dossier temporaire (et la vidéo envoyée) est supprimé ici.
    except ValueError as err:
        bar.empty()
        st.error(f"Analyse impossible : {err}")
        st.stop()
    bar.progress(1.0, text="Terminé")
    report = build_report(analysis, config, upload.name)
    st.session_state.update(analysis=analysis, report=report, annotated=annotated,
                            report_html=to_html(report, analysis, config))

if "report" not in st.session_state:
    st.stop()

report = st.session_state["report"]
verdict = report.verdict
st.divider()
st.subheader(f"Rapport : {report.video_name} (vue de {VIEW_LABELS[report.view]})")

for w in report.warnings:
    st.warning(w)

st.markdown("#### Verdict global")
box = {"bad": st.warning, "good": st.success}.get(verdict.tendency, st.info)
box(f"**{verdict.title}**\n\n{verdict.explanation}")
st.caption(f"Performance mesurée pour cette vue : {verdict.performance}.")
with st.expander("Pourquoi seulement une « tendance faible » ?"):
    for note in verdict.notes:
        st.markdown(f"- {note}")

st.markdown("#### Défauts retenus (au moins 2 répétitions concernées)")
retained = report.rules[report.rules["video_error"].astype(bool)]
if retained.empty:
    st.success("Aucun défaut retenu par les règles applicables à cette vue.")
for r in retained.itertuples():
    st.markdown(f"- **{r.label}** : {r.reps_with_error}/{r.reps_measured} répétitions. "
                f"Fiabilité de la règle : *{r.reliability_level}*.")
page_link("app/pages/results.py", label="Détail par répétition, fiabilité et corrections",
             icon="📋")

st.markdown("#### Vidéo annotée")
st.caption("Squelette (vert : côté mesuré), angles, numéro de répétition ; cadre rouge et "
           "défaut écrit pendant une répétition fautive.")
st.video(st.session_state["annotated"])
stem = Path(report.video_name).stem
c1, c2 = st.columns(2)
c1.download_button("Télécharger la vidéo annotée (mp4)", st.session_state["annotated"],
                   file_name=f"{stem}_annotee.mp4", mime="video/mp4")
c2.download_button("Télécharger le rapport (HTML)", st.session_state["report_html"],
                   file_name=f"{stem}_rapport.html", mime="text/html",
                   help="Fichier autonome. Pour un PDF : ouvrez-le puis Imprimer > PDF.")
