"""Page d'accueil : présentation et avertissements (complétée en phase C)."""

import streamlit as st

from src.config import load_config

config = load_config()

st.title("Gym-exe-analyzer")
st.warning(
    f"**{config['app']['disclaimer']}**\n\n"
    "Conçu comme aide pédagogique ; utilité non évaluée."
)
st.markdown(
    "Analyse de la technique du **squat** à partir d'une vidéo : points du squelette "
    "(MediaPipe Pose), angles articulaires, découpage en répétitions et règles issues "
    "de la littérature (voir *docs/SOURCES.md*)."
)
st.info("Application en cours de construction : les pages d'analyse arrivent en phase C.")
