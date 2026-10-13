"""Point d'entrée de l'application Streamlit (local et Streamlit Community Cloud).

Lancement : streamlit run streamlit_app.py
Les pages sont dans app/pages/. La page d'annotation en aveugle n'apparaît
qu'en local, quand l'index du dataset existe : en ligne, aucune vidéo du
dataset n'est disponible (upload uniquement).
"""

import streamlit as st

from src.config import ROOT, load_config

st.set_page_config(page_title="Gym-exe-analyzer", page_icon="🏋️", layout="wide")

pages = {
    "Gym-exe-analyzer": [
        st.Page("app/pages/home.py", title="Accueil et avertissements", icon="🏠", default=True),
        st.Page("app/pages/analyze.py", title="Analyse d'une vidéo", icon="🎥"),
        st.Page("app/pages/results.py", title="Résultats par répétition", icon="📋"),
    ],
    "Comprendre l'outil": [
        st.Page("app/pages/training.py", title="Entraînement", icon="🧠"),
        st.Page("app/pages/evaluation.py", title="Évaluation", icon="📊"),
        st.Page("app/pages/method.py", title="Méthode et limites", icon="⚙️"),
        st.Page("app/pages/data.py", title="Données et licences", icon="📚"),
    ],
}
if (ROOT / load_config()["dataset"]["index_csv"]).exists():
    pages["Local uniquement"] = [
        st.Page("app/pages/annotation.py", title="Annotation (aveugle)", icon="✍️")]
st.navigation(pages).run()
