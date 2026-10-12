"""Point d'entrée de l'application Streamlit (local et Streamlit Community Cloud).

Lancement : streamlit run streamlit_app.py
Les pages sont dans app/pages/. Les pages d'analyse, de résultats, d'entraînement
et d'évaluation seront ajoutées en phase C.
"""

import streamlit as st

st.set_page_config(page_title="Gym-exe-analyzer", page_icon="🏋️", layout="wide")

pages = [
    st.Page("app/pages/home.py", title="Accueil", icon="🏠", default=True),
    st.Page("app/pages/annotation.py", title="Annotation (aveugle)", icon="✍️"),
]
st.navigation(pages).run()
