"""Page « Méthode et limites » : lue directement depuis docs/PIPELINE.md et docs/SOURCES.md."""

import streamlit as st

from app.common import disclaimer, doc_section, read_doc

st.title("Méthode et limites")
disclaimer()

st.subheader("Pipeline")
st.markdown(doc_section("PIPELINE.md", "Schéma"))
st.markdown(
    "Mesures en **2D** (projection dans le plan de l'image) : chaque règle n'est valable que dans "
    "les vues où le mouvement se fait face à la caméra. Un défaut est retenu s'il apparaît sur au "
    "moins 2 répétitions (Myer et al., 2014)."
)

st.subheader("Limites")
for heading in ["Limites déjà identifiées", "Limites ajoutées par la phase B",
                "Limites ajoutées par la validation externe"]:
    with st.expander(heading, expanded=heading.startswith("Limites déjà")):
        st.markdown(doc_section("PIPELINE.md", heading))

st.subheader("Sources des critères et des seuils")
with st.expander("docs/SOURCES.md"):
    st.markdown(read_doc("SOURCES.md"))
