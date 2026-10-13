"""Page « Données et licences » : attribution exacte et règles de confidentialité."""

import streamlit as st

from app.common import disclaimer, read_doc
from src.config import load_config

config = load_config()
ds = config["dataset"]

st.title("Données et licences")
disclaimer()

st.subheader("Dataset d'entraînement")
st.markdown(f"> {ds['citation']}\n\nURL : {ds['url']} · Licence : [{ds['license']}]"
            f"({ds['license_url']}) · Consulté le 12/10/2026.\n\n"
            "**Données transformées** (extraction de landmarks et d'angles).")
st.info(
    "**Aucune vidéo d'un dataset n'est publiée**, ni dans le dépôt GitHub ni dans "
    "l'application en ligne : les personnes filmées sont identifiables, et la licence CC BY 4.0 "
    "ne couvre pas le droit à l'image (section 2(b)(1)). L'application en ligne analyse "
    "seulement les vidéos que vous envoyez. Elles sont traitées dans un dossier temporaire "
    "supprimé après l'analyse.",
    icon="🔒",
)
st.subheader("Détail : docs/DATA_LICENSES.md")
st.markdown(read_doc("DATA_LICENSES.md"))
