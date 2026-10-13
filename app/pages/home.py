"""Page d'accueil : ce que fait l'outil, ce qu'il ne fait pas, ce qu'ont montré les évaluations."""

import streamlit as st

from app.common import disclaimer, page_link
from src.config import load_config

config = load_config()

st.title("Gym-exe-analyzer")
disclaimer()
st.warning(
    "**Ce n'est pas un dispositif médical.** L'outil ne prédit aucune blessure et ne donne aucun "
    "conseil médical. En cas de douleur, consultez un professionnel de santé. Les corrections "
    "proposées ne remplacent pas un coach qualifié.",
    icon="🩺",
)

st.markdown(
    """
Analyse de la technique du **squat** à partir d'une vidéo filmée avec un téléphone :

1. **Squelette** : 33 points du corps détectés sur chaque image (MediaPipe Pose) ;
2. **Mesures** : angles et positions (profondeur, talons, tronc, tête, genoux, bassin) ;
3. **Répétitions** : découpage automatique de la vidéo ;
4. **Règles** : critères de la grille de Myer et al. (2014), avec leur source ;
5. **Rapport** : défauts par répétition, fiabilité réelle de chaque règle, corrections écrites,
   vidéo annotée et rapport HTML.
"""
)

st.subheader("Ce que les évaluations ont montré (à lire avant d'utiliser l'outil)")
st.markdown(
    """
- **Le verdict global est une tendance faible.** Le modèle fait à peine mieux que le hasard
  (exactitude équilibrée 0,61) et ses probabilités ne sont pas fiables : aucune n'est affichée.
- **La profondeur est la règle la mieux validée** (4,7 % de fausses alertes sur des squats
  « standard » d'un autre laboratoire). Les règles d'angle (tronc, talons, tête) sont **peu
  informatives** sur des squats réels ou **dépendent du format d'image**.
- Chaque règle affiche sa fiabilité, sa provenance et son statut (*concluant*, *indicatif*,
  *descriptif*). Seuls les résultats *concluants* sont présentés comme des résultats.
"""
)
st.info(
    "Conseils de tournage : **vue de profil** (les règles y sont les mieux évaluées), corps entier "
    "visible, une seule personne à l'image, au moins 3 répétitions, téléphone posé et immobile. "
    "Filmez uniquement des personnes qui ont donné leur consentement.",
    icon="🎥",
)
page_link("app/pages/analyze.py", label="Analyser une vidéo", icon="▶️")
st.caption(f"Données d'entraînement : {config['dataset']['citation']}. "
           f"Licence {config['dataset']['license']}.")
