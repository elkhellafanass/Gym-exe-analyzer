"""Page d'annotation en aveugle des 50 vidéos de profil du dataset.

Ce que l'annotateur NE voit PAS : le label good/bad, le nom du fichier, le
numéro du sujet, le squelette, les résultats des règles. Voir src/annotation.py
pour la justification. Fonctionne en local uniquement (le dataset n'est
jamais mis en ligne).
"""

import pandas as pd
import streamlit as st

from src.annotation import (
    ERROR_TYPES,
    FREQUENCIES,
    OVERALL,
    blind_order,
    ensure_h264,
    load_annotations,
    save_annotation,
)
from src.config import ROOT, load_config

ANNOT_PATH = ROOT / "data" / "annotations" / "side_blind.csv"
VIDEO_CACHE = ROOT / "data" / "annotation_videos"

config = load_config()
st.title("Annotation en aveugle : vues de profil")

index_path = ROOT / config["dataset"]["index_csv"]
if not index_path.exists():
    st.error("data/index.csv introuvable. Cet outil fonctionne uniquement en local, "
             "après `python scripts/prepare_dataset.py`.")
    st.stop()

order = blind_order(pd.read_csv(index_path, dtype={"subject": str}))
annotations = load_annotations(ANNOT_PATH)
done = set(annotations["anon_id"])

st.caption(
    "Les 50 vidéos de profil (good et bad mélangées) sont présentées dans un ordre "
    "aléatoire fixe. Le label, le nom du fichier et les détections automatiques sont "
    "masqués. Annotez ce que vous VOYEZ, sans chercher à deviner le label."
)
st.progress(len(done) / len(order), text=f"{len(done)}/{len(order)} vidéos annotées")

with st.expander("Définitions des erreurs (à relire avant de commencer)"):
    for label, definition in ERROR_TYPES.values():
        st.markdown(f"- **{label}** : {definition}")
    st.markdown("Source des critères : Myer et al. (2014), voir docs/SOURCES.md.")

ids = order["anon_id"].tolist()
if "annot_pos" not in st.session_state:
    remaining = [i for i, a in enumerate(ids) if a not in done]
    st.session_state.annot_pos = remaining[0] if remaining else 0

col_prev, col_sel, col_next = st.columns([1, 3, 1])
if col_prev.button("◀ Précédente", width="stretch"):
    st.session_state.annot_pos = max(0, st.session_state.annot_pos - 1)
if col_next.button("Suivante ▶", width="stretch"):
    st.session_state.annot_pos = min(len(ids) - 1, st.session_state.annot_pos + 1)
pos = col_sel.selectbox(
    "Vidéo", range(len(ids)), index=st.session_state.annot_pos,
    format_func=lambda i: f"{ids[i]} {'✓' if ids[i] in done else ''}")
st.session_state.annot_pos = pos

row = order.iloc[pos]
anon = row["anon_id"]
previous = annotations[annotations["anon_id"] == anon]
previous = previous.iloc[0].to_dict() if len(previous) else {}

col_video, col_form = st.columns([1, 1])
with col_video:
    with st.spinner("Préparation de la vidéo (conversion H.264, première fois seulement)…"):
        st.video(str(ensure_h264(row["path"], VIDEO_CACHE)))

with col_form, st.form(f"form_{anon}"):
    st.subheader(f"Vidéo {anon}")
    values = {}
    for key, (label, _) in ERROR_TYPES.items():
        default = previous.get(key) or "absent"
        values[key] = st.radio(label, FREQUENCIES, index=FREQUENCIES.index(default),
                               horizontal=True, key=f"{anon}_{key}")
    overall_default = previous.get("overall") or "incertain"
    overall = st.radio("Mon avis global sur l'exécution", OVERALL,
                       index=OVERALL.index(overall_default), horizontal=True,
                       key=f"{anon}_overall")
    comment = st.text_input("Commentaire (obligatoire si « Autre »)",
                            value=previous.get("comment", ""), key=f"{anon}_comment")
    if st.form_submit_button("Enregistrer et passer à la suivante", type="primary"):
        if values["other"] != "absent" and not comment.strip():
            st.error("Précisez le défaut « Autre » dans le commentaire.")
        else:
            save_annotation(ANNOT_PATH, {"anon_id": anon, "filename": row["filename"],
                                         "overall": overall, **values, "comment": comment})
            st.session_state.annot_pos = min(len(ids) - 1, pos + 1)
            st.rerun()
