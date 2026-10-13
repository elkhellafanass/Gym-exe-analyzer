"""Inspection des deux jeux de données externes (validation externe).

Pourquoi ? Avant d'évaluer quoi que ce soit, il faut savoir ce que contiennent
réellement les fichiers, au lieu de croire la description de la page Zenodo.
Ce script DÉCRIT seulement : il ne calcule aucune règle, aucun résultat.

  A) Teng 2025 (Zenodo 10.5281/zenodo.17558630) : images de profil,
     classes Good / Bad Back / Bad Heel.
  B) Dai 2026 (Zenodo 10.5281/zenodo.18859797) : points MediaPipe en CSV,
     classes Standard / Shallow / GoodMorning.

Étapes :
  1) A : dossiers, fichiers par classe, formats, résolutions
  2) A : doublons exacts (même fichier dans train et test ?)
  3) A : planche de quelques images par classe, enregistrée HORS du dépôt
  4) B : fichiers par classe, sujets, essais
  5) B : colonnes, plages de valeurs, nombre d'images, valeurs manquantes
  6) B : doublons entre les zip
  7) ce qui n'est pas documenté

Sortie : la console + outputs/external/*.csv (non versionné).
Usage : python scripts/inspect_external.py
"""

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.config import external_data_dir, load_config  # noqa: E402

config = load_config()
ext = external_data_dir(config)
A_DIR = ext / config["external"]["teng_squat"]["dir"]
B_DIR = ext / config["external"]["dai_squat"]["dir"]
OUT = ROOT / "outputs" / "external"
OUT.mkdir(parents=True, exist_ok=True)
INSPECTION = Path(config["external"]["inspection_dir"])
# Garde-fou : les images de personnes ne doivent jamais arriver dans le dépôt
assert ROOT not in INSPECTION.resolve().parents, "la planche doit être hors du dépôt"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


print("=== 1) A (Teng) : structure, classes, formats, résolutions ===")
rows = []
for f in sorted(p for p in A_DIR.rglob("*") if p.is_file()):
    rel = f.relative_to(A_DIR)
    with Image.open(f) as im:
        w, h, mode, fmt = im.width, im.height, im.mode, im.format
    rows.append({"split": rel.parts[0], "folder": rel.parts[1],
                 # la casse des dossiers varie (Bad Back / Bad back) : classe normalisée
                 "class": rel.parts[1].lower(), "file": rel.name,
                 "ext": f.suffix.lower(), "format": fmt, "mode": mode,
                 "width": w, "height": h, "bytes": f.stat().st_size, "md5": md5(f)})
A = pd.DataFrame(rows)
A.drop(columns="md5").to_csv(OUT / "teng_images.csv", index=False)
print(f"Dossier : {A_DIR}")
print(f"{len(A)} fichiers. Dossiers tels qu'écrits sur le disque :")
print(A.groupby(["split", "folder"]).size().to_string())
print("\nPar classe (casse normalisée) :")
print(pd.crosstab(A["class"], A["split"], margins=True).to_string())
print(f"\nFormats : {A.groupby(['ext', 'format', 'mode']).size().to_dict()}")
res = A.groupby(["width", "height"]).size().sort_values(ascending=False)
print(f"Résolutions distinctes : {len(res)} ; les plus fréquentes :")
print(res.head(8).to_string())
print(f"Images carrées (1:1, annoncé sur Zenodo) : {(A['width'] == A['height']).mean():.1%}")
w = A["width"]
print(f"Largeur : min {w.min()}, médiane {w.median():.0f}, max {w.max()}")
# Les noms de fichier contiennent-ils un identifiant de sujet ?
patterns = A["file"].str.replace(r"\d+", "#", regex=True).value_counts()
print(f"Motifs de noms de fichier : {patterns.to_dict()}")

print("\n=== 2) A : doublons exacts (contenu identique) ===")
dup = A[A.duplicated("md5", keep=False)].sort_values("md5")
groups = dup.groupby("md5")
cross_split = sum(g["split"].nunique() > 1 for _, g in groups)
cross_class = sum(g["class"].nunique() > 1 for _, g in groups)
print(f"Groupes de fichiers identiques : {groups.ngroups} "
      f"({len(dup)} fichiers) ; à cheval sur train/test : {cross_split} ; "
      f"dans deux classes différentes : {cross_class}")
dup.drop(columns="md5").assign(group=dup["md5"].factorize()[0]).to_csv(
    OUT / "teng_duplicates.csv", index=False)

print("\n=== 3) A : planche d'images (hors du dépôt) ===")
INSPECTION.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
classes = sorted(A["class"].unique())
n_per_class, size = 8, 220
sheet = Image.new("RGB", (n_per_class * size, len(classes) * (size + 24)), "white")
draw = ImageDraw.Draw(sheet)
for i, c in enumerate(classes):
    sub = A[A["class"] == c]
    picks = sub.iloc[rng.choice(len(sub), n_per_class, replace=False)]
    draw.text((4, i * (size + 24) + 4), f"classe : {c}  ({len(sub)} images)", fill="black")
    for j, r in enumerate(picks.itertuples()):
        with Image.open(A_DIR / r.split / r.folder / r.file) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((size, size))
        sheet.paste(thumb, (j * size, i * (size + 24) + 24))
sheet_path = INSPECTION / "A_teng_planche.jpg"
sheet.save(sheet_path, quality=85)
print(f"Planche : {sheet_path} (8 images tirées au hasard par classe, graine 42)")

print("\n=== 4) B (Dai) : classes, sujets, essais ===")
NAME = re.compile(r"^p(?P<subject>\d+)_(?P<cls>[A-Za-z]+)_(?P<trial>\d+)\.csv$")
rows = []
for f in sorted(B_DIR.rglob("*.csv")):
    rel = f.relative_to(B_DIR)
    m = NAME.match(f.name)
    df = pd.read_csv(f)
    coords = df.filter(regex=r"^[xyzv]_\d+$")
    rows.append({
        "zip": rel.parts[0], "folder": rel.parts[1], "file": f.name,
        "subject": m["subject"] if m else None, "cls": m["cls"] if m else None,
        "trial": m["trial"] if m else None,
        "n_frames": len(df), "n_cols": df.shape[1],
        "other_cols": ",".join(c for c in df.columns if c not in coords.columns),
        "frame_first": df["Frame"].iloc[0] if "Frame" in df else None,
        "frame_step_unique": ",".join(map(str, sorted(df["Frame"].diff().dropna().unique())))
        if "Frame" in df else None,
        "x_min": df.filter(regex=r"^x_").min().min(), "x_max": df.filter(regex=r"^x_").max().max(),
        "y_min": df.filter(regex=r"^y_").min().min(), "y_max": df.filter(regex=r"^y_").max().max(),
        "z_min": df.filter(regex=r"^z_").min().min(), "z_max": df.filter(regex=r"^z_").max().max(),
        "v_min": df.filter(regex=r"^v_").min().min(), "v_max": df.filter(regex=r"^v_").max().max(),
        "n_missing": int(coords.isna().sum().sum()),
        "md5": md5(f),
    })
B = pd.DataFrame(rows)
B.drop(columns="md5").to_csv(OUT / "dai_files.csv", index=False)
print(f"Dossier : {B_DIR}")
print(f"{len(B)} fichiers CSV (tous zip confondus). Noms hors motif p<sujet>_<classe>_<essai> : "
      f"{B['subject'].isna().sum()}")
print(B.groupby(["zip", "folder"]).size().to_string())

print("\n=== 6) B : doublons entre les zip (avant de dédoublonner) ===")
same = B.groupby("file")["md5"].nunique()
print(f"Fichiers présents dans plusieurs zip : {(B['file'].value_counts() > 1).sum()} ; "
      f"dont contenu différent : {(same > 1).sum()}")
B = B.drop_duplicates("md5").reset_index(drop=True)
print(f"Après suppression des copies identiques : {len(B)} fichiers uniques")

print("\n=== 4 bis) B : sujets et essais par classe (fichiers uniques) ===")
print(B.groupby("cls").agg(fichiers=("file", "size"), sujets=("subject", "nunique"),
                           essais_par_sujet_min=("subject", lambda s: s.value_counts().min()),
                           essais_par_sujet_max=("subject", lambda s: s.value_counts().max()))
      .to_string())
subjects = {c: set(g["subject"]) for c, g in B.groupby("cls")}
all_subjects = set().union(*subjects.values())
print(f"Sujets distincts (tous fichiers) : {len(all_subjects)} -> {sorted(all_subjects)}")
for c, s in subjects.items():
    print(f"  {c:12s} : {len(s):2d} sujets {sorted(s)}")
print(f"Sujets présents dans les 3 classes : {len(set.intersection(*subjects.values()))}")

print("\n=== 5) B : colonnes, plages, nombre d'images, valeurs manquantes ===")
print(f"Nombre de colonnes : {B['n_cols'].value_counts().to_dict()}")
print(f"Colonnes autres que x/y/z/v : {B['other_cols'].value_counts().to_dict()}")
print("Colonnes d'angles (hanche/genou annoncés sur Zenodo) : "
      f"{'oui' if B['other_cols'].str.contains('angle|hip|knee', case=False).any() else 'AUCUNE'}")
print(f"Première valeur de Frame : {B['frame_first'].value_counts().to_dict()} ; "
      f"pas entre images : {B['frame_step_unique'].value_counts().to_dict()}")
print("Images par fichier, par classe :")
print(B.groupby("cls")["n_frames"].describe()[["min", "50%", "max"]].to_string())
for k in "xyzv":
    print(f"  {k} : min {B[f'{k}_min'].min():.3f}  max {B[f'{k}_max'].max():.3f}")
print(f"Valeurs manquantes (x/y/z/v) : {B['n_missing'].sum()}")

print("\n=== 7) Non documenté / non vérifiable ===")
print("Voir la synthèse dans le message de fin d'étape et docs/DATA_LICENSES.md.")
