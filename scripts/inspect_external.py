"""Inspection des deux jeux de données externes (validation externe).

Pourquoi ? Avant d'évaluer quoi que ce soit, il faut savoir ce que contiennent
réellement les fichiers, au lieu de croire la description de la page Zenodo.
Ce script DÉCRIT seulement : il ne calcule aucune règle, aucun résultat.

  A) Teng 2025 (Zenodo 10.5281/zenodo.17558630) : images de profil,
     classes Good / Bad Back / Bad Heel.
  B) Dai 2026 (Zenodo v2 10.5281/zenodo.19550070, v1 18859797) : points en CSV,
     classes Standard / Shallow / GoodMorning.

Étapes :
  1) A : dossiers, fichiers par classe, formats, résolutions
  2) A : doublons exacts (même fichier dans train et test ?)
  3) A : planche de quelques images par classe, jamais versionnée (data/ ou hors du dépôt)
  4) A : images voisines quasi identiques -> blocs (corrélation)
  5) A : détection MediaPipe par résolution et par classe (aucune règle)
  6) A : phase du mouvement (angle du genou, descriptif)
  7) B : v1 et v2 : fichiers, sujets, doublons, colonnes, comparaison
  8) B : géométrie (sens de y, vue, côté visible, répétitions par fichier)
  9) B : rapport largeur/hauteur : hypothèses testées, méthode validée sur Mendeley
  10) ce qui n'est pas documenté

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
from scipy.signal import find_peaks  # noqa: E402

from src.analysis import model_path  # noqa: E402
from src.angles import angle_3pts  # noqa: E402
from src.config import external_data_dir, inspection_dir, load_config  # noqa: E402
from src.external import ASPECT_GRID, estimate_aspect, length_instability, read_normalized  # noqa: E402
from src.pose import LM, PoseSeries, extract_pose_images  # noqa: E402

config = load_config()
ext = external_data_dir(config)
A_DIR = ext / config["external"]["teng_squat"]["dir"]
OUT = ROOT / "outputs" / "external"
OUT.mkdir(parents=True, exist_ok=True)
INSPECTION = inspection_dir(config).resolve()
# Garde-fou : les images de personnes ne doivent jamais être versionnées :
# hors du dépôt, ou dans data/ (ignoré par .gitignore)
assert ROOT not in INSPECTION.parents or (ROOT / "data") in INSPECTION.parents, (
    "la planche doit être hors du dépôt ou dans data/"
)


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


print("\n=== 4) A : images voisines quasi identiques -> blocs ===")
# Les fichiers sont numérotés dans l'ordre du temps (vérifié ci-dessous) :
# ce sont des images extraites de vidéos. Deux images voisines ne sont pas
# deux observations indépendantes. On mesure la différence moyenne entre
# vignettes 32x32 en niveaux de gris (0 = identiques, 1 = opposées).
A["num"] = A["file"].str.extract(r"(\d+)", expand=False).astype(int)
A = A.sort_values(["split", "folder", "num"]).reset_index(drop=True)


def thumb(r) -> np.ndarray:
    with Image.open(A_DIR / r.split / r.folder / r.file) as im:
        return np.asarray(im.convert("L").resize((32, 32)), dtype=float) / 255


T = np.stack([thumb(r) for r in A.itertuples()])
same_folder = np.r_[False, (A["split"].values[1:] == A["split"].values[:-1])
                    & (A["folder"].values[1:] == A["folder"].values[:-1])]
step = np.r_[np.nan, np.abs(T[1:] - T[:-1]).mean(axis=(1, 2))]
rng = np.random.default_rng(42)
i, j = rng.integers(0, len(T), 5000), rng.integers(0, len(T), 5000)
random_pairs = np.abs(T[i] - T[j]).mean(axis=(1, 2))
print("Différence entre images de numéros consécutifs (même dossier), quantiles 10/50/90/99 % : "
      f"{np.round(np.quantile(step[same_folder], [.1, .5, .9, .99]), 3)}")
print("Différence entre paires tirées au hasard, quantiles 1/10/50 % : "
      f"{np.round(np.quantile(random_pairs, [.01, .1, .5]), 3)}")
print("-> la numérotation suit le temps : images voisines presque identiques.")
for t in (0.02, 0.03, 0.05, 0.08):
    # nouveau bloc quand on change de dossier OU quand l'image change brusquement
    A[f"block_{t}"] = np.cumsum(~same_folder | (step > t))
    sizes = A.groupby(f"block_{t}").size()
    per_class = A.groupby("class")[f"block_{t}"].nunique().to_dict()
    print(f"  seuil {t:.2f} : {sizes.size:4d} blocs (taille médiane {sizes.median():.0f}, "
          f"max {sizes.max()}) ; par classe {per_class}")

# Planche : première image de chaque bloc (seuil 0,05) -> vérifier à l'œil
# qu'un bloc = une prise de vue continue. Hors du dépôt.
block_size = A.groupby("block_0.05").size()
firsts = A.groupby("block_0.05").head(1)
size, ncols = 150, 12
nrows = int(np.ceil(len(firsts) / ncols))
sheet = Image.new("RGB", (ncols * size, nrows * (size + 14)), "white")
draw = ImageDraw.Draw(sheet)
for k, r in enumerate(firsts.itertuples()):
    with Image.open(A_DIR / r.split / r.folder / r.file) as im:
        th = im.convert("RGB")
        th.thumbnail((size, size))
    x, y = (k % ncols) * size, (k // ncols) * (size + 14)
    sheet.paste(th, (x, y + 14))
    n = int(block_size[A.loc[r.Index, "block_0.05"]])
    draw.text((x + 2, y), f"{r.split[:2]} {r.folder[:8]} n={n}", fill="black")
blocks_path = INSPECTION / "A_teng_blocs_0.05.jpg"
sheet.save(blocks_path, quality=80)
print(f"Planche des blocs (seuil 0,05) : {blocks_path}")
print("Constat visuel : dans 'test/Bad Back', la numérotation alterne entre deux prises de vue "
      "(blocs d'une image) ; une même scène se retrouve dans plusieurs classes et dans train/test.")

print("\n=== 4 bis) A : regroupement par SCÈNE (personne + lieu + tenue, approximatif) ===")
# Pourquoi ? Les blocs consécutifs coupent une même prise de vue en morceaux.
# Une "scène" regroupe les images qui se ressemblent globalement (vignette
# couleur 16x16, classification hiérarchique, lien moyen). C'est le plus proche
# d'un "sujet" que l'on puisse construire, mais ce n'est PAS un sujet vérifié :
# une personne peut changer de tenue ou de lieu, deux personnes peuvent
# partager un lieu.
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402


def color_thumb(r) -> np.ndarray:
    with Image.open(A_DIR / r.split / r.folder / r.file) as im:
        return np.asarray(im.convert("RGB").resize((16, 16)), dtype=float).ravel() / 255


tree = linkage(np.stack([color_thumb(r) for r in A.itertuples()]), "average", metric="cityblock")
for k in (6, 8, 10, 12, 15, 20):
    A[f"scene_{k}"] = fcluster(tree, k, "maxclust")
    ct = pd.crosstab(A[f"scene_{k}"], A["class"])
    print(f"  {k:2d} scènes : tailles {sorted(ct.sum(axis=1).tolist(), reverse=True)} ; "
          f"scènes présentes dans les 3 classes : {int((ct > 0).all(axis=1).sum())}")
print("Images par scène et par classe (10 scènes) :")
print(pd.crosstab(A["scene_10"], [A["class"]]).to_string())
size, per = 130, 8
sheet = Image.new("RGB", (per * size, 10 * (size + 14)), "white")
draw = ImageDraw.Draw(sheet)
for i, (k, g) in enumerate(A.groupby("scene_10")):
    picks = g.iloc[np.random.default_rng(42).choice(len(g), min(per, len(g)), replace=False)]
    draw.text((2, i * (size + 14)), f"scène {k} : {len(g)} images, classes "
              f"{g['class'].value_counts().to_dict()}", fill="black")
    for j, r in enumerate(picks.itertuples()):
        with Image.open(A_DIR / r.split / r.folder / r.file) as im:
            th = im.convert("RGB")
            th.thumbnail((size, size))
        sheet.paste(th, (j * size, i * (size + 14) + 14))
scenes_path = INSPECTION / "A_teng_scenes_10.jpg"
sheet.save(scenes_path, quality=80)
print(f"Planche des scènes : {scenes_path}")

print("\n=== 5) A : détection MediaPipe par résolution et par classe ===")
# Les règles ne sont pas appliquées ici : on mesure seulement si les points
# nécessaires sont détectés. Cache = coordonnées seulement (outputs/, non versionné).
cache = OUT / "teng_landmarks.npz"
names = [f"{r.split}/{r.folder}/{r.file}" for r in A.itertuples()]
if cache.exists():
    d = np.load(cache)
    assert list(d["files"]) == names
    xy, vis = d["xy"], d["vis"]
else:
    xy, vis, _ = extract_pose_images(
        [A_DIR / n for n in names], model_path(config), config["pose"]["max_people"],
        progress=lambda n, t: print(f"  {n}/{t}", end="\r") if n % 100 == 0 else None)
    np.savez_compressed(cache, xy=xy, vis=vis, files=np.array(names))
MIN_VIS = config["pose"]["min_visibility"]


def side_ok(points: list[str]) -> np.ndarray:
    """Vrai si tous les points listés sont visibles (>= seuil) d'au moins un côté."""
    ok = np.zeros(len(vis), dtype=bool)
    for s in ("l", "r"):
        ok |= np.all(vis[:, [LM[f"{n}_{s}"] for n in points]] >= MIN_VIS, axis=1)
    return ok


A["person_found"] = ~np.isnan(xy[:, LM["hip_l"], 0])
A["trunk_tibia_points"] = side_ok(["shoulder", "hip", "knee", "ankle"])
A["heel_points"] = side_ok(["heel", "foot", "ankle"])
A["all_side_points"] = side_ok(["shoulder", "hip", "knee", "ankle", "heel", "foot", "ear"])
A["resolution"] = A["width"].astype(str) + "x" + A["height"].astype(str)
checks = ["person_found", "trunk_tibia_points", "heel_points", "all_side_points"]
fail = 1 - A.groupby(["resolution", "class"])[checks].mean()
fail.insert(0, "n", A.groupby(["resolution", "class"]).size())
print("Taux d'ÉCHEC (part des images où la personne ou les points ne sont pas détectés) :")
print(fail.round(3).to_string())
print("\nPar classe seulement :")
print((1 - A.groupby("class")[checks].mean()).round(3).to_string())
print("\nPar résolution seulement :")
print((1 - A.groupby("resolution")[checks].mean()).round(3).to_string())
fail.to_csv(OUT / "teng_detection_failure.csv")

print("\n=== 6) A : phase du mouvement (descriptif, aucune règle) ===")
# Angle du genou (hanche-genou-cheville) du côté le plus visible : 180° = debout.
left = vis[:, [LM["hip_l"], LM["knee_l"], LM["ankle_l"]]].mean(1) >= \
    vis[:, [LM["hip_r"], LM["knee_r"], LM["ankle_r"]]].mean(1)


def pt(name: str) -> np.ndarray:
    return np.where(left[:, None], xy[:, LM[f"{name}_l"]], xy[:, LM[f"{name}_r"]])


A["knee_angle"] = angle_3pts(pt("hip"), pt("knee"), pt("ankle"))
A.loc[~A["trunk_tibia_points"], "knee_angle"] = np.nan
A["phase"] = pd.cut(A["knee_angle"], [0, 90, 120, 150, 181], right=False,
                    labels=["< 90° (bas)", "90-120°", "120-150°", ">= 150° (quasi debout)"])
print("Répartition des images par angle du genou (lignes = classes) :")
print(pd.crosstab(A["class"], A["phase"], normalize="index").round(2).to_string())
print(A.groupby("class")["knee_angle"].describe()[["count", "25%", "50%", "75%"]].round(0))
A.drop(columns="md5").to_csv(OUT / "teng_images.csv", index=False)

print("\n=== 7) B (Dai) : v1 et v2, fichiers, sujets, doublons ===")
NAME = re.compile(r"^p(?P<subject>\d+)_(?P<cls>[A-Za-z]+)_(?P<trial>\d+)\.csv$")


def describe_b(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.rglob("*.csv")):
        rel = f.relative_to(root)
        m = NAME.match(f.name)
        df = pd.read_csv(f)
        coords = df.filter(regex=r"^[xyzv]_\d+$")
        row = {"zip": rel.parts[0], "folder": rel.parts[1], "file": f.name, "path": str(f),
               "subject": m["subject"] if m else None, "cls": m["cls"] if m else None,
               "trial": m["trial"] if m else None, "n_frames": len(df), "n_cols": df.shape[1],
               "other_cols": ",".join(c for c in df.columns if c not in coords.columns),
               "frame_first": int(df["Frame"].iloc[0]),
               "frame_steps": ",".join(map(str, sorted(df["Frame"].diff().dropna().unique()))),
               "n_missing": int(coords.isna().sum().sum()), "md5": md5(f)}
        for k in "xyzv":
            values = df.filter(regex=fr"^{k}_\d+$").to_numpy()
            row[f"{k}_min"], row[f"{k}_max"] = np.nanmin(values), np.nanmax(values)
        rows.append(row)
    return pd.DataFrame(rows)


versions = {"v1": ext / config["external"]["dai_squat"]["v1"]["dir"],
            "v2": ext / config["external"]["dai_squat"]["dir"]}
tables = {}
for v, root in versions.items():
    raw = describe_b(root)
    copies = raw.groupby("md5")["zip"].nunique()
    conflict = raw.groupby(["cls", "subject", "trial"])["md5"].nunique()
    t = raw.drop_duplicates("md5").reset_index(drop=True)
    tables[v] = t
    print(f"\n--- {v} : {root}")
    print(f"{len(raw)} CSV extraits ; {int((copies > 1).sum())} contenus présents dans plusieurs "
          f"zip (copies identiques) ; {len(t)} fichiers uniques ; noms hors motif : "
          f"{t['subject'].isna().sum()} ; même nom mais contenu différent : {(conflict > 1).sum()}")
    print(t.groupby("cls").agg(fichiers=("file", "size"), sujets=("subject", "nunique"),
                               essais_min=("subject", lambda s: s.value_counts().min()),
                               essais_max=("subject", lambda s: s.value_counts().max()),
                               images_min=("n_frames", "min"),
                               images_med=("n_frames", "median"),
                               images_max=("n_frames", "max")).to_string())
    subj = {c: sorted(g["subject"].unique()) for c, g in t.groupby("cls")}
    for c, s in subj.items():
        print(f"  {c:9s} sujets : {s}")
    every = set.intersection(*map(set, subj.values()))
    print(f"  sujets distincts : {t['subject'].nunique()} ; présents dans les 3 classes : "
          f"{len(every)} {sorted(every)}")
    print(f"  colonnes : {t['n_cols'].value_counts().to_dict()} ; autres que x/y/z/v : "
          f"{t['other_cols'].value_counts().to_dict()} ; valeurs manquantes : "
          f"{t['n_missing'].sum()}")
    print(f"  Frame : premier numéro {t['frame_first'].value_counts().head(5).to_dict()} ; "
          f"pas {t['frame_steps'].value_counts().to_dict()}")
    print("  plages : " + " ; ".join(f"{k} [{t[f'{k}_min'].min():.2f}, {t[f'{k}_max'].max():.2f}]"
                                     for k in "xyzv"))
    t.drop(columns="md5").to_csv(OUT / f"dai_files_{v}.csv", index=False)

v1, v2 = tables["v1"], tables["v2"]
key = ["cls", "subject", "trial"]
both = v1.merge(v2, on=key, suffixes=("_v1", "_v2"))
print("\n--- Comparaison v1 / v2 (même classe, sujet, essai)")
print(f"Fichiers communs : {len(both)} ; contenu identique : "
      f"{int((both['md5_v1'] == both['md5_v2']).sum())} ; nombre d'images identique : "
      f"{int((both['n_frames_v1'] == both['n_frames_v2']).sum())} ; seulement en v1 : "
      f"{len(v1) - len(both)} ; seulement en v2 : {len(v2) - len(both)}")
print(f"Sujets v1 absents de v2 : {sorted(set(v1['subject']) - set(v2['subject']))}")
print(f"Sujets nouveaux en v2 : {sorted(set(v2['subject']) - set(v1['subject']))}")

print("\n=== 8) B (v2) : géométrie et contenu de chaque fichier ===")
geo = []
for r in v2.itertuples():
    xn, vv, _ = read_normalized(Path(r.path))
    hip_y = (xn[:, LM["hip_l"], 1] + xn[:, LM["hip_r"], 1]) / 2
    amp = hip_y.max() - hip_y.min()
    peaks, _ = find_peaks(hip_y, prominence=0.3 * amp)
    geo.append({
        "cls": r.cls, "subject": r.subject, "file": r.file, "n_bottoms": len(peaks),
        "starts_up": hip_y[0] < hip_y.min() + 0.2 * amp,
        "ends_up": hip_y[-1] < hip_y.min() + 0.2 * amp,
        "nose_above_ankles": float(np.mean(xn[:, LM["nose"], 1] < xn[:, LM["ankle_l"], 1])),
        "vis_left_leg": float(np.median(vv[:, [LM["hip_l"], LM["knee_l"], LM["ankle_l"]]])),
        "vis_right_leg": float(np.median(vv[:, [LM["hip_r"], LM["knee_r"], LM["ankle_r"]]])),
        # Écart gauche/droite des hanches en x, rapporté à la hauteur du tronc en y :
        # petit en vue de profil (indicatif : dépend du rapport largeur/hauteur)
        "hip_sep_x": float(np.median(np.abs(xn[:, LM["hip_l"], 0] - xn[:, LM["hip_r"], 0]))),
        "torso_y": float(np.median(np.abs(xn[:, LM["shoulder_l"], 1] - xn[:, LM["hip_l"], 1]))),
        "aspect_est": estimate_aspect(xn, vv),
        **{f"instab_{name}": length_instability(xn, vv, a)
           for name, a in [("9:16", 9 / 16), ("3:4", 3 / 4), ("1:1", 1.0),
                           ("4:3", 4 / 3), ("16:9", 16 / 9)]},
    })
G = pd.DataFrame(geo)
G["visible_side"] = np.where(G["vis_left_leg"] >= G["vis_right_leg"], "gauche", "droite")
G.to_csv(OUT / "dai_v2_geometry.csv", index=False)
print("Points bas (répétitions) par fichier :", G["n_bottoms"].value_counts().to_dict())
print(f"Commence debout : {G['starts_up'].mean():.0%} ; finit debout : {G['ends_up'].mean():.0%}")
print(f"Nez au-dessus des chevilles (y vers le bas) : {G['nose_above_ankles'].mean():.0%} "
      "des images")
print("Côté le plus visible (jambe), par classe :")
print(pd.crosstab(G["cls"], G["visible_side"]).to_string())
print("Côté le plus visible, par sujet ayant plusieurs classes :")
multi = G.groupby("subject")["cls"].nunique()
print(pd.crosstab([G["subject"]], [G["cls"], G["visible_side"]])
      .loc[multi[multi > 1].index].to_string())
print(f"Écart des hanches en x / hauteur du tronc en y (médiane) : "
      f"{(G['hip_sep_x'] / G['torso_y']).median():.2f}")

print("\n=== 9) B (v2) : quel rapport largeur/hauteur ? (hypothèses testées) ===")
index = pd.read_csv(ROOT / config["dataset"]["index_csv"], dtype={"subject": str})
val = []
for r in index[index["view"] == "side"].itertuples():
    f = ROOT / "outputs" / "landmarks" / f"{Path(r.filename).stem}.npz"
    if f.exists():
        p = PoseSeries.load(f)
        val.append((p.width / p.height,
                    estimate_aspect(p.xy / np.array([p.width, p.height]), p.vis)))
val = np.array(val)
err = val[:, 1] - val[:, 0]
print("Validation sur le dataset Mendeley (résolution connue, vue de profil) : "
      f"{len(val)} vidéos ; vrai rapport {np.unique(val[:, 0].round(4))} ; erreur médiane "
      f"{np.median(err):+.3f}, min {err.min():+.3f}, max {err.max():+.3f}")
print("Estimation sur B v2 (rapport qui stabilise le plus cuisse + tibia) :")
print(G.groupby("cls")["aspect_est"].describe()[["count", "25%", "50%", "75%"]].round(2))
print(f"Toutes classes : médiane {G['aspect_est'].median():.2f}, quantiles 10-90 % "
      f"{np.round(G['aspect_est'].quantile([.1, .9]).to_numpy(), 2)}")
print("Instabilité médiane des longueurs selon l'hypothèse (plus petit = plus plausible) :")
print(G[[c for c in G if c.startswith("instab_")]].median().round(3).to_string())
print(f"(grille testée : {ASPECT_GRID.min()} à {ASPECT_GRID.max()})")

print("\n=== 10) Non documenté / non vérifiable ===")
print("Synthèse dans le message de fin d'étape et docs/DATA_LICENSES.md.")
