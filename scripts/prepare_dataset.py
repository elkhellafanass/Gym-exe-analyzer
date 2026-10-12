"""PHASE 0 : indexation du dataset de squats -> data/index.csv

Usage (depuis la racine du projet, venv activé) :
    python scripts/prepare_dataset.py

Le dossier du dataset vient de config.yaml, ou de la variable
d'environnement SQUAT_DATA_DIR si elle est définie.
Aucune vidéo n'est copiée : on écrit seulement un tableau de métadonnées.
"""

import sys
from pathlib import Path

# Permet d'importer src/ quand on lance le script directement.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config, squat_data_dir  # noqa: E402
from src.dataset import build_index, missing_combinations  # noqa: E402


def main() -> None:
    config = load_config()
    data_dir = squat_data_dir(config)

    print("Étape 1 - Dossier du dataset")
    print(f"  {data_dir}")
    if not data_dir.exists():
        sys.exit("  ERREUR : dossier introuvable. Vérifiez config.yaml ou SQUAT_DATA_DIR.")

    print("Étape 2 - Lecture des noms de fichiers et des en-têtes vidéo")
    index, skipped = build_index(data_dir, config["dataset"]["filename_pattern"])
    print(f"  {len(index)} vidéos indexées, {len(skipped)} fichier(s) ignoré(s)")
    for path in skipped:
        print(f"  ignoré (nom inattendu ou dossier incohérent) : {path}")

    print("Étape 3 - Écriture de l'index")
    out = ROOT / config["dataset"]["index_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / config["dataset"]["own_videos_dir"]).mkdir(parents=True, exist_ok=True)
    index.to_csv(out, index=False)
    print(f"  -> {out.relative_to(ROOT)}")

    print("\n=== RÉSUMÉ ===")
    subjects = sorted(index["subject"].unique())
    print(f"Sujets : {len(subjects)} ({subjects[0]} à {subjects[-1]})")
    print("\nNombre de vidéos par qualité et par vue :")
    print(index.pivot_table(index="quality", columns="view", values="filename",
                            aggfunc="count", margins=True, margins_name="total"))
    missing = missing_combinations(index)
    print(f"\nCombinaisons sujet/qualité/vue manquantes : {missing or 'aucune'}")
    print("\nFPS observés :", index["fps"].value_counts().to_dict())
    res = (index["width"].astype(str) + "x" + index["height"].astype(str)).value_counts()
    print("Résolutions (largeur x hauteur) :", res.to_dict())
    d = index["duration_s"]
    print(f"Durée (s) : min {d.min():.1f} / médiane {d.median():.1f} / max {d.max():.1f}"
          f" / total {d.sum() / 60:.1f} min")


if __name__ == "__main__":
    main()
