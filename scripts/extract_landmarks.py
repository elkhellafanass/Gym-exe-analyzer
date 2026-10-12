"""Extrait les points MediaPipe des vidéos de data/index.csv vers outputs/landmarks/.

Pourquoi un script séparé ? L'extraction est l'étape lente (~9 s par vidéo,
~25 min pour 150 vidéos). Une fois en cache, train.py et evaluate.py
s'exécutent en quelques secondes. Le cache ne contient que des coordonnées.

Usage : python scripts/extract_landmarks.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.analysis import load_or_extract  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402

CACHE_DIR = ROOT / "outputs" / "landmarks"


def main() -> None:
    config = load_config()
    index = pd.read_csv(ROOT / config["dataset"]["index_csv"], dtype={"subject": str})
    t0 = time.time()
    for n, row in enumerate(index.itertuples(), 1):
        cached = (CACHE_DIR / f"{Path(row.filename).stem}.npz").exists()
        load_or_extract(Path(row.path), config, CACHE_DIR)
        print(f"[{n}/{len(index)}] {row.filename} {'(cache)' if cached else ''}", flush=True)
    print(f"Terminé en {time.time() - t0:.0f} s -> {CACHE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
