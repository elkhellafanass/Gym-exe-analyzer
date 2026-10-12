# Inspection des vidéos "bad" (phase 0)

Le dataset donne un seul label par vidéo (good/bad), **sans décrire le type d'erreur**. Avant d'écrire des
règles, on a regardé les vidéos.

**Méthode** (`scripts/inspect_bad_videos.py`) : MediaPipe Pose sur les 150 vidéos. Le point bas de chaque
répétition est le maximum local de la hauteur de hanche. On garde l'image du point bas de la répétition
médiane, puis on compare good et bad pour le même sujet et la même vue. Les planches d'images restent en local
(`docs/inspection/`, non versionné, personnes identifiables). Les indicateurs sont bruts (non lissés) et
exploratoires.

## 1. Observations visuelles (planches, 25 sujets x 3 vues)

| Motif observé dans les vidéos "bad" | Où on le voit | Fréquence (estimation visuelle) |
|---|---|---|
| Talons décollés / appui sur l'avant du pied, genoux très en avant | profil | fréquent (≈ 10 sujets sur 25 : 001, 004, 005, 006, 008, 014, 016, 018, 019, 021, 022) |
| Tronc très penché / dos arrondi, tête baissée | profil, diagonale | plusieurs sujets (002, 003, 007, 010, 015, 024) |
| Écartement des pieds plus étroit, genoux qui se rapprochent (valgus) | face | quelques sujets (004, 017, 018 nets) |
| Squat partiel (moins profond) | profil | quelques sujets (009, 017) |

Ces fréquences sont estimées visuellement sur **une seule image par vidéo** par un non-expert. Ce sont des
hypothèses, pas des annotations.

## 2. Comparaison chiffrée appariée (même sujet, même vue ; médiane des répétitions ; test de Wilcoxon)

Vue de **profil** (la seule où les angles du plan sagittal sont lisibles en 2D) :

| Indicateur | good (médiane) | bad (médiane) | sujets où bad > good | p |
|---|---|---|---|---|
| Angle du genou au point bas (°) | 70.0 | 63.3 | 10/25 | 0.49 |
| Angle de hanche (°) | 57.1 | 65.8 | 15/25 | 0.23 |
| Inclinaison du tronc / verticale (°) | 39.6 | 35.7 | 8/25 | 0.19 |
| Profondeur (hanche sous genou, / longueur de cuisse) | -0.16 | -0.17 | 13/25 | 0.41 |

Vue de **face** : ratio largeur genoux / largeur chevilles 1.35 (good) contre 1.34 (bad), 15/25, p = 0.44.
Les angles genou/hanche « de face » diffèrent significativement (p < 0.01), mais ce sont des **projections 2D**
d'angles sagittaux, faussées par la perspective. Ils ne sont pas interprétables comme des angles réels.

## 3. Conclusions

1. **"Bad" est hétérogène** : plusieurs types d'erreur différents selon les sujets, souvent combinés.
2. **Ni la profondeur ni l'inclinaison du tronc ne séparent good et bad** au niveau du groupe (profil, p > 0.18).
   Une règle « profondeur insuffisante » seule ne suffira pas à retrouver le label.
3. Le motif le plus récurrent à l'œil, le **décollement des talons**, n'a pas encore été mesuré. MediaPipe fournit
   des points talon (29/30) et pointe du pied (31/32) : c'est la première vérification de la phase A.
4. Le dos arrondi n'est **pas mesurable directement** : MediaPipe n'a aucun point sur la colonne (seulement épaules
   et hanches). Au mieux, on approxime avec l'inclinaison du tronc et la position de la tête.

## 4. Problèmes de données repérés

- **Plusieurs personnes à l'image** (salles de sport fréquentées) : MediaPipe peut suivre la mauvaise personne.
- **Segments hors squat** : par exemple, sujet 007 good diagonale avec ramassage d'haltère. Sujet 021 good profil :
  1 seule répétition détectée en 12 s (la personne reste debout l'essentiel du temps).
- 4 résolutions différentes (de 348x624 à 810x1440), 15 fps, durées de 8 à 39 s.
- Taux de détection de pose ≥ 98.6 % sur chaque vidéo (mais détection ≠ bonne personne ni points exacts).

## 5. Vérification des règles sur les 150 vidéos (phase A)

Script : `scripts/explore_rules.py`. Le pipeline complet a été appliqué aux 150 vidéos. Pour chaque sujet et chaque
vue, on compare la médiane (sur les répétitions) entre la vidéo good et la vidéo bad (test de Wilcoxon apparié,
n = 25 sujets ; 24 en diagonale, car aucune répétition n'est détectée pour le sujet 007 good).

| Vue | Règle | Médiane good → bad | Sujets où bad > good | p | Vidéos avec défaut retenu (good / bad) |
|---|---|---|---|---|---|
| profil | profondeur (`depth`) | -0,16 → -0,17 | 9/25 | 0,14 | 92 % / 100 % |
| profil | talons (`heel_rise`) | 0,11 → 0,16 | 15/25 | 0,09 | 80 % / 76 % |
| profil | tronc − tibia (`trunk_vs_tibia`) | 11,6° → 5,9° | 5/25 | **0,013** | 60 % / 60 % |
| profil | tête (`head_position`) | 11,8° → 11,6° | 13/25 | 0,39 | 24 % / 16 % |
| face | valgus (`knee_valgus`) | 1,20 → 1,20 | 13/25 | 0,51 | 16 % / 28 % |
| face | bassin (`hip_shift`) | 2,8° → 3,1° | 14/25 | 0,40 | 0 % / 16 % |
| diagonale | valgus (approx.) | 1,14 → 1,09 | 11/24 | 0,78 | 25 % / 48 % |
| diagonale | bassin (approx.) | 9,7° → 12,3° | 13/24 | 0,27 | 75 % / 84 % |

**Lecture honnête :**
1. **Les règles issues de Myer et al. (2014) ne reproduisent pas le label good/bad du dataset.** Elles se
   déclenchent presque autant sur les vidéos good que bad.
2. **Profondeur : la mesure est correcte, mais le standard diffère.** Une vérification visuelle (squelette superposé,
   sujets 004 et 008 good) confirme que la hanche reste au-dessus du genou. Le « good » du dataset n'exige pas
   la profondeur « cuisses parallèles ».
3. **Tronc − tibia : seule différence significative, et dans le sens inverse de la règle.** Les vidéos bad ont un
   tronc *moins* penché par rapport au tibia (5/25 sujets bad > good, p = 0,013). C'est cohérent avec
   l'observation visuelle « genoux très en avant / talons décollés » : le tibia s'incline davantage. Avec 8 tests,
   ce p isolé doit être lu avec prudence (comparaisons multiples).
4. **Talons : mesure peu fiable.** Une référence « sol » unique sur toute la vidéo déclenchait la règle sur 100 % des
   vidéos good, car le point talon de MediaPipe descend au point bas. D'où le passage à une mesure relative au début de
   chaque répétition. Mais celle-ci ne voit pas une personne **déjà** sur la pointe des pieds au départ (sujet 016
   bad, vérifié visuellement). L'angle du pied (talon → pointe) a été testé : il ne sépare pas good et bad (p = 0,17),
   y compris pour le sujet 016 (14° good contre 19° bad). Conclusion : à ces résolutions, les points du pied ne sont
   pas assez précis pour mesurer les talons.
5. **Diagonale :** la règle du bassin se déclenche sur 75 % des vidéos good, ce qui confirme que les mesures
   frontales en vue diagonale sont faussées par la perspective.

Conséquence pour la phase B : les règles restent un **retour pédagogique** basé sur la littérature. Elles ne sont
**pas** un détecteur du label du dataset. La phase B mesure formellement (leave-one-subject-out) si un classifieur
appris sur ces indicateurs fait mieux que le hasard.
