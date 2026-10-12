# Pipeline de bout en bout

> Outil pédagogique d'aide à l'analyse technique. Ce n'est pas un dispositif médical et il ne prédit aucune
> blessure. **Conçu comme aide pédagogique ; utilité non évaluée.**

## Schéma

```
 vidéo (mp4/mov)
   │
   ▼  src/pose.py ─ MediaPipe Pose Landmarker (33 points / image, en pixels)
   │                · plusieurs personnes → on garde la plus grande
   │                · visibilité < 0,5 → point absent ; trous ≤ 3 images interpolés
   │                · lissage Savitzky-Golay (7 images)
   ▼  src/angles.py ─ indicateurs image par image (2D)
   │                · profil : genou, hanche, tronc, tibia, profondeur, talon, cou
   │                · face   : ratio genoux/chevilles, bascule du bassin, asymétrie
   ▼  src/reps.py ─ répétitions = pics de la hauteur de hanche (find_peaks) + tempo
   │
   ▼  src/rules.py ─ règles de config.yaml (source + statut), filtrées par vue
   │                · défaut retenu si présent sur ≥ 2 répétitions (Myer et al., 2014)
   ▼
 résultats par répétition + synthèse vidéo + vidéo annotée (src/video.py)
```

Le même code (`src/analysis.py`) sert à l'application et aux scripts d'évaluation.

**Évaluation (phase B, décidée le 12/10/2026) :**
- **Analyse principale (option A) :** les seuils des règles restent figés (littérature). Un classifieur
  (régression logistique, forêt aléatoire) apprend le label good/bad à partir des **valeurs continues** des
  indicateurs. Il est validé en *leave-one-subject-out* : les 6 vidéos d'un sujet sont toujours dans le même pli,
  la normalisation est apprise dans chaque pli, et les hyperparamètres sont fixés à l'avance.
- **Analyse secondaire (option B), séparée :** les seuils sont recalibrés à l'intérieur de chaque pli
  leave-one-subject-out. Ces seuils sont étiquetés « calibrés sur le dataset », jamais « sourcés ».
- **Annotation en aveugle** des 50 vidéos de profil par l'auteur (label caché, ordre aléatoire fixe, aucune
  détection affichée). Elle permet d'évaluer chaque règle contre un type d'erreur.

## Limites déjà identifiées

1. **Contamination partielle de la conception.** Les indicateurs et leur façon de mesurer ont été conçus *après*
   avoir exploré les 25 sujets (phase A, `scripts/explore_rules.py`). En particulier, le passage à une mesure du
   talon relative au début de chaque répétition a été décidé en voyant que la mesure absolue se déclenchait sur
   100 % des vidéos good. Aucune donnée « jamais vue » ne reste pour tester ces choix. Les résultats de la phase B
   peuvent donc être légèrement optimistes, et toute recalibration (option B) est en partie contaminée.
2. **Désaccord de référentiel sur la profondeur.** La règle suit Myer et al. (2014) : hanche au moins au niveau du
   genou. La mesure a été vérifiée visuellement : elle est correcte. Mais les squats étiquetés « good » du dataset
   ne respectent souvent pas ce critère (92 % des vidéos good signalées). Le dataset applique un critère **non
   documenté**. Ce n'est pas une erreur de mesure, mais deux standards différents. Aucun des deux n'est une vérité
   absolue.
3. **Talons non mesurables de façon fiable à ces résolutions.**
   - Mesure absolue : le point talon de MediaPipe descend au point bas du squat, ce qui produisait des faux
     signaux.
   - Mesure relative au début de répétition : elle ne voit pas une personne déjà sur la pointe des pieds (sujet
     016 bad, vérifié visuellement), et elle se déclenche sur 80 % des good contre 76 % des bad.
   - Angle du pied (talon → pointe) : il ne sépare pas good et bad (p = 0,17).

   Résolutions du dataset : de 348x624 à 810x1440.
4. **Aucun écart good/bad en vue de profil.** Profondeur, talons, tronc − tibia et tête ne séparent pas good et
   bad. La seule différence significative (tronc − tibia, p = 0,013, sur 8 tests) va dans le sens **inverse** de
   la règle : les vidéos bad ont les genoux plus en avant. Seules les règles de face montrent un écart (valgus
   16 % contre 28 %, bassin 0 % contre 16 % des vidéos), non significatif.

5. **Annotation seulement en partie aveugle.** Avant d'annoter, l'auteur a vu les planches d'inspection et la
   description de certains sujets (ex. 016 bad sur la pointe des pieds, 004 et 008 good pas assez profonds). Il peut
   donc reconnaître certaines personnes malgré l'anonymisation (V01 à V50).

Autres limites, détaillées dans `docs/INSPECTION.md` et `docs/SOURCES.md` :
- 15 fps : tempo à ±0,07 s ;
- plusieurs personnes à l'image, cas hors du périmètre prévu par la fiche du modèle ;
- mesures 2D (projections) ;
- 25 sujets seulement ;
- un label par vidéo, sans type d'erreur ;
- inspection visuelle faite par un non-expert.

## Plan des phases B et C (mis à jour le 12/10/2026, cahier des charges « rapport de correction »)

L'utilisateur envoie une vidéo de squat (importée ou filmée sur son téléphone) et reçoit un rapport de correction
complet. Aucune vidéo « idéale » n'est générée. **Toute fiabilité affichée vient de l'évaluation réelle**
(leave-one-subject-out et annotation en aveugle), jamais d'une estimation a priori.

**Phase B : évaluation (après la fin de l'annotation)**
1. `src/features.py` : un seul module construit le vecteur d'indicateurs d'une vidéo (médianes par répétition). Il
   sert à l'entraînement ET à l'app, pour garantir les mêmes calculs.
2. `scripts/train.py` / `scripts/evaluate.py`, en étapes numérotées :
   - régression logistique et forêt aléatoire, en leave-one-subject-out ;
   - résultats par vue ;
   - précision, rappel, F1, matrice de confusion, IC par bootstrap **par sujet**.
3. **Probabilités et incertitude.** On évalue la calibration des probabilités en LOSO (score de Brier, courbe de
   fiabilité). Une zone « incertain » autour de 0,5 est proposée : sa largeur est un choix d'ingénierie, déclaré
   comme tel.
4. **Fiabilité de chaque règle** → `docs/results/rule_reliability.csv`, lu par l'app :
   - règles de profil : sensibilité et spécificité contre l'annotation en aveugle (n = 50 vidéos, avec IC) ;
   - règles de face et de diagonale : pas d'annotation, donc seulement les taux good/bad. Elles sont affichées
     « fiabilité non évaluée contre annotation ».
5. **Analyse secondaire (option B),** clairement séparée : seuils recalibrés en LOSO.
6. **Modèle publié dans le dépôt** (`models/*.joblib`, quelques Ko, appris sur des indicateurs dérivés, sans
   image). C'est nécessaire pour la version en ligne : `.gitignore` sera ajusté pour ce seul fichier.

**Phase C : application et rapport**
1. **Verdict global :** probabilité « bad » + zone d'incertitude + performance LOSO mesurée pour la vue choisie.
   Jamais formulé comme une certitude. La vue (face/profil/diagonale) est déclarée par l'utilisateur.
2. **Détail par répétition :** indicateurs, règles déclenchées, fiabilité de chaque règle (table de la phase B, plus
   les mentions « fiabilité réduite en diagonale » et « talons : indicateur peu fiable »).
   Le modèle donne un verdict **par vidéo** seulement : le dataset n'a pas de label par répétition.
3. **Correction écrite pour chaque défaut :**
   - texte en français simple, tiré de `config.yaml`, avec la source du seuil, son statut (« à valider par un
     expert » si besoin) et sa fiabilité mesurée ;
   - aucun conseil médical.
4. **Vidéo annotée :**
   - squelette et angles ;
   - surlignage des répétitions fautives (bandeau rouge pendant la répétition) ;
   - frise chronologique Plotly des répétitions et des défauts sous la vidéo.
5. **Comparaison côte à côte avec une vidéo good du dataset** (local uniquement, masquée si le dataset est absent,
   donc toujours en ligne) :
   - même vue ;
   - images du point bas mises côte à côte ;
   - avertissement : une vidéo « good » du dataset n'est pas un modèle idéal (par exemple, beaucoup ne respectent
     pas la profondeur de Myer).
6. **Export du rapport :**
   - **HTML autonome** en priorité (graphiques intégrés) ;
   - PDF via l'impression du navigateur, ou en option si le temps le permet ;
   - le rapport ne contient jamais d'image du dataset ;
   - la vidéo de l'utilisateur est traitée dans un dossier temporaire puis supprimée.
7. **Filmer dans l'app** (webcam, streamlit-webrtc) reste un bonus. Le parcours prévu est : filmer avec le téléphone,
   puis importer.

**Ce qui est prioritaire si le temps manque (délai : 2 semaines) :** 1 à 4 et l'export HTML sont indispensables.
La comparaison côte à côte, le PDF natif et la webcam sont optionnels.

## Protocole de la phase B (figé et committé AVANT la première exécution)

Ce protocole a été écrit et committé avant d'exécuter `train.py` et `evaluate.py` sur les données réelles. Le
commit le prouve (historique git). Toute modification ultérieure serait signalée comme telle.

| Élément | Choix fixé d'avance |
|---|---|
| Unité d'analyse | la vidéo (un label par vidéo) ; une ligne = médiane des répétitions de chaque indicateur |
| Indicateurs | ceux des règles applicables à la vue + rapport de tempo (`src/features.py`) |
| Modèles | un par vue ; régression logistique (C = 1) et forêt aléatoire (500 arbres, 3 vidéos min. par feuille) |
| Modèle de l'app | régression logistique (explicabilité), quel que soit le résultat ; la forêt sert de comparaison |
| Validation | leave-one-subject-out, 25 plis, les 6 vidéos d'un sujet dans le même pli |
| Prétraitement | imputation (médiane) et normalisation dans un Pipeline, apprises dans chaque pli |
| Hyperparamètres | aucune recherche ; valeurs de `config.yaml` (`evaluation`) |
| Décision | « bad » si probabilité > 0,5 ; zone « incertain » [0,35 ; 0,65] (choix d'ingénierie, non optimisé) |
| Détecteur à seuils (option A) | vidéo « bad » si ≥ 1 règle applicable à la vue retient un défaut (≥ 2 répétitions) |
| Vidéo sans répétition | « non analysable », exclue des métriques et comptée |
| IC | bootstrap percentile 95 %, 2000 tirages **de sujets** (graine 42) |
| Références | hasard (pile ou face, 10 000 tirages), « toujours bad », « toujours good », probabilité constante 0,5 (Brier = 0,25) |
| Test de permutation | régression logistique, 1000 permutations good/bad à l'intérieur de chaque paire (sujet, vue) |
| Calibration | score de Brier ; courbe de fiabilité en 5 groupes de taille égale |
| Accord annotateur / label | kappa de Cohen ; principal : « incertain » exclu ; sensibilité : « incertain » = désaccord / = accord |
| Règle vs annotation | défaut « présent » = noté sur quelques répétitions ou la plupart (principal) ; « la plupart » seulement (secondaire) ; correspondance : depth↔depth, heel_rise↔heel_rise, trunk_vs_tibia↔trunk_lean, head_position↔head |
| Option B (secondaire, script séparé) | seuil choisi dans chaque pli par l'indice de Youden, sens de la règle inchangé, égalité → seuil le plus proche de la littérature |

## Questions probables du jury

*(Les 15 questions et leurs réponses seront rédigées à partir des résultats réels de la phase B.)*
