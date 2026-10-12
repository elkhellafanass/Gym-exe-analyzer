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

**Phase B : évaluation (faite le 12/10/2026, résultats plus bas)**
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

## Résultats de la phase B (12/10/2026)

Commandes : `python scripts/train.py`, puis `python scripts/evaluate.py` (≈ 4 min, dont le test de permutation),
puis `python scripts/evaluate_option_b.py`. Tableaux dans `docs/results/`, figures dans `docs/figures/`.
Aucune modification du protocole après exécution. Une vidéo est non analysable (0 répétition détectée :
sujet 007, good, diagonale) : 149 vidéos évaluées.

**Lecture rapide : les résultats sont faibles.** Le classifieur fait un peu mieux que le hasard, de façon
statistiquement détectable, mais ses probabilités ne sont pas plus fiables qu'un « 50 % » constant. Les règles à
seuils figés ne séparent pas good et bad, et ne détectent pas les défauts que l'annotateur voit.

### 1. Accord de l'annotateur avec le label du dataset (50 vidéos de profil)

| | label good | label bad |
|---|---|---|
| annoté « bonne » | 2 | 0 |
| annoté « mauvaise » | 11 | 24 |
| annoté « incertain » | 12 | 1 |

| Convention | n | Kappa de Cohen [IC 95 %] | Accord brut |
|---|---|---|---|
| « incertain » exclu (principale) | 37 | **0,19 [0,00 ; 0,44]** | 0,70 |
| « incertain » = désaccord (pire cas) | 50 | 0,04 [−0,08 ; 0,16] | 0,52 |
| « incertain » = accord (meilleur cas, artificiel) | 50 | 0,56 [0,36 ; 0,76] | 0,78 |

L'accord est faible. L'annotateur juge presque toutes les vidéos imparfaites (35 « mauvaise », 2 « bonne ») :
il a un standard plus strict que le dataset. Son doute se concentre sur les vidéos good (12 « incertain » sur 13).
Ni l'annotateur ni le dataset ne sont une vérité de référence : ce sont deux juges non experts (pour le dataset,
le protocole d'étiquetage n'est pas documenté).

Mais l'annotateur voit **plus de défauts dans les vidéos bad** (`annotation_by_label.csv`, sur 25 vidéos) :
profondeur 14 contre 7, talons 9 contre 3, tronc 19 contre 10, genoux en avant 20 contre 11, tête 22 contre 20.
Une différence visible à l'œil existe donc, au moins en partie (annotation seulement en partie aveugle, voir
limite 5).

### 2. Règles de profil contre l'annotation en aveugle (seuils figés)

Défaut « présent » = noté sur quelques répétitions ou la plupart. Détection = part des vidéos où l'annotateur voit
le défaut et où la règle le signale. Fausses alertes = part des vidéos où l'annotateur ne voit rien et où la règle
signale quand même.

| Règle | Présent (annotation) | Détection [IC 95 %] | Fausses alertes [IC 95 %] | Kappa |
|---|---|---|---|---|
| depth | 21/50 | 0,95 [0,85 ; 1,00] | 0,97 [0,90 ; 1,00] | −0,01 |
| heel_rise | 12/50 | 0,75 [0,40 ; 1,00] | 0,79 [0,67 ; 0,89] | −0,02 |
| trunk_vs_tibia | 29/50 | 0,62 [0,41 ; 0,81] | 0,57 [0,31 ; 0,81] | 0,05 |
| head_position | 42/50 | 0,24 [0,10 ; 0,39] | 0,00 (0/8)* | 0,09 |

\* Le bootstrap donne [0 ; 0] quand aucun événement n'est observé : c'est un artefact. Avec 0 sur 8 vidéos, la
borne haute exacte (Clopper-Pearson, sans tenir compte des sujets) est 0,37.

**Aucune règle n'est en accord avec l'annotateur** (kappa entre −0,02 et 0,09). Trois règles signalent presque
autant quand le défaut est absent que quand il est présent : elles ne transmettent aucune information. La règle de
la tête est l'inverse : elle ne se trompe pas quand elle signale (0/8), mais elle manque 3 défauts sur 4.

Attention : les définitions de l'annotation ne sont pas exactement celles des règles. Exemple : profondeur =
« hanche *nettement* au-dessus du genou » dans l'outil d'annotation, « hanche au moins au niveau du genou » chez
Myer. Une partie des fausses alertes de `depth` vient de cet écart de standard. Résultats avec la définition
secondaire (« la plupart des répétitions ») : mêmes conclusions (`rule_vs_annotation.csv`).

### 3. Performances contre le label du dataset (leave-one-subject-out)

Toutes vues confondues (149 vidéos), IC 95 % par bootstrap sur les sujets :

| Méthode | Précision | Rappel | F1 | Exactitude équilibrée | AUC |
|---|---|---|---|---|---|
| Régression logistique (modèle de l'app) | 0,62 [0,55 ; 0,71] | 0,57 [0,47 ; 0,68] | 0,60 [0,52 ; 0,67] | **0,61 [0,55 ; 0,67]** | **0,60 [0,54 ; 0,67]** |
| Forêt aléatoire | 0,51 [0,45 ; 0,57] | 0,55 [0,43 ; 0,67] | 0,53 [0,45 ; 0,60] | 0,51 [0,45 ; 0,57] | 0,55 [0,48 ; 0,63] |
| Détecteur à seuils (règles, option A) | 0,55 [0,51 ; 0,59] | 0,76 [0,69 ; 0,83] | 0,64 [0,59 ; 0,68] | 0,56 [0,51 ; 0,62] | – |
| Trivial : toujours « bad » | 0,50 | 1,00 | **0,67** | 0,50 | – |
| Trivial : toujours « good » | non définie | 0,00 | 0,00 | 0,50 | – |
| Hasard (pile ou face) | 0,50 [0,42 ; 0,58] | 0,49 [0,39 ; 0,61] | 0,50 [0,41 ; 0,59] | 0,50 [0,42 ; 0,58] | 0,50 |

Par vue (exactitude équilibrée [IC 95 %]) :

| Vue | Régression logistique | Forêt aléatoire | Détecteur à seuils | Matrice du détecteur (TN FP / FN TP) |
|---|---|---|---|---|
| Profil (n = 50) | 0,58 [0,48 ; 0,68] | 0,46 [0,34 ; 0,58] | 0,52 [0,50 ; 0,56] | 1 24 / 0 25 |
| Face (n = 50) | 0,64 [0,54 ; 0,74] | 0,56 [0,44 ; 0,70] | 0,60 [0,48 ; 0,72] | 21 4 / 16 9 |
| Diagonale (n = 49) | 0,61 [0,50 ; 0,72] | 0,51 [0,39 ; 0,63] | 0,56 [0,48 ; 0,66] | 5 19 / 2 23 |

Matrices complètes : `docs/figures/confusion_matrices.png`.

Ce qu'il faut retenir :
- **Le F1 trompe ici.** « Toujours bad » obtient F1 = 0,67, mieux que tous les modèles, sans rien regarder. Avec
  des classes équilibrées, l'exactitude équilibrée et l'AUC sont les comparaisons honnêtes (référence : 0,5).
- **Régression logistique : au-dessus du hasard, mais de peu.** Test de permutation (1000 permutations
  good/bad à l'intérieur de chaque paire sujet-vue, LOSO refait à chaque fois) : AUC observée 0,60 contre 0,57 au
  95e centile sous l'hypothèse nulle, p = 0,002 ; exactitude équilibrée 0,61, p = 0,002. Par vue, les IC touchent
  ou incluent 0,5 : on ne peut pas dire quelle vue fonctionne le mieux.
- **La forêt aléatoire ne fait pas mieux que le hasard** (IC incluant 0,5). Avec 48 vidéos par entraînement, un
  modèle plus flexible apprend surtout le bruit.
- **Le détecteur à seuils de profil équivaut à « toujours bad »** : il signale 24 vidéos good sur 25. De face,
  il est spécifique (21/25 good non signalées) mais manque 16 bad sur 25.

**Ce que la régression logistique a appris** (`feature_importance.png`, coefficients standardisés, moyenne des
25 plis) : le tempo est l'indicateur le plus utilisé dans les trois vues (descente relativement plus courte →
« bad »). De profil : moins de profondeur → « bad » (sens de Myer) ; tronc moins penché que le tibia → « bad »
(sens **inverse** de la règle, comme en phase A) ; tête → « bad » (sens de Myer). Le valgus change de signe entre
face (+0,13) et diagonale (−0,14) : il n'apporte pas d'information stable. Les coefficients varient peu d'un pli à
l'autre, mais c'est attendu (deux plis partagent 23 sujets sur 24) : cela ne prouve pas qu'ils se généralisent.

### 4. Fiabilité des probabilités

| Modèle | Brier (toutes vues) [IC 95 %] | Référence « 0,5 constant » | Vidéos en zone « incertain » [0,35 ; 0,65] | Exactitude hors zone |
|---|---|---|---|---|
| Régression logistique | 0,251 [0,229 ; 0,274] | 0,250 | 94/149 (63 %) | 0,64 (n = 55) |
| Forêt aléatoire | 0,263 [0,243 ; 0,284] | 0,250 | 90/149 (60 %) | 0,59 (n = 59) |

**Les probabilités ne sont pas plus fiables que de répondre « 50 % » à chaque vidéo** (Brier égal à la
référence). La courbe de fiabilité (`reliability_curve.png`) n'est pas monotone : le groupe prédit à 0,49 contient
31 % de bad, celui prédit à 0,42 en contient 37 %, celui à 0,30 en contient 50 %. Conséquence pour l'app : la
probabilité ne doit pas être présentée comme « X % de chances d'être mauvais ». Le verdict sera « incertain »
pour environ 2 vidéos sur 3, et même hors de la zone, il se trompe environ 1 fois sur 3.

### 5. Analyse secondaire, option B (seuils recalibrés en LOSO) : séparée de l'analyse principale

Seuils choisis dans chaque pli (indice de Youden), appliqués au sujet retiré. **Ces seuils sont « calibrés sur le
dataset », pas « sourcés », et ils ne vont pas dans `config.yaml`.** Ils sont aussi en partie contaminés (limite 1).

| Vue | Méthode | Exactitude équilibrée [IC 95 %] | Seuil Myer → seuil calibré (médiane des plis) |
|---|---|---|---|
| Profil | depth | 0,62 [0,50 ; 0,74] | 0 → −0,43 (hanche à 0,43 cuisse **au-dessus** du genou) |
| Profil | heel_rise | 0,44 [0,36 ; 0,52] | 0,15 → 0,35 (instable : 0,15 à 0,37) |
| Profil | trunk_vs_tibia | 0,54 [0,44 ; 0,64] | 15° → 18,9° |
| Profil | head_position | 0,48 [0,36 ; 0,60] | 30° → 14,4° |
| Profil | ≥ 1 règle | 0,56 [0,50 ; 0,62] | – |
| Face | ≥ 1 règle | 0,64 [0,50 ; 0,78] | valgus 1 → 0,98 ; bassin 10° → 5,2° |
| Diagonale | ≥ 1 règle | 0,69 [0,58 ; 0,80] | valgus 1 → 1,02 ; bassin 10° → 11,8° |
| Toutes | ≥ 1 règle | 0,63 [0,56 ; 0,71] | – |

Même recalibrées, les règles de profil restent au niveau du hasard, sauf la profondeur. Le seuil de profondeur que
choisissent les données (hanche nettement au-dessus du genou) confirme le désaccord de référentiel (limite 2) : le
dataset tolère des squats bien moins profonds que Myer. Le meilleur chiffre (diagonale, 0,69) est obtenu parmi
12 combinaisons testées, sur des indicateurs conçus après avoir vu les données : c'est une piste, pas un résultat.

### 6. Table de fiabilité des règles pour l'app (`docs/results/rule_reliability.csv`)

| Règle | Vue | Signalée sur les vidéos good | … sur les bad | Fiabilité affichée dans l'app |
|---|---|---|---|---|
| depth | profil | 92 % | 100 % | détection 0,95 / fausses alertes 0,97 contre l'annotation |
| heel_rise | profil | 80 % | 76 % | détection 0,75 / fausses alertes 0,79 contre l'annotation |
| trunk_vs_tibia | profil | 60 % | 60 % | détection 0,62 / fausses alertes 0,57 contre l'annotation |
| head_position | profil | 24 % | 16 % | détection 0,24 / fausses alertes 0/8 contre l'annotation |
| knee_valgus | face | 16 % | 28 % | non évaluée contre annotation |
| knee_valgus | diagonale | 25 % | 48 % | non évaluée contre annotation (fiabilité réduite) |
| hip_shift | face | 0 % | 16 % | non évaluée contre annotation |
| hip_shift | diagonale | 75 % | 84 % | non évaluée contre annotation (fiabilité réduite) |

### Limites ajoutées par la phase B

6. **Pas de vérité de référence.** Le label du dataset et l'annotation de l'auteur sont en faible accord
   (kappa 0,19). Toute « performance » mesure l'accord avec un juge non expert, pas la qualité réelle du squat.
7. **Le signal du classifieur vient beaucoup du tempo**, un indicateur sans règle, mesuré à 15 fps (±0,07 s). Les
   vidéos bad seraient-elles simplement faites plus vite ? Le tempo fait partie de la grille de Myer, mais ce n'est
   pas ce que l'utilisateur attend d'un « verdict technique ».
8. **Probabilités non fiables** (Brier = référence constante). Le verdict de l'app doit être présenté comme une
   tendance faible, avec la performance LOSO mesurée à côté.
9. **Petits effectifs.** 25 sujets ; IC larges (souvent ±0,10 à ±0,20) ; 8 vidéos seulement sans défaut de tête
   selon l'annotateur. Les comparaisons par vue ne sont pas concluantes.
10. **Un seul annotateur, non expert**, sans mesure de reproductibilité (pas de deuxième annotation).

## Questions probables du jury

**1. Votre modèle fait-il mieux que le hasard ?**
Oui, mais de peu. AUC 0,60 [0,54 ; 0,67] et exactitude équilibrée 0,61 en leave-one-subject-out, p = 0,002 au
test de permutation. Ses probabilités ne sont pas mieux calibrées qu'un 50 % constant (Brier 0,251 contre 0,250).
C'est un signal faible, pas un outil de décision.

**2. Pourquoi le F1 de « toujours bad » (0,67) est-il supérieur à celui de votre modèle (0,60) ?**
Le F1 ne regarde que la classe positive et ignore les vrais négatifs. Prédire « bad » partout donne un rappel de
1 et une précision de 0,5, d'où F1 = 0,67. Avec 50 % de bad, c'est l'exactitude équilibrée et l'AUC qu'il faut
comparer au hasard (0,5). J'ai gardé le F1 parce qu'il était demandé, mais il ne sert pas à conclure.

**3. Pourquoi leave-one-subject-out et pas une validation croisée classique ?**
Chaque sujet a 6 vidéos. Avec un découpage aléatoire, une vidéo du sujet serait à l'entraînement et une autre au
test : le modèle reconnaîtrait la personne (morphologie, caméra) plutôt que la technique. Le LOSO met les 6
vidéos d'un sujet dans le même pli. Un test automatique vérifie que le sujet testé n'est jamais vu à
l'entraînement.

**4. Comment avez-vous évité la fuite d'information ?**
Quatre précautions : LOSO par sujet ; imputation et normalisation dans un Pipeline réappris à chaque pli ;
hyperparamètres fixés d'avance sans recherche ; protocole committé avant la première exécution (commit
`6e89270`). Il reste une fuite que je ne peux pas éliminer : les indicateurs ont été conçus après avoir exploré les
25 sujets (limite 1).

**5. Pourquoi les intervalles de confiance par bootstrap sur les sujets ?**
Les 6 vidéos d'un même sujet ne sont pas indépendantes. Rééchantillonner les vidéos ferait comme si on avait 150
observations indépendantes et donnerait des IC trop étroits. On tire 25 sujets avec remise et on garde toutes
leurs vidéos.

**6. Vos règles viennent de la littérature. Pourquoi ne fonctionnent-elles pas ?**
Trois raisons. (1) Le dataset applique un autre standard : 92 % des squats good n'atteignent pas la profondeur
de Myer. (2) Certaines mesures ne sont pas fiables à ces résolutions (talons). (3) La règle tronc−tibia va dans le
sens inverse des données. Une règle de la littérature n'est pas automatiquement valide pour un autre dataset,
une autre caméra et un autre juge.

**7. Votre propre annotation contredit le dataset (kappa 0,19). Qui a raison ?**
Personne ne peut le dire. Ni le protocole du dataset ni mon expertise ne sont documentés comme référence. Je juge
35 vidéos sur 50 « mauvaises », donc mon standard est plus strict. Mais je vois plus de défauts dans les vidéos bad
(par exemple tronc 19 contre 10) : nous ne sommes pas totalement indépendants. Il faudrait un expert certifié, et
deux annotateurs pour mesurer la reproductibilité.

**8. Pourquoi ne pas avoir recalibré les seuils pour améliorer les résultats ?**
Je l'ai fait, mais en analyse secondaire (option B), en LOSO pour ne pas tricher. Même recalibrées, les règles de
profil restent au hasard sauf la profondeur, et le seuil choisi par les données (hanche à 0,43 cuisse au-dessus du
genou) n'est pas un critère technique défendable. Ces seuils sont étiquetés « calibrés sur le dataset », jamais
« sourcés ».

**9. Que signifie la probabilité affichée par l'app ?**
La probabilité que le label du dataset soit « bad », pas la probabilité que le squat soit dangereux ou mauvais.
Elle est mal calibrée : l'app affiche donc un verdict en trois niveaux (good / incertain / bad), la performance
mesurée (exactitude équilibrée 0,61), et « incertain » pour environ 2 vidéos sur 3.

**10. Pourquoi la zone « incertain » est-elle [0,35 ; 0,65] ?**
C'est un choix d'ingénierie, fixé avant les résultats et non optimisé. Optimiser la largeur sur les mêmes données
aurait été une fuite. Hors de la zone, l'exactitude est de 0,64 (55 vidéos), à peine meilleure.

**11. Pourquoi une régression logistique plutôt qu'un modèle plus puissant ?**
Choix fixé d'avance pour l'explicabilité : chaque coefficient se lit (« moins de profondeur → plus de chances
d'être bad »). Avec 48 vidéos d'entraînement par vue, un modèle flexible surapprend : la forêt aléatoire fait
moins bien (AUC 0,55, compatible avec le hasard).

**12. Qu'est-ce que le modèle a vraiment appris ?**
Surtout le tempo (descente relativement plus courte → bad), puis la profondeur et le rapport tronc/tibia. Le sens
du tronc/tibia contredit Myer. Le valgus change de signe selon la vue. Le modèle a donc appris des corrélations du
dataset, pas une connaissance biomécanique.

**13. Pourquoi une vidéo est-elle exclue ?**
Le sujet 007 good diagonale : aucune répétition détectée. Elle est comptée comme « non analysable » et exclue des
métriques, comme le prévoyait le protocole. L'app fera de même : pas de verdict sans répétition.

**14. Votre outil est-il utile ?**
Je ne l'ai pas démontré. Les résultats montrent qu'il ne reproduit ni le label du dataset ni mon annotation de
façon fiable. Il reste conçu comme aide pédagogique : il montre des mesures, leur source, et leur fiabilité réelle.
Son utilité devrait être évaluée par une étude avec des utilisateurs et un expert.

**15. Que feriez-vous avec plus de temps ?**
(1) Faire annoter les vidéos par un coach certifié et par un deuxième annotateur (accord inter-juges).
(2) Filmer à 30 ou 60 fps et en plus haute résolution, pour le tempo et les talons. (3) Tester sur des sujets
jamais vus pendant la conception, pour supprimer la contamination. (4) Obtenir des labels par répétition et par
type d'erreur. (5) Évaluer l'utilité pédagogique avec de vrais utilisateurs.
