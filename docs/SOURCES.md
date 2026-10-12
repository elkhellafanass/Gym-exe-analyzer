# Sources des critères et des seuils

Règle du projet : **aucun seuil inventé**. Chaque règle de `config.yaml` porte une clé `source` (ci-dessous) et
un `status` :
- `sourcé` : le critère ET la valeur viennent de la source ;
- `critère sourcé, seuil à valider par un expert` : la source décrit le critère de façon qualitative ("le pied
  reste au sol"). La tolérance chiffrée nécessaire pour la mesure vidéo est un choix du projet, non sourcé ;
- `à valider par un expert` : ni critère ni valeur sourcés.

Toutes les sources ont été consultées le 12/10/2026.

---

## `myer2014` : grille d'évaluation du squat (Back Squat Assessment)

Myer GD, Kushner AM, et al. (2014). *The back squat: a proposed assessment of functional deficits and technical
factors that limit performance.* Strength and Conditioning Journal, 36(6), 4-27.
DOI : [10.1519/SSC.0000000000000103](https://doi.org/10.1519/SSC.0000000000000103). Texte intégral :
[PMC4262933](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262933).

Critères cités (tableau 1 de l'article, citations exactes) et règle correspondante :

| Règle (`config.yaml`) | Citation de la source | Ce que mesure le projet | Statut |
|---|---|---|---|
| `depth` | "At apex of depth, the tops of thighs are at least parallel to the ground." Et dans le texte : "the hip joint is at least level with or slightly below the knee joint." | hanche (point MediaPipe) au moins au niveau du genou → `depth ≥ 0` | sourcé |
| `heel_rise` | "Entire foot remains in contact with the ground." / "Lifting heels or toes off of the ground at any time is suboptimal" | montée du talon > 15 % de la longueur du pied | critère sourcé, seuil à valider |
| `trunk_vs_tibia` | "Trunk is parallel to tibia, while maintaining slightly lordotic lumbar spine." | inclinaison tronc − inclinaison tibia > 15° | critère sourcé, seuil à valider |
| `head_position` | "Line of neck is perpendicular to the ground and gaze is aimed forward." | inclinaison épaule→oreille / verticale > 30° | critère sourcé, seuil à valider |
| `knee_valgus` | "Lateral aspect of knee does not cross medial malleolus for either leg." | ratio séparation genoux / chevilles < 1 (approximation, voir `mizner2012`) | critère sourcé, seuil à valider |
| `hip_shift` | "Line of hips is parallel to ground in frontal plane throughout squat." | inclinaison de la ligne des hanches > 10° | critère sourcé, seuil à valider |
| `min_reps_for_video_error` | un défaut est noté s'il apparaît "during 2 or more repetitions" (sur 10 répétitions) | défaut retenu pour la vidéo si ≥ 2 répétitions concernées | sourcé (le dataset a ~10-20 répétitions par vidéo, contre 10 dans la source) |

**Vues de caméra :** l'article recommande de filmer de face, de dos et de profil ("anterior, posterior, and lateral").
D'où le filtrage des règles par vue dans `config.yaml`.

**Tempo : pas de règle.** L'article écrit "no less than 2:1 (descent: ascent)", donc une descente au moins deux
fois plus longue que la remontée. Mais son tableau 1 écrit "Ascent: Descent timing ratio is at least 2:1", et le
texte parle de "velocity". La formulation est ambiguë : le tempo est **affiché sans verdict**. De plus, à 15 fps,
une durée est connue à ±0,067 s près.

**Limites de l'adaptation :**
- Myer et al. décrivent un squat arrière avec un bâton (dowel) évalué **à l'œil** par un professionnel. Le dataset
  montre des squats au poids du corps, mesurés par un modèle de pose en 2D.
- Le point "hanche" de MediaPipe approxime le centre articulaire ; le pli de l'aine n'est pas détecté.
- Le critère `knee_valgus` parle du bord latéral du genou et de la malléole médiale. MediaPipe ne donne que les
  centres du genou et de la cheville. Le ratio < 1 est donc une approximation, qui signale plus tôt que le
  critère d'origine.

## `mizner2012` : ratio de séparation genoux / chevilles

Mizner RL, Chmielewski TL, Toepke JJ, Tofte KB (2012). *Comparison of 2-dimensional measurement techniques for
predicting knee angle and moment during a drop vertical jump.*
Texte intégral : [PMC3340567](https://pmc.ncbi.nlm.nih.gov/articles/PMC3340567).

Ce qu'on en retient : le ratio distance inter-genoux / distance inter-chevilles, mesuré en vidéo 2D de face, est
mieux associé à l'abduction du genou mesurée en 3D (r² = 0,35) que l'angle de projection frontal (r² = 0,15). La
fiabilité inter-évaluateurs est bonne (ICC 0,89-0,94). Population : 36 athlètes féminines, **saut en contrebas,
pas squat**.
**Aucun seuil validé n'est donné** : le ratio n'est utilisé que comme mesure, et la valeur 1 vient de la
géométrie du critère de Myer (genou à l'intérieur de la cheville).

## `mediapipe_card` : modèle de pose

*Model Card : MediaPipe BlazePose GHUM 3D*, Google, 16/04/2021.
[PDF](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf). Licence
Apache 2.0. Guide : [Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).

Ce qu'on en retient pour le projet :
- 33 points, dont talons et pointes de pied. C'est ce qui permet la règle `heel_rise`.
- Usage prévu : "Fitness and repetition counting", "3D pose measurements (angles / distances)".
- **Hors périmètre** : "Multiple people in an image", "People too far away from the camera (e.g. further than
  14 feet/4 meters)". Le dataset contient souvent d'autres personnes en arrière-plan (voir
  [INSPECTION.md](INSPECTION.md)).
- Précision (PDJ / PCK@0.2) du modèle *full* : 91,8 % en moyenne. Un point est compté juste si son erreur est
  inférieure à 20 % du diamètre du torse, ce qui est une tolérance large pour mesurer des angles fins.
- "not intended for human life-critical decisions", ce qui va dans le sens de l'avertissement du projet.

## Paramètres techniques (non sourcés, choix d'ingénierie)

Ce ne sont pas des critères de posture : ils règlent la mesure. Ils sont documentés ici pour la transparence.

| Paramètre | Valeur | Justification |
|---|---|---|
| `pose.min_visibility` | 0,5 | Visibilité MediaPipe = probabilité que le point soit visible ; 0,5 = plus probable que non |
| `pose.max_gap_frames` | 3 (0,2 s) | Un trou plus long pourrait masquer un mouvement : on ne l'invente pas |
| `pose.smooth_window` / `polyorder` | 7 / 2 | ≈ 0,5 s à 15 fps ; Savitzky-Golay conserve mieux les pics qu'une moyenne |
| `pose.max_people` | 3 | Personne analysée = la plus grande à l'image |
| `reps.min_travel` | 0,3 cuisse | Ignore les petites oscillations ; une répétition même partielle descend davantage |
| `reps.min_interval_s` | 0,8 s | Empêche de compter deux points bas dans la même répétition |
| `reps.edge_fraction` | 0,1 | Début/fin du mouvement, en excluant les pauses debout |

## Sources consultées mais NON utilisées comme référence

- **Règlement IPF (profondeur en powerlifting)** : je n'ai pu consulter que des sources secondaires
  ([USAPL via mass-lift.com](https://mass-lift.com/blog/2020/07/22/usapl-squats-rules-of-performance/),
  [zeolla.org](https://www.zeolla.org/Fitness/powerlifting/article/disqualification.htm)). Elles ne sont pas citées
  comme référence ; le critère de profondeur vient de Myer et al. (2014).
- **Études de validité de MediaPipe pour les angles du squat** : la recherche a trouvé des travaux (préprint JMIR,
  études métrologiques) que je n'ai pas lus en entier. **À compléter** avant de les citer dans le mémoire.
