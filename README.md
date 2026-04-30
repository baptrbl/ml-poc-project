# Student Project

Dataset : https://archive.ics.uci.edu/dataset/320/student+performance

## Préambule

Notre projet initial était d’essayer d’anticiper la probabilité qu’une personne étrangère se fasse arrêter par un agent de l’ICE selon la ville dans laquelle elle habite, dans le but qu’elle puisse éviter de se faire arrêter.

Cependant, après avoir récupéré certaines données, nous n’avions pas suffisamment de features pour réellement comprendre ce qui motive une arrestation. De plus, il nous manquait des données comparatives concernant les personnes étrangères ne se faisant pas arrêter. Le manque de précision dû à la création de données artificielles nous a donc conduits à abandonner ce projet et à le réorienter.

## Pourquoi ce projet ?

L’un des problèmes qui nous semble aujourd’hui les plus importants est l’inégalité des chances dans l’éducation. En effet, la méritocratie apparaît désormais comme largement illusoire. Il est donc pertinent de se demander comment comprendre ces inégalités et quels paramètres de notre quotidien influencent notre avenir.

Cet outil est destiné à l’administration d’un lycée afin de prédire la probabilité qu’un élève souhaite poursuivre ou non des études supérieures à la fin de sa scolarité. L’objectif est d’identifier les élèves ayant le plus besoin d’accompagnement afin de pouvoir les soutenir en priorité.

## Création de nouvelles features

Pour améliorer notre modèle, nous avons créé 10 nouvelles features à partir des données déjà présentes dans le dataset. L’objectif n’était pas d’inventer de nouvelles informations, mais de mieux représenter certaines situations qui peuvent influencer l’envie ou la possibilité de poursuivre des études supérieures.

Nous avons regroupé le niveau d’éducation des parents avec `parents_edu_sum`, car le contexte familial peut avoir un impact important sur l’ambition scolaire et l’accompagnement de l’élève. Nous avons aussi créé des variables liées à l’organisation de l’élève, comme `study_travel_balance` et `study_effort_index`, afin de comparer le temps consacré au travail scolaire avec le temps de trajet.

Nous avons également résumé le soutien reçu par l’élève grâce à `school_support_count` et `no_support`, car un élève sans aide scolaire ou familiale peut être plus fragile. D’autres variables, comme `alcohol_total` et `risk_behavior_score`, permettent de représenter certains comportements pouvant nuire à la scolarité.

Enfin, nous avons ajouté des indicateurs scolaires plus directs : `academic_avg_g1_g2`, `early_academic_risk` et `older_than_cohort`. Ces variables permettent de repérer plus facilement les élèves en difficulté dès les premières notes, ou ceux qui sont plus âgés que la majorité de leur classe.

## Lancer le projet

```bash
pip install -r requirements.txt
python3 scripts/main.py
```

## Organisation du projet

```text
.
├── data/
│   ├── raw/                 # Données sources non modifiées
│   └── processed/           # Données nettoyées et enrichies
├── deliverables/            # Rendus, PDF et documents finaux
├── models/                  # Modèles entraînés et sérialisés
├── notebooks/               # Exploration, feature engineering, entraînement
├── reports/
│   └── figures/             # Graphiques exportés
├── results/                 # Métriques et sorties d'évaluation
├── scripts/                 # Points d'entrée exécutables
└── src/                     # Code réutilisable du projet
```

## Fichiers importants

- `data/raw/student-mat.csv` et `data/raw/student-por.csv` : données originales UCI.
- `data/processed/student_data.csv` : données fusionnées.
- `data/processed/student_data_features.csv` : données avec features créées.
- `models/best_model.pkl` : modèle final entraîné.
- `notebooks/code_b_annote.ipynb` : notebook d'analyse et d'entraînement.
- `deliverables/assignment1.md`, `deliverables/read_me.pdf`, `deliverables/read_me.txt` : livrables.
