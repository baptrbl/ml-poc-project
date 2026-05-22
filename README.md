# Student Higher Education Prediction

Dataset : https://archive.ics.uci.edu/dataset/320/student+performance

## Description rapide du projet

Ce projet est un proof of concept de machine learning qui prédit si un élève est
à risque de ne pas poursuivre vers les études supérieures. L'objectif métier est
d'aider une équipe pédagogique à prioriser les élèves qui pourraient bénéficier
d'un accompagnement d'orientation ou d'un suivi renforcé.

Le modèle final est intégré dans une application Streamlit disponible dans
`src/app.py`. Les livrables demandés sont disponibles directement à la racine du
repo :

- `assignment1.md` : explication rapide du projet et des données.
- `assignment2.md` : feature engineering et dataset préprocessé.
- `assignment3.md` : description et comparaison des modèles.
- `plots/` : minimum 3 graphiques exportés et versionnés.
- `models/best_model.pkl` : modèle final sérialisé.

## Guide pour récupérer les données

Les données nécessaires sont déjà versionnées dans le repository :

- `data/raw/student-mat.csv`
- `data/raw/student-por.csv`
- `data/processed/student_data.csv`
- `data/processed/student_data_features.csv`

Pour repartir de zéro, la source officielle est le UCI Student Performance
Dataset :

https://archive.ics.uci.edu/dataset/320/student+performance

Télécharger les fichiers `student-mat.csv` et `student-por.csv`, puis les placer
dans `data/raw/`. Le dataset enrichi utilisé par le modèle est disponible dans
`data/processed/student_data_features.csv`.

## Project objective

This machine learning proof of concept predicts whether a student is likely to
pursue higher education after secondary school. The project is designed as a
decision-support tool for a school administration: students identified as at
risk can receive additional guidance, academic support, or personalized follow-up.

The model is not intended to replace human judgment. It is a prioritization
signal that should be interpreted with educational context.

## Project background

The initial project idea was to predict the probability that a foreign person
could be arrested by ICE depending on the city where they live. After early data
collection, the available variables were not sufficient to explain the event
properly, and there was no reliable comparison group for people who were not
arrested. Because that setup would have required artificial data and weak
assumptions, the project was redirected toward education inequality.

Education is a more appropriate topic for this proof of concept because the UCI
Student Performance dataset provides structured, documented, and reproducible
features about students, family background, support, and academic results.

## Feature engineering

The processed dataset includes engineered features derived from the original UCI
variables. These features do not invent new information; they summarize existing
signals in a form that is easier for the model to use.

Examples:

- `parents_edu_sum`: combined parental education level.
- `study_travel_balance` and `study_effort_index`: relationship between study time and travel time.
- `school_support_count` and `no_support`: level of academic or family support.
- `alcohol_total` and `risk_behavior_score`: summarized behavioral risk signals.
- `academic_avg_g1_g2`, `early_academic_risk`, and `older_than_cohort`: early academic indicators.

## Run the project

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/main.py
streamlit run src/app.py
```

`scripts/main.py` evaluates the registered model and writes
`results/model_metrics.csv` and `results/feature_importance.csv`. The Streamlit
app presents the business objective, dataset, latest metrics, model
explainability, and an interactive prediction demo.

## Model explainability

Model explainability helps understand why the model makes its predictions.
This project includes two levels of explanation:

- **Global explainability**: identifies which features matter most for the model
  in general. For the final Random Forest model, this uses
  `feature_importances_`, aggregated back to readable feature names.
- **Local explainability**: explains one individual prediction. The app compares
  the selected student with a reference student from the test set and estimates
  which variables increase or decrease the probability of `higher = yes`.

To generate the global feature importance file:

```bash
python3 scripts/main.py
```

This creates:

```text
results/feature_importance.csv
```

In the Streamlit app, open:

- `Explicabilité du modèle` to see the global feature importance chart and table.
- `Prédiction interactive` to see the probability of pursuing higher education
  and the local factors pushing the prediction toward `yes` or `no`.

## Repository structure

```text
.
├── data/
│   ├── raw/                 # Données sources non modifiées
│   └── processed/           # Données nettoyées et enrichies
├── deliverables/            # Rendus, PDF et documents finaux
├── models/                  # Modèles entraînés et sérialisés
├── notebooks/               # Exploration, feature engineering, entraînement
├── plots/                   # Graphiques exportés
├── results/                 # Métriques et sorties d'évaluation
├── scripts/                 # Points d'entrée exécutables
└── src/                     # Code réutilisable du projet
```

## Important files

- `data/raw/student-mat.csv` et `data/raw/student-por.csv` : données originales UCI.
- `data/processed/student_data.csv` : données fusionnées.
- `data/processed/student_data_features.csv` : données avec features créées.
- `models/best_model.pkl` : modèle final entraîné.
- `results/model_metrics.csv` : métriques générées par `python3 scripts/main.py`.
- `results/model_benchmark.csv` : comparaison des modèles testés dans le notebook.
- `results/feature_importance.csv` : importance globale des variables.
- `plots/target_distribution.svg` : distribution de la cible.
- `plots/model_comparison.svg` : comparaison visuelle des modèles.
- `plots/feature_importance.svg` : importance globale des variables.
- `src/config.py` : chemins du projet et modèle enregistré.
- `src/data.py` : chargement du dataset et split train/test.
- `src/explainability.py` : fonctions d'explicabilité globale et locale.
- `src/metrics.py` : métriques de classification.
- `src/app.py` : application Streamlit.
- `scripts/main.py` : point d'entrée d'évaluation.
- `notebooks/code_b_annote.ipynb` : notebook d'analyse et d'entraînement.
- `assignment1.md`, `assignment2.md`, `assignment3.md` : livrables demandés.
