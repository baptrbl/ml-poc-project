# Assignment 3 - Description et comparaison des modèles

## Modèles testés

Le notebook `notebooks/code_b_annote.ipynb` compare plusieurs familles de modeles :

- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- SVM

Le modele sauvegarde sur GitHub est :

`models/best_model.pkl`

Il s'agit du pipeline final retenu pour l'application Streamlit.

## Trois modèles principaux

### Logistic Regression

La regression logistique sert de baseline interpretable. Elle est rapide a entrainer et donne une lecture simple des effets globaux, mais elle capte moins bien les relations non lineaires entre les variables.

Dans le benchmark, elle obtient un recall eleve sur la classe de risque, ce qui est interessant pour reperer les eleves fragiles, mais sa precision reste faible.

### Decision Tree

L'arbre de decision est utile pour produire des regles comprehensibles. Il permet d'expliquer facilement certains chemins de decision, mais il est plus instable et peut sur-apprendre si sa profondeur n'est pas bien controlee.

Dans ce projet, il sert surtout de modele explicatif de comparaison. Ses performances sont inferieures aux modeles d'ensemble.

### Random Forest

Le Random Forest est le modele final retenu. Il combine plusieurs arbres, gere bien les donnees mixtes et offre un meilleur compromis entre performance, robustesse et explicabilite globale via les importances de variables.

Le modele final est optimise avec une validation croisee sensible au cout metier, en penalisation plus forte des faux negatifs.

## Comparaison des modèles

Les scores ci-dessous proviennent du benchmark du notebook, sauvegarde dans `results/model_benchmark.csv`.

| Modele | Accuracy | Precision | Recall | F1 | ROC AUC | Business cost score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdaBoost | 0.920 | 0.521 | 0.327 | 0.395 | 0.857 | -42.2 |
| Logistic Regression | 0.817 | 0.277 | 0.691 | 0.393 | 0.833 | -43.8 |
| SVM | 0.889 | 0.362 | 0.410 | 0.376 | 0.857 | -43.8 |
| Gradient Boosting | 0.922 | 0.593 | 0.255 | 0.342 | 0.863 | -44.8 |
| Random Forest | 0.921 | 0.800 | 0.070 | 0.128 | 0.882 | -52.8 |
| Decision Tree | 0.871 | 0.218 | 0.228 | 0.219 | 0.579 | -54.6 |

## Résultat du modèle final

Après tuning du Random Forest, les resultats sur le test set sont :

- Accuracy : 0.9187
- Precision : 0.5238
- Recall : 0.6111
- F1-score : 0.5641
- ROC AUC : 0.9287

Matrice de confusion :

```text
[[181, 10],
 [  7, 11]]
```

Le choix final du Random Forest vient du compromis entre performance, robustesse, capacite a gerer les variables mixtes et explicabilite via les importances globales.

## Fichiers modèles

Le modele final serialise est disponible sur GitHub :

- `models/best_model.pkl`

Le chargement du modele est gere par :

- `src/model_io.py`
- `src/config.py`
