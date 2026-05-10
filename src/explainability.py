"""Explainability helpers for the student higher-education model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import FEATURE_IMPORTANCE_FILE, MODELS
from model_io import load_model

YES_CLASS = 0
NO_CLASS = 1

FEATURE_LABELS = {
    "G1": "note du premier trimestre",
    "G2": "note du second trimestre",
    "G3": "note finale",
    "grade_avg_12": "moyenne des deux premières notes",
    "grade_trend_12": "progression entre G1 et G2",
    "grade_momentum": "dynamique de progression scolaire",
    "studytime": "temps d'étude",
    "failures": "échecs scolaires passés",
    "failure_flag": "présence d'échecs scolaires",
    "high_failure_flag": "échecs scolaires répétés",
    "absences": "absences",
    "high_absence_flag": "niveau d'absence élevé",
    "schoolsup": "soutien scolaire",
    "famsup": "soutien familial",
    "support_index": "niveau global de soutien",
    "parent_edu_avg": "niveau moyen d'éducation des parents",
    "parent_edu_gap": "écart d'éducation entre les parents",
    "Medu": "éducation de la mère",
    "Fedu": "éducation du père",
    "goout": "fréquence des sorties",
    "Dalc": "consommation d'alcool en semaine",
    "Walc": "consommation d'alcool le week-end",
    "alc_index": "niveau moyen d'alcool",
    "social_index": "vie sociale",
    "risk_behavior_index": "comportements à risque",
    "traveltime": "temps de trajet",
    "internet": "accès internet",
    "age": "âge",
}


def load_trained_model(model_path: Path | None = None) -> Any:
    """Load the trained model used by the project."""

    return load_model(model_path or MODELS["best_model"]["path"])


def get_final_estimator(model: Any) -> Any:
    """Return the estimator that makes predictions inside a pipeline."""

    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def get_feature_names(model: Any) -> list[str]:
    """Return the input feature names expected by the model."""

    if hasattr(model, "feature_names_in_"):
        return [str(feature) for feature in model.feature_names_in_]
    return []


def _get_transformed_feature_names(model: Any) -> list[str]:
    """Return feature names after preprocessing when available."""

    if hasattr(model, "named_steps") and "prep" in model.named_steps:
        preprocessor = model.named_steps["prep"]
        if hasattr(preprocessor, "get_feature_names_out"):
            return [str(feature) for feature in preprocessor.get_feature_names_out()]

    return get_feature_names(model)


def _map_transformed_to_original(
    transformed_feature: str,
    original_features: list[str],
) -> str:
    """Map a transformed feature name back to a readable original feature."""

    clean_name = transformed_feature.split("__", maxsplit=1)[-1]
    for original_feature in sorted(original_features, key=len, reverse=True):
        if clean_name == original_feature or clean_name.startswith(f"{original_feature}_"):
            return original_feature
    return clean_name


def _importance_values(model: Any) -> np.ndarray | None:
    """Return raw importance values from supported model families."""

    estimator = get_final_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            return coef
        if coef.shape[0] == 1:
            return coef[0]
        return coef.mean(axis=0)

    return None


def compute_global_feature_importance(model: Any | None = None) -> pd.DataFrame:
    """Compute global feature importance as a clean DataFrame.

    The function supports tree-based models with ``feature_importances_`` and
    linear models with ``coef_``. If the estimator is not directly interpretable,
    it returns the expected input features with zero importance instead of
    failing.
    """

    model = model or load_trained_model()
    original_features = get_feature_names(model)
    transformed_features = _get_transformed_feature_names(model)
    raw_importance = _importance_values(model)

    if raw_importance is None or len(raw_importance) != len(transformed_features):
        rows = [
            {"feature": feature, "importance": 0.0, "importance_abs": 0.0}
            for feature in original_features
        ]
        importance_df = pd.DataFrame(rows)
    else:
        mapped_features = [
            _map_transformed_to_original(feature, original_features)
            for feature in transformed_features
        ]
        raw_df = pd.DataFrame(
            {
                "feature": mapped_features,
                "importance": raw_importance,
                "importance_abs": np.abs(raw_importance),
            }
        )
        importance_df = (
            raw_df.groupby("feature", as_index=False)
            .agg({"importance": "sum", "importance_abs": "sum"})
            .sort_values("importance_abs", ascending=False)
            .reset_index(drop=True)
        )

    if importance_df.empty:
        return pd.DataFrame(columns=["feature", "importance", "importance_abs", "rank"])

    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    return importance_df[["feature", "importance", "importance_abs", "rank"]]


def write_global_feature_importance(model: Any | None = None) -> pd.DataFrame:
    """Write global feature importance to ``results/feature_importance.csv``."""

    importance_df = compute_global_feature_importance(model)
    importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    return importance_df


def predict_class_probabilities(model: Any, sample: pd.DataFrame) -> dict[str, float]:
    """Return readable class probabilities for ``higher``.

    In this project, ``higher = yes`` is encoded as class ``0`` and
    ``higher = no`` as class ``1``.
    """

    if not hasattr(model, "predict_proba"):
        return {}

    probabilities = model.predict_proba(sample)[0]
    classes = list(getattr(model, "classes_", []))
    if not classes:
        classes = list(range(len(probabilities)))

    readable_probabilities: dict[str, float] = {}
    for class_value, probability in zip(classes, probabilities):
        class_label = str(class_value).strip().lower()
        if class_label in {str(YES_CLASS), "yes", "true"}:
            readable_probabilities["yes"] = float(probability)
        elif class_label in {str(NO_CLASS), "no", "false"}:
            readable_probabilities["no"] = float(probability)

    if "yes" not in readable_probabilities and len(probabilities) > 0:
        readable_probabilities["yes"] = float(probabilities[0])
    if "no" not in readable_probabilities and len(probabilities) > 1:
        readable_probabilities["no"] = float(probabilities[1])

    return readable_probabilities


def predict_probability_yes(model: Any, sample: pd.DataFrame) -> float | None:
    """Return the probability that ``higher`` is predicted as ``yes``."""

    probabilities = predict_class_probabilities(model, sample)
    return probabilities.get("yes")


def build_reference_row(X_reference: pd.DataFrame) -> pd.DataFrame:
    """Build a median/mode reference student used for local explanations."""

    values: dict[str, object] = {}
    for column in X_reference.columns:
        series = X_reference[column].dropna()
        if series.empty:
            values[column] = 0
        elif pd.api.types.is_numeric_dtype(series):
            values[column] = float(series.median())
        else:
            values[column] = series.mode().iloc[0]

    return pd.DataFrame([values], columns=X_reference.columns)


def explain_local_prediction(
    model: Any,
    sample: pd.DataFrame,
    X_reference: pd.DataFrame,
    top_n: int = 6,
) -> tuple[pd.DataFrame, float | None]:
    """Estimate local feature contributions for one prediction.

    This is a lightweight, model-agnostic fallback: each feature is changed from
    a reference student to the selected student's value, one feature at a time.
    The contribution is the change in probability of ``higher = yes``.
    """

    if sample.shape[0] != 1:
        raise ValueError("Local explanation expects exactly one sample row.")

    reference = build_reference_row(X_reference)
    base_probability = predict_probability_yes(model, reference)
    final_probability = predict_probability_yes(model, sample)

    if base_probability is None or final_probability is None:
        return pd.DataFrame(
            columns=["feature", "label", "value", "reference", "contribution"]
        ), final_probability

    global_importance = compute_global_feature_importance(model)
    feature_order = [
        feature
        for feature in global_importance["feature"].tolist()
        if feature in sample.columns
    ]
    feature_order.extend(
        feature for feature in sample.columns if feature not in set(feature_order)
    )

    rows: list[dict[str, object]] = []
    for feature in feature_order:
        candidate = reference.copy()
        candidate.loc[:, feature] = sample[feature].iloc[0]
        candidate_probability = predict_probability_yes(model, candidate)
        if candidate_probability is None:
            continue

        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": sample[feature].iloc[0],
                "reference": reference[feature].iloc[0],
                "contribution": candidate_probability - base_probability,
            }
        )

    contribution_df = pd.DataFrame(rows)
    if contribution_df.empty:
        return contribution_df, final_probability

    contribution_df["contribution_abs"] = contribution_df["contribution"].abs()
    contribution_df = (
        contribution_df.sort_values("contribution_abs", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return contribution_df, final_probability
