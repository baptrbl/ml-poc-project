"""Streamlit application for the student higher-education prediction project."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import FEATURES_DATA_FILE, MODEL_METRICS_FILE, MODELS
from data import load_dataset_split
from model_io import load_model


def _load_features() -> pd.DataFrame:
    """Load the processed student dataset for display."""

    return pd.read_csv(FEATURES_DATA_FILE)


def _load_metrics() -> pd.DataFrame:
    """Load the latest model evaluation results."""

    return pd.read_csv(MODEL_METRICS_FILE)


def _load_test_rows() -> pd.DataFrame:
    """Load the test features used by scripts/main.py."""

    _, X_test, _, _ = load_dataset_split()
    return X_test.reset_index(drop=True)


def _load_best_model():
    """Load the final model registered in config.MODELS."""

    return load_model(MODELS["best_model"]["path"])


def _format_risk_label(prediction: int) -> str:
    """Return a clear user-facing label for the encoded prediction."""

    if int(prediction) == 1:
        return "At risk of not pursuing higher education"
    return "Likely to pursue higher education"


def build_app() -> None:
    """Render the Streamlit application."""

    st.set_page_config(
        page_title="Student Higher Education Prediction",
        layout="wide",
    )

    st.title("Student Higher Education Prediction")
    st.write(
        "This proof of concept predicts whether a student is likely to pursue "
        "higher education. The goal is to help a school identify students who "
        "may need extra guidance before the final decision point."
    )

    objective_tab, data_tab, model_tab, demo_tab = st.tabs(
        ["Objective", "Data", "Model", "Demo"]
    )

    with objective_tab:
        st.header("Business objective")
        st.write(
            "The project supports school administrations in reducing educational "
            "inequalities. The prediction is a decision-support signal: it can "
            "help prioritize guidance, academic support, and personalized follow-up."
        )
        st.info(
            "The model should support human review. It must not be used as an "
            "automatic decision system for students."
        )

    with data_tab:
        st.header("Dataset")
        st.write(
            "The project uses the UCI Student Performance dataset, enriched with "
            "features about academic trajectory, family background, support, and "
            "risk behaviors."
        )

        if not FEATURES_DATA_FILE.exists():
            st.warning(
                "Processed dataset not found. Run the notebook or preprocessing first."
            )
            return

        df = _load_features()

        metric_cols = st.columns(4)
        metric_cols[0].metric("Rows", f"{df.shape[0]:,}")
        metric_cols[1].metric("Columns", df.shape[1])
        metric_cols[2].metric("Target", "higher")
        metric_cols[3].metric("Positive class", "no")

        st.subheader("Preview")
        st.dataframe(df.head(20), width="stretch", hide_index=True)

        if "higher" in df.columns:
            st.subheader("Target distribution")
            target_counts = df["higher"].value_counts().rename_axis("higher")
            st.bar_chart(target_counts)

        if "G3" in df.columns and "higher" in df.columns:
            st.subheader("Academic signal")
            avg_grade = (
                df.groupby("higher", as_index=False)["G3"]
                .mean()
                .rename(columns={"G3": "average_final_grade"})
            )
            st.write(
                "Average final grade depending on whether the student wants to "
                "pursue higher education."
            )
            st.dataframe(avg_grade, width="stretch", hide_index=True)
            st.bar_chart(avg_grade.set_index("higher"))

    with model_tab:
        st.header("Model evaluation")

        if not MODEL_METRICS_FILE.exists():
            st.info(
                "Run `python3 scripts/main.py` to generate `results/model_metrics.csv`."
            )
            return

        metrics_df = _load_metrics()
        st.dataframe(metrics_df, width="stretch", hide_index=True)

        if "f1" in metrics_df.columns:
            best_model = metrics_df.sort_values("f1", ascending=False).iloc[0]
            st.success(
                f"Best model: {best_model['model_name']} "
                f"with F1-score = {best_model['f1']:.3f}"
            )

        st.caption(
            "The target is encoded as 1 for students at risk of not pursuing "
            "higher education and 0 otherwise."
        )

    with demo_tab:
        st.header("Prediction demo")
        st.write(
            "Select a student from the evaluation split to inspect the model output "
            "on realistic data."
        )

        try:
            X_test = _load_test_rows()
            model = _load_best_model()
            selected_row = st.number_input(
                "Student row",
                min_value=0,
                max_value=max(len(X_test) - 1, 0),
                value=0,
                step=1,
            )
            sample = X_test.iloc[[int(selected_row)]]
            prediction = int(model.predict(sample)[0])
            st.metric("Prediction", _format_risk_label(prediction))
            st.dataframe(sample, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Prediction demo unavailable: {exc}")


if __name__ == "__main__":
    build_app()
