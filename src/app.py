"""Streamlit dashboard for the student higher-education prediction project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    FEATURE_IMPORTANCE_FILE,
    FEATURES_DATA_FILE,
    MODEL_METRICS_FILE,
    MODELS,
)
from data import load_dataset_split
from explainability import (
    explain_local_prediction,
    predict_class_probabilities,
)
from model_io import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINISTRY_LOGO_FILE = PROJECT_ROOT / "assets" / "ministere-education-nationale.svg"
TARGET_COL = "higher"
MODEL_KEY = "best_model"
PAPER = "#F6F7FB"
PAPER_2 = "#FFFFFF"
SURFACE = "#FFFFFF"
INK = "#1E1E1E"
INK_2 = "#3A3A3A"
INK_3 = "#666666"
RULE_HAIR = "rgba(0, 0, 145, 0.14)"
SIGNAL = "#000091"
SIGNAL_DEEP = "#00006D"
POSITIVE = "#18753C"
NEGATIVE = "#E1000F"
SAND = "#FDCF41"
SLATE = "#6A6A6A"
PAGE_OPTIONS = [
    "Accueil",
    "Vue d'ensemble",
    "Dataset",
    "Insights",
    "Performance du modèle",
    "Explicabilité du modèle",
    "Prédiction interactive",
    "Conclusion",
]
MODEL_SUMMARIES = [
    {
        "name": "Logistic Regression",
        "role": "Baseline interprétable",
        "plus": "Simple, rapide, facile à expliquer devant un jury.",
        "minus": "Capte mal les relations non linéaires entre variables.",
        "status": "Testé comme point de comparaison.",
    },
    {
        "name": "Decision Tree",
        "role": "Modèle explicatif",
        "plus": "Très lisible et utile pour comprendre quelques règles de décision.",
        "minus": "Risque d'overfitting et performance moins stable.",
        "status": "Testé, mais pas retenu comme modèle final.",
    },
    {
        "name": "Random Forest",
        "role": "Modèle final choisi",
        "plus": "Robuste, performant sur données mixtes et moins sensible au bruit.",
        "minus": "Moins interprétable qu'une régression ou un arbre simple.",
        "status": "Retenu après comparaison et tuning.",
    },
    {
        "name": "AdaBoost",
        "role": "Ensemble boosting",
        "plus": "Peut améliorer des modèles faibles en combinant plusieurs arbres.",
        "minus": "Sensible au bruit et aux observations atypiques.",
        "status": "Testé dans le benchmark.",
    },
    {
        "name": "Gradient Boosting",
        "role": "Boosting performant",
        "plus": "Bon candidat pour capter des relations complexes.",
        "minus": "Demande davantage de tuning et peut sur-apprendre.",
        "status": "Testé dans le benchmark.",
    },
    {
        "name": "SVM",
        "role": "Frontière de décision flexible",
        "plus": "Intéressant pour séparer des classes avec marge.",
        "minus": "Moins lisible et plus coûteux à régler.",
        "status": "Testé comme modèle alternatif.",
    },
]


def _load_features() -> pd.DataFrame:
    """Load the processed student dataset for display."""

    return pd.read_csv(FEATURES_DATA_FILE)


def _load_metrics() -> pd.DataFrame:
    """Load the latest model evaluation results."""

    return pd.read_csv(MODEL_METRICS_FILE)


def _load_feature_importance() -> pd.DataFrame:
    """Load the latest global feature importance results."""

    return pd.read_csv(FEATURE_IMPORTANCE_FILE)


def _load_test_rows() -> pd.DataFrame:
    """Load the test features used by scripts/main.py."""

    _, X_test, _, _ = load_dataset_split()
    return X_test.reset_index(drop=True)


def _build_new_student_profile(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Create a realistic editable profile from the reference population."""

    values: dict[str, object] = {}
    for column in reference_df.columns:
        series = reference_df[column].dropna()
        if series.empty:
            values[column] = 0
        elif pd.api.types.is_numeric_dtype(series):
            values[column] = series.median()
        else:
            values[column] = series.mode().iloc[0]

    return pd.DataFrame([values], columns=reference_df.columns)


def _load_best_model() -> Any:
    """Load the final model registered in config.MODELS."""

    return load_model(MODELS[MODEL_KEY]["path"])


def _format_risk_label(prediction: int) -> str:
    """Return a clear user-facing label for the encoded prediction."""

    if int(prediction) == 1:
        return "Risque de ne pas poursuivre vers les études supérieures"
    return "Profil susceptible de poursuivre vers les études supérieures"


def _risk_message(prediction: int) -> str:
    """Return a non-technical interpretation of the prediction."""

    if int(prediction) == 1:
        return (
            "Le modèle recommande une attention prioritaire. Cet élève pourrait "
            "bénéficier d'un échange d'orientation, d'un suivi académique ou "
            "d'un accompagnement personnalisé."
        )

    return (
        "Le modèle ne détecte pas de signal de risque fort. Le suivi habituel "
        "reste pertinent, avec une vigilance sur les facteurs scolaires et sociaux."
    )


def _inject_css() -> None:
    """Apply lightweight dashboard styling."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --paper: #F6F7FB;
            --paper-2: #FFFFFF;
            --paper-3: #E5E5F4;
            --surface: #FFFFFF;
            --ink: #1E1E1E;
            --ink-2: #3A3A3A;
            --ink-3: #666666;
            --ink-4: #929292;
            --signal: #000091;
            --signal-deep: #00006D;
            --positive: #18753C;
            --negative: #E1000F;
            --republic-red: #E1000F;
            --republic-blue: #000091;
            --rule-hair: rgba(0, 0, 145, 0.14);
            --serif: 'Fraunces', 'Times New Roman', Georgia, serif;
            --sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --mono: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--paper);
            color: var(--ink);
            font-family: var(--sans);
        }
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0 0 auto 0;
            height: 6px;
            background: linear-gradient(
                90deg,
                var(--republic-blue) 0 33.33%,
                #FFFFFF 33.33% 66.66%,
                var(--republic-red) 66.66% 100%
            );
            z-index: 999;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: var(--ink-2);
        }
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] b {
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(246, 247, 251, 0.92);
            border-bottom: 1px solid var(--rule-hair);
            backdrop-filter: blur(12px);
        }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid rgba(0, 0, 145, 0.18);
        }
        [data-testid="stSidebar"] * {
            font-family: var(--sans);
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: var(--ink) !important;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }
        .republic-strip {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            height: 5px;
            margin: 0.75rem 0 1.1rem;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid rgba(0, 0, 0, 0.06);
        }
        .republic-strip span:nth-child(1) {
            background: var(--republic-blue);
        }
        .republic-strip span:nth-child(2) {
            background: #FFFFFF;
        }
        .republic-strip span:nth-child(3) {
            background: var(--republic-red);
        }
        h1, h2, h3 {
            font-family: var(--serif) !important;
            color: var(--ink) !important;
            letter-spacing: 0;
            max-width: 100%;
            overflow-wrap: anywhere;
            text-wrap: balance;
            white-space: normal;
        }
        h1 {
            font-size: 2.55rem !important;
            line-height: 1.08 !important;
            font-weight: 500 !important;
        }
        h2 {
            font-size: 2.15rem !important;
            line-height: 1.12 !important;
            font-weight: 500 !important;
        }
        h3 {
            font-size: 1.45rem !important;
            line-height: 1.16 !important;
            font-weight: 600 !important;
        }
        div[data-testid="column"],
        div[data-testid="column"] > div,
        [data-testid="stMarkdownContainer"] {
            min-width: 0;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 2.05rem !important;
            line-height: 1.05 !important;
            white-space: normal;
            overflow-wrap: normal;
        }
        p, li, div, span {
            font-family: var(--sans);
        }
        code, pre, .numeric, .metric-value, .metric-label, .eyebrow, .pill {
            font-family: var(--mono) !important;
        }
        .metric-card {
            border: 1px solid var(--rule-hair);
            border-radius: 8px;
            padding: 0.95rem 0.85rem;
            background: var(--surface);
            box-shadow: 0 2px 8px rgba(0, 0, 145, 0.05);
            min-height: 112px;
            min-width: 0;
            overflow: hidden;
        }
        .metric-card,
        .metric-card * {
            color: var(--ink);
        }
        .metric-label {
            color: var(--ink-3);
            font-size: 0.58rem;
            font-weight: 500;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
            max-width: 100%;
            line-height: 1.25;
            overflow-wrap: normal;
            white-space: normal;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 500;
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
            max-width: 100%;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        .metric-value-long {
            font-size: 0.92rem;
            line-height: 1.18;
            overflow-wrap: normal;
        }
        .probability-card {
            border: 1px solid rgba(0, 0, 145, 0.22);
            border-radius: 10px;
            padding: 1.05rem 1.15rem;
            background: #F5F5FE;
            margin: 0.65rem 0 0.85rem;
        }
        .probability-card,
        .probability-card * {
            color: var(--ink);
        }
        .probability-label {
            color: var(--ink-3);
            font-family: var(--mono);
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }
        .probability-number {
            color: var(--signal-deep);
            font-family: var(--mono);
            font-size: 2.55rem;
            font-weight: 700;
            line-height: 1;
            font-variant-numeric: tabular-nums;
        }
        .probability-helper {
            color: var(--ink-2);
            margin-top: 0.55rem;
            font-size: 0.92rem;
        }
        .section-card {
            border: 1px solid var(--rule-hair);
            border-radius: 8px;
            padding: 1.25rem 1.35rem;
            background: var(--surface);
            margin-bottom: 1rem;
        }
        .section-card,
        .section-card * {
            color: var(--ink-2);
        }
        .section-card strong,
        .section-card b {
            color: var(--ink);
        }
        .value-card {
            min-height: 176px;
            height: 100%;
            border-radius: 8px;
            padding: 1.35rem 1.45rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.8rem;
            border: 1px solid rgba(0, 0, 145, 0.16);
            box-shadow: 0 2px 10px rgba(0, 0, 145, 0.05);
        }
        .value-card h3 {
            font-family: var(--sans) !important;
            font-size: 1.05rem !important;
            line-height: 1.22 !important;
            font-weight: 800 !important;
            margin: 0;
            color: inherit !important;
            white-space: normal;
            overflow-wrap: normal;
        }
        .value-card p {
            margin: 0;
            font-size: 0.98rem;
            line-height: 1.48;
            color: inherit;
        }
        .value-card-blue {
            background: var(--republic-blue);
            color: #FFFFFF;
            border-color: var(--republic-blue);
        }
        .value-card-white {
            background: #FFFFFF;
            color: var(--ink);
            border-top: 5px solid var(--republic-blue);
            border-bottom: 5px solid var(--republic-red);
            padding-top: calc(1.35rem - 4px);
            padding-bottom: calc(1.35rem - 4px);
        }
        .value-card-red {
            background: var(--republic-red);
            color: #FFFFFF;
            border-color: var(--republic-red);
        }
        .value-card-blue *,
        .value-card-red * {
            color: #FFFFFF !important;
        }
        .hero-card {
            border: 1px solid rgba(0, 0, 145, 0.20);
            border-radius: 12px;
            padding: 3rem;
            background:
                linear-gradient(90deg, #000091 0 8px, transparent 8px),
                linear-gradient(180deg, #FFFFFF 0%, #F5F5FE 100%);
            color: var(--ink);
            margin-bottom: 1.35rem;
            box-shadow: 0 8px 28px rgba(0, 0, 145, 0.08);
        }
        .hero-card h1 {
            color: var(--ink);
            font-family: var(--serif);
            font-size: 2.9rem;
            line-height: 1.08;
            font-weight: 500;
            letter-spacing: 0;
            margin-bottom: 1rem;
            max-width: 100%;
            overflow-wrap: normal;
            text-wrap: balance;
            white-space: normal;
        }
        .hero-card p {
            color: var(--ink-2);
            font-family: var(--serif);
            font-size: 1.18rem;
            line-height: 1.55;
            max-width: 820px;
        }
        .hero-mark {
            display: inline-flex;
            align-items: center;
            gap: 0.7rem;
            color: var(--signal);
            font-family: var(--mono);
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }
        .hero-mark::before {
            content: "";
            width: 2.2rem;
            height: 2px;
            background: var(--signal);
            display: inline-block;
        }
        .pill {
            display: inline-block;
            border: 1px solid rgba(0, 0, 145, 0.24);
            border-radius: 4px;
            padding: 0.28rem 0.7rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            color: var(--signal-deep);
            background: #F5F5FE;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .republic-badge {
            border-left: 5px solid var(--republic-red);
            background: #FFFFFF;
            padding: 0.8rem 1rem;
            margin: 0.7rem 0 1rem;
            color: var(--ink);
            box-shadow: 0 1px 4px rgba(0, 0, 145, 0.06);
        }
        .republic-badge strong {
            color: var(--republic-blue);
        }
        .muted {
            color: var(--ink-3);
            font-size: 0.82rem;
            margin-top: 0.55rem;
        }
        .stButton > button {
            min-height: 44px;
            border-radius: 8px;
            border: 1px solid var(--signal);
            background: var(--signal);
            color: white;
            font-family: var(--sans);
            font-weight: 700;
            box-shadow: none;
            transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
        }
        .stButton > button *,
        .stButton > button p,
        .stButton > button span {
            color: white !important;
        }
        .stButton > button:hover {
            background: var(--signal-deep);
            border-color: var(--signal-deep);
            color: white;
        }
        .stButton > button:active {
            transform: translateY(1px);
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--rule-hair);
            border-radius: 8px;
            background: var(--surface);
            padding: 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetric"] * {
            color: var(--ink) !important;
        }
        div[data-testid="stMetric"] label {
            font-family: var(--mono);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--ink-3);
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-family: var(--mono);
            color: var(--ink);
            font-variant-numeric: tabular-nums;
        }
        div[data-testid="stProgress"] * {
            color: var(--ink) !important;
        }
        .stAlert {
            border-radius: 8px;
            border: 1px solid var(--rule-hair);
            background: var(--surface) !important;
        }
        .stAlert,
        .stAlert * {
            color: var(--ink) !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--rule-hair);
            border-radius: 8px;
        }
        input,
        textarea,
        select,
        [data-baseweb="input"] * {
            color: var(--ink) !important;
        }
        [data-baseweb="select"] > div {
            background: #272832 !important;
            border-color: #272832 !important;
            color: #FFFFFF !important;
        }
        [data-baseweb="select"] *,
        [data-baseweb="popover"] [role="listbox"] *,
        [data-baseweb="popover"] [role="option"] * {
            color: #FFFFFF !important;
        }
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [role="option"] {
            background: #272832 !important;
        }
        [data-baseweb="popover"] [role="option"]:hover {
            background: var(--signal) !important;
        }
        input,
        textarea {
            background: var(--surface) !important;
        }
        hr {
            border-color: var(--rule-hair);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, help_text: str | None = None) -> None:
    """Render a compact KPI card."""

    help_markup = f'<div class="muted">{help_text}</div>' if help_text else ""
    value_class = "metric-value metric-value-long" if len(value) > 16 else "metric-value"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="{value_class}">{value}</div>
            {help_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _probability_card(
    prediction: int,
    probability_yes: float | None,
    probability_no: float | None,
) -> None:
    """Render prediction confidence and both class probabilities."""

    if probability_yes is None:
        st.info(
            "Le modèle ne fournit pas de probabilité exploitable pour ce profil. "
            "La décision reste affichée, mais sans pourcentage associé."
        )
        return

    probability_yes = max(0.0, min(1.0, float(probability_yes)))
    probability_no = (
        max(0.0, min(1.0, float(probability_no)))
        if probability_no is not None
        else 1 - probability_yes
    )

    if int(prediction) == 1:
        main_probability = probability_no
        main_label = "Probabilité du résultat prédit"
        main_text = "Risque estimé que l'élève ne poursuive pas ses études."
        progress_text = "de confiance sur la prédiction `higher = no`"
    else:
        main_probability = probability_yes
        main_label = "Probabilité du résultat prédit"
        main_text = "Chance estimée que l'élève poursuive des études supérieures."
        progress_text = "de confiance sur la prédiction `higher = yes`"

    percent_main = main_probability * 100
    st.markdown(
        f"""
        <div class="probability-card">
            <div class="probability-label">{main_label}</div>
            <div class="probability-number">{percent_main:.1f} %</div>
            <div class="probability-helper">
                {main_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(
        main_probability,
        text=f"{percent_main:.1f} % {progress_text}",
    )

    yes_col, no_col = st.columns(2)
    yes_col.metric("higher = yes", f"{probability_yes * 100:.1f} %")
    no_col.metric("higher = no", f"{probability_no * 100:.1f} %")


def _best_model_row(metrics_df: pd.DataFrame) -> pd.Series | None:
    """Return the best model row using F1-score when available."""

    if metrics_df.empty:
        return None
    if "f1" in metrics_df.columns:
        return metrics_df.sort_values("f1", ascending=False).iloc[0]
    return metrics_df.iloc[0]


def _format_metric(value: object) -> str:
    """Format metric values for KPI cards."""

    if pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def _plotly_layout(fig: Any, height: int = 360) -> Any:
    """Apply a consistent visual style to Plotly figures."""

    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=48, b=24),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK, family="Inter, sans-serif"),
        title_font=dict(size=18, family="Fraunces, serif", color=INK),
        legend_title_text="",
        colorway=[SIGNAL, NEGATIVE, INK, SLATE, SAND],
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=INK,
        tickfont=dict(color=INK_3, family="JetBrains Mono, monospace"),
        title_font=dict(color=INK_2),
    )
    fig.update_yaxes(
        gridcolor=RULE_HAIR,
        linecolor=INK,
        tickfont=dict(color=INK_3, family="JetBrains Mono, monospace"),
        title_font=dict(color=INK_2),
    )
    return fig


def _target_distribution_chart(df: pd.DataFrame) -> Any:
    """Build the target distribution chart."""

    target_counts = (
        df[TARGET_COL]
        .value_counts()
        .rename_axis("Poursuite d'études")
        .reset_index(name="Nombre d'élèves")
    )
    fig = px.bar(
        target_counts,
        x="Poursuite d'études",
        y="Nombre d'élèves",
        color="Poursuite d'études",
        color_discrete_map={"yes": SIGNAL, "no": NEGATIVE},
        title="Distribution de la cible higher",
    )
    return _plotly_layout(fig)


def _grade_distribution_chart(df: pd.DataFrame) -> Any:
    """Build a final-grade distribution chart."""

    fig = px.histogram(
        df,
        x="G3",
        color=TARGET_COL,
        nbins=20,
        barmode="overlay",
        color_discrete_map={"yes": SIGNAL, "no": NEGATIVE},
        labels={"G3": "Note finale", TARGET_COL: "Poursuite d'études"},
        title="Distribution des notes finales",
    )
    fig.update_traces(opacity=0.72)
    return _plotly_layout(fig)


def _student_comparison_chart(df: pd.DataFrame) -> Any:
    """Compare key student indicators by target value."""

    comparison_cols = [
        "G1",
        "G2",
        "G3",
        "studytime",
        "failures",
        "absences",
        "parents_edu_sum",
        "risk_behavior_score",
    ]
    available_cols = [column for column in comparison_cols if column in df.columns]
    comparison = (
        df.groupby(TARGET_COL)[available_cols]
        .mean(numeric_only=True)
        .reset_index()
        .melt(id_vars=TARGET_COL, var_name="Indicateur", value_name="Moyenne")
    )
    fig = px.bar(
        comparison,
        x="Indicateur",
        y="Moyenne",
        color=TARGET_COL,
        barmode="group",
        color_discrete_map={"yes": SIGNAL, "no": NEGATIVE},
        labels={TARGET_COL: "Poursuite d'études"},
        title="Comparaison moyenne des profils selon higher",
    )
    return _plotly_layout(fig, height=430)


def _metrics_chart(metrics_df: pd.DataFrame) -> Any:
    """Build a model metrics chart."""

    metric_cols = [
        column
        for column in ["accuracy", "precision", "recall", "f1"]
        if column in metrics_df.columns
    ]
    model_col = "model_name" if "model_name" in metrics_df.columns else "model_key"
    chart_df = metrics_df[[model_col, *metric_cols]].melt(
        id_vars=model_col,
        var_name="Métrique",
        value_name="Score",
    )
    fig = px.bar(
        chart_df,
        x="Métrique",
        y="Score",
        color="Métrique",
        color_discrete_sequence=[SIGNAL, NEGATIVE, INK, SLATE],
        text="Score",
        facet_col=model_col if metrics_df.shape[0] > 1 else None,
        title="Métriques de performance du modèle",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_yaxes(range=[0, 1])
    return _plotly_layout(fig)


def _correlation_heatmap(df: pd.DataFrame) -> Any:
    """Build a compact correlation heatmap for numeric indicators."""

    selected_cols = [
        "age",
        "Medu",
        "Fedu",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2",
        "G3",
        "parents_edu_sum",
        "risk_behavior_score",
        "academic_avg_g1_g2",
    ]
    available_cols = [column for column in selected_cols if column in df.columns]
    corr = df[available_cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[[0, "#FCA5A5"], [0.5, "#F8FAFC"], [1, "#86EFAC"]],
        zmin=-1,
        zmax=1,
        title="Heatmap de corrélation des variables numériques clés",
    )
    return _plotly_layout(fig, height=560)


def _feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> Any:
    """Build a global feature importance chart."""

    chart_df = (
        importance_df.sort_values("importance_abs", ascending=False)
        .head(top_n)
        .sort_values("importance_abs", ascending=True)
    )
    fig = px.bar(
        chart_df,
        x="importance_abs",
        y="feature",
        orientation="h",
        text="importance_abs",
        labels={"importance_abs": "Importance", "feature": "Feature"},
        title=f"Top {min(top_n, len(chart_df))} des variables les plus importantes",
    )
    fig.update_traces(marker_color=SIGNAL, texttemplate="%{text:.3f}")
    return _plotly_layout(fig, height=520)


def _update_engineered_features(sample: pd.DataFrame) -> pd.DataFrame:
    """Keep simple engineered features aligned after user edits."""

    row = sample.copy()

    if {"G1", "G2"}.issubset(row.columns):
        row["grade_avg_12"] = row[["G1", "G2"]].mean(axis=1)
        row["grade_trend_12"] = row["G2"] - row["G1"]
        row["grade_momentum"] = (row["G2"] - row["G1"]) / (row["G1"] + 1)

    if "absences" in row.columns:
        row["high_absence_flag"] = (row["absences"] >= 20).astype(int)

    if {"Dalc", "Walc"}.issubset(row.columns):
        row["alc_index"] = (row["Dalc"] + row["Walc"]) / 2

    if {"goout", "freetime"}.issubset(row.columns):
        row["social_index"] = (row["goout"] + row["freetime"]) / 2

    if {"alc_index", "social_index"}.issubset(row.columns):
        row["risk_behavior_index"] = (row["alc_index"] + row["social_index"]) / 2

    support_cols = [
        column
        for column in ["schoolsup", "famsup", "paid", "activities", "internet", "nursery"]
        if column in row.columns
    ]
    if support_cols:
        row["support_index"] = row[support_cols].replace({"yes": 1, "no": 0}).sum(axis=1)

    if {"Medu", "Fedu"}.issubset(row.columns):
        row["parent_edu_avg"] = row[["Medu", "Fedu"]].mean(axis=1)
        row["parent_edu_gap"] = (row["Medu"] - row["Fedu"]).abs()

    if "failures" in row.columns:
        row["failure_flag"] = (row["failures"] > 0).astype(int)
        row["high_failure_flag"] = (row["failures"] >= 2).astype(int)

    return row


def _render_sidebar() -> str:
    """Render dashboard navigation and return the selected page."""

    if "page" not in st.session_state or st.session_state["page"] not in PAGE_OPTIONS:
        st.session_state["page"] = "Accueil"

    if MINISTRY_LOGO_FILE.exists():
        st.sidebar.image(str(MINISTRY_LOGO_FILE), width=150)
    st.sidebar.markdown(
        """
        <div class="republic-strip">
            <span></span><span></span><span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("Dashboard ML")
    st.sidebar.caption("Prédiction de poursuite d'études")
    page = st.sidebar.radio(
        "Navigation",
        PAGE_OPTIONS,
        key="page",
    )
    st.sidebar.divider()
    st.sidebar.info(
        "Projet académique de proof of concept ML. Les résultats servent à "
        "prioriser l'accompagnement, pas à automatiser une décision."
    )
    st.sidebar.caption(
        "Logo : bloc-marque Éducation nationale, source Wikimedia Commons / "
        "Gouvernement de la République française."
    )
    return page


def _go_to_prediction() -> None:
    """Navigate to the interactive prediction page."""

    st.session_state["page"] = "Prédiction interactive"
    st.rerun()


def _render_kpis(df: pd.DataFrame | None, metrics_df: pd.DataFrame | None) -> None:
    """Render KPI cards used on the overview page."""

    best_model = _best_model_row(metrics_df) if metrics_df is not None else None
    cols = st.columns(6)

    with cols[0]:
        _metric_card("Lignes", f"{df.shape[0]:,}" if df is not None else "N/A")
    with cols[1]:
        _metric_card("Variables", str(df.shape[1]) if df is not None else "N/A")
    with cols[2]:
        _metric_card("Target", TARGET_COL)
    with cols[3]:
        model_name = best_model["model_name"] if best_model is not None else "N/A"
        _metric_card("Meilleur modèle", str(model_name))
    with cols[4]:
        accuracy = best_model["accuracy"] if best_model is not None else None
        _metric_card("Accuracy", _format_metric(accuracy))
    with cols[5]:
        f1_score = best_model["f1"] if best_model is not None else None
        _metric_card("F1-score", _format_metric(f1_score))


def _render_home(df: pd.DataFrame | None, metrics_df: pd.DataFrame | None) -> None:
    """Render the landing page shown first when opening the app."""

    if MINISTRY_LOGO_FILE.exists():
        st.image(str(MINISTRY_LOGO_FILE), width=190)
    st.markdown(
        """
        <div class="republic-strip">
            <span></span><span></span><span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-mark">République française · Éducation</div>
            <h1>Prédire la poursuite d'études supérieures</h1>
            <p>
                Dashboard ML académique pour identifier les élèves qui pourraient
                avoir besoin d'un accompagnement renforcé avant leur orientation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.25, 0.75])
    with left_col:
        st.subheader("Intitulé du projet")
        st.markdown(
            """
            <div class="republic-badge">
                <strong>Proof of Concept ML</strong><br>
                Une interface de démonstration inspirée des codes visuels
                institutionnels : sobriété, lisibilité et aide à la décision.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            "Construire un proof of concept capable de prédire si un élève est "
            "susceptible de poursuivre des études supérieures, à partir de ses "
            "caractéristiques scolaires, familiales et comportementales."
        )
        st.markdown(
            """
            <span class="pill">Machine Learning</span>
            <span class="pill">Éducation</span>
            <span class="pill">Aide à la décision</span>
            <span class="pill">Dashboard Streamlit</span>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.subheader("Tester un étudiant")
        st.write(
            "Remplis quelques caractéristiques simples pour obtenir une prédiction "
            "claire et interprétable."
        )
        st.button(
            "Commencer le test",
            type="primary",
            width="stretch",
            on_click=_go_to_prediction,
        )

    st.divider()
    _render_kpis(df, metrics_df)

    st.subheader("Pourquoi ce projet est utile")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            """
            <article class="value-card value-card-blue">
                <h3>Repérer plus tôt</h3>
                <p>
                    Mettre en évidence des signaux scolaires ou sociaux pouvant
                    justifier un accompagnement.
                </p>
            </article>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <article class="value-card value-card-white">
                <h3>Prioriser l'action</h3>
                <p>
                    Aider les équipes à concentrer leur suivi sur les profils
                    les plus fragiles.
                </p>
            </article>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            """
            <article class="value-card value-card-red">
                <h3>Garder l'humain au centre</h3>
                <p>
                    Le modèle propose un signal, mais la décision reste
                    pédagogique et contextuelle.
                </p>
            </article>
            """,
            unsafe_allow_html=True,
        )


def _render_overview(df: pd.DataFrame | None, metrics_df: pd.DataFrame | None) -> None:
    """Render the overview page."""

    st.title("Prédiction de poursuite d'études supérieures")
    st.markdown(
        """
        Ce dashboard présente un proof of concept de machine learning destiné à
        aider un établissement scolaire à repérer les élèves qui pourraient avoir
        besoin d'un accompagnement renforcé avant leur orientation.
        """
    )

    _render_kpis(df, metrics_df)

    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Problème business")
        st.write(
            "La poursuite d'études supérieures peut dépendre de signaux scolaires, "
            "familiaux et comportementaux. Le modèle transforme ces signaux en "
            "indicateur de risque pour aider les équipes à prioriser leurs actions."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Usage recommandé")
        st.write(
            "Le score doit déclencher une discussion, pas une décision automatique. "
            "Il s'utilise avec l'avis des enseignants, des conseillers et de l'élève."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if df is not None and TARGET_COL in df.columns:
        st.plotly_chart(_target_distribution_chart(df), width="stretch")


def _render_dataset(df: pd.DataFrame | None) -> None:
    """Render the dataset page."""

    st.title("Dataset")
    if df is None:
        st.warning(
            "Jeu de données préparé introuvable. Lance d'abord le notebook ou "
            "l'étape de prétraitement."
        )
        return

    st.write(
        "Le projet utilise le dataset UCI Student Performance. Il contient des "
        "informations scolaires, familiales et sociales sur des élèves, puis une "
        "cible `higher` indiquant leur intention de poursuivre des études supérieures."
    )

    cols = st.columns(4)
    cols[0].metric("Lignes", f"{df.shape[0]:,}")
    cols[1].metric("Colonnes", df.shape[1])
    cols[2].metric("Valeurs manquantes", int(df.isna().sum().sum()))
    cols[3].metric("Target", TARGET_COL)

    st.subheader("Lecture simple des données")
    good_col, bad_col = st.columns(2)
    with good_col:
        st.success("Points positifs")
        st.markdown(
            """
            - Dataset structuré et documenté.
            - Données réelles issues d'un contexte éducatif.
            - Mélange utile de variables scolaires, familiales et comportementales.
            - Très peu ou pas de valeurs manquantes dans le fichier préparé.
            - Présence de features créées pour enrichir l'analyse.
            """
        )
    with bad_col:
        st.warning("Points de vigilance")
        st.markdown(
            """
            - La classe `no` est minoritaire : le dataset est déséquilibré.
            - Certaines variables peuvent être sensibles socialement.
            - La cible est une intention déclarée, pas une décision future observée.
            - Le dataset reste limité en taille pour généraliser à tous les lycées.
            - Certaines notes peuvent créer un risque de fuite d'information selon l'usage.
            """
        )

    st.subheader("Aperçu simple")
    display_cols = [
        column
        for column in [
            "age",
            "sex",
            "studytime",
            "failures",
            "absences",
            "G1",
            "G2",
            "G3",
            "parents_edu_sum",
            "risk_behavior_score",
            TARGET_COL,
        ]
        if column in df.columns
    ]
    st.dataframe(df[display_cols].head(30), width="stretch", hide_index=True)

    st.subheader("Types de variables")
    type_counts = (
        df.dtypes.astype(str)
        .value_counts()
        .rename_axis("Type")
        .reset_index(name="Nombre de colonnes")
    )
    st.dataframe(type_counts, width="stretch", hide_index=True)


def _render_insights(df: pd.DataFrame | None) -> None:
    """Render exploratory insights."""

    st.title("Insights")
    if df is None:
        st.warning("Impossible d'afficher les insights sans dataset préparé.")
        return

    if TARGET_COL not in df.columns:
        st.warning(f"La colonne cible `{TARGET_COL}` est absente du dataset.")
        return

    chart_col, text_col = st.columns([1.5, 0.8])
    with chart_col:
        st.plotly_chart(_target_distribution_chart(df), width="stretch")
    with text_col:
        st.subheader("Lecture métier")
        st.write(
            "La classe `no` représente les élèves qui déclarent ne pas vouloir "
            "poursuivre vers les études supérieures. Elle est minoritaire, ce "
            "qui rend le F1-score important pour évaluer le modèle."
        )

    if "G3" in df.columns:
        st.plotly_chart(_grade_distribution_chart(df), width="stretch")

    st.plotly_chart(_student_comparison_chart(df), width="stretch")
    st.plotly_chart(_correlation_heatmap(df), width="stretch")


def _render_model_performance(metrics_df: pd.DataFrame | None) -> None:
    """Render model performance details."""

    st.title("Performance du modèle")
    if metrics_df is None:
        st.info("Lance `python3 scripts/main.py` pour générer les métriques.")
        return

    best_model = _best_model_row(metrics_df)
    _render_kpis(None, metrics_df)

    st.subheader("Modèles testés pendant le projet")
    st.write(
        "Le notebook compare plusieurs familles de modèles. Le fichier final "
        "`models/best_model.pkl` contient le modèle retenu et sauvegardé pour "
        "l'application."
    )

    for start in range(0, len(MODEL_SUMMARIES), 3):
        cols = st.columns(3)
        for col, model_info in zip(cols, MODEL_SUMMARIES[start : start + 3]):
            with col:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown(f"**{model_info['name']}**")
                st.caption(model_info["role"])
                st.markdown(f"**Plus :** {model_info['plus']}")
                st.markdown(f"**Moins :** {model_info['minus']}")
                st.markdown(f"**Statut :** {model_info['status']}")
                st.markdown("</div>", unsafe_allow_html=True)

    st.plotly_chart(_metrics_chart(metrics_df), width="stretch")
    st.subheader("Table des métriques")
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    if best_model is not None:
        st.subheader("Modèle choisi")
        st.success(f"Modèle final : `{best_model['model_name']}`")
        st.write(
            "Le Random Forest a été choisi car il offre un bon compromis pour ce "
            "projet : il gère correctement les variables numériques et catégorielles "
            "après preprocessing, il capte des relations non linéaires, il est plus "
            "robuste qu'un arbre seul, et ses performances finales restent solides "
            "sur la classe minoritaire suivie par le F1-score."
        )
        st.write(
            f"Dans l'évaluation finale, il obtient une accuracy de "
            f"{_format_metric(best_model.get('accuracy'))} et un F1-score de "
            f"{_format_metric(best_model.get('f1'))}."
        )

    st.caption(
        "La cible est encodée à 1 pour les élèves à risque de ne pas poursuivre "
        "vers les études supérieures, et à 0 sinon."
    )


def _render_explainability(importance_df: pd.DataFrame | None) -> None:
    """Render global model explainability."""

    st.title("Explicabilité du modèle")
    st.write(
        "L'explicabilité sert à comprendre les signaux utilisés par le modèle. "
        "Ici, on regarde quelles variables influencent le plus la prédiction "
        "`higher` au niveau global."
    )

    if importance_df is None:
        st.info(
            "Lance `python3 scripts/main.py` pour générer "
            "`results/feature_importance.csv`."
        )
        return

    if importance_df.empty or importance_df["importance_abs"].sum() == 0:
        st.warning(
            "Le modèle sauvegardé ne fournit pas d'importance directement "
            "interprétable. Le fichier existe, mais les valeurs sont nulles."
        )
        st.dataframe(importance_df, width="stretch", hide_index=True)
        return

    st.subheader("Importance globale des variables")
    st.plotly_chart(_feature_importance_chart(importance_df), width="stretch")

    top_features = importance_df.sort_values("importance_abs", ascending=False).head(5)
    readable_features = ", ".join(top_features["feature"].tolist())
    st.markdown(
        f"""
        **Lecture métier :** les variables les plus utilisées par le modèle sont
        `{readable_features}`. Cela ne veut pas dire qu'elles causent directement
        la poursuite d'études, mais qu'elles aident fortement le modèle à séparer
        les profils dans ce dataset.
        """
    )

    st.subheader("Table complète")
    st.dataframe(importance_df, width="stretch", hide_index=True)

    st.caption(
        "Pour le Random Forest, l'importance correspond à la contribution moyenne "
        "des variables aux décisions des arbres. Les variables catégorielles encodées "
        "sont regroupées sous leur nom d'origine."
    )


def _render_interactive_prediction() -> None:
    """Render a simple prediction workflow."""

    st.title("Prédiction interactive")
    st.write(
        "Cette démo permet de partir d'un élève réel du jeu d'évaluation ou "
        "d'ajouter un nouveau profil. Les champs ci-dessous modifient quelques "
        "signaux clés sans reconstruire tout le dataset."
    )

    try:
        X_test = _load_test_rows()
        model = _load_best_model()
    except Exception as exc:
        st.warning(f"Démo de prédiction indisponible : {exc}")
        return

    add_student_option = "Ajouter un élève"
    profile_options: list[int | str] = [add_student_option, *list(range(len(X_test)))]
    selected_profile = st.selectbox(
        "Profil de départ",
        options=profile_options,
        format_func=lambda value: value if isinstance(value, str) else f"Élève #{value}",
    )
    if selected_profile == add_student_option:
        sample = _build_new_student_profile(X_test)
        st.caption(
            "Nouveau profil initialisé avec des valeurs représentatives du dataset. "
            "Ajuste les paramètres ci-dessous pour créer ton élève."
        )
    else:
        sample = X_test.iloc[[int(selected_profile)]].copy()

    form_col, result_col = st.columns([1.2, 0.8])
    with form_col:
        st.subheader("Paramètres ajustables")
        if "age" in sample.columns:
            sample.loc[:, "age"] = st.slider("Âge", 15, 22, int(sample["age"].iloc[0]))
        if "studytime" in sample.columns:
            sample.loc[:, "studytime"] = st.slider(
                "Temps d'étude hebdomadaire", 1, 4, int(sample["studytime"].iloc[0])
            )
        if "failures" in sample.columns:
            sample.loc[:, "failures"] = st.slider(
                "Nombre d'échecs scolaires passés", 0, 4, int(sample["failures"].iloc[0])
            )
        if "absences" in sample.columns:
            sample.loc[:, "absences"] = st.slider(
                "Nombre d'absences", 0, 80, int(sample["absences"].iloc[0])
            )

        grade_col_1, grade_col_2 = st.columns(2)
        with grade_col_1:
            if "G1" in sample.columns:
                sample.loc[:, "G1"] = st.slider("Note G1", 0, 20, int(sample["G1"].iloc[0]))
        with grade_col_2:
            if "G2" in sample.columns:
                sample.loc[:, "G2"] = st.slider("Note G2", 0, 20, int(sample["G2"].iloc[0]))

        behavior_col_1, behavior_col_2, behavior_col_3 = st.columns(3)
        with behavior_col_1:
            if "goout" in sample.columns:
                sample.loc[:, "goout"] = st.slider(
                    "Sorties", 1, 5, int(sample["goout"].iloc[0])
                )
        with behavior_col_2:
            if "Dalc" in sample.columns:
                sample.loc[:, "Dalc"] = st.slider(
                    "Alcool semaine", 1, 5, int(sample["Dalc"].iloc[0])
                )
        with behavior_col_3:
            if "Walc" in sample.columns:
                sample.loc[:, "Walc"] = st.slider(
                    "Alcool week-end", 1, 5, int(sample["Walc"].iloc[0])
                )

    sample = _update_engineered_features(sample)
    prediction = int(model.predict(sample)[0])
    probabilities = predict_class_probabilities(model, sample)
    probability_yes = probabilities.get("yes")
    probability_no = probabilities.get("no")
    contribution_df, explained_probability = explain_local_prediction(
        model=model,
        sample=sample,
        X_reference=X_test,
        top_n=8,
    )
    if probability_yes is None:
        probability_yes = explained_probability
        probability_no = 1 - explained_probability if explained_probability is not None else None

    with result_col:
        st.subheader("Résultat")
        _probability_card(prediction, probability_yes, probability_no)
        st.metric("Décision modèle", _format_risk_label(prediction))
        if prediction == 1:
            st.warning(_risk_message(prediction))
        else:
            st.success(_risk_message(prediction))
        st.caption(
            "Cette prédiction est une aide à l'analyse. Elle doit être complétée "
            "par un échange humain et par le contexte réel de l'élève."
        )

    st.subheader("Pourquoi ce score ?")
    st.write(
        "Les facteurs ci-dessous comparent l'élève sélectionné à un profil de "
        "référence du jeu de test. Une contribution positive augmente la probabilité "
        "de `higher = yes`; une contribution négative la diminue."
    )

    if contribution_df.empty:
        st.info(
            "Le modèle ne permet pas de calculer une explication locale détaillée "
            "pour cette prédiction."
        )
    else:
        positive_factors = contribution_df[contribution_df["contribution"] > 0]
        negative_factors = contribution_df[contribution_df["contribution"] < 0]

        yes_col, no_col = st.columns(2)
        with yes_col:
            st.success("Facteurs qui tirent vers `yes`")
            if positive_factors.empty:
                st.write("Aucun facteur positif dominant détecté.")
            else:
                for _, row in positive_factors.head(4).iterrows():
                    st.write(
                        f"- {row['label']} : +{row['contribution'] * 100:.1f} pts"
                    )

        with no_col:
            st.warning("Facteurs qui tirent vers `no`")
            if negative_factors.empty:
                st.write("Aucun facteur négatif dominant détecté.")
            else:
                for _, row in negative_factors.head(4).iterrows():
                    st.write(
                        f"- {row['label']} : {row['contribution'] * 100:.1f} pts"
                    )

        display_explanation = contribution_df[
            ["label", "value", "reference", "contribution"]
        ].rename(
            columns={
                "label": "facteur",
                "value": "valeur_etudiant",
                "reference": "valeur_reference",
                "contribution": "impact_sur_proba_yes",
            }
        )
        st.dataframe(display_explanation, width="stretch", hide_index=True)

    with st.expander("Voir les variables envoyées au modèle"):
        st.dataframe(sample, width="stretch", hide_index=True)


def _render_conclusion() -> None:
    """Render the conclusion page."""

    st.title("Conclusion")
    st.markdown(
        """
        Ce projet montre comment un modèle de classification peut aider à repérer
        des élèves potentiellement fragiles dans leur projection vers les études
        supérieures.

        **Ce que le dashboard apporte :**

        - une vue synthétique du dataset et de la cible ;
        - des indicateurs simples pour interpréter les performances ;
        - des insights visuels sur les signaux scolaires et sociaux ;
        - une démo de prédiction compréhensible pour un utilisateur non technique.

        **Limite importante :** le modèle ne doit jamais remplacer l'analyse
        pédagogique. Il doit servir de support de discussion et d'aide à la
        priorisation.
        """
    )


def build_app() -> None:
    """Render the Streamlit dashboard."""

    st.set_page_config(
        page_title="Dashboard ML - Poursuite d'études",
        layout="wide",
    )
    _inject_css()

    page = _render_sidebar()
    df = _load_features() if FEATURES_DATA_FILE.exists() else None
    metrics_df = _load_metrics() if MODEL_METRICS_FILE.exists() else None
    importance_df = (
        _load_feature_importance() if FEATURE_IMPORTANCE_FILE.exists() else None
    )

    if page == "Accueil":
        _render_home(df, metrics_df)
    elif page == "Vue d'ensemble":
        _render_overview(df, metrics_df)
    elif page == "Dataset":
        _render_dataset(df)
    elif page == "Insights":
        _render_insights(df)
    elif page == "Performance du modèle":
        _render_model_performance(metrics_df)
    elif page == "Explicabilité du modèle":
        _render_explainability(importance_df)
    elif page == "Prédiction interactive":
        _render_interactive_prediction()
    else:
        _render_conclusion()


if __name__ == "__main__":
    build_app()
