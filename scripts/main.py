"""Main project entry point for model evaluation and optional app launch."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import APP_ENTRYPOINT, MODELS, STREAMLIT_HOST, STREAMLIT_PORT
from data import load_dataset_split
from metrics import compute_metrics
from model_io import load_model
from results import write_metrics


def _load_module(module_name: str, module_path: Path) -> Any:
    """Load a Python module from a file path."""

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_app_entrypoint() -> None:
    """Validate that the Streamlit app keeps the template contract."""

    if not APP_ENTRYPOINT.exists():
        raise FileNotFoundError(f"Streamlit entry point not found: {APP_ENTRYPOINT}")

    app_module = _load_module("project_app", APP_ENTRYPOINT)
    if not hasattr(app_module, "build_app") or not callable(app_module.build_app):
        raise TypeError("src/app.py must expose a callable `build_app()` function.")


def _validate_models_config() -> None:
    """Validate the model registry defined in src/config.py."""

    if not MODELS:
        raise ValueError("config.MODELS is empty. Register at least one trained model.")

    for model_id, model_config in MODELS.items():
        if "path" not in model_config:
            raise ValueError(f"Missing `path` for model `{model_id}` in config.MODELS.")


def _evaluate_models(X_test: Any, y_test: Any) -> list[dict[str, object]]:
    """Evaluate every configured model on the same test split."""

    rows: list[dict[str, object]] = []

    for model_id, model_config in MODELS.items():
        model_path = Path(model_config["path"])
        model = load_model(model_path)

        if not hasattr(model, "predict"):
            raise TypeError(
                f"Loaded object for model `{model_id}` does not expose a predict method."
            )

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("compute_metrics() must return a non-empty dictionary.")

        row: dict[str, object] = {
            "model_key": model_id,
            "model_name": model_config.get("name", model_id),
            "model_path": str(model_path),
        }
        row.update(
            {
                metric_name: float(metric_value)
                for metric_name, metric_value in metrics.items()
            }
        )
        rows.append(row)

    return rows


def _streamlit_env() -> dict[str, str]:
    """Build an environment where src modules resolve as top-level imports."""

    env = os.environ.copy()
    pythonpath_entries = [str(SRC_DIR)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _launch_streamlit() -> None:
    """Launch the Streamlit app used for the project demo."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_ENTRYPOINT),
            "--server.address",
            STREAMLIT_HOST,
            "--server.port",
            str(STREAMLIT_PORT),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=_streamlit_env(),
    )


def main(launch_app: bool = False) -> None:
    """Load the data split, evaluate configured models, and write metrics."""

    _validate_app_entrypoint()
    _validate_models_config()
    _, X_test, _, y_test = load_dataset_split()
    metrics_df = write_metrics(_evaluate_models(X_test, y_test))
    print("Model evaluation completed. Metrics saved to results/model_metrics.csv")
    print(metrics_df.to_string(index=False))

    if launch_app:
        print(f"\nLaunching Streamlit on http://{STREAMLIT_HOST}:{STREAMLIT_PORT} ...")
        _launch_streamlit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate project models.")
    parser.add_argument(
        "--launch-app",
        action="store_true",
        help="Launch the Streamlit app after model evaluation.",
    )
    args = parser.parse_args()
    main(launch_app=args.launch_app)
