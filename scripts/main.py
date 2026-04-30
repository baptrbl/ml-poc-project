"""Evaluate configured models on the processed student dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODELS
from data import load_dataset_split
from metrics import compute_metrics
from model_io import load_model
from results import write_metrics


def main() -> None:
    """Load the data split, evaluate each model, and write metrics."""

    _, X_test, _, y_test = load_dataset_split()
    rows: list[dict[str, object]] = []

    for model_id, model_config in MODELS.items():
        model = load_model(model_config["path"])
        y_pred = model.predict(X_test)

        rows.append(
            {
                "model_id": model_id,
                "model_name": model_config["name"],
                **compute_metrics(y_test, y_pred),
            }
        )

    metrics_df = write_metrics(rows)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
