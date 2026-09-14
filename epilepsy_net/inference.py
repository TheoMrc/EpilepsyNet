"""Path-independent PyTorch inference for the released EpilepsyNet model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epilepsy_net.models import DEVICE, EpilepsyNet

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "model.json"
DEFAULT_WEIGHTS = REPOSITORY_ROOT / "models" / "epilepsynet.pt"


def load_model(
    weights_path: str | Path = DEFAULT_WEIGHTS,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | torch.device = DEVICE,
) -> tuple[EpilepsyNet, dict]:
    """Build EpilepsyNet from its released configuration and load its weights."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    features = config["features"]
    labels = config["label_cols"]
    model = EpilepsyNet(
        features=features,
        hidden_channels=config["HIDDEN_CHANNELS"],
        num_layers=config["N_LAYERS"],
        kernel_size=config["KERNEL_SIZE"],
        dropout_rate=config["DROPOUT_RATE"],
        n_output_states=len(labels),
    ).to(device)
    state = torch.load(Path(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


def predict_array(
    feature_values: np.ndarray,
    *,
    model: EpilepsyNet | None = None,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class indices and probabilities for an array shaped (time, features)."""
    if model is None or config is None:
        model, config = load_model()
    values = np.asarray(feature_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(config["features"]):
        raise ValueError(
            f"Expected (time, {len(config['features'])}) values, got {values.shape}"
        )
    values = np.abs(values)
    device = next(model.parameters()).device
    inputs = torch.from_numpy(values).transpose(0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0].transpose(0, 1)
    probabilities_np = probabilities.cpu().numpy()
    return probabilities_np.argmax(axis=1), probabilities_np


def predict_dataframe(
    dataframe: pd.DataFrame,
    *,
    model: EpilepsyNet | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Predict one state per millisecond from a DanioTracker time-series table."""
    if model is None or config is None:
        model, config = load_model()
    missing = [name for name in config["features"] if name not in dataframe.columns]
    if missing:
        raise KeyError("Missing EpilepsyNet features: " + ", ".join(missing))
    indices, probabilities = predict_array(
        dataframe[config["features"]].to_numpy(), model=model, config=config
    )
    result = pd.DataFrame({"state": [config["label_cols"][i] for i in indices]})
    for index, label in enumerate(config["label_cols"]):
        result[f"probability_{label}"] = probabilities[:, index]
    return result


def find_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive intervals for True runs in a one-dimensional mask."""
    intervals: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(mask) - 1))
    return intervals


def predict_video_timestamps(
    experiment: str,
    condition: str,
    video_number: str,
    data_dir: str | Path | None = None,
) -> list[dict]:
    """Return annotation-tool intervals for one DanioTracker fish time series."""
    if data_dir is None:
        from epilepsy_net.utils import DATA_DIR

        data_dir = DATA_DIR
    csv_path = (
        Path(data_dir)
        / experiment
        / condition
        / "time_series"
        / f"fish_{video_number}_time_series.csv"
    )
    if not csv_path.is_file():
        return []

    prediction = predict_dataframe(pd.read_csv(csv_path))
    output = []
    export_names = {"not_moving": "stationnary", "cbm": "cbm"}
    for label, export_name in export_names.items():
        for start, end in find_intervals(prediction["state"].to_numpy() == label):
            output.append({"state": export_name, "start": start, "end": end})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("epilepsynet_predictions.csv")
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=str(DEVICE))
    args = parser.parse_args()

    model, config = load_model(args.weights, args.config, args.device)
    prediction = predict_dataframe(
        pd.read_csv(args.input_csv), model=model, config=config
    )
    prediction.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
