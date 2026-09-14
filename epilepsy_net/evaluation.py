"""
Evaluate fish-level metrics for the trained EpilepsyNet model on the validation split.

Metrics (seizure channel focused):
- Fish-level presence classification (has any seizure vs none): precision, recall, F1, confusion counts.
- Average seizure duration per fish: labeled vs predicted.
- Extras: duration MAE, duration correlation (Pearson), presence accuracy.

Outputs a summary to stdout and writes JSON and XLSX files to explicit paths.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epilepsy_net.inference import load_model as load_released_model
from epilepsy_net.load_data import (
    FEATURES,
    LABEL_COLS,
    TimeSeriesDataset,
    train_test_split_hash,
)
from epilepsy_net.models import DEVICE
from epilepsy_net.utils import DATA_DIR

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPOSITORY_ROOT / "models" / "epilepsynet.pt"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "model.json"


def load_model():
    """Load the released checkpoint used by DanioTracker."""
    return load_released_model(
        weights_path=WEIGHTS_PATH,
        config_path=CONFIG_PATH,
        device=DEVICE,
    )[0]


def argmax_along_classes(array_bool_ch_time: np.ndarray) -> np.ndarray:
    """Convert one-hot-like boolean channels (C, T) into class indices (T,)."""
    if array_bool_ch_time.ndim != 2:
        raise ValueError("Expected shape (C, T)")
    # In case of non-exclusive labels, prefer the channel with the highest value
    return np.argmax(array_bool_ch_time, axis=0)


def compute_presence_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Binary presence metrics given boolean arrays per fish."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def find_total_duration(class_indices: np.ndarray, target_class: int) -> int:
    """Sum of timesteps where class == target_class."""
    return int(np.sum(class_indices == target_class))


def evaluate_dataset(dataset: list[dict]) -> dict[str, float | int | str]:
    # Load model
    model = load_model()

    # Determine class index for the seizure state
    label_cols = LABEL_COLS
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf8") as f:
            cfg = json.load(f)
            label_cols = cfg.get("label_cols", LABEL_COLS)
    if "cbm" not in label_cols:
        raise ValueError("'cbm' not found in LABEL_COLS")
    cbm_index = label_cols.index("cbm")

    presence_true: list[int] = []
    presence_pred: list[int] = []
    duration_true: list[int] = []
    duration_pred: list[int] = []

    for fish in dataset:
        df = fish["data"]
        if df is None:
            continue
        values = np.abs(
            df[FEATURES].values.astype(float)
        )  # match training preprocessing
        features = (
            torch.tensor(values, dtype=torch.float32)
            .transpose(0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            logits = model(features)[0]  # (C, T)
            pred_classes = torch.argmax(logits, dim=0).cpu().numpy()

        label_ch_time = df[label_cols].values.astype(float).T  # (C, T)
        true_classes = argmax_along_classes(label_ch_time)

        # Fish-level seizure presence flag
        has_cbm_true = int(np.any(true_classes == cbm_index))
        has_cbm_pred = int(np.any(pred_classes == cbm_index))

        presence_true.append(has_cbm_true)
        presence_pred.append(has_cbm_pred)

        # Total seizure duration per fish
        duration_true.append(find_total_duration(true_classes, cbm_index))
        duration_pred.append(find_total_duration(pred_classes, cbm_index))

    presence_true_arr = np.array(presence_true, dtype=int)
    presence_pred_arr = np.array(presence_pred, dtype=int)
    duration_true_arr = np.array(duration_true, dtype=float)
    duration_pred_arr = np.array(duration_pred, dtype=float)

    presence = compute_presence_metrics(presence_true_arr, presence_pred_arr)

    avg_duration_true = (
        float(np.mean(duration_true_arr)) if len(duration_true_arr) > 0 else 0.0
    )
    avg_duration_pred = (
        float(np.mean(duration_pred_arr)) if len(duration_pred_arr) > 0 else 0.0
    )
    duration_mae = (
        float(np.mean(np.abs(duration_true_arr - duration_pred_arr)))
        if len(duration_true_arr) > 0
        else 0.0
    )
    if (
        len(duration_true_arr) > 1
        and np.std(duration_true_arr) > 0
        and np.std(duration_pred_arr) > 0
    ):
        duration_corr = float(np.corrcoef(duration_true_arr, duration_pred_arr)[0, 1])
    else:
        duration_corr = 0.0

    summary = {
        "n_fish": int(len(presence_true_arr)),
        "cbm_presence_precision": presence["precision"],
        "cbm_presence_recall": presence["recall"],
        "cbm_presence_f1": presence["f1"],
        "cbm_presence_accuracy": presence["accuracy"],
        "cbm_presence_tp": presence["tp"],
        "cbm_presence_fp": presence["fp"],
        "cbm_presence_fn": presence["fn"],
        "cbm_presence_tn": presence["tn"],
        "avg_cbm_duration_true": avg_duration_true,
        "avg_cbm_duration_pred": avg_duration_pred,
        "cbm_duration_mae": duration_mae,
        "cbm_duration_corr": duration_corr,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate fish-level seizure metrics")
    split = parser.add_mutually_exclusive_group()
    split.add_argument(
        "--on-val", action="store_true", help="Evaluate on validation split only"
    )
    split.add_argument(
        "--on-train", action="store_true", help="Evaluate on training split only"
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--annotations-dir", default="annotations")
    parser.add_argument(
        "--output-json", type=Path, default=Path("outputs/fish_level_metrics.json")
    )
    parser.add_argument(
        "--output-xlsx", type=Path, default=Path("outputs/fish_level_metrics.xlsx")
    )
    args = parser.parse_args()

    all_data = TimeSeriesDataset.load_dataset(args.data_dir, args.annotations_dir)
    train_data, val_data = train_test_split_hash(all_data, test_size=0.2)
    if args.on_val:
        dataset = val_data
        split_name = "val"
    elif args.on_train:
        dataset = train_data
        split_name = "train"
    else:
        dataset = all_data
        split_name = "all"

    summary = evaluate_dataset(dataset)
    summary["split"] = split_name
    print(json.dumps(summary, indent=2))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf8") as output_file:
        json.dump(summary, output_file, indent=2)
    print(f"Saved fish-level metrics to {args.output_json}")

    # Write XLSX with descriptions
    descriptions = {
        "split": "Dataset subset used: 'train', 'val', or 'all'",
        "n_fish": "Number of fish evaluated in this subset",
        "cbm_presence_precision": "Among fish predicted to have any seizure, fraction that truly have a seizure (TP / (TP + FP))",
        "cbm_presence_recall": "Among fish that truly have any seizure, fraction predicted to have a seizure (TP / (TP + FN))",
        "cbm_presence_f1": "Harmonic mean of presence precision and recall (2PR/(P+R))",
        "cbm_presence_accuracy": "Overall correctness for presence classification ((TP + TN) / total fish)",
        "cbm_presence_tp": "Fish with a seizure present and predicted present",
        "cbm_presence_fp": "Fish with no seizure but predicted present",
        "cbm_presence_fn": "Fish with a seizure present but predicted absent",
        "cbm_presence_tn": "Fish with no seizure and predicted absent",
        "avg_cbm_duration_true": "Average number of frames labeled as seizure per fish",
        "avg_cbm_duration_pred": "Average number of frames predicted as seizure per fish",
        "cbm_duration_mae": "Mean absolute error between labeled and predicted seizure duration per fish",
        "cbm_duration_corr": "Pearson correlation between labeled and predicted seizure durations across fish",
    }

    rows = []
    for key, value in summary.items():
        rows.append(
            {"metric": key, "value": value, "explanation": descriptions.get(key, "")}
        )
    df = pd.DataFrame(rows, columns=["metric", "value", "explanation"])
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.output_xlsx, index=False)
    print(f"Saved fish-level metrics table to {args.output_xlsx}")


if __name__ == "__main__":
    main()
