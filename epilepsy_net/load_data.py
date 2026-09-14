"""Script used to load the database for the EpilepsyNet model."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from epilepsy_net.utils import (
    ANNOTATIONS_DIR,
    get_flattened_annotations,
    load_all_annotations,
)

FEATURES = [
    "rolling state",
    "head speed (mm/s)",
    # "point 2 speed (mm/s)",
    # "point 3 speed (mm/s)",
    # "point 4 speed (mm/s)",
    # "point 5 speed (mm/s)",
    # "point 6 speed (mm/s)",
    # "point 7 speed (mm/s)",
    # "point 8 speed (mm/s)",
    "tail speed (mm/s)",
    "midline speed (mm/s)",
    "bending speed 1 (°/ms)",
    "bending speed 2 (°/ms)",
    "bending speed 3 (°/ms)",
    "bending speed 4 (°/ms)",
    "bending speed 5 (°/ms)",
    "bending speed 6 (°/ms)",
    "bending speed 7 (°/ms)",
    "mean bending speed (°/ms)",
    "total bending speed (°/ms)",
    "total curvature (°)",
    # "bending angle 1 (°)",
    # "bending angle 2 (°)",
    # "bending angle 3 (°)",
    # "bending angle 4 (°)",
    # "bending angle 5 (°)",
    # "bending angle 6 (°)",
    # "bending angle 7 (°)",
]
LABEL_COLS = ["not_moving", "cbm", "swimming"]


class TimeSeriesDataset(Dataset):
    """Dataset class for the EpilepsyNet model."""

    def __init__(self, fishes_data: list[dict]):
        """Initialize dataset with fish data."""
        self.fishes_features = [
            torch.tensor(np.abs(fish_data["data"][FEATURES].values.astype(float)))
            .float()
            .T
            for fish_data in fishes_data
            if isinstance(fish_data["data"], pd.DataFrame)
        ]
        self.fishes_label = [
            torch.tensor(fish_data["data"][LABEL_COLS].values.astype(float)).float().T
            for fish_data in fishes_data
            if isinstance(fish_data["data"], pd.DataFrame)
        ]

        # matplotlib plot each feature of each fish
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))
        # for fish_data in fishes_data:
        #     for feature in FEATURES:
        #         plt.plot(fish_data["data"][feature])
        #         plt.title(feature)
        #         plt.show()
        self.fishes_metadata = fishes_data

    @staticmethod
    def load_dataset(
        database_dir: str, annotations_dir: str | None = None
    ) -> list[dict]:
        """Load the database from annotations and time series Excel files."""
        annotations_path = (
            Path(annotations_dir) if annotations_dir is not None else ANNOTATIONS_DIR
        )
        all_annotations = load_all_annotations(annotations_path)
        annotations: list[dict] = get_flattened_annotations(
            all_annotations, database_dir
        )

        dataset = []
        for annotation in tqdm(annotations, desc="Loading fish database"):
            time_series_path = annotation["time_series_path"]
            if not os.path.isfile(time_series_path):
                continue

            fish_data = pd.read_csv(time_series_path)
            if len(fish_data) < 30:
                continue
            fish_data = add_label_columns(fish_data, annotation["timestamps"]).iloc[1:]
            annotation["data"] = fish_data
            dataset.append(annotation)

        print(f"Loaded {len(dataset)} fish data from annotations.")
        print(f"Number of lost annotations: {len(annotations) - len(dataset)}")

        return dataset

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.fishes_features)

    def __getitem__(self, idx):
        """Get features and labels for a specific index."""
        features = self.fishes_features[idx]  # (n_features, time)
        labels = self.fishes_label[idx]  # (n_pred_ch, time)
        return features, labels


def add_label_columns(df: pd.DataFrame, timestamps) -> pd.DataFrame:
    """Add label columns to the DataFrame based on timestamps."""
    not_moving_state = np.zeros(len(df), dtype=np.float32)
    cbm_state = np.zeros(len(df), dtype=np.float32)
    for interval in timestamps:
        start, end = interval["start"], interval["end"]
        if interval["state"] == "cbm":
            cbm_state[start : end + 1] = 1
        else:
            not_moving_state[start : end + 1] = 1

    df["not_moving"] = not_moving_state
    df["cbm"] = cbm_state
    df["swimming"] = 1 - (not_moving_state + cbm_state)
    return df


def train_test_split_hash(
    dataset: list[dict], test_size: float = 0.2
) -> tuple[list[dict], list[dict]]:
    """Split dataset into train and test sets using a hash-based method."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    train_set = []
    test_set = []
    for fish in dataset:
        fish_path_hash_fraction = int(
            hashlib.sha256(fish["time_series_path"].encode()).hexdigest(), 16
        ) / (2**256 - 1)
        if fish_path_hash_fraction < test_size:
            test_set.append(fish)
        else:
            train_set.append(fish)
    return train_set, test_set
