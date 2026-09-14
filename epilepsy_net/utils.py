"""
Helper functions for the BreakdanceFish project.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = Path(
    os.environ.get("EPILEPSYNET_ANNOTATIONS_DIR", REPOSITORY_ROOT / "annotations")
)
DATA_DIR = Path(os.environ.get("EPILEPSYNET_DATA_DIR", REPOSITORY_ROOT / "data"))
USERS_FILE = Path(
    os.environ.get(
        "EPILEPSYNET_USERS_FILE", REPOSITORY_ROOT / "annotator" / "users.json"
    )
)


def sorting_folder(folder: str) -> tuple[str, int, str]:
    """Sort folders by molecule name and their numeric suffix if present, otherwise return 0"""
    parts = folder.split("_")
    molecule_name = parts[0]
    if len(parts) > 1 and parts[1].isdigit():
        concentration = int(parts[1])
    else:
        concentration = 0
    return molecule_name, concentration, folder


def get_all_videos() -> dict:
    """Iterate through all videos in all conditions in all experiments"""
    videos_dict: defaultdict[str, dict[str, list[str]]] = defaultdict(dict)
    if not os.path.isdir(DATA_DIR):
        return videos_dict

    for experiment in sorted(os.listdir(DATA_DIR)):
        experiment_path = os.path.join(DATA_DIR, experiment)
        if not os.path.isdir(experiment_path):
            continue
        folders = [
            folder
            for folder in os.listdir(experiment_path)
            if os.path.isdir(os.path.join(experiment_path, folder))
        ]
        exp_conditions = sorted(folders, key=sorting_folder)
        for condition in exp_conditions:
            condition_path = os.path.join(experiment_path, condition, "fish_videos")
            if not os.path.isdir(condition_path):
                continue
            videos_dict[experiment][condition] = sorted(
                [
                    video
                    for video in os.listdir(condition_path)
                    if video.endswith(".webm")
                ],
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
    return videos_dict


def get_unannotated_videos(
    videos_dict: dict, annotations: dict
) -> list[tuple[str, str, str]]:
    """Get unannotated videos by comparing the videos dict and the annotations dict"""
    unannotated_videos: list[tuple[str, str, str]] = []
    for experiment, conditions in videos_dict.items():
        for condition, videos_list in conditions.items():
            annotated_videos = annotations.get(experiment, {}).get(condition, [])
            unannotated_videos.extend(
                (experiment, condition, video)
                for video in videos_list
                if video not in annotated_videos
            )
    return unannotated_videos


def get_leaderboard_data(annotations: dict, users: list[str]) -> dict:
    """Get the leaderboard data from the annotations"""
    leaderboard_data = {i: 0 for i in users}
    for experiment in annotations:
        for condition in annotations[experiment]:
            videos = annotations[experiment][condition]
            for video in videos:
                user = annotations[experiment][condition][video]["username"]
                leaderboard_data[user] = leaderboard_data.get(user, 0) + 1
    leaderboard_data = dict(
        sorted(leaderboard_data.items(), key=lambda x: x[1], reverse=True)
    )
    return leaderboard_data


def load_all_annotations(
    annotations_dir: str | os.PathLike = ANNOTATIONS_DIR,
) -> defaultdict:
    """Load every EFAS annotation JSON file from a directory."""
    annotations: defaultdict[str, dict[str, list[str]]] = defaultdict(dict)
    annotations_dir = Path(annotations_dir)
    if not annotations_dir.is_dir():
        return annotations
    for annotation_path in sorted(annotations_dir.glob("*_annotations.json")):
        experiment_name = annotation_path.name.removesuffix("_annotations.json")
        with annotation_path.open("r", encoding="utf-8") as annotation_file:
            annotations[experiment_name].update(json.load(annotation_file))
    return annotations


def load_experiment_annotations(experiment: str) -> dict:
    """Load the annotation file and return its content for a specific experiment"""
    annotation_path = os.path.join(ANNOTATIONS_DIR, f"{experiment}_annotations.json")
    if os.path.exists(annotation_path):
        with open(annotation_path, encoding="utf-8") as fp:
            annotations = json.load(fp)
    else:
        annotations = {}
    return annotations


def load_users() -> list[str]:
    """Load configured annotator names."""
    if not os.path.isfile(USERS_FILE):
        return ["annotator"]
    with open(USERS_FILE, encoding="utf-8") as f:
        users = json.load(f)
    return users


def save_annotations(annotations: dict, experiment: str) -> None:
    """Save the annotations to the file"""
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    annotation_path = os.path.join(ANNOTATIONS_DIR, f"{experiment}_annotations.json")
    with open(annotation_path, "w", encoding="utf-8") as fp:
        json.dump(annotations, fp, indent=4)


def get_flattened_annotations(
    all_annotations: dict[str, dict[str, dict[str, dict]]], database_dir: str
) -> list[dict]:
    """Flatten all annotations into a list of dicts instead of a multi entry dict"""
    flattened_annotations = []
    for experiment, conditions in all_annotations.items():
        for condition, fishes in conditions.items():
            for fish_name, annotation in fishes.items():
                fish_annotations = {}
                timestamps = annotation["timestamps"]
                if timestamps is None:
                    continue
                fish_annotations["experiment"] = experiment
                fish_annotations["condition"] = condition
                fish_annotations["timestamps"] = timestamps
                fish_annotations["time_series_path"] = os.path.join(
                    database_dir,
                    experiment,
                    condition,
                    "time_series",
                    f"{fish_name.removesuffix('.webm')}_time_series.csv",
                )
                flattened_annotations.append(fish_annotations)
    return flattened_annotations


def plot_pred_and_labels(proba_preds, val_labels, n_examples, file_path):
    """Plotting function to visualize model performance on a few examples"""
    n_pred_channels = proba_preds[0].shape[1]

    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 4 * n_examples))
    for i in range(n_examples):
        probs = proba_preds[i]
        labels = val_labels[i]
        probs[0, 0] = 1 - probs[0, 0]
        labels[0, 0] = 1 - labels[0, 0]
        t = np.arange(probs.shape[-1])
        for ch, label_col in zip(
            range(n_pred_channels), ("Moving state", "CBM state"), strict=True
        ):
            ax = axes[i, ch]
            ax.plot(t, probs[0, ch], color="blue", lw=2)
            ax.plot(t, labels[0, ch], color="red", lw=2)
            ax.set_xlabel("Time (ms)", fontdict={"family": "Arial", "size": 14})
            ax.set_ylabel(label_col, fontdict={"family": "Arial", "size": 14})
            ax.set_ylim((-0.05, 1.05))

            ax.grid(True, alpha=0.3)

    plt.savefig(file_path, bbox_inches="tight")
    plt.close(fig)


def get_annotations_to_review(all_annotations, target_user):
    """Get a list of annotations from a user to be reviewed"""
    annotations_to_review = []
    total_annotations_for_target = 0
    for exp, conds in all_annotations.items():
        for cond, vids in conds.items():
            for vid_name, data in vids.items():
                if data.get("username") == target_user:
                    total_annotations_for_target += 1
                    if not data.get("reviewed"):
                        annotations_to_review.append((exp, cond, vid_name, data))

    return annotations_to_review, total_annotations_for_target


def get_annotations_to_review_start_only(all_annotations, target_user):
    """Get unreviewed annotations for a user where the first timestamp does not start at 0.

    Notes:
    - Excludes items with no timestamps (None or empty list), since there's nothing to check at start.
    - Includes only those with timestamps present and the first segment start strictly greater than 0.
    """
    annotations_to_review = []
    total_candidates = 0
    for exp, conds in all_annotations.items():
        for cond, vids in conds.items():
            for vid_name, data in vids.items():
                if data.get("username") != target_user:
                    continue
                timestamps = data.get("timestamps")
                if not timestamps:
                    # No timestamps or empty list → skip from start-only review scope
                    continue
                # timestamps is a list of dicts like {state, start, end}
                first = timestamps[0]
                if isinstance(first, dict) and first.get("start", 0) != 0:
                    total_candidates += 1
                    if not data.get("reviewed"):
                        annotations_to_review.append((exp, cond, vid_name, data))

    return annotations_to_review, total_candidates
