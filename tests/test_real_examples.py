from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("flask")


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    (
        "2025_01_21_CBZ_ToCP_PTZ_PCX_n=3",
        "PCX_200_μM_3h_1",
        "4",
    ),
    (
        "2025_01_14_CBZ_ToCP_PTZ_PCX_n=2",
        "PCX_400_μM_CBZ_4h_1",
        "4",
    ),
    (
        "2025_02_12_PCX_Gamme_n=2",
        "PCX_200_μM_4h_1",
        "10",
    ),
)
GALLERY_EXAMPLES = (
    "CBM_1.webm",
    "CBM_2.webm",
    "CBM_3.webm",
    "CBM_4.webm",
    "CBM_5.webm",
    "Spaghetti.webm",
)
UNANNOTATED_EXPERIMENT = "EFAS_demo_unannotated"
UNANNOTATED_VIDEOS = (
    ("Demo_video_1", "1"),
    ("Demo_video_2", "2"),
    ("Demo_video_3", "3"),
)


def _load_example_app(monkeypatch):
    monkeypatch.setenv("EPILEPSYNET_DATA_DIR", str(ROOT / "examples" / "data"))
    monkeypatch.setenv(
        "EPILEPSYNET_ANNOTATIONS_DIR", str(ROOT / "examples" / "annotations")
    )
    import epilepsy_net.utils as utils

    importlib.reload(utils)
    import annotator.app as app_module

    return importlib.reload(app_module).app


def test_real_examples_are_complete_and_browsable(monkeypatch):
    for experiment, condition, video_number in EXAMPLES:
        annotations = json.loads(
            (
                ROOT / "examples" / "annotations" / f"{experiment}_annotations.json"
            ).read_text(encoding="utf-8")
        )
        video_name = f"fish_{video_number}.webm"
        condition_dir = ROOT / "examples" / "data" / experiment / condition
        assert (condition_dir / "fish_videos" / video_name).stat().st_size > 0
        series_path = (
            condition_dir / "time_series" / f"fish_{video_number}_time_series.csv"
        )
        assert len(pd.read_csv(series_path)) > 0
        assert annotations[condition][video_name]["timestamps"]
        assert any(
            interval["state"] == "cbm"
            for interval in annotations[condition][video_name]["timestamps"]
        )
    for gallery_name in GALLERY_EXAMPLES:
        assert (
            ROOT / "annotator" / "static" / "example_videos" / gallery_name
        ).stat().st_size > 0

    client = _load_example_app(monkeypatch).test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert all(gallery_name.encode() in home.data for gallery_name in GALLERY_EXAMPLES)
    assert client.get("/experiment_choice").status_code == 200
    experiment, condition, video_number = EXAMPLES[0]
    with client.session_transaction() as session:
        session["chosen_experiment"] = experiment
        session["username"] = "Reviewer"
    assert client.get("/videos").status_code == 200

    assert (
        client.get(f"/annotate/{experiment}/{condition}/{video_number}").status_code
        == 200
    )
    video_response = client.get(f"/serve_video/{experiment}/{condition}/{video_number}")
    assert video_response.status_code == 200
    assert video_response.mimetype == "video/webm"


def test_real_example_supports_model_assisted_annotation(monkeypatch):
    client = _load_example_app(monkeypatch).test_client()
    experiment, condition, video_number = EXAMPLES[0]
    response = client.get(f"/predict/{experiment}/{condition}/{video_number}")
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)


def test_unannotated_demo_videos_are_available(monkeypatch):
    app = _load_example_app(monkeypatch)
    client = app.test_client()

    with client.session_transaction() as session:
        session["chosen_experiment"] = UNANNOTATED_EXPERIMENT
        session["username"] = "Curious Guppy"

    assert client.get("/videos").status_code == 200
    for condition, video_number in UNANNOTATED_VIDEOS:
        annotation_page = client.get(
            f"/annotate/{UNANNOTATED_EXPERIMENT}/{condition}/{video_number}"
        )
        assert annotation_page.status_code == 200
        video_response = client.get(
            f"/serve_video/{UNANNOTATED_EXPERIMENT}/{condition}/{video_number}"
        )
        assert video_response.status_code == 200
        assert video_response.mimetype == "video/webm"
