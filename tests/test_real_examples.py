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
        "pcx_200uM_3h_fish_4_cbm.webm",
    ),
    (
        "2025_01_14_CBZ_ToCP_PTZ_PCX_n=2",
        "PCX_400_μM_CBZ_4h_1",
        "4",
        "pcx_400uM_cbz_4h_fish_4_cbm.webm",
    ),
    (
        "2025_02_12_PCX_Gamme_n=2",
        "PCX_200_μM_4h_1",
        "10",
        "pcx_200uM_4h_fish_10_cbm.webm",
    ),
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
    for experiment, condition, video_number, gallery_name in EXAMPLES:
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
        assert (
            ROOT / "annotator" / "static" / "example_videos" / gallery_name
        ).stat().st_size > 0

    client = _load_example_app(monkeypatch).test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert all(gallery_name.encode() in home.data for _, _, _, gallery_name in EXAMPLES)
    assert client.get("/experiment_choice").status_code == 200
    experiment, condition, video_number, _ = EXAMPLES[0]
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
    experiment, condition, video_number, _ = EXAMPLES[0]
    response = client.get(f"/predict/{experiment}/{condition}/{video_number}")
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)
