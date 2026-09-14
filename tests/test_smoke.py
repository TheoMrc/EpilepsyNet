from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from epilepsy_net.inference import (
    load_model,
    predict_array,
    predict_dataframe,
)

ROOT = Path(__file__).resolve().parents[1]


def test_released_checkpoint_runs():
    model, config = load_model(device="cpu")
    values = np.zeros((128, len(config["features"])), dtype=np.float32)
    classes, probabilities = predict_array(values, model=model, config=config)

    assert classes.shape == (128,)
    assert probabilities.shape == (128, len(config["label_cols"]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert np.isfinite(probabilities).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 36579


def test_dataframe_uses_released_feature_order():
    model, config = load_model(device="cpu")
    frame = pd.DataFrame(
        np.zeros((32, len(config["features"]))), columns=config["features"]
    )
    output = predict_dataframe(frame, model=model, config=config)

    assert len(output) == len(frame)
    assert set(output["state"]).issubset(set(config["label_cols"]))
    assert list(output.columns) == [
        "state",
        *[f"probability_{label}" for label in config["label_cols"]],
    ]


@pytest.mark.slow
def test_onnx_matches_pytorch():
    ort = pytest.importorskip("onnxruntime")
    model, config = load_model(device="cpu")
    rng = np.random.default_rng(4321)
    values = rng.normal(size=(96, len(config["features"]))).astype(np.float32)
    _, expected_probabilities = predict_array(values, model=model, config=config)

    session = ort.InferenceSession(
        str(ROOT / "models" / "epilepsynet.onnx"),
        providers=["CPUExecutionProvider"],
    )
    logits = session.run(
        ["logits"],
        {"features": np.abs(values).T[np.newaxis].astype(np.float32)},
    )[0]
    actual = torch.softmax(torch.from_numpy(logits), dim=1)[0].T.numpy()
    np.testing.assert_allclose(expected_probabilities, actual, rtol=1e-4, atol=1e-4)


def test_model_metadata_matches_checkpoint():
    config = json.loads((ROOT / "configs" / "model.json").read_text())
    assert len(config["features"]) == config["n_features"] == 14
    assert config["label_cols"] == ["not_moving", "cbm", "swimming"]
    assert config["train_samples"] == 764
    assert config["val_samples"] == 194
