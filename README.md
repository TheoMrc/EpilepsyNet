# EpilepsyNet

## Authors

- [Théo Mercé, PhD](https://github.com/TheoMrc)
- [Emilien Reaud, PhD student](https://github.com/EmilienRD)
- [Etienne Windels, Data scientist](https://github.com/ewindels)

This repository contains the EpilepsyNet temporal convolutional architecture,
training and evaluation code, preprocessing utilities, the Epileptic Fish
Annotation System (EFAS), and the exact weights used by DanioTracker for the
manuscript revision.

EpilepsyNet receives 14 DanioTracker behavioral features at millisecond
resolution and assigns one of three mutually exclusive states to each sample:
not_moving, cbm (seizure-related convulsive body movement), or swimming.

## Released model

- PyTorch state dictionary: models/epilepsynet.pt
- ONNX graph: models/epilepsynet.onnx
- Architecture and feature order: configs/model.json
- Selected training configuration: configs/training.json
- Validation metrics from the selected run: models/validation_metrics.json
- PyTorch checkpoint SHA-256:
  e23dabf8e93056be259d85171bb6b36c2358f901f9b749c0e9b28bf468b9fc83
- ONNX graph SHA-256:
  4ce9b33f62b014921aa8f3691ffa6184e934fe683883b58ebc24c635eaec1e9d

The PyTorch file is byte for byte identical to
BreakdanceFish/models/best_model/best_model.pt and
DanioTracker/models/epilepsynet.pt.

## Repository contents

```text
EpilepsyNet/
├── epilepsy_net/             Core model, data loading, inference, evaluation, and training code
├── annotator/                EFAS Flask annotation and review application
├── configs/                  Released model and training configurations
├── models/                   Released PyTorch/ONNX weights and validation metrics
├── examples/                 Small real-data videos, time series, and EFAS annotations
├── scripts/                  Data preparation utilities
├── tests/                    Smoke tests and real-example integration tests
├── pyproject.toml            Project metadata, dependencies, and Ruff/Ty configuration
├── setup.py                  Setuptools compatibility entry point and console scripts
└── tox.ini                   `tox -e format` automation
```

The `epilepsy_net/` package is the reusable library. `inference.py` loads the
released checkpoint and predicts a state for each time-series sample;
`training.py` contains the reproducible training loop; `evaluation.py` computes
fish-level validation metrics; `load_data.py` implements the training dataset;
and `utils.py` contains feature definitions, path handling, plotting, and
interval utilities.

`annotator/` is the Epileptic Fish Annotation System (EFAS). It provides the
browser interface used to select experiments, annotate convulsive body
movement (CBM) intervals, review annotations, and request model-assisted
predictions. The application writes annotations to the directory configured by
`EPILEPSYNET_ANNOTATIONS_DIR` and reads videos/time series from
`EPILEPSYNET_DATA_DIR`.

The released `models/` files are deliberately kept together with the exact
JSON configuration and checksums needed to reproduce the published inference
path. The `examples/` tree is only a compact demonstration dataset; it is not
the full training database.

## Installation and dependencies

Python 3.10 or newer is required. The commands below install the complete
runtime, annotation, training, test, and formatting toolchain.

### Windows PowerShell

    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[annotation,training,test,format]"

### Linux or macOS

    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[annotation,training,test,format]"

The dependency groups are:

- Core inference and training: NumPy, pandas, and PyTorch.
- `annotation`: Flask, Matplotlib, OpenPyXL, and tqdm for EFAS and data preparation.
- `training`: Matplotlib, OpenPyXL, tqdm, and Weights & Biases.
- `test`: pytest and ONNX Runtime for the released-graph comparison.
- `format`: tox, Ruff, and Ty for formatting, lint fixes, and type checking.

The setup metadata installs the complete runtime dependency set by default;
the extras make the intended workflows explicit and reproducible. FFmpeg is
also required on `PATH` when converting MP4 files for EFAS; it is an external
program and is not installed by pip. Verify it with `ffmpeg -version`.

To install only the runtime package, use:

    python -m pip install -e .

For development checks, run:

    python -m tox -e format
    python -m pytest -q

`tox -e format` runs `ruff format`, applies safe Ruff lint fixes, and then
runs `ty check`. The configuration is stored in `pyproject.toml` so the same
commands can be used locally and in continuous integration.

## Inference

Input is a DanioTracker per-fish CSV containing the 14 columns listed, in exact
order, in configs/model.json. The implementation applies the absolute-value
transformation used during training.

    python -m epilepsy_net.inference fish_time_series.csv --output states.csv

The output reports the predicted state and the probability of each state for
every input row. The default model and feature configuration are resolved from
the repository's `models/` and `configs/` directories, so the command can be
run from the repository root without copying model files into the working
directory.

## Training

The full database is intentionally excluded; only the compact examples
described below are included. Arrange the DanioTracker time series and EFAS JSON
annotations using the following structure:

    data/
      experiment_name/
        condition_name/
          fish_videos/
            fish_1.webm
          time_series/
            fish_1_time_series.csv
    annotations/
      experiment_name_annotations.json

Each CSV contains one row per millisecond and all 14 feature columns listed in
`configs/model.json`. Annotation JSON files are organized by condition and
video; each video record contains inclusive `timestamps` intervals whose
`state` is `cbm` or `stationnary` (the historical spelling retained by EFAS).
All other samples are treated as swimming. The files under `examples/` provide
three complete real instances of this structure. To train, run:

    python -m epilepsy_net.training --config configs/training.json --data-dir path/to/data --annotations-dir path/to/annotations --output-dir outputs

The deterministic SHA-256 path split assigns approximately 80 percent of fish
to training and 20 percent to validation. Only the first recording from each
fish group is included, while individual-fish time series from one source video
can be assigned to different datasets. The selected database produced 764
training fish and 194 validation fish. Seed 42, gradient accumulation, class
loss, latency metrics, early stopping, and all checkpoint selection criteria
remain in epilepsy_net/training.py. Fish-level evaluation is available with:

    python -m epilepsy_net.evaluation --on-val --data-dir path/to/data --annotations-dir path/to/annotations

Weights & Biases logging is
disabled by default and can be enabled by setting WANDB_MODE to online in the
training JSON.

## Annotation tool

EFAS is included under annotator. Set the data and annotation
locations, then start it:

    set EPILEPSYNET_DATA_DIR=C:\path\to\data
    set EPILEPSYNET_ANNOTATIONS_DIR=C:\path\to\annotations
    python -m annotator.app

The tool supports manual intervals, model-assisted intervals, comments, user
tracking, and review. Replace annotator/users.json with the desired
annotator names. Use a strong EPILEPSYNET_SECRET_KEY for a shared deployment.

To prepare videos and XLSX exports in the expected layout:

    python -m scripts.prepare_data --experiments path/to/experiment --output path/to/data

### Included real-data examples

Three short real videos, their matching DanioTracker time series, and existing
EFAS annotations are included under `examples/`. The same three videos are also
shown in the application's CBM example gallery. From PowerShell, launch EFAS on
the included example dataset with:

    $env:EPILEPSYNET_DATA_DIR = (Resolve-Path "examples/data")
    $env:EPILEPSYNET_ANNOTATIONS_DIR = (Resolve-Path "examples/annotations")
    epilepsynet-annotate

Then open http://127.0.0.1:5000. The examples are intended only to exercise and
demonstrate EFAS; they are not an additional training or validation dataset.

## Verification

    python -m pytest -q

The formatting and static-analysis check is:

    python -m tox -e format

The tests load the released checkpoint, run synthetic temporal inference,
validate CSV output, and compare the ONNX graph with PyTorch when ONNX Runtime
is installed.

## Data availability

Apart from the compact EFAS examples documented above, no videos, annotations,
time-series tables, database exports, or study metadata are included.

## License

The source repositories did not contain a license file. Add the authors'
selected license before public release.



