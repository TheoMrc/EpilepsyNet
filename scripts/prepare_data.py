"""Prepare the directory layout used by the EpilepsyNet annotation and training code."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from epilepsy_net.utils import DATA_DIR


def convert_mp4_to_webm(input_file: str | Path, output_file: str | Path) -> None:
    """Convert an MP4 video to VP9 WebM using an installed ffmpeg executable."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-c:v",
            "libvpx-vp9",
            "-cpu-used",
            "4",
            "-threads",
            "4",
            str(output_path),
        ],
        check=True,
    )


def convert_xlsx_sheets_to_csv(input_file: str | Path, output_dir: str | Path) -> None:
    """Write every workbook sheet as a DanioTracker time-series CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    excel = pd.ExcelFile(input_file)
    for sheet_name in excel.sheet_names:
        dataframe = excel.parse(sheet_name)
        safe_name = sheet_name.replace(" ", "_").replace("/", "_").lower()
        dataframe.to_csv(output_path / f"{safe_name}_time_series.csv", index=False)


def copy_files(
    source_paths: list[str | Path], destination: str | Path = DATA_DIR
) -> None:
    """Convert source experiment folders into the documented public layout."""
    destination = Path(destination)
    for source in tqdm(source_paths, desc="Experiments"):
        source_path = Path(source)
        experiment_name = source_path.name
        conditions = [
            folder
            for folder in source_path.iterdir()
            if folder.is_dir()
            and folder.name not in {"training_data", "Grouped results"}
        ]
        for condition in tqdm(conditions, desc="Conditions", leave=False):
            condition_output = destination / experiment_name / condition.name
            video_output = condition_output / "fish_videos"
            series_output = condition_output / "time_series"
            for source_file in condition.iterdir():
                if source_file.suffix.lower() == ".xlsx":
                    series_output.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, series_output / source_file.name)
                    convert_xlsx_sheets_to_csv(source_file, series_output)
                elif (
                    source_file.suffix.lower() == ".mp4"
                    and source_file.stat().st_size >= 5000
                ):
                    target = video_output / source_file.with_suffix(".webm").name
                    if not target.exists():
                        convert_mp4_to_webm(source_file, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", type=Path, help="JSON list of experiment folders")
    source.add_argument("--experiments", nargs="+", help="Experiment folders")
    parser.add_argument("--output", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    if args.json:
        source_paths = json.loads(args.json.read_text(encoding="utf-8"))
    else:
        source_paths = args.experiments
    copy_files(source_paths, args.output)


if __name__ == "__main__":
    main()
