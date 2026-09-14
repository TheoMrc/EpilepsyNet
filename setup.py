"""Setuptools compatibility entry point for EpilepsyNet."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

CORE_DEPENDENCIES = [
    "numpy>=1.24",
    "pandas>=2.0",
    "torch>=2.0",
]
ANNOTATION_DEPENDENCIES = [
    "Flask>=3.0",
    "matplotlib>=3.7",
    "openpyxl>=3.1",
    "tqdm>=4.65",
]
TRAINING_DEPENDENCIES = [
    "matplotlib>=3.7",
    "openpyxl>=3.1",
    "tqdm>=4.65",
    "wandb>=0.16",
]

setup(
    name="epilepsynet-zebrafish",
    version="1.0.0",
    description=(
        "EpilepsyNet training, inference, evaluation, and annotation tools "
        "for zebrafish larvae"
    ),
    url="https://github.com/TheoMrc/EpilepsyNet",
    long_description=README,
    long_description_content_type="text/markdown",
    author=(
        "Théo Mercé, PhD; Emilien Reaud, PhD student; Etienne Windels, Data scientist"
    ),
    packages=find_packages(exclude=["tests", ".github"]),
    include_package_data=True,
    package_data={
        "annotator": [
            "templates/*.html",
            "static/*.css",
            "static/*.js",
            "static/*.ico",
            "static/*.png",
            "static/example_videos/*.webm",
        ],
    },
    install_requires=CORE_DEPENDENCIES
    + ANNOTATION_DEPENDENCIES
    + TRAINING_DEPENDENCIES,
    extras_require={
        "annotation": ANNOTATION_DEPENDENCIES,
        "training": TRAINING_DEPENDENCIES,
        "test": ["onnxruntime>=1.17", "pytest>=8"],
        "format": ["ruff>=0.11", "ty>=0.0.1", "tox>=4"],
    },
    entry_points={
        "console_scripts": [
            "epilepsynet-infer=epilepsy_net.inference:main",
            "epilepsynet-train=epilepsy_net.training:main",
            "epilepsynet-annotate=annotator.app:main",
            "epilepsynet-evaluate=epilepsy_net.evaluation:main",
            "epilepsynet-prepare-data=scripts.prepare_data:main",
        ]
    },
)
