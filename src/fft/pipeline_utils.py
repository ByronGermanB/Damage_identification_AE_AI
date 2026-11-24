from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mylib import (  # type: ignore
    parse_config_args as _mylib_parse_config_args,
)

from utils.paths import (
    DATASETS_DIR,
    FIGURES_DIR,
    PARAMETERS_DIR,
    RESULTS_DIR,
)

DEFAULT_CONFIG_SUBDIR = PARAMETERS_DIR / "example_data"
DEFAULT_CONFIG_NAME = "trial_1.json"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_SUBDIR / DEFAULT_CONFIG_NAME


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "get_fft_paths",
    "parse_config_args",
]


def get_fft_paths(
    config: dict[str, Any],
    create: bool = True,
) -> dict[str, Path]:
    """Build standard dataset, trial, model, and figure paths for a config."""
    dataset_name = config["dataset_name"]
    trial_name = config["trial_name"]

    dataset_dir = DATASETS_DIR / dataset_name / trial_name
    trial_results_dir = RESULTS_DIR / dataset_name / "models" / "trials" / trial_name
    trial_figures_dir = FIGURES_DIR / dataset_name / trial_name
    features_path = dataset_dir / "features.csv"
    force_data_path = dataset_dir / "force_data.csv"
    preprocessed_data_path = dataset_dir / "preprocessed_data.joblib"

    if create:
        for directory in (dataset_dir, trial_results_dir, trial_figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

    return {
        "dataset_dir": dataset_dir,
        "trial_dir": trial_results_dir,
        "trial_figures_dir": trial_figures_dir,
        "features_path": features_path,
        "force_data_path": force_data_path,
        "preprocessed_data_path": preprocessed_data_path,
    }


def parse_config_args(
    description: str,
) -> argparse.Namespace:
    """Parse `-c` / `--config` (defaulting to `DEFAULT_CONFIG_PATH`) ."""

    return _mylib_parse_config_args(
        description=description,
        default_config=DEFAULT_CONFIG_PATH,
    )
