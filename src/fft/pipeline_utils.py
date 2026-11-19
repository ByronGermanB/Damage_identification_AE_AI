from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mylib import load_config as _mylib_load_config  # type: ignore
from mylib import save_config  #type: ignore

from utils.paths import (
    DATASETS_DIR,
    FIGURES_DIR,
    PARAMETERS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
)

DEFAULT_CONFIG_SUBDIR = PARAMETERS_DIR / "arcan"
DEFAULT_CONFIG_NAME = "trial_2.json"

__all__ = [
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_CONFIG_SUBDIR",
    "get_experiment_paths",
    "load_config",
    "parse_config_arg",
    "resolve_config_path",
    "resolve_path",
    "save_config",
]


def resolve_path(raw_path: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve an absolute path or a path relative to `base_dir`."""
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def resolve_config_path(config_arg: str | Path | None = None) -> Path:
    """
    Resolve a configuration file path (`.json`, `.yaml`, or `.yml`).

    Accepts:
    - `None`: defaults to `parameters/arcan/trial_2.json`
    - A bare filename (e.g., `trial_2.json`): looked up in `parameters/arcan/`
    - A path relative to `PROJECT_ROOT` (e.g., `parameters/arcan/trial_2.json`)
    - An absolute path
    """
    if config_arg is None:
        return DEFAULT_CONFIG_SUBDIR / DEFAULT_CONFIG_NAME

    candidate = Path(config_arg)
    if candidate.is_absolute():
        return candidate
    if (PROJECT_ROOT / candidate).exists():
        return PROJECT_ROOT / candidate
    return DEFAULT_CONFIG_SUBDIR / candidate


def load_config(config_arg: str | Path | None = None) -> dict[str, Any]:
    """Resolve the configuration path and load it via `mylib.load_config`."""
    config_path = resolve_config_path(config_arg)
    return _mylib_load_config(config_path)


def get_experiment_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Build standard dataset, trial, model, and figure paths for a config."""
    dataset_name = config["dataset_name"]
    trial_name = config["trial_name"]

    dataset_dir = DATASETS_DIR / dataset_name
    trial_dir = RESULTS_DIR / dataset_name / "models" / "trials" / trial_name
    trial_figures_dir = FIGURES_DIR / dataset_name / trial_name
    preprocessed_data_path = (
        dataset_dir / "trial_data" / trial_name / "preprocessed_data.joblib"
    )

    return {
        "dataset_dir": dataset_dir,
        "trial_dir": trial_dir,
        "trial_figures_dir": trial_figures_dir,
        "preprocessed_data_path": preprocessed_data_path,
    }


def parse_config_arg(description: str) -> str | None:
    """Parse an optional `--config` CLI argument for standalone stage execution."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path or filename of the config file (default: parameters/arcan/trial_2.json).",
    )
    args = parser.parse_args()
    return args.config
