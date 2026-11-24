from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mylib import (  # type: ignore
    parse_config_args as _mylib_parse_config_args,
)
from mylib import resolve_path  # type: ignore

from utils.paths import (
    DATASETS_DIR,
    EXAMPLE_TRADB,
    FIGURES_DIR,
    PARAMETERS_DIR,
    RESULTS_DIR,
)

DEFAULT_CONFIG_SUBDIR = PARAMETERS_DIR / "example_data"
DEFAULT_CONFIG_NAME = "segmentation.json"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_SUBDIR / DEFAULT_CONFIG_NAME


PARAM_COLUMNS = [
    "fs",
    "threshold_dbae",
    "pre_trigger_us",
    "smoothing_us",
    "min_separation_us",
    "min_prominence_ratio",
    "valley_decay_ratio",
    "min_secondary_ratio",
]


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "get_segmentation_paths",
    "parse_config_args",
]


def get_segmentation_paths(
    config: dict[str, Any],
    create: bool = True,
) -> dict[str, Path]:
    """Build (and optionally create) standard dataset, result, and figure paths."""
    dataset_name = str(config["dataset_name"])
    test_id = str(config["test_id"])

    raw_tradb = config.get("tradb_path")
    tradb_path = resolve_path(raw_tradb) if raw_tradb else EXAMPLE_TRADB

    dataset_dir = DATASETS_DIR / dataset_name
    results_dir = RESULTS_DIR / dataset_name / "segmentation" / test_id
    figures_dir = FIGURES_DIR / dataset_name / "segmentation" / test_id

    if create:
        for directory in (dataset_dir, results_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

    return {
        "tradb_path": tradb_path,
        "dataset_dir": dataset_dir,
        "results_dir": results_dir,
        "figures_dir": figures_dir,
    }


def parse_config_args(
    description: str,
) -> argparse.Namespace:
    """Parse `-c` / `--config` (defaulting to `DEFAULT_CONFIG_PATH`) ."""

    return _mylib_parse_config_args(
        description=description,
        default_config=DEFAULT_CONFIG_PATH,
    )
