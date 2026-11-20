from __future__ import annotations

from pathlib import Path
from typing import Any

from mylib import configure_plots, load_config, save_config  # type: ignore

from src.preprocessing.pipeline_utils import (
    DEFAULT_CONFIG_PATH,
    get_segmentation_paths,
    parse_config_args,
)
from utils.segmentation import run_segmentation_grid_search

configure_plots(font="IBM Plex Sans")


def _extract_trais_and_expectations(
    config: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Read `(trai_list, expected_segments)` from `selections` or flat arrays."""
    selections = config.get("selections")
    if selections:
        trai_list = [int(item["trai"]) for item in selections]
        expected_segments = [int(item["expected_segments"]) for item in selections]
        return trai_list, expected_segments

    trai_list = [int(t) for t in config.get("trai_list", [])]
    expected_segments = [int(e) for e in config.get("expected_segments", [])]
    return trai_list, expected_segments


def main(
    config_arg: str | Path | None = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the sub-hit segmentation grid search from a stage-01 selection JSON."""
    config, cfg_path = load_config(config_arg, return_path=True)
    paths = get_segmentation_paths(config, create=True)

    dataset_name = str(config["dataset_name"])
    test_id = str(config["test_id"])
    fs = float(config.get("fs", 5e6))

    trai_list, expected_segments = _extract_trais_and_expectations(config)
    if not trai_list:
        raise ValueError(
            f"No TRAIs found in '{cfg_path}'. Run 01_select_waveforms.py first."
        )

    tradb_path = paths["tradb_path"]
    if not tradb_path.exists():
        raise FileNotFoundError(f"Transient database not found: {tradb_path}")

    param_grid = config.get("param_grid")
    if not param_grid:
        raise ValueError(f"Missing 'param_grid' dictionary in '{cfg_path}'.")

    gs_cfg = config.get("grid_search", {})
    save_valid_only = bool(gs_cfg.get("save_valid_only", True))
    img_format = str(gs_cfg.get("img_format", "png"))
    figsize_mm = tuple(gs_cfg.get("figsize_mm", [140, 90]))
    save_figures = bool(gs_cfg.get("save_figures", True))
    show_figures = bool(gs_cfg.get("show_figures", False))
    save_txt = bool(gs_cfg.get("save_txt", True))

    print(f"[INFO] Loaded config from   : {cfg_path}")
    print(f"[INFO] Dataset / Test ID    : {dataset_name} / {test_id}")
    print(f"[INFO] TRAIs                : {trai_list}")
    print(f"[INFO] Expected segments    : {expected_segments}")
    print(f"[INFO] Figures directory    : {paths['figures_dir']}")
    print(f"[INFO] Results directory    : {paths['results_dir']}")

    summary_df = run_segmentation_grid_search(
        tradb_path=tradb_path,
        trai_list=trai_list,
        param_grid=param_grid,
        fig_dir=paths["figures_dir"],
        results_dir=paths["results_dir"],
        expected_segments=expected_segments,
        save_valid_only=save_valid_only,
        test_id=test_id,
        fs=fs,
        img_format=img_format,
        figsize=figsize_mm,
        save=save_figures,
        show=show_figures,
        save_txt=save_txt,
    )

    # Save an exact snapshot of the configuration used for reproducibility
    save_config(config, paths["results_dir"] / "grid_search_config_used.json")
    print(f"[INFO] Completed {len(summary_df)} total evaluations.")


if __name__ == "__main__":
    args = parse_config_args(
        "Stage 2: Run sub-hit segmentation grid search from selected TRAIs JSON.",
    )
    main(
        config_arg=args.config,
    )
