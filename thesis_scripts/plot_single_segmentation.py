from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import vallenae as vae
from mylib import configure_plots, load_config, parse_config_args  # type: ignore

from src.preprocessing.pipeline_utils import get_segmentation_paths
from thesis_scripts.thesis_utils import DEFAULT_CONFIG_DIR
from utils.segmentation import (
    PARAM_COLUMNS,
    compute_smoothed_envelope,
    plot_subhit_segmentation,
    segment_merged_hits,
)

configure_plots(font="IBM Plex Sans")

DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "single_segmentation.json"


def load_combo_params(csv_path: str, test_id: str, combo_id: str) -> dict[str, Any]:
    """Read a specific `combo_id` parameter row from `grid_search_summary.csv`."""
    df = pd.read_csv(csv_path)
    match = df[df["combo_id"] == combo_id]
    if match.empty:
        raise ValueError(f"Combo '{combo_id}' not found in {csv_path}.")

    row = match.iloc[0]
    return {
        col: (row[col].item() if hasattr(row[col], "item") else row[col])
        for col in PARAM_COLUMNS
        if col in row and pd.notna(row[col])
    }


def main(
    config_arg: str | Path | None = DEFAULT_CONFIG_PATH,
) -> None:
    config = load_config(config_arg, return_path=False)
    paths = get_segmentation_paths(config, create=False)

    # 1. Choose specimen, single TRAI, and output styling for the thesis
    test_id = config["test_id"]
    trai = config["trai"]
    combo_id: str | None = config.get("combo_id", None)

    manual_params: dict[str, Any] = {
        "sampling_rate": 5e6,
        "threshold_dbae": 30.0,
        "pre_trigger_us": 40.0,
        "smoothing_us": 100.0,
        "min_separation_us": 300.0,
        "min_prominence_ratio": 0.25,
        "valley_decay_ratio": 0.35,
        "min_secondary_ratio": 0.1,
    }

    # 2. Establish the parameters
    csv_path = paths["results_dir"] / "grid_search_summary.csv"
    params = (
        load_combo_params(csv_path=csv_path, test_id=test_id, combo_id=combo_id)
        if combo_id is not None
        else manual_params
    )

    # 3. Load the single transient and segment it
    tradb_path = paths["tradb_path"]
    tradb = vae.io.TraDatabase(tradb_path)
    amp_v, time = tradb.read_wave(trai, time_axis=True)
    waveform = amp_v * 1e3

    sampling_rate = float(params.get("sampling_rate", 5.0))
    smooth_env = compute_smoothed_envelope(
        waveform, sampling_rate=sampling_rate, smoothing_us=float(params.get("smoothing_us", 100.0))
    )
    segments = segment_merged_hits(waveform=waveform, **params)

    # 4. Plot & save without main title (caption goes in LaTeX/Word)
    plot_cfg = config["plot"]
    img_format = plot_cfg.get("img_format", "pdf")
    figsize = tuple(plot_cfg.get("figsize_mm", [120, 90]))
    fig_path = paths["figures_dir"] / f"trai_{trai}_segmentation.{img_format}"
    plot_subhit_segmentation(
        waveform=waveform,
        time=time,
        segments=segments,
        params=params,
        figure_path=fig_path,
        smooth_env=smooth_env,
        trai=trai,
        test_id=test_id,
        show_title=plot_cfg.get("show_tile", False),
        show_subplot_titles=plot_cfg.get("show_subplot_tiles", True),
        legend_loc=plot_cfg.get("legend_loc", "lower right"),
        figsize=figsize,
        save=plot_cfg.get("save_figures", False),
        show=plot_cfg.get("show_figures", True),
        save_txt=False,
    )


if __name__ == "__main__":
    args = parse_config_args(
        description="Single Segmention: Run sub-hit segmentation from selected TRAI.",
        default_config=DEFAULT_CONFIG_PATH,
    )
    main(
        config_arg=args.config,
    )
