from __future__ import annotations

from typing import Any

import pandas as pd
import vallenae as vae

from mylib import configure_figures_dir, configure_plots
from src.preprocessing.segmentation import (
    compute_smoothed_envelope,
    plot_subhit_segmentation,
    segment_merged_hits,
)
from utils.paths import FIGURES_DIR, RAW_EA_DIR, RESULTS_DIR

PARAM_COLUMNS = [
    "fs",
    "threshold_dbae",
    "pre_trigger_us",
    "smoothing_us",
    "min_separation_us",
    "min_prominence_ratio",
    "valley_decay_ratio",
    "min_secondary_ratio",
    "onset_rise_fraction",
    "keep_full_tail",
]


def load_combo_params(test_id: str, combo_id: str) -> dict[str, Any]:
    """Read a specific `combo_id` parameter row from `grid_search_summary.csv`."""
    summary_path = (
        RESULTS_DIR / "segmentation_grid_search" / test_id / "grid_search_summary.csv"
    )
    df = pd.read_csv(summary_path)
    match = df[df["combo_id"] == combo_id]
    if match.empty:
        raise ValueError(f"Combo '{combo_id}' not found in {summary_path}.")

    row = match.iloc[0]
    return {
        col: (row[col].item() if hasattr(row[col], "item") else row[col])
        for col in PARAM_COLUMNS
        if col in row and pd.notna(row[col])
    }


def main() -> None:
    # 1. Choose specimen, single TRAI, and output styling for the thesis
    test_id = "FV_mod1_1"
    trai = 7006
    combo_id: str | None = "combo_012"  # Set to None to use `manual_params` below

    manual_params: dict[str, Any] = {
        "fs": 5e6,
        "threshold_dbae": 45.0,
        "pre_trigger_us": 40.0,
        "smoothing_us": 80.0,
        "min_separation_us": 250.0,
        "min_prominence_ratio": 0.20,
        "valley_decay_ratio": 0.30,
        "min_secondary_ratio": 0.15,
        "onset_rise_fraction": 0.10,
        "keep_full_tail": False,
    }

    params = (
        load_combo_params(test_id, combo_id)
        if combo_id is not None
        else manual_params
    )

    # 2. Configure publication figure style & output folder
    out_dir = FIGURES_DIR / "thesis" / "ch4_segmentation"
    configure_plots(font="Times New Roman", font_size=8, dpi=150, save_dpi=300)
    configure_figures_dir(out_dir)

    # 3. Load the single transient and segment it
    tradb_path = RAW_EA_DIR / test_id / f"{test_id}.tradb"
    tradb = vae.io.TraDatabase(tradb_path)
    amp_v, time = tradb.read_wave(trai, time_axis=True)
    waveform = amp_v * 1e3

    fs = float(params.get("fs", 5e6))
    smooth_env = compute_smoothed_envelope(
        waveform, fs=fs, smoothing_us=float(params.get("smoothing_us", 80.0))
    )
    segments = segment_merged_hits(waveform=waveform, **params)

    # 4. Plot & save without main title (caption goes in LaTeX/Word)
    fig_path = out_dir / f"{test_id}_trai_{trai}_segmentation.pdf"
    plot_subhit_segmentation(
        waveform=waveform,
        time=time,
        segments=segments,
        params=params,
        figure_path=fig_path,
        smooth_env=smooth_env,
        trai=trai,
        test_id=test_id,
        show_title=False,
        show_subplot_titles=True,
        legend_loc="lower right",
        figsize=(140, 85),
        save=True,
        show=True,
        save_txt=False,
    )


if __name__ == "__main__":
    main()