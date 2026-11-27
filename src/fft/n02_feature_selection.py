from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from mylib import load_config  # type: ignore

from src.fft.pipeline_utils import (
    DEFAULT_CONFIG_PATH,
    get_fft_paths,
    parse_config_args,
)
from utils.feature_selection import (
    analyze_collinearity,
    get_candidate_features,
    save_pairwise_feature_plots,
    screen_univariate_modality,
)


def main(config_arg: str | Path | None = DEFAULT_CONFIG_PATH) -> None:
    config: dict[str, Any] = load_config(config_arg)
    sel_cfg = config.get("feature_selection", {})

    paths = get_fft_paths(config, create=True)
    feat_path = paths["features_path"]
    if not feat_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {feat_path}")

    df = pd.read_csv(feat_path)
    out_dir = paths["feature_selection_results_dir"]
    fig_dir = paths["feature_selection_figures_dir"]

    candidates = get_candidate_features(
        df,
        exclude=sel_cfg.get("exclude_features"),
    )

    corr_threshold = float(sel_cfg.get("corr_threshold", 0.85))

    # 1. Pearson & Spearman correlation matrices + high-correlation pair report
    collinear_pairs_df = analyze_collinearity(
        df=df,
        candidate_features=candidates,
        threshold=corr_threshold,
        out_dir=out_dir,
    )

    # 2. 1D histograms & univariate summary
    modality_df = screen_univariate_modality(
        df=df,
        features=candidates,
        bins=int(sel_cfg.get("hist_bins", 40)),
        fig_path=fig_dir / "hist_features.png",
    )
    modality_df.to_csv(out_dir / "univariate_screening.csv", index=False)

    # 3. Pairwise feat_vs_feat scatter plots for visual inspection
    plot_cfg = sel_cfg["plots"]
    if plot_cfg.get("save_pairwise_plots", True):
        plot_cols = candidates
        figsize = plot_cfg.get("figsize", (90, 60))
        save_pairwise_feature_plots(
            df=df,
            features=plot_cols,
            out_dir=fig_dir / "feat_vs_feat",
            figsize=figsize,
            )

    summary_payload = {
        "candidates": candidates,
        "corr_threshold": corr_threshold,
        "collinear_pairs": collinear_pairs_df.to_dict(orient="records"),
    }
    
    (out_dir / "selected_features.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    print(f"[INFO] Candidate features ({len(candidates)}): {candidates}")
    print(f"[INFO] Pairs with |r| > {corr_threshold}: {len(collinear_pairs_df)}")
    print(f"[INFO] Diagnostics saved to: {out_dir}")


if __name__ == "__main__":
    args = parse_config_args("Feature selection for unsupervised AE clustering.")
    main(config_arg=args.config)
