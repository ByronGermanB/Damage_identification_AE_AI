from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mylib import img_size_inches, save_fig  # type: ignore

from utils.fft_analysis import plot_feat_vs_feat

META_OR_NON_FEATURE_COLS = {
    "trai",
    "sub_trai",
    "hit_id",
    "test_id",
    "label",
    "time",
    "time_norm",
    "Count",
}


def get_candidate_features(
    df: pd.DataFrame,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Return valid numeric feature columns excluding metadata/ID fields."""
    blocked = set(META_OR_NON_FEATURE_COLS)
    if exclude:
        blocked.update(exclude)

    cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in blocked
    ]

    # Keep only columns with non-zero variance
    return [c for c in cols if df[c].nunique(dropna=True) > 1]


def analyze_collinearity(
    df: pd.DataFrame,
    candidate_features: list[str],
    threshold: float = 0.85,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlation matrices and return a table of
    feature pairs where |Pearson| > threshold or |Spearman| > threshold.
    """
    cols = candidate_features
    if len(cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson", "spearman"])

    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    p_corr = sub.corr(method="pearson").abs()
    s_corr = sub.corr(method="spearman").abs()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.corr(method="pearson").to_csv(out_dir / "corr_pearson.csv")
        sub.corr(method="spearman").to_csv(out_dir / "corr_spearman.csv")

    pairs: list[dict[str, object]] = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            p_val = p_corr.loc[col_a, col_b]
            s_val = s_corr.loc[col_a, col_b]
            p_abs = abs(float(p_val)) if pd.notna(p_val) else 0.0
            s_abs = abs(float(s_val)) if pd.notna(s_val) else 0.0
            if max(p_abs, s_abs) > threshold:
                pairs.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "pearson": float(p_val) if pd.notna(p_val) else np.nan,
                        "spearman": float(s_val) if pd.notna(s_val) else np.nan,
                    }
                )

    pairs_df = pd.DataFrame(pairs)
    if out_dir is not None:
        pairs_df.to_csv(out_dir / "collinear_pairs.csv", index=False)

    return pairs_df


def screen_univariate_modality(
    df: pd.DataFrame,
    features: list[str],
    bins: int = 40,
    log_positive_skewed: tuple[str, ...] = ("energy",),
    fig_path: Path | None = None,
) -> pd.DataFrame:
    """Plot 1D histograms and compute basic distribution summary stats."""
    rows: list[dict[str, float | int | str]] = []
    plot_series: dict[str, np.ndarray] = {}

    for col in features:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue

        if col in log_positive_skewed and np.all(vals > 0):
            vals = np.log10(vals)

        counts, _ = np.histogram(vals, bins=bins)
        peaks = int(np.sum((counts[1:-1] > counts[:-2]) & (counts[1:-1] >= counts[2:])))
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        cv = float(std_val / abs(mean_val)) if abs(mean_val) > 1e-12 else 0.0

        plot_series[col] = vals
        rows.append(
            {
                "feature": col,
                "hist_peaks": peaks,
                "std": std_val,
                "cv": cv,
                "skew": float(pd.Series(vals).skew()),
            }
        )

    summary_df = pd.DataFrame(rows)

    if fig_path is not None and plot_series:
        n_cols = 3
        n_rows = int(np.ceil(len(plot_series) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=img_size_inches((180, 45 * n_rows)),
            constrained_layout=True,
        )
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, (col, vals) in zip(axes_flat, plot_series.items()):
            ax.hist(vals, bins=bins, density=True, alpha=0.8)
            title_lbl = f"log10({col})" if col in log_positive_skewed else col
            ax.set_title(title_lbl)
            ax.set_xlabel(title_lbl)
            ax.set_ylabel("Density [-]")
        for ax in axes_flat[len(plot_series) :]:
            ax.set_visible(False)

        fig_path.parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, fig_path)
        plt.close(fig)

    return summary_df


def save_pairwise_feature_plots(
    df: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    label_col: str = "test_id",
    fig_format: str = "png",
    figsize: tuple = (90, 60),
) -> None:
    """Save pairwise `feat_vs_feat` scatter plots for all feature combinations."""
    valid_cols = [
        c for c in features if c in df.columns and c not in META_OR_NON_FEATURE_COLS
    ]
    if len(valid_cols) < 2:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    y = df[label_col] if label_col in df.columns else pd.Series(0, index=df.index)

    for feat_x, feat_y in combinations(valid_cols, 2):
        fig_path = out_dir / f"{feat_x}_vs_{feat_y}.{fig_format}"
        plot_feat_vs_feat(
            X=df,
            y=y,
            feat_x=feat_x,
            feat_y=feat_y,
            fig_path=fig_path,
            figsize=figsize,
            save=True,
        )
