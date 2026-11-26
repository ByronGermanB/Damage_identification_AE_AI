from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import vallenae as vae
from mylib import Profiler, img_size_inches, resolve_path, save_fig  # type: ignore
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert

from utils.paths import FIGURES_DIR, RESULTS_DIR

PARAM_COLUMNS = [
    "sampling_rate",
    "threshold_dbae",
    "pre_trigger_us",
    "smoothing_us",
    "min_separation_us",
    "min_prominence_ratio",
    "valley_decay_ratio",
    "min_secondary_ratio",
]


@dataclass(frozen=True)
class SubHitSegment:
    """Boundaries and landmark indices for an isolated sub-hit within a transient."""

    sub_trai: int
    start_idx: int
    end_idx: int
    onset_idx: int
    peak_idx: int


def dbae_to_mV(dbae: float) -> float:
    """Convert acoustic emission amplitude from dBAE (ref. 1 uV) to mV."""
    return float(10.0 ** ((dbae / 20.0) - 3.0))


def compute_smoothed_envelope(
    waveform: npt.NDArray[np.floating],
    sampling_rate: float = 5.0,
    smoothing_us: float = 80.0,
) -> npt.NDArray[np.float64]:
    """
    Compute the zero-phase smoothed Hilbert envelope [mV] of a transient signal.
    Parameters:
    - waveform: 1D array of transient signal amplitudes [mV].
    - sampling_rate: Sampling frequency [MHz].
    - smoothing_us: Smoothing window width [µs] for the uniform filter.

    Returns:
    - smooth_env: 1D array of the smoothed Hilbert envelope [mV].
    """

    dt_us = 1 / sampling_rate
    smooth_samples = max(1, round(smoothing_us / dt_us))
    analytic_env = np.abs(hilbert(waveform))
    return uniform_filter1d(
        analytic_env.astype(np.float64), size=smooth_samples, mode="nearest"
    )


def _find_preceding_valley(
    smooth_env: npt.NDArray[np.float64],
    peak_idx: int,
) -> int:
    """Walk backward from `peak_idx` to locate the nearest local minimum."""
    idx = peak_idx
    while idx > 0 and smooth_env[idx - 1] <= smooth_env[idx]:
        idx -= 1
    return idx


def _refine_arrival_onset(
    waveform: npt.NDArray[np.floating],
    smooth_env: npt.NDArray[np.float64],
    valley_idx: int,
    peak_idx: int,
    rise_fraction: float = 0.15,
) -> int:
    """Pinpoint the arrival onset (t0) between a preceding valley and burst peak."""
    search_slice = np.abs(waveform[valley_idx : peak_idx + 1])
    if search_slice.size == 0:
        return valley_idx

    baseline_amp = float(smooth_env[valley_idx])
    peak_raw_amp = float(np.max(search_slice))
    onset_level = baseline_amp + rise_fraction * (peak_raw_amp - baseline_amp)

    above_onset = np.flatnonzero(search_slice >= onset_level)
    onset_offset = int(above_onset[0]) if above_onset.size > 0 else 0
    return valley_idx + onset_offset


def segment_merged_hits(
    waveform: npt.NDArray[np.floating],
    sampling_rate: float = 5.0,
    threshold_dbae: float = 30.0,
    pre_trigger_us: float = 40.0,
    smoothing_us: float = 100.0,
    min_separation_us: float = 300.0,
    min_prominence_ratio: float = 0.25,
    valley_decay_ratio: float = 0.35,
    min_secondary_ratio: float = 0.10,
    onset_rise_fraction: float = 0.15,
) -> list[SubHitSegment]:
    """
    Split a transient `waveform` [mV] into individual `SubHitSegment` objects when
    multiple AE bursts overlap within the same acquisition window.
    """
    n_samples = waveform.size
    if n_samples == 0:
        return []

    # 1. Compute zero-phase smoothed Hilbert envelope [mV]
    dt_us = 1 / sampling_rate
    threshold_mV = dbae_to_mV(threshold_dbae)
    smooth_env = compute_smoothed_envelope(
        waveform, sampling_rate=sampling_rate, smoothing_us=smoothing_us
    )

    # 2. Detect candidate peaks using both height and prominence to reject coda ripples
    min_dist_samples = max(1, round(min_separation_us / dt_us))
    pre_trigger_samples = max(0, round(pre_trigger_us / dt_us))
    min_prominence = max(threshold_mV * 0.5, min_prominence_ratio * threshold_mV)

    candidate_peaks, _ = find_peaks(
        smooth_env,
        height=threshold_mV,
        distance=min_dist_samples,
        prominence=min_prominence,
    )

    if candidate_peaks.size == 0:
        peak_idx = int(np.argmax(smooth_env))
        return [
            SubHitSegment(
                sub_trai=1,
                start_idx=0,
                end_idx=n_samples,
                onset_idx=min(pre_trigger_samples, peak_idx),
                peak_idx=peak_idx,
            )
        ]

    # 3. Validate true multi-hit boundaries via valley decay and relative amplitude
    valid_peaks = [int(candidate_peaks[0])]
    split_valleys: list[int] = []

    for raw_peak in candidate_peaks[1:]:
        curr_peak = int(raw_peak)
        prev_peak = valid_peaks[-1]

        valley_rel_idx = int(np.argmin(smooth_env[prev_peak:curr_peak]))
        valley_idx = prev_peak + valley_rel_idx
        valley_amp = float(smooth_env[valley_idx])
        prev_amp = float(smooth_env[prev_peak])
        curr_amp = float(smooth_env[curr_peak])
        smaller_peak_amp = min(prev_amp, curr_amp)

        has_deep_valley = valley_amp <= valley_decay_ratio * smaller_peak_amp
        is_significant_burst = curr_amp >= min_secondary_ratio * prev_amp

        if has_deep_valley and is_significant_burst:
            valid_peaks.append(curr_peak)
            split_valleys.append(valley_idx)
        elif curr_amp > prev_amp:
            # Replace dominant peak if an early precursor preceded the main burst
            valid_peaks[-1] = curr_peak

    # 4. Locate physical arrival onsets (t0) and apply pre-trigger buffer
    onsets: list[int] = []
    starts: list[int] = []

    # First burst: realign t0 if a sub-threshold precursor triggered early
    first_peak = valid_peaks[0]
    if first_peak * dt_us > (pre_trigger_us + min_separation_us):
        initial_valley = _find_preceding_valley(smooth_env, first_peak)
        first_onset = _refine_arrival_onset(
            waveform, smooth_env, initial_valley, first_peak, onset_rise_fraction
        )
    else:
        first_onset = min(pre_trigger_samples, first_peak)

    onsets.append(first_onset)
    starts.append(max(0, first_onset - pre_trigger_samples))

    # Subsequent bursts: clamp pre-trigger at the preceding valley to avoid tail leakage
    for idx, valley_idx in enumerate(split_valleys):
        next_peak = valid_peaks[idx + 1]
        next_onset = _refine_arrival_onset(
            waveform, smooth_env, valley_idx, next_peak, onset_rise_fraction
        )
        onsets.append(next_onset)
        starts.append(max(0, next_onset - pre_trigger_samples))

    # 5. Build SubHitSegment objects
    segments: list[SubHitSegment] = []
    n_bursts = len(valid_peaks)

    for i in range(n_bursts):
        if i == n_bursts - 1:
            end_idx = n_samples
        else:
            end_idx = starts[i + 1]

        segments.append(
            SubHitSegment(
                sub_trai=i + 1,
                start_idx=starts[i],
                end_idx=end_idx,
                onset_idx=onsets[i],
                peak_idx=valid_peaks[i],
            )
        )

    return segments


def _format_segmentation_report(
    waveform: npt.NDArray[np.floating],
    time: npt.NDArray[np.floating],
    segments: list[SubHitSegment],
    params: dict[str, Any],
    trai: int | None = None,
    test_id: str | None = None,
) -> str:
    """Format a plain-text report summarizing parameters and extracted sub-hits."""
    sampling_rate = float(params.get("sampling_rate", 5.0))
    dt_us = 1 / sampling_rate
    lines = [
        "=" * 72,
        "SUB-HIT SEGMENTATION REPORT",
        "=" * 72,
        f"Test ID          : {test_id or 'N/A'}",
        f"TRAI             : {trai if trai is not None else 'N/A'}",
        f"Total Samples    : {len(waveform)} ({len(waveform) * dt_us:.2f} us)",
        f"Segments Found   : {len(segments)}",
        "-" * 72,
        "PARAMETERS:",
    ]
    for key, val in params.items():
        lines.append(f"  {key:<22}: {val}")

    lines.extend(["-" * 72, "SEGMENTS:"])
    for seg in segments:
        t_start = time[seg.start_idx] * 1e6
        t_onset = time[seg.onset_idx] * 1e6
        t_peak = time[seg.peak_idx] * 1e6
        t_end = time[seg.end_idx - 1] * 1e6
        dur_us = (seg.end_idx - seg.start_idx) * dt_us
        pre_trig_us = (seg.onset_idx - seg.start_idx) * dt_us
        peak_mv = float(np.max(np.abs(waveform[seg.start_idx : seg.end_idx])))
        lines.append(
            f"  Sub-hit {seg.sub_trai}: [{seg.start_idx}:{seg.end_idx}] | "
            f"Start={t_start:.2f} us | Onset={t_onset:.2f} us (PreTrig={pre_trig_us:.2f} us) | "
            f"Peak={t_peak:.2f} us ({peak_mv:.2f} mV) | End={t_end:.2f} us | Dur={dur_us:.2f} us"
        )
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


def plot_subhit_segmentation(
    waveform: npt.NDArray[np.floating],
    time: npt.NDArray[np.floating],
    segments: list[SubHitSegment],
    params: dict[str, Any],
    figure_path: Path | str = "subhit_segmentation.png",
    smooth_env: npt.NDArray[np.float64] | None = None,
    trai: int | None = None,
    test_id: str | None = None,
    title: str | None = None,
    show_title: bool = True,
    show_subplot_titles: bool = True,
    legend_loc: str = "lower right",
    figsize: tuple[float, float] = (140, 90),
    save: bool = True,
    show: bool = True,
    save_txt: bool = True,
) -> np.ndarray:
    """Plot transient envelope boundaries and extracted sub-hits using mylib.
    Parameters:
    - waveform: 1D array of transient signal amplitudes [mV].
    - time: 1D array of time values [s] corresponding to `waveform`.
    - segments: List of `SubHitSegment` objects defining extracted sub-hits.
    - params: Dictionary of segmentation parameters used for this analysis.
    - figure_path: Path to save the generated figure.
    - smooth_env: Optional pre-computed smoothed Hilbert envelope [mV].
    - trai: Optional transient ID for labeling.
    - test_id: Optional test identifier for labeling.
    - title: Optional custom title for the figure.
    - figsize: Tuple specifying figure size in inches (width, height).
    - save: Whether to save the figure to `figure_path`.
    - show: Whether to display the figure interactively.
    - save_txt: Whether to save a plain-text segmentation report alongside the figure.
    Returns:
    - axs: The matplotlib Axes array containing the generated subplots.
    """

    fig_path = Path(figure_path)
    time_us = time * 1e6
    sampling_rate = float(params.get("sampling_rate", 5.0))
    smoothing_us = float(params.get("smoothing_us", 80.0))
    threshold_dbae = float(params.get("threshold_dbae", 45.0))
    threshold_mV = dbae_to_mV(threshold_dbae)

    if smooth_env is None:
        smooth_env = compute_smoothed_envelope(
            waveform, sampling_rate=sampling_rate, smoothing_us=smoothing_us
        )

    fig, axs = plt.subplots(
        2, 1, figsize=img_size_inches(figsize), sharex=True, tight_layout=True
    )

    if show_title:
        if title is None:
            id_prefix = f"{test_id} - " if test_id else ""
            trai_label = f"TRAI {trai}" if trai is not None else "Transient"
            fig.suptitle(
                f"Sub-hit Segmentation: {id_prefix}{trai_label} ({len(segments)} segments)",
            )
        else:
            fig.suptitle(title)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    axs[0].plot(time_us, waveform, color="0.65", linewidth=0.5, label="Raw waveform")
    axs[0].plot(
        time_us, smooth_env, color="darkorange", linewidth=0.9, label="Hilbert envelope"
    )
    axs[0].axhline(
        threshold_mV,
        color="crimson",
        linestyle=":",
        linewidth=0.7,
        label=f"Threshold ({threshold_dbae:.1f} dBAE)",
    )
    axs[0].axhline(
        -threshold_mV,
        color="crimson",
        linestyle=":",
        linewidth=0.7,
    )

    for idx, seg in enumerate(segments):
        color = colors[idx % len(colors)]
        axs[0].axvspan(
            time_us[seg.start_idx], time_us[seg.end_idx - 1], color=color, alpha=0.12
        )
        axs[0].axvline(
            time_us[seg.onset_idx],
            color=color,
            linestyle="--",
            linewidth=0.7,
            label="Onset ($t_0$)" if idx == 0 else None,
        )
        axs[0].plot(
            time_us[seg.peak_idx],
            smooth_env[seg.peak_idx],
            marker="v",
            markersize=4,
            color=color,
            linestyle="None",
            label="Detected peak" if idx == 0 else None,
        )

    axs[0].set_ylabel("Amplitude [mV]")
    if show_subplot_titles:
        axs[0].set_title("Envelope Detection & Event Landmarks", fontsize=8)
    axs[0].grid(True, linestyle="-", linewidth=0.3, alpha=0.7)
    axs[0].set_axisbelow(True)
    axs[0].legend(loc=legend_loc, fontsize=6.5, ncol=2)

    for idx, seg in enumerate(segments):
        color = colors[idx % len(colors)]
        axs[1].plot(
            time_us[seg.start_idx : seg.end_idx],
            waveform[seg.start_idx : seg.end_idx],
            color=color,
            linewidth=0.65,
            label=f"Sub-hit {seg.sub_trai}",
        )
        axs[1].axvline(
            time_us[seg.onset_idx], color=color, linestyle="--", linewidth=0.6
        )

    axs[1].set_xlabel("Time [us]")
    axs[1].set_ylabel("Amplitude [mV]")
    if show_subplot_titles:
        axs[1].set_title("Extracted Sub-hit Waveforms", fontsize=8)
    axs[1].grid(True, linestyle="-", linewidth=0.3, alpha=0.7)
    axs[1].set_axisbelow(True)
    axs[1].legend(loc=legend_loc, fontsize=6.5)

    if save:
        save_fig(fig, fig_path)

    if save_txt:
        txt_path = fig_path.with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(
            _format_segmentation_report(
                waveform=waveform,
                time=time,
                segments=segments,
                params=params,
                trai=trai,
                test_id=test_id,
            ),
            encoding="utf-8",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return axs


def _write_valid_combos_report(
    summary_df: pd.DataFrame,
    expected_map: dict[int, int],
    report_path: Path,
    param_keys: list[str],
    test_id: str = "",
) -> list[str]:
    """Identify combinations matching `expected_map` across all signals and write a report."""
    trai_order = list(expected_map.keys())
    valid_combos: list[str] = []
    combo_scores: list[tuple[int, int, str]] = []

    for combo_id, group in summary_df.groupby("combo_id", sort=False):
        by_trai = dict(zip(group["trai"], group["n_segments"]))
        matches = sum(1 for t, exp in expected_map.items() if by_trai.get(t) == exp)
        abs_error = sum(
            abs(int(by_trai.get(t, 0)) - exp) for t, exp in expected_map.items()
        )
        combo_scores.append((matches, -abs_error, str(combo_id)))

        if matches == len(expected_map) and len(by_trai) == len(expected_map):
            valid_combos.append(combo_id)

    total_combos = summary_df["combo_id"].nunique()
    lines = [
        "=" * 76,
        "GRID SEARCH VALIDATION REPORT",
        "=" * 76,
        f"Test ID            : {test_id or 'N/A'}",
        f"TRAI List          : {trai_order}",
        f"Expected Segments  : {[expected_map[t] for t in trai_order]}",
        f"Combos Evaluated   : {total_combos}",
        f"Valid Combos Found : {len(valid_combos)}",
        "=" * 76,
        "",
    ]

    if valid_combos:
        for combo_id in valid_combos:
            sub = summary_df[summary_df["combo_id"] == combo_id].set_index("trai")
            first_row = sub.iloc[0]
            param_dict = {
                k: (
                    first_row[k].item()
                    if hasattr(first_row[k], "item")
                    else first_row[k]
                )
                for k in param_keys
                if k in first_row
            }
            lines.append(f"[{combo_id}]  ->  {param_dict}")
            for t in trai_order:
                row = sub.loc[t]
                lines.append(
                    f"    TRAI {t:<6} | detected={int(row['n_segments'])} "
                    f"(expected={expected_map[t]}) | "
                    f"durations_us={row['durations_us']} | "
                    f"file={row['figure_file'] or 'not_saved'}"
                )
            lines.append("-" * 76)
    else:
        lines.append("No parameter combination matched all expected segment counts.")
        lines.append(
            "Top 5 closest combinations (ranked by matched signals & lowest error):"
        )
        lines.append("-" * 76)
        combo_scores.sort(reverse=True)
        for matches, neg_err, combo_id in combo_scores[:5]:
            sub = summary_df[summary_df["combo_id"] == combo_id].set_index("trai")
            first_row = sub.iloc[0]
            param_dict = {
                k: (
                    first_row[k].item()
                    if hasattr(first_row[k], "item")
                    else first_row[k]
                )
                for k in param_keys
                if k in first_row
            }
            detected_str = ", ".join(
                f"{t}:{int(sub.loc[t, 'n_segments'])}/{expected_map[t]}"
                for t in trai_order
                if t in sub.index
            )
            lines.append(
                f"[{combo_id}] matched {matches}/{len(expected_map)} "
                f"(total_err={-neg_err}) | {detected_str}"
            )
            lines.append(f"    params = {param_dict}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return valid_combos


def run_segmentation_grid_search(
    tradb_path: Path | str,
    trai_list: list[int],
    param_grid: dict[str, list[Any]],
    fig_dir: Path | str = FIGURES_DIR,
    results_dir: Path | str = RESULTS_DIR,
    expected_segments: list[int] | None = None,
    save_valid_only: bool = False,
    test_id: str = "",
    sampling_rate: float = 5.0,
    img_format: str = "png",
    show_title: bool = True,
    show_subplot_titles: bool = True,
    legend_loc: str = "lower right",
    figsize: tuple[float, float] = (140, 90),
    save: bool = True,
    show: bool = False,
    save_txt: bool = True,
) -> pd.DataFrame:
    """Run a parametric grid search for `segment_merged_hits` across selected transients."""
    expected_map: dict[int, int] | None = None
    if expected_segments is not None:
        if len(expected_segments) != len(trai_list):
            raise ValueError(
                f"expected_segments length ({len(expected_segments)}) must match "
                f"trai_list length ({len(trai_list)})."
            )
        expected_map = dict(zip(trai_list, expected_segments))
    elif save_valid_only:
        raise ValueError(
            "expected_segments must be provided when save_valid_only=True."
        )

    profiler = Profiler()
    with profiler.section("Load .tradb waveforms"):
        tradb = vae.io.TraDatabase(Path(tradb_path))
        waveforms: dict[
            int, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = {}
        for trai in trai_list:
            amp_v, time = tradb.read_wave(trai, time_axis=True)
            waveforms[trai] = (amp_v * 1e3, time)

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    summary_rows: list[dict[str, Any]] = []
    dt_us = 1 / sampling_rate

    t_seg_total = 0.0
    t_plot_total = 0.0

    for combo_idx, values in enumerate(combinations, start=1):
        params = {"sampling_rate": sampling_rate, **dict(zip(keys, values))}

        combo_evals: list[tuple[int, list[SubHitSegment], list[float]]] = []
        with profiler.section("Envelope & segmentation"):
            for trai in trai_list:
                waveform, _ = waveforms[trai]
                segments = segment_merged_hits(waveform=waveform, **params)
                durations_us = [
                    round((s.end_idx - s.start_idx) * dt_us, 2) for s in segments
                ]
                combo_evals.append((trai, segments, durations_us))
        t_seg_total += profiler._times["Envelope & segmentation"]

        if expected_map is not None:
            combo_is_valid = all(
                len(segs) == expected_map[t] for t, segs, _ in combo_evals
            )
        else:
            combo_is_valid = True

        should_export = (not save_valid_only) or combo_is_valid

        for trai, segments, durations_us in combo_evals:
            waveform, time = waveforms[trai]
            prefix = f"{test_id}_" if test_id else ""
            stem = f"{prefix}trai_{trai}_{combo_idx}"
            fig_name = f"{stem}.{img_format}"
            fig_path = fig_dir / fig_name

            if should_export and (save or save_txt or show):
                with profiler.section("Plot & export reports"):
                    smooth_env = compute_smoothed_envelope(
                        waveform,
                        sampling_rate=sampling_rate,
                        smoothing_us=float(params.get("smoothing_us", 80.0)),
                    )
                    plot_subhit_segmentation(
                        waveform=waveform,
                        time=time,
                        segments=segments,
                        params=params,
                        figure_path=fig_path,
                        smooth_env=smooth_env,
                        trai=trai,
                        test_id=test_id,
                        show_title=show_title,
                        show_subplot_titles=show_subplot_titles,
                        legend_loc=legend_loc,
                        figsize=figsize,
                        save=save,
                        show=show,
                        save_txt=save_txt,
                    )
                t_plot_total += profiler._times["Plot & export reports"]

            row: dict[str, Any] = {
                "combo_id": combo_idx,
                "test_id": test_id,
                "trai": trai,
                "n_segments": len(segments),
            }
            if expected_map is not None:
                row["expected_segments"] = expected_map[trai]
                row["valid_signal"] = len(segments) == expected_map[trai]
                row["valid_combo"] = combo_is_valid

            row.update(
                {
                    "durations_us": str(durations_us),
                    "figure_file": fig_name if (should_export and save) else "",
                    **params,
                }
            )
            summary_rows.append(row)

    profiler._times["Envelope & segmentation"] = t_seg_total
    profiler._times["Plot & export reports"] = t_plot_total
    profiler.report()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "grid_search_summary.csv", index=False)

    if expected_map is not None:
        valid_report_path = results_dir / "valid_combinations.txt"
        valid_ids = _write_valid_combos_report(
            summary_df=summary_df,
            expected_map=expected_map,
            report_path=valid_report_path,
            param_keys=["sampling_rate", *keys],
            test_id=test_id,
        )
        print(
            f"[INFO] {len(valid_ids)}/{len(combinations)} combinations matched all "
            f"expected segments. Report saved at: {valid_report_path}"
        )

    return summary_df


def load_segmentation_params(
    seg_cfg: dict[str, Any],
    dataset_name: str,
    default_test_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Resolve `segment_merged_hits()` keyword arguments from the `"segmentation"`
    config block, reading `grid_search_summary.csv` when needed.
    """
    if not seg_cfg.get("enabled", False):
        return None

    # 1. Direct parameter dictionary override
    if seg_cfg.get("parameters"):
        return dict(seg_cfg["parameters"])

    # 2. Locate grid_search_summary.csv
    if seg_cfg.get("summary_csv"):
        summary_path = resolve_path(seg_cfg["summary_csv"])
    else:
        source_test_id = seg_cfg.get("source_test_id") or default_test_id
        if not source_test_id:
            raise ValueError(
                "Specify 'source_test_id' or 'summary_csv' under extraction.segmentation."
            )
        summary_path = (
            RESULTS_DIR
            / dataset_name
            / "segmentation"
            / source_test_id
            / "grid_search_summary.csv"
        )

    if not summary_path.exists():
        raise FileNotFoundError(f"Grid search summary not found: {summary_path}")

    df = pd.read_csv(summary_path)
    combo_id = seg_cfg.get("combo_id")

    if combo_id and str(combo_id).lower() != "best":
        matches = df[df["combo_id"] == combo_id]
        if matches.empty:
            raise ValueError(f"Combo '{combo_id}' not found in {summary_path}.")
    else:
        if "valid_combo" in df.columns and df["valid_combo"].any():
            matches = df[df["valid_combo"]]
        else:
            raise ValueError(
                f"No valid combos found in {summary_path}. Specify an explicit 'combo_id'."
            )

    row = matches.iloc[0]
    params: dict[str, Any] = {}
    for col in PARAM_COLUMNS:
        if col in row and pd.notna(row[col]):
            val = row[col]
            params[col] = val.item() if hasattr(val, "item") else val

    return params
