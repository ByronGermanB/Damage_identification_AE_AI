"""
Created on Thu Nov 16 10:24:10 2023

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================

# Main libraries
import os
import re
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import vallenae as vae
from mylib import (  # type: ignore
    IMG_SIZE,
    MARKER_LIST,
    configure_plots,
    get_active_palette,
    img_size_inches,
    resolve_save_target,
    save_fig,
)

# Libraries for signal processing
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hamming

# Model training
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Feature visualization
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
)

# Libraries for hyperparameter optimization
# Libraries for metrics and model evaluation
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_predict,
    train_test_split,
)

# Data processing and normalization
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from utils.segmentation import SubHitSegment, segment_merged_hits

configure_plots(font="IBM Plex Sans")

# =============================================================================
# Feature extraction
# =============================================================================


class Features:
    def __init__(
        self,
        path: str | Path,
        lower_freq: float,
        upper_freq: float,
        sampling_rate: float,
        N_samp: int,
        N_seg: int,
        signal_offset: float,
        load_offset: float = 0,
        force: pd.DataFrame | None = None,
        segmentation_params: dict[str, Any] | None = None,
        sub_signal_offset_us: float = 0.0,
    ):
        """
        Parameters
        ----------
        path : str or Path
            Directory where the `.pridb` and `.tradb` files are located.
        lower_freq : float
            Lower frequency limit in kHz.
        upper_freq : float
            Upper frequency limit in kHz.
        sampling_rate : float
            Data acquisition frequency per second in MHz.
        N_samp : int
            Number of samples in each segment for Fourier analysis (power of 2).
        N_seg : int
            Total number of overlapping segments to analyze.
        signal_offset : float
            Start offset in microseconds [µs] depending on pretrigger.
        load_offset : float, default=0
            Offset between AE and MTS time bases [s].
        force : DataFrame, optional
            MTS force data for automatic time alignment.
        """
        self.path = Path(path)
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        self.sampling_rate = sampling_rate
        self.N_samp = N_samp
        self.N_seg = N_seg
        self.signal_offset_us = signal_offset
        self.signal_offset = int(signal_offset * sampling_rate)
        self.load_offset = load_offset
        self.force = force

        for item in os.listdir(self.path):
            if item.endswith(".tradb"):
                self.tradb_path = self.path / item
            elif item.endswith(".pridb"):
                self.pridb_path = self.path / item

        self.res_freq = sampling_rate * 1000.0 / N_samp
        self.time_samp = 1.0 / sampling_rate
        self.time_seg = self.time_samp * N_samp

        self.lower_freq_index = int(self.lower_freq // self.res_freq)
        self.upper_freq_index = int(self.upper_freq // self.res_freq + 1)
        self.N_feat_seg = 9

        self.window = hamming(self.N_samp)
        self.freq_fft = (fftfreq(self.N_samp, self.time_samp * 1e-6) / 1000.0)[
            : self.N_samp // 2
        ]
        self.freq_fft_rango = self.freq_fft[
            self.lower_freq_index : self.upper_freq_index + 1
        ]
        self.segmentation_params = segmentation_params
        self.sub_signal_offset = int(sub_signal_offset_us * sampling_rate)

    def _extract_fft_vector(
        self,
        waveform_mv: np.ndarray,
        offset_pts: int,
        bin_edges: list[int],
    ) -> np.ndarray:
        """Compute the 1D spectral feature vector (`9 * N_seg`) for one waveform or sub-hit."""
        min_required = int(((self.N_seg - 1) / 2 + 1) * self.N_samp) + offset_pts
        if waveform_mv.size < min_required:
            waveform_mv = np.pad(waveform_mv, (0, min_required - waveform_mv.size))

        seg_total = np.zeros((self.N_seg, self.N_samp))
        for i in range(self.N_seg):
            start = int(i * self.N_samp / 2 + offset_pts)
            end = int((i / 2 + 1) * self.N_samp + offset_pts)
            seg_total[i] = waveform_mv[start:end]

        amp_fft = np.abs(fft(seg_total * self.window))
        amp_fft_rango = amp_fft[:, self.lower_freq_index : self.upper_freq_index + 1]

        peak_freq = np.zeros(self.N_seg)
        centroid_freq = np.zeros(self.N_seg)
        part_powers = np.zeros((6, self.N_seg))

        for i in range(self.N_seg):
            spectrum = amp_fft_rango[i]
            spec_sum = np.sum(spectrum)
            if spec_sum > 0:
                peak_freq[i] = self.freq_fft_rango[np.argmax(spectrum)]
                centroid_freq[i] = np.sum(spectrum * self.freq_fft_rango) / spec_sum

            power_spec = spectrum**2
            denominator = np.sum(power_spec)
            if denominator > 0:
                for band in range(6):
                    part_powers[band, i] = (
                        np.sum(power_spec[bin_edges[band] : bin_edges[band + 1]])
                        * 100.0
                        / denominator
                    )

        w_peak_freq = (peak_freq * centroid_freq) ** 0.5
        return np.hstack(
            (
                peak_freq,
                w_peak_freq,
                centroid_freq,
                part_powers[0],
                part_powers[1],
                part_powers[2],
                part_powers[3],
                part_powers[4],
                part_powers[5],
            )
        )

    def read_transient(self, trai: int) -> tuple[np.ndarray, np.ndarray]:
        """Load `(time_us, amp_mv)` for a given `trai` from `self.tradb_path`."""
        return load_transient_wave(self.tradb_path, trai)

    def feature_extr(
        self,
        threshold: float,
        counts: int,
        label: int | str,
        test_id: str,
        max_trai: int | None = None,
        min_trai: int = 1,
    ) -> pd.DataFrame:
        """Extract frequency and time-domain features for all filtered transients."""
        pridb = vae.io.PriDatabase(str(self.pridb_path))
        tradb = vae.io.TraDatabase(str(self.tradb_path))
        df_hits = pridb.read_hits()

        threshold_V = 10 ** (threshold / 20 - 6)
        threshold_mv = threshold_V * 1e3
        no_saturation = 10 ** (94 / 20 - 6)

        mask = (
            (df_hits["channel"] >= 1)
            & (df_hits["amplitude"] >= threshold_V)
            & (df_hits["amplitude"] <= no_saturation)
            & (df_hits["trai"] >= min_trai)
            & (df_hits["counts"] >= counts)
        )
        if max_trai is not None:
            mask &= df_hits["trai"] <= max_trai

        df_hits_filtro = df_hits[mask]
        if df_hits_filtro.empty:
            print(
                f"[WARNING] No transients found for label '{label}' with the given filtering criteria."
            )
            return pd.DataFrame()

        # trai = df_hits_filtro["trai"].to_numpy()
        # N_trai = trai.size
        # v_features_total = np.zeros((N_trai, self.N_seg * self.N_feat_seg))

        n_bins = len(self.freq_fft_rango)
        bin_edges = [int(n_bins * k / 6) for k in range(7)]

        feature_rows: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []

        for hit in df_hits_filtro.itertuples():
            trans = int(hit.trai)
            amp_v, _ = tradb.read_wave(trans, time_axis=False)
            amp_mv = amp_v * 1e3
            total_sq_energy = float(np.sum(amp_mv**2))

            if self.segmentation_params is not None:
                segments = segment_merged_hits(
                    waveform=amp_mv,
                    **self.segmentation_params,
                )

                # Fallback to single full segment if no peak crossed the segmentation threshold
                if not segments:
                    segments = [
                        SubHitSegment(
                            sub_trai=1,
                            start_idx=0,
                            end_idx=len(amp_mv),
                            onset_idx=0,
                            peak_idx=int(np.argmax(np.abs(amp_mv))),
                        )
                    ]

                for seg in segments:
                    sub_amp = amp_mv[seg.start_idx : seg.end_idx]
                    if sub_amp.size == 0:
                        continue

                    # Count positive threshold crossings in this sub-hit
                    above = sub_amp >= threshold_mv
                    sub_counts = int(np.sum(above[1:] & ~above[:-1]) + int(above[0]))
                    if sub_counts < counts:
                        continue

                    peak_mv = float(np.max(np.abs(sub_amp)))
                    if peak_mv <= 0:
                        continue

                    sub_amp_dbae = 20.0 * (np.log10(peak_mv * 1e-3) + 6.0)
                    energy_ratio = (
                        float(np.sum(sub_amp**2)) / total_sq_energy
                        if total_sq_energy > 0
                        else 1.0
                    )
                    sub_time_s = float(hit.time) + (
                        seg.onset_idx * self.time_samp * 1e-6
                    )
                    sub_rise_us = max(
                        0.0, (seg.peak_idx - seg.onset_idx) * self.time_samp
                    )

                    feat_vec = self._extract_fft_vector(
                        waveform_mv=sub_amp,
                        offset_pts=self.sub_signal_offset,
                        bin_edges=bin_edges,
                    )
                    feature_rows.append(feat_vec)
                    meta_rows.append(
                        {
                            "rise_time": sub_rise_us,
                            "time": sub_time_s,
                            "amplitude": sub_amp_dbae,
                            "energy": float(hit.energy) * energy_ratio,
                            "counts": sub_counts,
                            "label": label,
                            "trai": trans,
                            "sub_trai": int(seg.sub_trai),
                            "hit_id": f"{trans}_{seg.sub_trai}",
                            "test_id": test_id,
                        }
                    )

            else:
                feat_vec = self._extract_fft_vector(
                    waveform_mv=amp_mv,
                    offset_pts=self.signal_offset,
                    bin_edges=bin_edges,
                )
                feature_rows.append(feat_vec)
                meta_rows.append(
                    {
                        "rise_time": float(hit.rise_time),
                        "time": float(hit.time),
                        "amplitude": 20.0 * (np.log10(float(hit.amplitude)) + 6.0),
                        "energy": float(hit.energy),
                        "counts": int(hit.counts),
                        "label": label,
                        "trai": trans,
                        "sub_trai": 1,
                        "hit_id": f"{trans}_1",
                        "test_id": test_id,
                    }
                )

        if not feature_rows:
            return pd.DataFrame()

        base_features = [
            "peak_freq",
            "w_peak_freq",
            "centroid_freq",
            *(f"p_power_{k}" for k in range(1, 7)),
        ]

        name_feature = (
            base_features
            if self.N_seg == 1
            else [
                f"{feat}_{seg}"
                for feat in base_features
                for seg in range(1, self.N_seg + 1)
            ]
        )

        df_features = pd.DataFrame(np.vstack(feature_rows), columns=name_feature)
        df_meta = pd.DataFrame(meta_rows)

        if self.force is not None:
            load_offset = df_meta["time"].iloc[-1] - self.force["Time [s]"].iloc[-1]
        else:
            load_offset = self.load_offset

        df_meta["time"] = df_meta["time"] - load_offset
        df_meta["time_norm"] = df_meta["time"] / df_meta["time"].max()

        # df_label = pd.DataFrame({"label": [label] * N_trai})
        # df_total = pd.concat([df_features, df_time, df_label, df_trai], axis=1)
        # df_total["test_id"] = test_id

        return pd.concat([df_features, df_meta], axis=1)

    def plot_signal(
        self,
        trai: int,
        name_figure: str,
        figure_path: str | Path | None = None,
        time_graph: float | None = None,
        title: str | None = None,
        x_label: str = "Time [µs]",
        y_label: str = "Amplitude [mV]",
        figsize: tuple[float, float] = IMG_SIZE,
        ax: plt.Axes | None = None,
        save: bool = False,
        show: bool = True,
    ) -> plt.Axes:
        """Convenience wrapper to plot a transient directly from `self.tradb_path`."""
        time_us, amp_mv = self.read_transient(trai)
        plot_title = title or f"Transient Wave Plot: {name_figure} - Transient: {trai}"
        target = resolve_save_target(figure_path, f"Transient_{name_figure}_{trai}.pdf")

        return plot_signal(
            time_us=time_us,
            amp_mv=amp_mv,
            figure_path=target,
            time_graph=time_graph,
            signal_offset_us=self.signal_offset_us,
            sampling_rate=self.sampling_rate,
            title=plot_title,
            x_label=x_label,
            y_label=y_label,
            figsize=figsize,
            ax=ax,
            save=save,
            show=show,
        )

    def plot_segmentation(
        self,
        trai: int,
        name_figure: str,
        figure_path: str | Path | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        save: bool = False,
        show: bool = True,
    ) -> np.ndarray:
        """Convenience wrapper to plot segmentation + FFT directly from `self.tradb_path`."""
        time_us, amp_mv = self.read_transient(trai)
        plot_title = title or f"Segmentation: {name_figure} - Transient: {trai}"
        target = resolve_save_target(
            figure_path, f"Segmentation_{name_figure}_{trai}.pdf"
        )

        return plot_segmentation(
            time_us=time_us,
            amp_mv=amp_mv,
            sampling_rate=self.sampling_rate,
            lower_freq=self.lower_freq,
            upper_freq=self.upper_freq,
            n_samp=self.N_samp,
            n_seg=self.N_seg,
            signal_offset_us=self.signal_offset_us,
            figure_path=target,
            title=plot_title,
            figsize=figsize,
            save=save,
            show=show,
        )


def load_transient_wave(
    tradb_path: str | Path,
    trai: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a single transient from a Vallen `.tradb` file.

    Returns
    -------
    time_us : np.ndarray
        Time vector in microseconds [µs].
    amp_mv : np.ndarray
        Signal amplitude in millivolts [mV].
    """
    tradb = vae.io.TraDatabase(str(tradb_path))
    amp_v, time_s = tradb.read_wave(trai, time_axis=True)
    return time_s * 1e6, amp_v * 1e3


def load_transient_csv(
    csv_path: str | Path,
    time_col: str | int = 0,
    amp_col: str | int = 1,
    time_scale: float = 1.0,
    amp_scale: float = 1.0,
    sep: str = ",",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a transient waveform from a `.csv` file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    time_col, amp_col : str or int, default=(0, 1)
        Column names or positional indices for time and amplitude.
    time_scale, amp_scale : float, default=1.0
        Multiplicative unit conversion factors (e.g., `1e6` for s -> µs, `1e3` for V -> mV).
    sep : str, default=","
        CSV delimiter.

    Returns
    -------
    time_us : np.ndarray
    amp_mv : np.ndarray
    """
    df = pd.read_csv(csv_path, sep=sep)
    t = (
        df[time_col].to_numpy()
        if isinstance(time_col, str)
        else df.iloc[:, time_col].to_numpy()
    )
    a = (
        df[amp_col].to_numpy()
        if isinstance(amp_col, str)
        else df.iloc[:, amp_col].to_numpy()
    )
    return t * time_scale, a * amp_scale


def compute_segmented_fft(
    amp_mv: np.ndarray,
    sampling_rate: float,
    n_samp: int = 1024,
    n_seg: int = 1,
    lower_freq: float = 100.0,
    upper_freq: float = 900.0,
    signal_offset_us: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Segment a waveform with 50% overlap, apply a Hamming window, and compute FFT spectra.

    Parameters
    ----------
    amp_mv : np.ndarray
        Waveform amplitude in mV.
    sampling_rate : float
        Sampling frequency in MHz (samples/µs).
    n_samp : int
        Samples per segment (power of 2).
    n_seg : int
        Number of 50%-overlapping segments.
    lower_freq, upper_freq : float
        Frequency band of interest in kHz.
    signal_offset_us : float
        Start offset in µs.

    Returns
    -------
    dict with keys:
        `seg_indices`, `seg_windowed`, `freq_range`, `amp_fft_range`,
        `peak_freq`, `centroid_freq`.
    """
    offset_pts = int(signal_offset_us * sampling_rate)
    res_freq = sampling_rate * 1000.0 / n_samp  # kHz
    time_samp = 1.0 / sampling_rate  # µs

    lower_idx = int(lower_freq // res_freq)
    upper_idx = int(upper_freq // res_freq + 1)

    window = hamming(n_samp)
    freq_fft = (fftfreq(n_samp, time_samp * 1e-6) / 1000.0)[: n_samp // 2]
    freq_range = freq_fft[lower_idx : upper_idx + 1]

    seg_total = np.zeros((n_seg, n_samp))
    seg_indices: list[tuple[int, int]] = []

    for i in range(n_seg):
        start = int(i * n_samp / 2) + offset_pts
        end = int((i / 2 + 1) * n_samp) + offset_pts
        seg_total[i] = amp_mv[start:end]
        seg_indices.append((start, end))

    seg_windowed = seg_total * window
    amp_fft = np.abs(fft(seg_windowed))
    amp_fft_range = amp_fft[:, lower_idx : upper_idx + 1]

    peak_freq = np.zeros(n_seg)
    centroid_freq = np.zeros(n_seg)

    for i in range(n_seg):
        peak_idx = int(np.argmax(amp_fft_range[i]))
        peak_freq[i] = freq_range[peak_idx]
        centroid_freq[i] = np.sum(amp_fft_range[i] * freq_range) / np.sum(
            amp_fft_range[i]
        )

    return {
        "seg_indices": np.array(seg_indices),
        "seg_windowed": seg_windowed,
        "freq_range": freq_range,
        "amp_fft_range": amp_fft_range,
        "peak_freq": peak_freq,
        "centroid_freq": centroid_freq,
    }


# =============================================================================
# Standalone Plotting Functions (mylib-compatible)
# =============================================================================
def plot_signal(
    time_us: np.ndarray,
    amp_mv: np.ndarray,
    figure_path: str | Path | None = None,
    time_graph: float | None = None,
    signal_offset_us: float = 0.0,
    sampling_rate: float | None = None,
    title: str | None = "Transient Wave Plot",
    subtitle: str | None = None,
    x_label: str = "Time [µs]",
    y_label: str = "Amplitude [mV]",
    figsize: tuple[float, float] = IMG_SIZE,
    ax: plt.Axes | None = None,
    i: int = 1,
    n_col: int = 1,
    save: bool = False,
    show: bool = True,
) -> plt.Axes:
    """
    Plot and optionally save a time-domain transient waveform.
    `figsize` is specified in millimeters `(width, height)`.
    """
    primary_color = get_active_palette(1)[0]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(f"{x_label}\n({string.ascii_lowercase[i - 1]})")
        if subtitle is not None:
            ax.set_title(subtitle)
        elif title is not None:
            ax.set_title(title)

    if time_graph is None:
        ax.plot(time_us, amp_mv, color=primary_color, linewidth=0.6)
    else:
        if sampling_rate is not None:
            start_idx = int(signal_offset_us * sampling_rate)
            end_idx = int(time_graph * sampling_rate + 1)
            t_slice = time_us[start_idx:end_idx]
            a_slice = amp_mv[start_idx:end_idx]
        else:
            mask = (time_us >= signal_offset_us) & (time_us <= time_graph)
            t_slice = time_us[mask]
            a_slice = amp_mv[mask]

        ax.plot(t_slice, a_slice, color=primary_color, linewidth=0.6)
        if a_slice.size > 0:
            ax.set_ylim(bottom=a_slice.min() * 1.6, top=a_slice.max() * 1.6)

    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        default_name = f"{title or 'Transient'}.pdf"
        save_fig(fig, resolve_save_target(figure_path, default_name))

    if show:
        plt.show()
    if own_fig:
        plt.close(fig)

    return ax


def plot_segmentation(
    time_us: np.ndarray,
    amp_mv: np.ndarray,
    sampling_rate: float,
    lower_freq: float = 100.0,
    upper_freq: float = 900.0,
    n_samp: int = 1024,
    n_seg: int = 1,
    signal_offset_us: float = 0.0,
    figure_path: str | Path | None = None,
    title: str | None = "Signal Segmentation & FFT",
    figsize: tuple[float, float] | None = None,
    save: bool = False,
    show: bool = True,
) -> np.ndarray:
    """
    Plot windowed time-domain segments (top row) and their FFT magnitude spectra
    with `f_peak` and `f_centroid` markers (bottom row).

    Works identically for `n_seg == 1` and `n_seg > 1`.
    `figsize` is specified in millimeters `(width, height)`.
    """
    if figsize is None:
        figsize = (90 * n_seg, 110)

    dsp = compute_segmented_fft(
        amp_mv=amp_mv,
        sampling_rate=sampling_rate,
        n_samp=n_samp,
        n_seg=n_seg,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        signal_offset_us=signal_offset_us,
    )

    seg_indices = dsp["seg_indices"]
    seg_windowed = dsp["seg_windowed"]
    freq_range = dsp["freq_range"]
    amp_fft_range = dsp["amp_fft_range"]
    peak_freq = dsp["peak_freq"]
    centroid_freq = dsp["centroid_freq"]

    palette = get_active_palette(3)
    sig_color, peak_color, centroid_color = palette[0], palette[1], palette[2]

    fig, axs = plt.subplots(
        2,
        n_seg,
        figsize=img_size_inches(figsize),
        squeeze=False,
    )
    if title is not None:
        fig.suptitle(title)

    seg_t_window_max = np.amax(np.abs(seg_windowed))
    amp_fft_max = np.amax(amp_fft_range)

    for j in range(n_seg):
        start, end = seg_indices[j]

        # Row 0: Windowed time-domain segment
        ax_time = axs[0, j]
        ax_time.plot(
            time_us[start:end], seg_windowed[j], color=sig_color, linewidth=0.8
        )
        ax_time.set_ylim(-1.1 * seg_t_window_max, 1.1 * seg_t_window_max)
        ax_time.set_xlabel("Time [µs]")
        if j == 0:
            ax_time.set_ylabel("Amplitude [mV]")
        ax_time.set_title(f"Segment {j + 1}")
        ax_time.grid(True, linestyle="-", alpha=0.7)
        ax_time.set_axisbelow(True)

        # Row 1: FFT spectrum + Peak & Centroid frequencies
        ax_fft = axs[1, j]
        ax_fft.plot(freq_range, amp_fft_range[j], color=sig_color, linewidth=0.8)
        ax_fft.set_ylim(0, amp_fft_max * 1.1)

        y_max = np.amax(amp_fft_range[j]) / (amp_fft_max * 1.1)
        ax_fft.axvline(
            x=peak_freq[j],
            ymax=min(y_max * 1.05, 1.0),
            color=peak_color,
            linestyle="--",
            linewidth=1.0,
            label=f"$f_{{peak}}$ ({peak_freq[j]:.0f} kHz)",
        )
        ax_fft.axvline(
            x=centroid_freq[j],
            ymax=min(y_max * 0.85, 1.0),
            color=centroid_color,
            linestyle=":",
            linewidth=1.0,
            label=f"$f_{{centroid}}$ ({centroid_freq[j]:.0f} kHz)",
        )

        ax_fft.set_xlabel("Frequency [kHz]")
        if j == 0:
            ax_fft.set_ylabel("FFT Magnitude [mV]")
        ax_fft.legend(loc="upper right")
        ax_fft.grid(True, linestyle="-", alpha=0.7)
        ax_fft.set_axisbelow(True)

    if save:
        default_name = f"{title or 'Segmentation'}.pdf"
        save_fig(fig, resolve_save_target(figure_path, default_name))

    if show:
        plt.show()
    plt.close(fig)

    return axs


def train_test_set(
    data,
    normalization=None,
    columns_to_transform=None,
    features=None,
    split=True,
    test_size=0.2,
):
    """
    Parameters
    ----------
    data : DataFrame
        DataFrame containing features, classes, and trais.
    normalization : str, optional
        Normalization for the features.
        'log': logarithmic.
        'std': standard.
        'min_max': min-max.
        'log-std': Logarithmic on columns_to_transform and standard on all. The default is None.
    columns_to_transform : list, optional
        List of columns for transformation.
        If not set, the entire DataFrame is transformed. The default is None.
    split : boolean, optional
        True: To split into train and test sets. The default is True.
    test_size : float, optional
        Percentage of the test set. The default is 0.2.

    Returns
    -------
    X_train : DataFrame
        Features for training.
    X_test : DataFrame
        Features for evaluation.
    y_train : Series
        Classes for training.
    y_test : Series
        Classes for evaluation.

    If split=False, only X_original, X_train, y_train, and hits
    (DataFrame for cumulative hits plot) are returned.
    """
    # Separate features from classes
    X = data.drop(
        [
            "label",
            "trai",
            "time",
            "test_id",
        ],
        axis=1,
    )  # Remove class and trais columns (also remove counts and time_norm)

    if features:
        X = X[features]

    y = data["label"].copy()  # Create a copy of the class column

    X_original = X.copy()

    # Feature names
    feature_names = X.columns.tolist()

    # If columns to transform are not defined, select all
    if columns_to_transform is None:
        columns_to_transform = feature_names

    # Normalization options
    if normalization == "log":
        log_transformer = FunctionTransformer(np.log)
        X[columns_to_transform] = log_transformer.transform(
            X[columns_to_transform]
        )  # Apply logarithm to features

    if normalization == "std":
        standardize = StandardScaler()
        X[columns_to_transform] = standardize.fit_transform(
            X[columns_to_transform]
        )  # Apply standardization to features

    if normalization == "min_max":
        min_max = MinMaxScaler()
        X[columns_to_transform] = min_max.fit_transform(
            X[columns_to_transform]
        )  # Apply min-max scaling to features

    if normalization == "log-std":
        log_transformer = FunctionTransformer(np.log)
        X[columns_to_transform] = log_transformer.transform(
            X[columns_to_transform]
        )  # Apply logarithm to features
        standardize = StandardScaler()
        X = standardize.fit_transform(X)  # Apply standardization to features

    # The DataFrame returned by the function depends on whether it is split or not
    if split:
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=data["label"],
            random_state=0,
            shuffle=True,
        )

        # The function returns 4 DataFrames
        return X_train, X_test, y_train, y_test

    else:
        X_train = X
        y_train = y

        # DataFrame for cumulative hits
        hits = data[
            [
                "test_id",
                "trai",
                "label",
                "time",
                "amplitude",
                "energy",
                "time_norm",
                "counts",
            ]
        ].copy()
        hits["Count"] = 1

        return X_original, X_train, y_train, hits


def backward_elimination(X_train, y_train):
    """
    Parameters
    ----------
    X_train : DataFrame
        Feature set for training.
    y_train : DataFrame
        Feature set for evaluation.

    Returns
    -------
    selected_feature_names : list
        List of selected feature names.

    """
    # Classifier
    model = RandomForestClassifier(random_state=11)

    # Perform backward elimination with cross-validation
    selector = RFECV(estimator=model, step=1, cv=5, scoring="f1_macro")
    selector.fit(X_train, y_train)

    # Get the selected feature names
    selected_feature_names = selector.get_feature_names_out().tolist()

    # Get the best score
    best_f1_score = selector.cv_results_["mean_test_score"][selector.n_features_ - 1]

    # Print the best feature subset and its performance
    print("", "Backward Elimination", sep="\n")
    print("Best Feature Subset:", selected_feature_names)
    print(f"Best F1 Score: {best_f1_score:.1%}")

    return selected_feature_names


def forest_clf(X_train, X_test, y_train, y_test, selected_features=None):
    """
    Parameters
    ----------
    X_train : dataframe, array
        Feature set for training.
    X_test : dataframe, array
        Feature set for evaluation.
    y_train : series, array
        Classes for training.
    y_test : series, array
        Classes for evaluation.
    selected_features : list, optional
        Names of the features to train the model. The default is None.

    Returns
    -------
    y_train_pred : array
        Predicted values for the training set.
    y_test_pred : array
        Predicted values for the evaluation set.
    best_features : dataframe
        DataFrame with features sorted by importance.
    model : ensemble._forest
        Machine learning model that can be exported for predictions.

    """
    # Feature selection (if backward elimination was used)
    if selected_features is not None:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    # Train the model
    model = RandomForestClassifier(random_state=11)  # Model name
    model.fit(X_train, y_train)  # Train the model

    # Model accuracy on the test set
    baseline_accuracy = model.score(X_test, y_test)

    # F1 score of the model on the test set
    y_test_pred = model.predict(X_test)
    f1_score_test = f1_score(
        y_test, y_test_pred, average="macro"
    )  # Calculate the F1 score
    print("", "Precision and F1 score on the test set", sep="\n")
    print(
        f"Baseline Accuracy: {baseline_accuracy:.1%}"
    )  # Print the model accuracy in % format with 1 decimal
    print(f"F1 score - test = {f1_score_test:.1%}")

    # Cross-validation that returns the model accuracy (scores) and predicted values (predict)
    y_train_pred = cross_val_predict(
        model, X_train, y_train, cv=5
    )  # Predicted values on the training set with cross-validation
    cross_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy
    cross_f1_score = f1_score(
        y_train, y_train_pred, average="macro"
    )  # Calculate the F1 score
    print("", "Cross-validation to check for overfitting", sep="\n")
    print(f"Cross-val Accuracy = {cross_accuracy:.1%}")
    print(f"Cross-val F1 score = {cross_f1_score:.1%}")

    # Calculate the importance of each feature for classification
    feature_importances = (model.feature_importances_ * 100).round(2)
    best_features = pd.DataFrame(
        sorted(
            zip(X_train.columns.tolist(), feature_importances),
            key=lambda x: x[1],
            reverse=True,
        ),
        columns=["Features", "Importances (%)"],
    )

    # Print the sorted DataFrame
    print("", "Feature Importance", sep="\n")
    print(best_features.head(5))

    return y_train_pred, y_test_pred, best_features, model


def forest_clf_hyper(X_train, X_test, y_train, y_test, selected_features=None):
    """
    Parameters
    ----------
    X_train : dataframe, array
        Feature set for training.
    X_test : dataframe, array
        Feature set for evaluation.
    y_train : series, array
        Classes for training.
    y_test : series, array
        Classes for evaluation.
    selected_features : list, optional
        Names of the features to train the model. The default is None.

    Returns
    -------
    y_train_pred : array
        Predicted values for the training set.
    y_test_pred : array
        Predicted values for the evaluation set.
    best_features : dataframe
        DataFrame with features sorted by importance.
    best_model : ensemble._forest
        Machine learning model that can be exported for predictions.

    """
    param_grid = {
        "max_features": [
            "sqrt",
            "log2",
        ],  # Maximum number of features to consider at each split
        "n_estimators": np.arange(100, 1000, 100),  # Number of trees in the forest
        "max_depth": [None] + list(np.arange(5, 30, 5)),  # Maximum depth of each tree
        "min_samples_split": np.arange(
            2, 11
        ),  # Minimum number of samples required to split an internal node
        "min_samples_leaf": np.arange(
            1, 5
        ),  # Minimum number of samples required to be at a leaf node
    }

    # Define the scoring metric
    scorer = make_scorer(f1_score, average="macro")

    model = RandomForestClassifier(random_state=42)  # Model name

    # Perform randomized search
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        scoring=scorer,
        cv=5,
        n_iter=10,
        random_state=42,
    )  # Optimizer
    random_search.fit(X_train, y_train)  # Train the optimizer
    best_params = random_search.best_params_  # Get the optimized parameters
    best_score = (
        random_search.best_score_
    )  # Get the F1 score from the optimizer training

    print("", "Hyperparameter tuning", sep="\n")
    print(f"Best parameters: {best_params}")
    print(f"Optimized F1 score: {best_score:.1%}")

    # Train the model with the best estimator
    best_model = random_search.best_estimator_  # Store the best estimator (best model)
    best_model.fit(X_train, y_train)  # Train the machine with the best estimator
    y_test_pred = best_model.predict(
        X_test
    )  # Predictions made with the optimized model on the test set

    # Model accuracy on the test set
    baseline_accuracy = accuracy_score(y_test, y_test_pred)

    # F1 score of the model on the test set
    f1_score_test = f1_score(
        y_test, y_test_pred, average="macro"
    )  # Calculate the F1 score
    print("", "Precision and F1 score on the test set", sep="\n")
    print(
        f"Baseline Accuracy: {baseline_accuracy:.1%}"
    )  # Print the model accuracy in % format with 1 decimal
    print(f"F1 score - test = {f1_score_test:.1%}")

    # Cross-validation that returns the model accuracy (scores) and predicted values (predict)
    y_train_pred = cross_val_predict(
        best_model, X_train, y_train, cv=5
    )  # Predicted values on the training set with cross-validation
    cross_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy
    cross_f1_score = f1_score(
        y_train, y_train_pred, average="macro"
    )  # Calculate the F1 score
    print("", "Cross-validation to check for overfitting", sep="\n")
    print(f"Cross-val Accuracy = {cross_accuracy:.1%}")
    print(f"Cross-val F1 score = {cross_f1_score:.1%}")

    # Calculate the importance of each feature for classification
    feature_importances = (best_model.feature_importances_ * 100).round(2)
    best_features = pd.DataFrame(
        sorted(
            zip(X_train.columns.tolist(), feature_importances),
            key=lambda x: x[1],
            reverse=True,
        ),
        columns=["Features", "Importances (%)"],
    )

    # Print the sorted DataFrame
    print("", "Feature Importance", sep="\n")
    print(best_features.head(5))

    return y_train_pred, y_test_pred, best_features, best_model


def plot_tsne(
    X,
    y,
    figure_path=None,
    title="t-SNE Scatter Plot",
    x_label="t-SNE Dimension 1",
    y_label="t-SNE Dimension 2",
    figsize: tuple[float, float] = IMG_SIZE,
    selected_features=None,
    save: bool = False,
    show: bool = True,
):
    """
    Plot and optionally save a 2D t-SNE scatter plot with class labels and markers.
    `figsize` is specified in millimeters `(width, height)`.
    """
    if selected_features is not None:
        X = X[selected_features]

    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X)

    tsne_df = pd.DataFrame(X_reduced, columns=[f"t-SNE_{i + 1}" for i in range(2)])
    y_series = pd.Series(y, name="Labels").reset_index(drop=True)
    tsne_df["Labels"] = y_series

    label_order = list(pd.unique(tsne_df["Labels"]))
    n_labels = len(label_order)
    markers = {
        label: MARKER_LIST[idx % len(MARKER_LIST)]
        for idx, label in enumerate(label_order)
    }
    palette = get_active_palette(n_labels)

    fig, ax = plt.subplots(figsize=img_size_inches(figsize))

    sns.scatterplot(
        x="t-SNE_1",
        y="t-SNE_2",
        hue="Labels",
        style="Labels",
        hue_order=label_order,
        style_order=label_order,
        data=tsne_df,
        palette=palette,
        markers=markers,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.set_axisbelow(True)

    if save:
        save_fig(fig, resolve_save_target(figure_path, f"{title}.pdf"))

    if show:
        plt.show()
    plt.close(fig)

    return ax


def plot_histogram(
    X,
    y,
    figure_path=None,
    figsize: tuple[float, float] = IMG_SIZE,
    bins: int = 30,
    feature_to_save: str = "",
    save: bool = False,
    show: bool = True,
):
    """
    Plot feature histograms grouped by class and optionally save `feature_to_save`.
    `figsize` is specified in millimeters `(width, height)`.
    """
    class_labels = list(pd.unique(y))
    palette = get_active_palette(len(class_labels))

    for feature in X.columns:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))

        for class_label, color in zip(class_labels, palette):
            feat_to_plot = X[y == class_label][feature]
            ax.hist(
                feat_to_plot,
                bins=bins,
                alpha=0.6,
                label=class_label,
                color=color,
                edgecolor="black",
            )

        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram - {feature}")
        ax.legend()

        if save and (feature == feature_to_save):
            save_fig(
                fig,
                resolve_save_target(figure_path, f"Histogram_{feature_to_save}.pdf"),
            )

        if show:
            plt.show()
        plt.close(fig)


def plot_feat_vs_feat(
    X,
    y,
    feat_x,
    feat_y,
    fig_path="Feat_vs_Feat.png",
    title=None,
    subtitle=None,
    x_label=None,
    y_label=None,
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    save=False,
    show=False,
):
    """
    Plot and optionally save a scatter plot of two features with class labels and markers.
    `figsize` is specified in millimeters `(width, height)`.
    """
    y = pd.Series(y, name="Labels")
    if "Labels" in X.columns:
        X = X.drop(["Labels"], axis=1)

    y.index = X.index
    data = pd.concat([X, y], axis=1)
    label_order = list(pd.unique(data["Labels"]))
    n_labels = len(label_order)

    markers = {
        label: MARKER_LIST[idx % len(MARKER_LIST)]
        for idx, label in enumerate(label_order)
    }
    palette = get_active_palette(n_labels)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"{feat_x} vs {feat_y}" if title is None else title)
        ax.set_xlabel(feat_x if x_label is None else x_label)
    else:
        fig = ax.get_figure()
        base_xlabel = feat_x if x_label is None else x_label
        ax.set_xlabel(f"{base_xlabel}\n({string.ascii_lowercase[i - 1]})")
        ax.set_title(subtitle if subtitle is not None else f"{feat_x} vs {feat_y}")

    sns.scatterplot(
        x=feat_x,
        y=feat_y,
        hue="Labels",
        style="Labels",
        hue_order=label_order,
        style_order=label_order,
        data=data,
        markers=markers,
        palette=palette,
        ax=ax,
        legend=True,
    )

    if (i - 1) % n_col == 0:
        ax.set_ylabel(feat_y if y_label is None else y_label)
    else:
        ax.set_ylabel("")

    if feat_x == "energy":
        ax.set_xscale("log")
    if feat_y == "energy":
        ax.set_yscale("log")

    ax.grid(True)
    ax.set_axisbelow(True)

    if n_labels > 6:
        ax.legend(
            title="Label",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )
        if own_fig:
            fig.subplots_adjust(right=0.72)

    if save:
        save_fig(fig, fig_path)
    if show:
        plt.show()
    if own_fig:
        plt.close(fig)

    return ax


def resolve_feature_columns(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    n_seg: int | None = None,
) -> list[str]:
    """
    Resolve base feature names against `data.columns` for both `N_seg == 1`
    and segmented (`N_seg > 1`) DataFrames.

    - If a name already exists in `data.columns` (e.g., `"energy"` or `"peak_freq"`
      when `N_seg == 1`, or an explicit `"peak_freq_2"`), it is kept as-is.
    - Otherwise, it expands to matching segmented columns (`f"{col}_{seg}"`).
    """
    resolved: list[str] = []
    missing: list[str] = []

    for col in feature_columns:
        if col in data.columns:
            resolved.append(col)
            continue

        if n_seg is not None and n_seg > 1:
            seg_cols = [
                f"{col}_{seg}"
                for seg in range(1, n_seg + 1)
                if f"{col}_{seg}" in data.columns
            ]
        else:
            pattern = re.compile(rf"^{re.escape(col)}_(\d+)$")
            matches = [
                (int(m.group(1)), c) for c in data.columns if (m := pattern.match(c))
            ]
            seg_cols = [c for _, c in sorted(matches, key=lambda item: item[0])]

        if seg_cols:
            resolved.extend(seg_cols)
        else:
            missing.append(col)

    if missing:
        raise KeyError(f"Feature column(s) not found in DataFrame: {missing}")

    # Preserve order while removing any accidental duplicates
    return list(dict.fromkeys(resolved))


def select_significant_indices(
    data: pd.DataFrame,
    label_column: str,
    feature_columns: Sequence[str],
    max_points: int = 20,
    n_seg: int | None = None,
) -> list:
    """
    Select the top `max_points` most significant (highest mean |robust z-score|)
    row indices per class in `label_column`, supporting both `N_seg == 1` and
    segmented (`N_seg > 1`) feature columns.
    """
    resolved_cols = resolve_feature_columns(data, feature_columns, n_seg=n_seg)
    selected_indices = []

    for _, group in data.groupby(label_column, sort=False):
        values = group[resolved_cols].apply(pd.to_numeric, errors="coerce")
        values = values.dropna(how="all")

        if values.empty:
            continue

        median = values.median()
        mad = (values - median).abs().median()
        mad = mad.mask(mad == 0, 1.0)

        robust_z = 0.6745 * (values - median) / mad
        significance = robust_z.abs().mean(axis=1)

        selected_indices.extend(
            significance.nlargest(min(max_points, len(significance))).index
        )

    return selected_indices
