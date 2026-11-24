"""
Created on Tue Jan  9 11:09:45 2024

@author: bbarmac
"""

from __future__ import annotations

import csv
import os
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mylib import (  # type: ignore
    IMG_SIZE,
    MARKER_LIST,
    img_size_inches,
    save_fig,
)


def _get_active_palette(n_colors: int) -> list[str]:
    """Return `n_colors` from the active Matplotlib color cycle set by `configure_plots()`."""
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle_colors:
        cycle_colors = sns.color_palette().as_hex()
    return [cycle_colors[idx % len(cycle_colors)] for idx in range(n_colors)]


def _resolve_save_target(
    figure_path: str | Path | None,
    default_filename: str,
) -> str | Path:
    """Resolve `figure_path` whether it is a directory, a full file path, or None."""
    if figure_path is None:
        return default_filename
    target = Path(figure_path)
    return target if target.suffix else target / default_filename


def force_data(
    path: str | Path,
    section: list[float] | None = None,
    test_id: str | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    path : str or Path
        Directory where the force MTS CSV files are stored.
    section : list or array, optional
        Width and thickness measurements of the specimen in mm. Defaults to [22, 2.4].
    test_id : str, optional
        Identifier assigned to the loaded test. Defaults to the CSV filename stem.

    Returns
    -------
    force_df : DataFrame
        DataFrame containing force, displacement, time, stress, and test_id data.
    """
    if section is None:
        section = [22, 2.4]

    files = os.listdir(path)
    forces_list: list[pd.DataFrame] = []

    for item in files:
        if not item.endswith(".csv"):
            continue

        force_filename_path = Path(path) / item

        # Find the first row whose first column is numeric
        with force_filename_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file, delimiter=";")

            for row_number, row in enumerate(reader):
                if not row:
                    continue

                first_value = row[0].strip().replace(",", ".")
                try:
                    float(first_value)
                    first_data_row = row_number
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"No numeric data found in {force_filename_path}")

        # Read from the first numeric row
        force = pd.read_csv(
            force_filename_path,
            sep=";",
            skiprows=first_data_row,
            header=None,
            names=["Crosshead [mm]", "Load [kN]", "Time [s]"],
            decimal=",",
        )

        # Add a row of zeros where the test starts
        new_row = pd.Series(0, index=force.columns)
        force = pd.concat([pd.DataFrame([new_row]), force], ignore_index=True)

        # Stress
        area = np.prod(section)
        force["Stress [MPa]"] = force["Load [kN]"] * 1000 / area

        # Add the test id column
        current_test_id = item.split(".")[0] if test_id is None else test_id
        force["test_id"] = current_test_id
        forces_list.append(force)

    if not forces_list:
        print(f"[INFO] No .csv files found in {path}. Please check the directory.")
        return pd.DataFrame()

    return pd.concat(forces_list, ignore_index=True)


def plot_stress_hits_cluster(
    data,
    force,
    test_id,
    figure_path="Stress and Cumulative hits vs Time - Clustering.pdf",
    plot_type="scatter",
    x="time",
    y="Cumulative_Label",
    x_label="Time [s]",
    y_label="Cumulative hits",
    y_label_right="Stress [MPa]",
    use_log_scale=False,
    subtitle=None,
    limits=None,
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    save=False,
    show=True,
):
    """
    Plot and optionally save cumulative hits per cluster and stress vs time.
    `figsize` is specified in millimeters `(width, height)`.
    """
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    condition = data["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    filtered_hits = data.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = filtered_hits.groupby("label")[
        "Count"
    ].cumsum()
    filtered_force = force[condition_force]

    num_labels = filtered_hits["label"].nunique()
    markers = [MARKER_LIST[idx % len(MARKER_LIST)] for idx in range(num_labels)]
    palette = _get_active_palette(num_labels)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"Stress and Cumulative hits vs Time - {test_id}")
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        ax.set_title(subtitle if subtitle is not None else test_id)

    if plot_type == "scatter":
        sns.scatterplot(
            x=x,
            y=y,
            hue="label",
            style="label",
            data=filtered_hits,
            markers=markers,
            palette=palette,
            linewidth=0.2,
            s=20,
            ax=ax,
        )
        if use_log_scale:
            ax.set_yscale("log")
    elif plot_type == "line":
        sns.lineplot(
            x=x,
            y=y,
            hue="label",
            style="label",
            data=filtered_hits,
            palette=palette,
            linewidth=2,
            ax=ax,
        )

    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)

    ax2 = ax.twinx()
    sns.lineplot(
        x="Time [s]",
        y="Stress [MPa]",
        data=filtered_force,
        ax=ax2,
        linewidth=1,
        color="black",
        label="Stress",
    )

    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force["Stress [MPa]"]) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)

    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel("")

    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    if i == 1:
        ax2.legend(handles, legends, loc="upper left")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()

    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        save_fig(fig, figure_path)

    if show:
        plt.show()
    if own_fig:
        plt.close(fig)

    return ax


def plot_stress_hits(
    hits,
    force,
    test_id,
    figure_path=None,
    plot_type="scatter",
    title="Stress and Cumulative hits vs Time",
    x="time",
    y="Cumulative_Label",
    x_label="Time [s]",
    y_label="Cumulative hits",
    y_label_right="Stress [MPa]",
    subtitle=None,
    limits=None,
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    n_row=1,
    save=False,
):
    """
    Plot and optionally save cumulative hits and stress vs time.
    `figsize` is specified in millimeters `(width, height)`.
    """
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    condition = hits["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    filtered_hits = hits.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = filtered_hits["Count"].cumsum()
    filtered_force = force[condition_force]

    primary_color = _get_active_palette(1)[0]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"Stress and Cumulative hits vs Time - {test_id}")
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        ax.set_title(subtitle if subtitle is not None else test_id)

    if plot_type == "scatter":
        sns.scatterplot(
            x=x,
            y=y,
            data=filtered_hits,
            color=primary_color,
            label="hits",
            linewidth=0.2,
            s=20,
            ax=ax,
        )
    elif plot_type == "line":
        sns.lineplot(
            x=x,
            y=y,
            label="hits",
            data=filtered_hits,
            color=primary_color,
            linewidth=2,
            ax=ax,
        )

    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)

    ax2 = ax.twinx()
    sns.lineplot(
        x="Time [s]",
        y="Stress [MPa]",
        data=filtered_force,
        ax=ax2,
        linewidth=1,
        color="black",
        label="Stress",
    )

    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force["Stress [MPa]"]) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)

    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel("")

    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    if i == 1:
        ax2.legend(handles, legends, loc="upper left")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()

    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        default_name = f"{title} - {test_id}.pdf" if own_fig else f"{title}.pdf"
        save_fig(fig, _resolve_save_target(figure_path, default_name))

    return ax


def plot_feature_time_histogram(
    data,
    figure_path=None,
    title="Feature distribution vs Normalized time",
    x="time_norm",
    y="peak_freq",
    x_label="Normalized time",
    y_label="Peak frequency [kHz]",
    subtitle=None,
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    save=False,
):
    """
    Plot and optionally save a 2D histogram of feature distribution vs normalized time.
    `figsize` is specified in millimeters `(width, height)`.
    """
    data = data[data["time_norm"] < 0.99]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"{y} vs Normalized time")
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        ax.set_title(subtitle if subtitle is not None else y)

    sns.histplot(
        data=data,
        x=x,
        y=y,
        bins=60,
        pmax=0.95,
        cmap="rainbow",
        cbar=True,
        ax=ax,
    )

    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        default_name = f"{title} - {y}.pdf" if own_fig else f"{title}.pdf"
        save_fig(fig, _resolve_save_target(figure_path, default_name))

    return ax


def plot_feature_histogram(
    data,
    figure_path=None,
    title="Feature distribution vs Normalized time",
    x="peak_freq",
    x_label="Peak frequency [kHz]",
    subtitle=None,
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    save=False,
):
    """
    Plot and optionally save a 1D histogram of a feature distribution.
    `figsize` is specified in millimeters `(width, height)`.
    """
    primary_color = _get_active_palette(1)[0]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"{x} Histogram")
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        ax.set_title(subtitle if subtitle is not None else x)

    sns.histplot(
        data=data,
        x=x,
        bins=40,
        pmax=0.95,
        color=primary_color,
        ax=ax,
    )

    if (i - 1) % n_col == 0:
        ax.set_ylabel("Event count")
    else:
        ax.set_ylabel("")

    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        default_name = f"{title} - {x}.pdf" if own_fig else f"{title}.pdf"
        save_fig(fig, _resolve_save_target(figure_path, default_name))

    return ax


def limit_finder(data, force, test_ids=None):
    """
    Compute upper axis limits [time_limit, hits_limit, stress_limit] across labeled tests.
    """
    if test_ids is None:
        test_ids = data["test_id"].unique()
    condition = data["test_id"].isin(test_ids)
    condition_force = force["test_id"].isin(test_ids)

    filtered_hits = data.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = None

    for test_id in test_ids:
        single_condition = filtered_hits["test_id"] == test_id
        filtered_hits.loc[single_condition, "Cumulative_Label"] = (
            filtered_hits[single_condition].groupby("label")["Count"].cumsum()
        )

    filtered_force = force[condition_force]

    max_value_time = filtered_hits["time"].max()
    max_value_hits = filtered_hits["Cumulative_Label"].max()
    max_value_stress = filtered_force["Stress [MPa]"].max()

    return [max_value_time, max_value_hits, max_value_stress]


def limit_finder_no_label(hits, force, test_ids):
    """
    Compute upper axis limits [time_limit, hits_limit, stress_limit] across unlabeled tests.
    """
    condition = hits["test_id"].isin(test_ids)
    condition_force = force["test_id"].isin(test_ids)

    filtered_hits = hits.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = None

    for test_id in test_ids:
        single_condition = filtered_hits["test_id"] == test_id
        filtered_hits.loc[single_condition, "Cumulative_Label"] = filtered_hits[
            single_condition
        ]["Count"].cumsum()

    filtered_force = force[condition_force]

    max_value_time = filtered_hits["time"].max()
    max_value_hits = filtered_hits["Cumulative_Label"].max()
    max_value_stress = filtered_force["Stress [MPa]"].max()

    return [max_value_time, max_value_hits, max_value_stress]


def reorder_hits(hits, y, y_pred):
    """
    Order hits and predicted class labels by ascending time.
    """
    y_pred_df = pd.Series(y_pred, name="Class")

    hits = hits.loc[y.index].reset_index(drop=True)
    hits = hits.drop("Class", axis=1)

    hits_and_pred = pd.concat([hits, y_pred_df], axis=1)
    hits_and_pred = hits_and_pred.sort_values(by="time", ascending=True)

    hits_ordered = hits_and_pred.drop("Class", axis=1)
    y_pred_ordered = hits_and_pred["Class"].copy()

    return hits_ordered, y_pred_ordered