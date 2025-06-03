# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:09:45 2024

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Dataframes
import os
import string

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Figures
import seaborn as sns

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["figure.dpi"] = 150


def force_data(path, section=[22, 2.4]):
    """
    Parameters
    ----------
    path : str
        Directory where the force mts csv files are stored.
    section : list or array
        Width and thickness measurements of the specimen in mm. The default is [22, 2.4]

    Returns
    -------
    force_df : Dataframe
        Dataframe containing force, displacement, time, and test_id data.

    """
    files = os.listdir(path)
    forces_list = []

    for item in files:
        # Check if the files are .csv
        if item.endswith(".csv"):
            # Path of the folder joined with the file
            force_filename_path = os.path.join(path, item)

            # Load the csv file
            force = pd.read_csv(force_filename_path, sep=";")

            # Rename the columns and remove the units row
            new_column_names = ["Crosshead [mm]", "Load [kN]", "Time [s]"]
            force.columns = new_column_names
            force = force.iloc[1:]

            # Add a row of 0 where the test starts
            new_row = pd.Series(0, index=force.columns)
            force = pd.concat([pd.DataFrame([new_row]), force], ignore_index=True)

            # Change all the , to .
            force = force.replace({",": "."}, regex=True)
            force = force.apply(pd.to_numeric, errors="coerce")

            # Stress
            area = np.prod(section)
            force["Stress [MPa]"] = force["Load [kN]"] * 1000 / area

            # Add the test id column
            test_id = item.split(".")[0]
            force["test_id"] = test_id
            forces_list.append(force)

    force_df = pd.concat(forces_list, ignore_index=True)
    return force_df


def plot_stress_hits_cluster(
    labels,
    hits,
    force,
    test_id,
    figure_path,
    plot_type="scatter",
    title="Stress and Cumulative hits vs Time - Clustering",
    x="time",
    y="Cumulative_Label",
    x_label="Time [s]",
    y_label="Cumulative hits",
    y_label_right="Stress [MPa]",
    subtitle=None,
    limits=None,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    n_row=1,
    save=False,
):
    """
    Parameters
    ----------
    labels : array
        Labels of the clusters obtained by k-means or dbscan.
    hits : Dataframe
        Set containing the characteristics over time of the hits and their id.
    force : Dataframe
        Set containing load displacement, time data, and their id.
    test_id : str
        Id of the test to be plotted.
    figure_path : str
        Directory to save the figure.
    plot_type : str, optional
        Type of plot to create ('scatter' or 'line'). The default is 'scatter'.
    title : str, optional
        Title of the image to display and save. The default is 'Cumulative hits vs Time'.
    x : str, optional
        Time characteristic for the x-axis. The default is 'time'.

        Other options: 'Cumulative_Label', 'Cumulative', 'amplitude', 'energy'
    y : str, optional
        Time characteristic for the y-axis. The default is 'Cumulative_Label'.

        Other options: 'time', 'Cumulative', 'amplitude', 'energy'
    x_label : str, optional
        Label for the x-axis. The default is 'Time [s]'.
    y_label : str, optional
        Label for the y-axis. The default is 'Cumulative hits'.
    y_label_right : str, optional
        Label for the right y-axis. The default is 'Stress [MPa]'.
    subtitle : str or list, optional
        Title for each subplot, if left as default the test_id is plotted. The default is None.
    limits : list, optional
        Upper limits for the axes [time_limit, hits_limit, stress_limit]. The default is None.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    ax : matplotlib axis object, optional
        Axis to plot on. If not provided, a new axis will be created.
    i : int, optional
        Index for subplot. Default is 1.
    n_col : int, optional
        Number of columns. The default is 1.
    n_row : int, optional
        Number of rows. The default is 1.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. Cluster vs Time {test_id}.pdf

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Plots and (saves) a diagram of cumulative hits and stress in pdf format.
    """
    # Extract the limits
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    # Create a DataFrame with the labels
    labels_df = pd.Series(labels, name="Labels")
    labels_df.index = hits.index

    # Concatenate the new DataFrame with the existing one
    hits_labels = pd.concat([hits, labels_df], axis=1)

    # Condition to filter rows based on the string column
    condition = hits_labels["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    # Filtered DataFrame
    filtered_hits = hits_labels.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = filtered_hits.groupby("Labels")[
        "Count"
    ].cumsum()

    # Force data
    filtered_force = force[condition_force]

    # Count the number of unique labels in the DataFrame
    num_labels = filtered_hits["Labels"].nunique()

    # Define custom markers for each class
    markers = ["o", "s", "^", "v", "D", "p"]  # Customize the markers as desired
    palette = sns.color_palette()
    # Customize the markers list to match the number of unique labels
    markers = markers[:num_labels]
    palette = palette[:num_labels]

    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_title(f"Stress and Cumulative hits vs Time - {test_id}")
        ax.set_xlabel(x_label)

    else:
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(test_id)

    # Create the scatter plot
    if plot_type == "scatter":
        sns.scatterplot(
            x=x,
            y=y,
            hue="Labels",
            style="Labels",
            data=filtered_hits,
            markers=markers,
            palette=palette,
            linewidth=0.2,
            s=20,
            ax=ax,
        )
    elif plot_type == "line":
        sns.lineplot(
            x=x,
            y=y,
            hue="Labels",
            style="Labels",
            data=filtered_hits,
            palette=palette,
            linewidth=2,
            ax=ax,
        )

    # Set the x-axis limit
    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    # Set the y-axis limit
    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)

    # Create a twin Axes sharing the x-axis
    ax2 = ax.twinx()

    # Create the second scatter plot on the twin Axes
    sns.lineplot(
        x="Time [s]",
        y="Stress [MPa]",
        data=filtered_force,
        ax=ax2,
        linewidth=1,
        color="black",
        label="Stress",
    )

    # Set the y-axis limit for the second plot
    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force["Stress [MPa]"]) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)

    # Set y-axis label only for the last plot of each row
    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel("")

    # Combine legends
    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    # Legend in the first plot
    if i == 1:
        # Create a new legend with combined handles and legends
        ax2.legend(handles, legends, loc="upper left")

        # Remove the separate legend from ax if it exists
        ax.get_legend().remove()
    else:
        ax.get_legend().remove()
        ax2.get_legend().remove()

    # Plot the grid
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        if ax is None:
            figure_filename = f"{title} - {test_id}.pdf"
        else:
            figure_filename = f"{title}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    return ax


def plot_stress_hits(
    hits,
    force,
    test_id,
    figure_path,
    plot_type="scatter",
    title="Stress and Cumulative hits vs Time",
    x="time",
    y="Cumulative_Label",
    x_label="Time [s]",
    y_label="Cumulative hits",
    y_label_right="Stress [MPa]",
    subtitle=None,
    limits=None,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    n_row=1,
    save=False,
):
    """
    Parameters
    ----------
    hits : Dataframe
        Set containing the characteristics over time of the hits and their id.
    force : Dataframe
        Set containing load displacement, time data, and their id.
    test_id : str
        Id of the test to be plotted.
    figure_path : str
        Directory to save the figure.
    plot_type : str, optional
        Type of plot to create ('scatter' or 'line'). The default is 'scatter'.
    title : str, optional
        Title of the image to display and save. The default is 'Stress and Cumulative hits vs Time'.
    x : str, optional
        Time characteristic for the x-axis. The default is 'time'.

        Other options: 'Cumulative_Label', 'Cumulative', 'amplitude', 'energy'
    y : str, optional
        Time characteristic for the y-axis. The default is 'Cumulative_Label'.

        Other options: 'time', 'Cumulative', 'amplitude', 'energy'
    x_label : str, optional
        Label for the x-axis. The default is 'Time [s]'.
    y_label : str, optional
        Label for the y-axis. The default is 'Cumulative hits'.
    y_label_right : str, optional
        Label for the right y-axis. The default is 'Stress [MPa]'.
    subtitle : str or list, optional
        Title for each subplot, if left as default the test_id is plotted. The default is None.
    limits : list, optional
        Upper limits for the axes [time_limit, hits_limit, stress_limit]. The default is None.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    ax : matplotlib axis object, optional
        Axis to plot on. If not provided, a new axis will be created.
    i : int, optional
        Index for subplot. Default is 1.
    n_col : int, optional
        Number of columns. The default is 1.
    n_row : int, optional
        Number of rows. The default is 1.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. Cluster vs Time {test_id}.pdf

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Plots and (saves) a diagram of cumulative hits and stress in pdf format.
    """
    # Extract the limits
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    # Condition to filter rows based on the string column
    condition = hits["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    # Filtered DataFrame
    filtered_hits = hits.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = filtered_hits["Count"].cumsum()

    # Force data
    filtered_force = force[condition_force]

    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_title(f"Stress and Cumulative hits vs Time - {test_id}")
        ax.set_xlabel(x_label)

    else:
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(test_id)

    # Create the scatter plot
    if plot_type == "scatter":
        sns.scatterplot(
            x=x, y=y, data=filtered_hits, label="hits", linewidth=0.2, s=20, ax=ax
        )
    elif plot_type == "line":
        sns.lineplot(x=x, y=y, label="hits", data=filtered_hits, linewidth=2, ax=ax)

    # Set the x-axis limit
    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    # Set the y-axis limit
    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)

    # Create a twin Axes sharing the x-axis
    ax2 = ax.twinx()

    # Create the second scatter plot on the twin Axes
    sns.lineplot(
        x="Time [s]",
        y="Stress [MPa]",
        data=filtered_force,
        ax=ax2,
        linewidth=1,
        color="black",
        label="Stress",
    )

    # Set the y-axis limit for the second plot
    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force["Stress [MPa]"]) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)

    # Set y-axis label only for the last plot of each row
    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel("")

    # Combine legends
    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    # Legend in the first plot
    if i == 1:
        # Create a new legend with combined handles and legends
        ax2.legend(handles, legends, loc="upper left")

        # Remove the separate legend from ax if it exists
        ax.get_legend().remove()
    else:
        ax.get_legend().remove()
        ax2.get_legend().remove()

    # Plot the grid
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        if ax is None:
            figure_filename = f"{title} - {test_id}.pdf"
        else:
            figure_filename = f"{title}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    return ax


def limit_finder(labels, hits, force, test_ids):
    """
    Parameters
    ----------
    labels : array
        Labels of the clusters obtained by k-means or dbscan.
    hits : Dataframe
        Dataframe containing the characteristics over time of the hits and their id.
    force : Dataframe
        Dataframe containing load displacement, time data, and their id.
    test_ids : list
        Ids of the tests considered for the calculation.

    Returns
    -------
    max_value_hits : float
        Maximum value for hits.
    max_value_stress : float
        Maximum value for stress.

    """
    # Create a DataFrame with the labels
    labels_df = pd.Series(labels, name="Labels")
    labels_df.index = hits.index

    # Concatenate the new DataFrame with the existing one
    hits_labels = pd.concat([hits, labels_df], axis=1)

    # Condition to filter rows based on the string column
    condition = hits_labels["test_id"].isin(test_ids)
    condition_force = force["test_id"].isin(test_ids)

    # Filtered DataFrame
    filtered_hits = hits_labels.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = None

    for test_id in test_ids:
        single_condition = filtered_hits["test_id"] == test_id
        filtered_hits.loc[single_condition, "Cumulative_Label"] = (
            filtered_hits[single_condition].groupby("Labels")["Count"].cumsum()
        )

    # Force data
    filtered_force = force[condition_force]

    # Maximum values of the df
    max_value_time = filtered_hits["time"].max()
    max_value_hits = filtered_hits["Cumulative_Label"].max()
    max_value_stress = filtered_force["Stress [MPa]"].max()

    return [max_value_time, max_value_hits, max_value_stress]


def limit_finder_no_label(hits, force, test_ids):
    """
    Parameters
    ----------
    hits : Dataframe
        Dataframe containing the characteristics over time of the hits and their id.
    force : Dataframe
        Dataframe containing load displacement, time data, and their id.
    test_ids : list
        Ids of the tests considered for the calculation.

    Returns
    -------
    max_value_hits : float
        Maximum value for hits.
    max_value_stress : float
        Maximum value for stress.

    """
    # Condition to filter rows based on the string column
    condition = hits["test_id"].isin(test_ids)
    condition_force = force["test_id"].isin(test_ids)

    # Filtered DataFrame
    filtered_hits = hits.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = None

    for test_id in test_ids:
        single_condition = filtered_hits["test_id"] == test_id
        filtered_hits.loc[single_condition, "Cumulative_Label"] = filtered_hits[
            single_condition
        ]["Count"].cumsum()

    # Force data
    filtered_force = force[condition_force]

    # Maximum values of the df
    max_value_time = filtered_hits["time"].max()
    max_value_hits = filtered_hits["Cumulative_Label"].max()
    max_value_stress = filtered_force["Stress [MPa]"].max()

    return [max_value_time, max_value_hits, max_value_stress]


def reorder_hits(hits, y, y_pred):
    """
    Parameters
    ----------
    hits : Dataframe
        Dataframe for plotting cumulative hits.
    y : series or array
        Classes.
    y_pred : series or array
        Predictions.

    Returns
    -------
    hits_ordered : Dataframe
        Hits ordered by ascending time.
    y_pred_ordered : series
        Predictions ordered by ascending time.

    """
    # Create a DataFrame with the predictions
    y_pred_df = pd.Series(y_pred, name="Class")

    # Select only the hits from the training or test set
    hits = hits.loc[y.index]
    hits = hits.reset_index(drop=True)

    # Remove the class column
    hits = hits.drop("Class", axis=1)

    # Concatenate hits and predictions dataframes
    hits_and_pred = pd.concat([hits, y_pred_df], axis=1)

    # Sort dataframe based on the time column
    hits_and_pred = hits_and_pred.sort_values(by="time", ascending=True)

    # Separate hits and predictions
    hits_ordered = hits_and_pred.drop("Class", axis=1)
    y_pred_ordered = hits_and_pred["Class"].copy()

    return hits_ordered, y_pred_ordered
