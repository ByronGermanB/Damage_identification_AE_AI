# Dataframes
import os

import matplotlib.pyplot as plt
import pandas as pd

from utils.force_mts import plot_feature_time_histogram, plot_feature_histogram


# Directories
base_dir = os.path.dirname(os.path.abspath(__file__))

datasets_path = os.path.join(base_dir, "datasets")
figures_path = os.path.join(base_dir, "Figures")

# Import datasets
data = pd.read_csv(os.path.join(datasets_path, "dataset_total.csv"))

# List of features to plot
features = [
    "peak_freq",
    "w_peak_freq",
    "centroid_freq",
    "p_power_1",
    "p_power_2",
    "p_power_3",
]

y_labels = [
    "Peak Frequency [kHz]",
    "Weighted Peak Frequency [kHz]",
    "Centroid Frequency [kHz]",
    "Partial Power 1 [%]",
    "Partial Power 2 [%]",
    "Partial Power 3 [%]",
]

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (180, 115)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Feature distribution vs Normalized time", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):

    plot_feature_time_histogram(
        data,
        figures_path,
        title="Feature distribution vs Normalized time",
        x="time_norm",
        y=features[i - 1],
        y_label=y_labels[i - 1],
        ax=ax,
        n_col=n_col,
        i=i,
        save=True,
    )
plt.show()

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Feature distribution histogram", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):

    plot_feature_histogram(
        data,
        figures_path,
        title="Feature distribution histogram",
        x=features[i - 1],
        ax=ax,
        x_label=y_labels[i - 1],
        n_col=n_col,
        i=i,
        save=True,
    )
plt.show()

print("Feature vs Time plot generated successfully.")