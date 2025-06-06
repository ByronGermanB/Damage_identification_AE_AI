# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:52:46 2024

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Dataframes
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# Functions for dataset splitting
from mis_funciones.analisis_AE import train_test_set
from mis_funciones.force_mts import (
    limit_finder,
    limit_finder_no_label,
    plot_stress_hits,
    plot_stress_hits_cluster,
)

# Functions for DBSCAN model
# Functions for plotting
from mis_funciones.no_supervisado import (
    plot_cluster_feat,
    plot_cluster_tsne,
    plot_cluster_tsne_3d,
    tsne,
    tsne_3d,
)

# =============================================================================
# Initial data
# =============================================================================
# Current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Directory to save images
figure_dir = os.path.join(base_dir, "Figures")
os.makedirs(figure_dir, exist_ok=True)

# Directory to save results
results_dir = os.path.join(base_dir, "Results")
os.makedirs(results_dir, exist_ok=True)

models_dir = os.path.join(base_dir, "models")
datasets_dir = os.path.join(base_dir, "datasets")
images_dir = os.path.join(datasets_dir, "images", "test")

# Import datasets
data = pd.read_csv(os.path.join(datasets_dir, "Dataset_total.csv"))
force = pd.read_csv(os.path.join(datasets_dir, "Datos_MTS.csv"))

# Separate features and labels
# X_original, X, y, hits = train_test_set(data, normalization='std', split=False)
X_original, X, y, hits = train_test_set(
    data, normalization="log-std", columns_to_transform=["energy"], split=False
)
test_ids = hits["test_id"].unique()

# T-SNE
X_reduced = tsne_3d(X, figure_dir)
X_reduced = tsne(X, figure_dir)

# =============================================================================
# Import images
# =============================================================================
# Set the parameters
target_size = (128, 128)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_data = test_datagen.flow_from_directory(
    images_dir,
    color_mode="grayscale",
    target_size=target_size,
    batch_size=batch_size,
    shuffle=False,
    class_mode=None,
)

# =============================================================================
# Predictions with model
# =============================================================================
# Load the saved model
model_name = "CNN_model_2"
model_path = os.path.join(models_dir, model_name + ".keras")
model = load_model(model_path)

# Get predictions
predictions = model.predict(test_data)
# Round predictions to 0 or 1
labels_cnn = np.round(predictions).astype(int)
# Flatten the predictions array
labels_cnn = labels_cnn.flatten()

# Save predictions to CSV
labels_cnn_df = data.copy()
labels_cnn_df["CNN label"] = labels_cnn

# Save the new data DataFrame with predictions
labels_cnn_df.to_csv(os.path.join(results_dir, f"labels_{model_name}.csv"), index=False)

# =============================================================================
# T-SNE plot
# =============================================================================
# List of labels and subtitle
labels_dbscan = pd.read_csv(os.path.join(results_dir, "labels_dbscan.csv"))
labels_dbscan = labels_dbscan["DBSCAN label"]

labels = [labels_cnn, labels_dbscan]
subtitle = ["CNN", "DBSCAN"]

plot_cluster_tsne_3d(
    labels_dbscan,
    X_reduced,
    figure_dir,
    subtitle=subtitle,
    title=f"t-SNE Clustering 3D- {model_name}",
    guardar=False,
)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 1)

# Image size in mm
width, height = (90, 130)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
title = f"t-SNE Clustering Comparison - {model_name}"

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    plot_cluster_tsne(
        labels[i - 1],
        X_reduced,
        figure_dir,
        subtitle=subtitle[i - 1],
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        guardar=False,
        title=title,
    )
plt.show()
# =============================================================================
# Feature vs Feature plot
# =============================================================================
plot_cluster_feat(
    labels_cnn,
    data,
    "p_power_3",
    "w_peak_freq",
    figure_dir,
    width=90,
    height=60,
    title=f"Feature vs. feature scatter plot - {model_name}",
    x_label="Partial power 3 [%]",
    y_label="Weighted peak frequency [kHz]",
    guardar=False,
)

# =============================================================================
# Feature vs Feature plot separated by cluster (0 and 1)
# =============================================================================
# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
title = f"Feature vs. feature scatter plot - {model_name}"

for i, ax in enumerate(axes.flat, start=1):
    plot_cluster_feat(
        labels_cnn[labels_cnn == i - 1],
        data[labels_cnn == i - 1],
        "p_power_3",
        "w_peak_freq",
        figure_dir,
        width=90,
        height=60,
        title=title,
        x_label="Partial power 3 [%]",
        y_label="Weighted peak frequency [kHz]",
        guardar=False,
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
    )

plt.show()

# =============================================================================
# Stress vs hits plots - Clustering
# =============================================================================
test_id_prueba = ["P0_2", "P90_3", "P0_90_3", "P0_90W_1", "P45_4", "PQ_3"]
limits = limit_finder(labels_cnn, hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (180, 115)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
title = f"Stress and Cumulative hits vs Time - Clustering {model_name}"
plt.suptitle(title, fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits_cluster(
        labels_cnn,
        hits,
        force,
        test_id_prueba[i - 1],
        figure_dir,
        plot_type="line",
        limits=limits,
        y_label_right="\u03c3 [MPa]",
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        guardar=False,
        title=title,
    )
plt.show()

# =============================================================================
# Stress vs hits plots
# =============================================================================
test_id_prueba = ["P0_2", "P90_3", "P0_90_3", "P0_90W_1", "P45_4", "PQ_3"]
limits = limit_finder_no_label(hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (180, 115)
figsize_inches = (width / 25.4, height / 25.4)
# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Stress and Cumulative hits vs Time", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits(
        hits,
        force,
        test_id_prueba[i - 1],
        figure_dir,
        plot_type="line",
        limits=limits,
        y_label_right="\u03c3 [MPa]",
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        guardar=False,
    )
plt.show()
