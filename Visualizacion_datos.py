# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:10:11 2024

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Dataframes
import os

import pandas as pd

# Functions for dataset splitting
# Functions for result visualization
from mis_funciones.analisis_AE import plot_feat_vs_feat, train_test_set
from mis_funciones.force_mts import plot_force_hits

# =============================================================================
# Initial data
# =============================================================================
# Current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory to save images
figure_path = os.path.join(script_dir, "Figures")
os.makedirs(figure_path, exist_ok=True)

models_path = os.path.join(script_dir, "models")
datasets_path = os.path.join(script_dir, "datasets")

dbscan_model_path = os.path.join(models_path, "dbscan_model.joblib")
kmeans_model_path = os.path.join(models_path, "kmeans_model.joblib")

# Import datasets
data = pd.read_csv(os.path.join(datasets_path, "Dataset_total.csv"))
force = pd.read_csv(os.path.join(datasets_path, "Datos_MTS.csv"))

# =============================================================================
# Dataset splitting
# =============================================================================
X_original, X, y, hits = train_test_set(data, split=False)
test_ids = hits["test_id"].unique()

# =============================================================================
# Data visualization - Segment = 1
# =============================================================================
plot_feat_vs_feat(data, y, "w_peak_freq", "centroid_freq", figure_path)
plot_feat_vs_feat(data, y, "time_norm", "amplitude", figure_path)
plot_feat_vs_feat(data, y, "time_norm", "energy", figure_path, save=True)
plot_feat_vs_feat(data, y, "w_peak_freq", "energy", figure_path, save=True)

# Class 0
data_0 = data[data["Class"] == "[0_A]"]
y_0 = y[y == "[0_A]"]
plot_feat_vs_feat(data_0, y_0, "time_norm", "w_peak_freq", figure_path)

# Class 90
data_90 = data[data["Class"] == "[90_A]"]
y_90 = y[y == "[90_A]"]
plot_feat_vs_feat(data_90, y_90, "time_norm", "w_peak_freq", figure_path)

# Force and hits
for test_id in test_ids:
    plot_force_hits(y, hits, force, test_id, figure_path)

for test_id in test_ids:
    condition = data["test_id"] == test_id
    plot_feat_vs_feat(
        data[condition], y[condition], "time_norm", "w_peak_freq", figure_path
    )

# =============================================================================
# Data visualization - Segment = 3
# =============================================================================
plot_feat_vs_feat(data, y, "w_peak_freq_1", "centroid_freq_1", figure_path)
plot_feat_vs_feat(data, y, "time_norm", "w_peak_freq_1", figure_path)

# Class 0
data_0 = data[data["Class"] == "[0_A]"]
y_0 = y[y == "[0_A]"]
plot_feat_vs_feat(data_0, y_0, "time_norm", "w_peak_freq_1", figure_path)

# Class 90
data_90 = data[data["Class"] == "[90_A]"]
y_90 = y[y == "[90_A]"]
plot_feat_vs_feat(data_90, y_90, "time_norm", "w_peak_freq_1", figure_path)
