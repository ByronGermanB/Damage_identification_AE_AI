# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:18:58 2023

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Dataframes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
import os

# Functions for dataset splitting
from mis_funciones.analisis_AE import train_test_set

# Functions for DBSCAN model
from mis_funciones.no_supervisado import grid_search_dbscan, kmeans_per_k, tsne

# Functions for plotting
from mis_funciones.no_supervisado import plot_kmeans_per_k, plot_cluster_feat, plot_cluster_time, plot_cluster_hits 
from mis_funciones.no_supervisado import plot_cluster_tsne, plot_dbi
from mis_funciones.force_mts import plot_stress_hits, limit_finder, plot_stress_hits_cluster, limit_finder_no_label

# =============================================================================
# Initial data
# =============================================================================
# Current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory to save images
figure_path = os.path.join(script_dir, 'Figuras')

# Import datasets
datasets_path = os.path.join(script_dir, 'datasets')
data = pd.read_csv(os.path.join(datasets_path, 'Dataset_total.csv'))
force = pd.read_csv(os.path.join(datasets_path, 'Datos_MTS.csv'))

# Separate features and labels
# X_original, X, y, hits = train_test_set(data, normalization='std', split=False)
X_original, X, y, hits = train_test_set(data, normalization='log-std', columns_to_transform=['energy'], split=False)
test_ids = hits['test_id'].unique()

# T-SNE
X_reduced = tsne(X, figure_path)

# List with dbscan models for k = 2 3 4 5 
dbscan_per_k = []
dbscan_results_per_k = []

# =============================================================================
# DBSCAN
# =============================================================================
# Parameters for DBSCAN
target_clusters = 2  # Specify the desired number of clusters
epsilon_values = np.linspace(1, 10, 10) 
min_samples_values = np.linspace(5, 20, 4, dtype=int)

# Model and adjusted parameters
dbscan_models, results = grid_search_dbscan(X_reduced, target_clusters, epsilon_values, min_samples_values)

# =============================================================================
# DBSCAN by k
# =============================================================================
# Index of the chosen model
index = -2

# Add to the lists
dbscan_per_k.append(dbscan_models[index])
dbscan_results_per_k.append(results[index])

# =============================================================================
# DBSCAN - DBI Scores
# =============================================================================
dbi_dbscan = []

for i in range(len(dbscan_results_per_k)):
    dbi_dbscan.append(dbscan_results_per_k[i]['DBI Score'])

# =============================================================================
# DBSCAN - Chosen model
# =============================================================================
dbscan_model = dbscan_per_k[0]
labels_dbscan = dbscan_model.labels_

# Get unique values and their counts
unique_labels, counts = np.unique(labels_dbscan, return_counts=True)

# Display the results
for label, count in zip(unique_labels, counts):
    percentage  = count / np.sum(counts)
    print(f'Occurrences of {label}: {count}')
    print(f'Percentage: {percentage:.1%}')
    print("-----")

# =============================================================================
# K-means by k
# =============================================================================
k = 6 # Maximum number of clusters
kmeans_models, silhouette_scores, dbi_kmeans = kmeans_per_k(X_reduced, k)
plot_kmeans_per_k(X_reduced, kmeans_models, silhouette_scores, figure_path)

# =============================================================================
# K-means - Chosen model
# =============================================================================
kmeans_model = kmeans_models[0]
labels_kmeans = kmeans_model.labels_

# =============================================================================
# DBI Plot
# =============================================================================
plot_dbi(dbi_kmeans, dbi_dbscan, figure_path, save=False) # Needs to have been executed for different target_clusters
# The same number of clusters as kmeans

# =============================================================================
# t-SNE Plot       
# =============================================================================
# List of labels and subtitle
labels = [labels_kmeans, labels_dbscan]
subtitle = ['k-means', 'DBSCAN']

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 1)

# Image size in mm
width, height = (90, 130)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True)
plt.suptitle('t-SNE Clustering Comparison', fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    plot_cluster_tsne(labels[i-1], X_reduced, figure_path, subtitle=subtitle[i-1], ax=ax, i=i, n_col=n_col, n_row=n_row, save=False)
    

# =============================================================================
# Force vs hits plots - Clustering
# =============================================================================
test_id_prueba = ['P0_3', 'P90_3', 'P0_90_3', 'P0_90W_1', 'P45_4', 'PQ_3']
hits_limit, stress_limit = limit_finder(labels_dbscan, hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (190, 110)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True)
plt.suptitle('Stress and Cumulative hits vs Time - Clustering', fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits_cluster(labels_dbscan, hits, force, test_id_prueba[i-1], figure_path, 
                    plot_type='line', hits_limit=hits_limit, stress_limit=stress_limit, 
                    ax=ax, i=i, n_col=n_col, n_row=n_row, save=False)    

# =============================================================================
# Force vs hits plots 
# =============================================================================
test_id_prueba = ['P0_3', 'P90_3', 'P0_90_3', 'P0_90W_1', 'P45_4', 'PQ_3']
hits_limit, stress_limit = limit_finder_no_label(hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (190, 110)
figsize_inches = (width / 25.4, height / 25.4)
# Create a new figure and set the size
fig, axes = plt.subplots(n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True)
plt.suptitle('Stress and Cumulative hits vs Time', fontsize=10)
    
# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits(hits, force, test_id_prueba[i-1], figure_path, 
                    plot_type='line', hits_limit=hits_limit, stress_limit=stress_limit, 
                    ax=ax, i=i, n_col=n_col, n_row=n_row, save=False)


# Create a new figure and set the size
fig, axes = plt.subplots(n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True)
plt.suptitle('Feature vs Feature Plot', fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_cluster_feat(labels_dbscan, data, 'time_norm', 'w_peak_freq', figure_path, x_label='test',
                      ax=ax, i=i, n_col=n_col, n_row=n_row, save=False)

# =============================================================================
# Other plots
# =============================================================================
# DBSCAN
plot_cluster_feat(labels_dbscan, data, 'p_power_1', 'w_peak_freq', figure_path)

for test_id in test_ids:
    plot_stress_hits_cluster(labels_dbscan, hits, force, test_id, figure_path)
    plot_stress_hits(hits, force, test_id, figure_path, plot_type='line')
    
for test_id in test_ids:
    condition = data['test_id'] == test_id
    plot_cluster_feat(labels_dbscan[condition], data[condition], 'time_norm', 'w_peak_freq', figure_path)

# kmeans
plot_cluster_tsne(labels_kmeans, X_reduced, figure_path)
plot_cluster_feat(labels_kmeans, X, 'p_power_3', 'w_peak_freq', figure_path)
for test_id in test_ids:
    plot_cluster_hits(labels_kmeans, hits, test_id, figure_path)
for test_id in test_ids:
    plot_cluster_time(labels_kmeans, hits, test_id, figure_path)
    
# =============================================================================
# Save Model
# =============================================================================
models_path = os.path.join(script_dir, 'models')
dbscan_path = os.path.join(models_path, 'dbscan_model.joblib')
kmeans_path = os.path.join(models_path, 'kmeans_model.joblib')

dump(dbscan_model, dbscan_path)
print('Dbscan model saved at: "./models/dbscan_model.joblib"')

dump(kmeans_model, kmeans_path)
print('k-means model saved at: "./models/kmeans_model.joblib"')

# =============================================================================
# Change class label according to clustering
# =============================================================================
data['Class'] = labels_dbscan
labeled_dataset_name = 'Dataset_0_y_90_labeled.csv'
data.to_csv(os.path.join(datasets_path, labeled_dataset_name), index=False)
print(f'Dataset labeled saved at: "./dataset/{labeled_dataset_name}"')  

# =============================================================================
# Filtered hits
# =============================================================================
hits_filtered = hits.drop(['Class', 'amplitude', 'energy', 'time_norm', 'counts'], axis=1).copy()
hits_filtered.to_csv('hits_filtered.csv', index=False)
