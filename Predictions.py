# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:52:46 2024

@author: bbarmac
"""
# =============================================================================
# Importamos las librerias necesarias
# =============================================================================
# Dataframes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os

# Funciones para division de datasets
from mis_funciones.analisis_AE import train_test_set

# Funciones para Modelo DBSCAN
from mis_funciones.no_supervisado import  tsne

# Funciones para graficar
from mis_funciones.no_supervisado import plot_cluster_feat
from mis_funciones.no_supervisado import plot_cluster_tsne
from mis_funciones.force_mts import plot_stress_hits, limit_finder, plot_stress_hits_cluster, limit_finder_no_label

# =============================================================================
# Datos iniciales
# =============================================================================
# Directorio actual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio para guardar imagenes
figure_path = os.path.join(base_dir, 'Figuras')
os.makedirs(figure_path, exist_ok=True)

# Directorio para guarta los resultados
results_path = os.path.join(base_dir, 'Results')
os.makedirs(results_path, exist_ok=True)

models_path = os.path.join(base_dir, 'models')
datasets_path = os.path.join(base_dir, 'datasets')

dbscan_model_path = os.path.join(models_path, 'dbscan_model.joblib')
kmeans_model_path = os.path.join(models_path, 'kmeans_model.joblib')

# Importacion de datasets
data = pd.read_csv(os.path.join(datasets_path, 'Dataset_total.csv'))
force = pd.read_csv(os.path.join(datasets_path, 'Datos_MTS.csv'))

# Separacion features labels
# X_original, X, y, hits = train_test_set(data, normalization='std', split=False)
X_original, X, y, hits = train_test_set(data, normalization='log-std', columns_to_transform=['energy'], split=False)
test_ids = hits['test_id'].unique()

# T-SNE
X_reduced = tsne(X, figure_path)

# =============================================================================
# Predicciones con modelo
# =============================================================================
dbscan_model = load(dbscan_model_path)
kmeans_model = load(kmeans_model_path)

labels_dbscan = dbscan_model.fit_predict(X_reduced)
labels_kmeans = kmeans_model.predict(X_reduced)

# Save the labels of DBSCAN and KMeans in CSV files
labels_dbscan_df = data.copy()
labels_dbscan_df['DBSCAN label'] = labels_dbscan

labels_kmeans_df = data.copy()
labels_kmeans_df['DBSCAN label'] = labels_dbscan

labels_dbscan_df.to_csv(os.path.join(results_path, 'labels_dbscan.csv'), index=False)
labels_kmeans_df.to_csv(os.path.join(results_path, 'labels_kmeans.csv'), index=False)

# Get unique values and their counts
unique_labels, counts = np.unique(labels_dbscan, return_counts=True)

# Display the results
for label, count in zip(unique_labels, counts):
    percentage  = count / np.sum(counts)
    print(f'Occurrences of {label}: {count}')
    print(f'Percentage: {percentage:.1%}')
    print("-----")

# =============================================================================
# Grafica de t-SNE       
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
    plot_cluster_tsne(labels[i-1], X_reduced, figure_path, subtitle=subtitle[i-1], ax=ax, i=i, n_col=n_col, n_row=n_row, guardar=False)

# =============================================================================
# Grafica Feat vs Feat
# =============================================================================
plot_cluster_feat(labels_dbscan, data, 'p_power_3', 'w_peak_freq', figure_path, 
                  width=90, height=60, title='Feature vs. feature scatter plot',
                  x_label='Partial power 3 [%]', y_label='Weighted peak frequency [kHz]', guardar=False)

# =============================================================================
# Graficas de fuerza vs hits - Clustering
# =============================================================================
test_id_prueba = ['P0_2', 'P90_3', 'P0_90_3', 'P0_90W_1', 'P45_4', 'PQ_3']
limits = limit_finder(labels_dbscan, hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (180, 115)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True)
plt.suptitle('Stress and Cumulative hits vs Time - Clustering', fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits_cluster(labels_dbscan, hits, force, test_id_prueba[i-1], figure_path, 
                    plot_type='line', limits=limits, y_label_right='\u03C3 [MPa]', 
                    ax=ax, i=i, n_col=n_col, n_row=n_row, guardar=True)    

# =============================================================================
# Graficas de fuerza vs hits 
# =============================================================================
test_id_prueba = ['P0_2', 'P90_3', 'P0_90_3', 'P0_90W_1', 'P45_4', 'PQ_3']
limits = limit_finder_no_label(hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (180, 115)
figsize_inches = (width / 25.4, height / 25.4)
# Create a new figure and set the size
fig, axes = plt.subplots(n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True)
plt.suptitle('Stress and Cumulative hits vs Time', fontsize=10)
    
# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits(hits, force, test_id_prueba[i-1], figure_path, 
                    plot_type='line', limits=limits, y_label_right='\u03C3 [MPa]', 
                    ax=ax, i=i, n_col=n_col, n_row=n_row, guardar=False)



