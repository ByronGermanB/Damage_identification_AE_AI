# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:10:11 2024

@author: bbarmac
"""

#%%
# =============================================================================
# Importamos las librerias necesarias
# =============================================================================
# Dataframes
import pandas as pd
import os

# Funciones para division de datas
from mis_funciones.analisis_AE import train_test_set

# Funciones para visualizacion de resultados
from mis_funciones.analisis_AE import plot_feat_vs_feat
from mis_funciones.force_mts import plot_force_hits

#%%
# =============================================================================
# Datos iniciales
# =============================================================================
# Directorio actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio para guardar imagenes
figure_path = os.path.join(script_dir, 'Figuras')

models_path = os.path.join(script_dir, 'models')
datasets_path = os.path.join(script_dir, 'datasets')

dbscan_model_path = os.path.join(models_path, 'dbscan_model.joblib')
kmeans_model_path = os.path.join(models_path, 'kmeans_model.joblib')


# Importacion de datasets
data = pd.read_csv(os.path.join(datasets_path, 'Dataset_total.csv'))
force = pd.read_csv(os.path.join(datasets_path, 'Datos_MTS.csv'))

#%%
# =============================================================================
# Division dataset
# =============================================================================
X_original, X, y, hits = train_test_set(data, split=False)
test_ids = hits['test_id'].unique()

#%%
# =============================================================================
# Visualizacion de datos - Segmento = 1
# =============================================================================
plot_feat_vs_feat(data, y, 'w_peak_freq', 'centroid_freq', figure_path)
plot_feat_vs_feat(data, y, 'time_norm', 'amplitude', figure_path)
plot_feat_vs_feat(data, y, 'time_norm', 'energy', figure_path, guardar=True)
plot_feat_vs_feat(data, y, 'w_peak_freq', 'energy', figure_path, guardar=True)

# Clase 0
data_0 = data[data['Clase'] == '[0_A]']
y_0 = y[y == '[0_A]']
plot_feat_vs_feat(data_0, y_0, 'time_norm', 'w_peak_freq', figure_path)

# Clase 90
data_90 = data[data['Clase'] == '[90_A]']
y_90 = y[y == '[90_A]']
plot_feat_vs_feat(data_90, y_90, 'time_norm', 'w_peak_freq', figure_path)

# Fuerza y hits
for test_id in test_ids:
    plot_force_hits(y, hits, force, test_id, figure_path)

for test_id in test_ids:
    condition = data['test_id'] == test_id
    plot_feat_vs_feat(data[condition], y[condition], 'time_norm', 'w_peak_freq', figure_path)
#%%
# =============================================================================
# Visualizacion de datos - Segmento = 3
# =============================================================================
plot_feat_vs_feat(data, y, 'w_peak_freq_1', 'centroid_freq_1', figure_path)
plot_feat_vs_feat(data, y, 'time_norm', 'w_peak_freq_1', figure_path)

# Clase 0
data_0 = data[data['Clase'] == '[0_A]']
y_0 = y[y == '[0_A]']
plot_feat_vs_feat(data_0, y_0, 'time_norm', 'w_peak_freq_1', figure_path)

# Clase 90
data_90 = data[data['Clase'] == '[90_A]']
y_90 = y[y == '[90_A]']
plot_feat_vs_feat(data_90, y_90, 'time_norm', 'w_peak_freq_1', figure_path)
