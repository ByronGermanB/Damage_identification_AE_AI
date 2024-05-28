# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:31:23 2024

@author: bbarmac
"""

#%%
# =============================================================================
# Importamos las librerias necesarias
# =============================================================================
# Directorios
import os
# Funciones para extraccion de features
from mis_funciones.analisis_AE import Features

#%%
# =============================================================================
# Datos iniciales
# =============================================================================
# Directorio actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio con datos de EA
EA_path = os.path.join(script_dir, 'Datos_EA')
# Directorio para guardar imagenes
figure_path = os.path.join(script_dir, 'Figuras')

# Directorio para lectura de datos EA
path_mina = os.path.join(EA_path, 'MINA_P0_90_2')  

# Constantes1
lower_freq = 0  # Limite inferior de frecuencia en kHz
upper_freq = 900 # Limite superior de frecuencia en kHz
sampling_rate = 5.0 # Frecuencia de adquisicion de datos por segundo en MHz
N_samp = 1024   # Numero de muestras (datos) en cada segmento
N_seg = 1  # Numero de segmentos (si se cambia se debe cambiar los nombres de las columnas)
desfase = 200 # El desfase va de [0 a pretrigger*sampling rate] [0-250]

# Valores de filtrado
umbral = 36
counts = 5

#%%
# =============================================================================
# Extraccion de features
# =============================================================================
# Inicializacion de clases
mina = Features(path_mina, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0)
mina_df = mina.feature_extr(umbral, counts, clase='[mina]', test_id='mina')

mina.plot_signal(trai=3, name_figure='PLB', time_graph=260, figure_path=figure_path, width=90, height=70, guardar=False)
mina.plot_segmentation_1(trai=3, name_figure='PLB', figure_path=figure_path, width=90, height=125, guardar=False)
