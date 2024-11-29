# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:31:23 2024

@author: bbarmac
"""

# =============================================================================
# Importamos las librerias necesarias
# =============================================================================
# Directorios
import os
# Funciones para extraccion de features
from mis_funciones.analisis_AE import Features

# =============================================================================
# Datos iniciales
# =============================================================================
# Directorio actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio con datos de EA
EA_path = os.path.join(script_dir, 'Datos_EA')
# Directorio para guardar imagenes
figure_path = os.path.join(script_dir, 'Figuras')
os.makedirs(figure_path, exist_ok=True)

# Directorio para lectura de datos EA
test_id = 'MINA_P0_90_2'
data = os.path.join(EA_path, test_id)  

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

# =============================================================================
# Extraccion de features
# =============================================================================
# Inicializacion de clases
trail = Features(data, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0)
trail_df = trail.feature_extr(umbral, counts, clase='[trial]', test_id=test_id)

trai = 3
trail.plot_signal(trai=trai, name_figure='PLB', time_graph=260, figure_path=figure_path, width=90, height=70, guardar=False)
trail.plot_segmentation_1(trai=trai, name_figure='PLB', figure_path=figure_path, width=90, height=125, guardar=True)
