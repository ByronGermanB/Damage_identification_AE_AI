# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:48:58 2023

@author: bbarmac
"""

# =============================================================================
# Importamos las librerias necesarias
# =============================================================================
# Directorios
import os
# Funciones para extraccion de features
from mis_funciones.analisis_AE import Features, unir_df

# Funciones para cargar datod de MTS
from mis_funciones.force_mts import force_data

# =============================================================================
# Datos iniciales
# =============================================================================
# Directorio actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio con datos de EA
EA_path = os.path.join(script_dir, 'Datos_EA')

# Directorios de ensayos
path_P0_1 = os.path.join(EA_path, 'P0_1-1')  # Es la carpeta P0_1-1 cambio de nombre por comodidad
path_P0_2 = os.path.join(EA_path,'P0_2')
path_P0_3 = os.path.join(EA_path, 'P0_3')
path_P0_4 = os.path.join(EA_path, 'P0_4')
path_P0_5 = os.path.join(EA_path, 'P0_5')
path_P0_6 = os.path.join(EA_path, 'P0_6')
path_P90_1 = os.path.join(EA_path, 'P90_1')
path_P90_2 = os.path.join(EA_path, 'P90_2')
path_P90_3 = os.path.join(EA_path, 'P90_3')
path_P90_4 = os.path.join(EA_path, 'P90_4')

path_P0_90_1 = os.path.join(EA_path, 'P0_90_1')
path_P0_90_2 = os.path.join(EA_path, 'P0_90_2')
path_P0_90_3 = os.path.join(EA_path, 'P0_90_3')
path_P0_90_4 = os.path.join(EA_path, 'P0_90_4')
path_P0_90W_1 = os.path.join(EA_path, 'P0_90W_1')
path_P0_90W_3 = os.path.join(EA_path, 'P0_90W_3')
path_P45_2 = os.path.join(EA_path, 'P45_2')
path_P45_3 = os.path.join(EA_path, 'P45_3')
path_P45_4 = os.path.join(EA_path, 'P45_4')
path_PQ_1 = os.path.join(EA_path, 'PQ_1')
path_PQ_2 = os.path.join(EA_path, 'PQ_2')
path_PQ_3 = os.path.join(EA_path, 'PQ_3')
path_PQ_4 = os.path.join(EA_path, 'PQ_4')

path_P0_A1 = os.path.join(EA_path,'P0_A1')
path_P0_A2 = os.path.join(EA_path,'P0_A2')
path_P0_A3 = os.path.join(EA_path,'P0_A3')
path_P0_A4 = os.path.join(EA_path,'P0_A4')
path_P90_A1 = os.path.join(EA_path,'P90_A1')
path_P90_A2 = os.path.join(EA_path,'P90_A2')
path_P90_A3 = os.path.join(EA_path,'P90_A3')
path_P90_A4 = os.path.join(EA_path,'P90_A4')

# Constantes1
lower_freq = 0  # Limite inferior de frecuencia en kHz
upper_freq = 900 # Limite superior de frecuencia en kHz
sampling_rate = 5.0 # Frecuencia de adquisicion de datos por segundo en MHz
N_samp = 1024   # Numero de muestras (datos) en cada segmento
N_seg = 1  # Numero de segmentos (si se cambia se debe cambiar los nombres de las columnas)
desfase = 200 # El desfase va de [0 a pretrigger*sampling rate] [0-250]

# =============================================================================
# Generacion de objetos
# =============================================================================
# Inicializacion de clases
P0_1 = Features(path_P0_1, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0)
P0_2 = Features(path_P0_2, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=16.875718)
P0_3 = Features(path_P0_3, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=14.416501)
P0_4 = Features(path_P0_4, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=11.668197)
P0_5 = Features(path_P0_5, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0.58955476)
P0_6 = Features(path_P0_6, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=1.548062)
P90_1 = Features(path_P90_1, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=4.4939521)
P90_2 = Features(path_P90_2, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=-0.61250454)
P90_3 = Features(path_P90_3, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=2.3312921)
P90_4 = Features(path_P90_4, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=-0.03841427)

P0_90_1 = Features(path_P0_90_1, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=6.1713131)
P0_90_2 = Features(path_P0_90_2, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0.85981781)
P0_90_3 = Features(path_P0_90_3, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0.065987506)
P0_90_4 = Features(path_P0_90_4, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=-1.3343409)
P0_90W_1 = Features(path_P0_90W_1, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0.53398163)
P0_90W_3 = Features(path_P0_90W_3, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0.15772048)
P45_2 = Features(path_P45_2, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=-13.833989)
P45_3 = Features(path_P45_3, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=7.7640255)
P45_4 = Features(path_P45_4, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0.191836)
PQ_1 = Features(path_PQ_1, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=6.1940507)
PQ_2 = Features(path_PQ_2, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=5.8247346)
PQ_3 = Features(path_PQ_3, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=7.9953848)
PQ_4 = Features(path_PQ_4, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0.0063383359)

P0_A1 = Features(path_P0_A1, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0.29895222)
P0_A2 = Features(path_P0_A2, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0.21784435)
P0_A3 = Features(path_P0_A3, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=-0.12090737)
P0_A4 = Features(path_P0_A4, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=-0.17492944)
P90_A1 = Features(path_P90_A1, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0)
P90_A2 = Features(path_P90_A2, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=-0.4175458)
P90_A3 = Features(path_P90_A3, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=-0.25912817)
P90_A4 = Features(path_P90_A4, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=-0.61885705)

# =============================================================================
# Extraccion de features
# =============================================================================
# Valores de filtrado
umbral = 36
counts = 5

# =============================================================================
# # Extraccion de features - Sin rotura
# =============================================================================
P0_1_df = P0_1.feature_extr(umbral, counts, clase='[0]', test_id='P0_1', max_trai=66)
P0_2_df = P0_2.feature_extr(umbral, counts, clase='[0]', test_id='P0_2',max_trai=111)
P0_3_df_previo = P0_3.feature_extr(umbral, counts, clase='[0]', test_id='P0_3', max_trai=86) # Reparar archivo
P0_3_df = P0_3_df_previo[50:] # Archivo 3 solo toma el segundo ensayo (100 depende del P0_3_previo)
P0_3_df['time'] = P0_3_df['time']- 375 # Cambio de tiempo por tomar solo el segundo ensayo
P0_4_df = P0_4.feature_extr(umbral, counts, clase='[0]', test_id='P0_4', max_trai=401)
P0_5_df = P0_5.feature_extr(umbral, counts, clase='[0]', test_id='P0_5', max_trai=385)
P0_6_df = P0_6.feature_extr(umbral, counts, clase='[0]', test_id='P0_6', max_trai=460)

P90_1_df = P90_1.feature_extr(umbral, counts, clase='[90]', test_id='P90_1', max_trai=28)
P90_2_df = P90_2.feature_extr(umbral, counts, clase='[90]', test_id='P90_2', max_trai=56)
P90_3_df = P90_3.feature_extr(umbral, counts, clase='[90]', test_id='P90_3', max_trai=130)
P90_4_df = P90_4.feature_extr(umbral, counts, clase='[90]', test_id='P90_4', max_trai=616)

P0_90_1_df = P0_90_1.feature_extr(umbral, counts, clase='[0-90]', test_id='P0_90_1', max_trai=200)
P0_90_2_df = P0_90_2.feature_extr(umbral, counts, clase='[0-90]', test_id='P0_90_2', max_trai=440)
P0_90_3_df = P0_90_3.feature_extr(umbral, counts, clase='[0-90]', test_id='P0_90_3', max_trai=474)
P0_90_4_df = P0_90_4.feature_extr(umbral, counts, clase='[0-90]', test_id='P0_90_4', max_trai=428)

P0_90W_1_df = P0_90W_1.feature_extr(umbral, counts, clase='[0-90W]', test_id='P0_90W_1', max_trai=220)
P0_90W_3_df = P0_90W_3.feature_extr(umbral, counts, clase='[0-90W]', test_id='P0_90W_3', max_trai=270)

P45_2_df = P45_2.feature_extr(umbral, counts, clase='[45]', test_id='P45_2', max_trai=100)
P45_3_df = P45_3.feature_extr(umbral, counts, clase='[45]', test_id='P45_3', max_trai=180) # Para tsne original estaba en 520
P45_4_df = P45_4.feature_extr(umbral, counts, clase='[45]', test_id='P45_4', max_trai=304)

PQ_1_df = PQ_1.feature_extr(umbral, counts, clase='[Q]', test_id='PQ_1', max_trai=211)
PQ_2_df = PQ_2.feature_extr(umbral, counts, clase='[Q]', test_id='PQ_2',  max_trai=109)
PQ_3_df = PQ_3.feature_extr(umbral, counts, clase='[Q]', test_id='PQ_3',  max_trai=166)
PQ_4_df = PQ_4.feature_extr(umbral, counts, clase='[Q]', test_id='PQ_4',  max_trai=318)

# =============================================================================
# # Extraccion de features - Con rotura
# =============================================================================
P0_1_df = P0_1.feature_extr(umbral, counts, clase='[0]', test_id='P0_1')
P0_2_df = P0_2.feature_extr(umbral, counts, clase='[0]', test_id='P0_2')
P0_3_df_previo = P0_3.feature_extr(umbral, counts, clase='[0]', test_id='P0_3', max_trai=187) # Reparar archivo
P0_3_df = P0_3_df_previo[111:] # Archivo 3 solo toma el segundo ensayo (100 depende del P0_3_previo)
P0_3_df['time'] = P0_3_df['time'] - 375 # Cambio de tiempo por tomar solo el segundo ensayo
P0_4_df = P0_4.feature_extr(umbral, counts, clase='[0]', test_id='P0_4')
P0_5_df = P0_5.feature_extr(umbral, counts, clase='[0]', test_id='P0_5')
P0_6_df = P0_6.feature_extr(umbral, counts, clase='[0]', test_id='P0_6')

P90_1_df = P90_1.feature_extr(umbral, counts, clase='[90]', test_id='P90_1', max_trai=100)
P90_2_df = P90_2.feature_extr(umbral, counts, clase='[90]', test_id='P90_2', max_trai=425)
P90_3_df = P90_3.feature_extr(umbral, counts, clase='[90]', test_id='P90_3')
P90_4_df = P90_4.feature_extr(umbral, counts, clase='[90]', test_id='P90_4')

P0_90_1_df = P0_90_1.feature_extr(umbral, counts, clase='[0-90]', test_id='P0_90_1', max_trai=417)
P0_90_2_df = P0_90_2.feature_extr(umbral, counts, clase='[0-90]', test_id='P0_90_2')
P0_90_3_df = P0_90_3.feature_extr(umbral, counts, clase='[0-90]', test_id='P0_90_3')
P0_90_4_df = P0_90_4.feature_extr(umbral, counts, clase='[0-90]', test_id='P0_90_4', max_trai=557)

P0_90W_1_df = P0_90W_1.feature_extr(umbral, counts, clase='[0-90W]', test_id='P0_90W_1')
P0_90W_3_df = P0_90W_3.feature_extr(umbral, counts, clase='[0-90W]', test_id='P0_90W_3')

P45_2_df = P45_2.feature_extr(umbral, counts, clase='[45]', test_id='P45_2', max_trai=143)
P45_3_df = P45_3.feature_extr(umbral, counts, clase='[45]', test_id='P45_3') # Para tsne original estaba en 520
P45_4_df = P45_4.feature_extr(umbral, counts, clase='[45]', test_id='P45_4')

PQ_1_df = PQ_1.feature_extr(umbral, counts, clase='[Q]', test_id='PQ_1')
PQ_2_df = PQ_2.feature_extr(umbral, counts, clase='[Q]', test_id='PQ_2')
PQ_3_df = PQ_3.feature_extr(umbral, counts, clase='[Q]', test_id='PQ_3')
PQ_4_df = PQ_4.feature_extr(umbral, counts, clase='[Q]', test_id='PQ_4', max_trai=431)


# =============================================================================
# Union de dataframes - Data set de ensayos validos 
# =============================================================================
# Directorio para guardar los datasets
dataset_path = os.path.join(script_dir, 'datasets') 

file_name = 'Dataset_0_90.csv'
df_0_90 = unir_df(P0_2_df, P0_3_df, P0_4_df, P0_5_df, P0_6_df,
                    P90_1_df, P90_2_df, P90_3_df)
df_0_90.to_csv(os.path.join(dataset_path, file_name), index=False)
print(f'Archivo guardado como: {file_name}')

file_name = 'Dataset_total.csv'
df_total = unir_df(df_0_90, P0_90_1_df, P0_90_2_df, P0_90_3_df, P0_90_4_df, 
                   P0_90W_1_df, P0_90W_3_df,
                   P45_2_df, P45_3_df, P45_4_df,
                   PQ_1_df, PQ_2_df, PQ_3_df, PQ_4_df)
df_total.to_csv(os.path.join(dataset_path, file_name), index=False)
print(f'Archivo guardado como: {file_name}')

# =============================================================================
# Dataset de datos de fuerza
# =============================================================================
force_path = os.path.join(script_dir, 'Datos_MTS') 
force_df = force_data(force_path)
force_df.to_csv(os.path.join(dataset_path, 'Datos_MTS.csv'), index=False)