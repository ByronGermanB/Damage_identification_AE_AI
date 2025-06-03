# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:48:58 2023

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Directories
import os

# Functions for feature extraction
from mis_funciones.analisis_AE import Features, unir_df

# Functions to load MTS data
from mis_funciones.force_mts import force_data

# =============================================================================
# Initial data
# =============================================================================
# Current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory with AE data
EA_path = os.path.join(script_dir, "Datos_EA")

# Test directories
path_P0_1 = os.path.join(
    EA_path, "P0_1-1"
)  # This is the folder P0_1-1, renamed for convenience
path_P0_2 = os.path.join(EA_path, "P0_2")
path_P0_3 = os.path.join(EA_path, "P0_3")
path_P0_4 = os.path.join(EA_path, "P0_4")
path_P0_5 = os.path.join(EA_path, "P0_5")
path_P0_6 = os.path.join(EA_path, "P0_6")
path_P90_1 = os.path.join(EA_path, "P90_1")
path_P90_2 = os.path.join(EA_path, "P90_2")
path_P90_3 = os.path.join(EA_path, "P90_3")
path_P90_4 = os.path.join(EA_path, "P90_4")

path_P0_90_1 = os.path.join(EA_path, "P0_90_1")
path_P0_90_2 = os.path.join(EA_path, "P0_90_2")
path_P0_90_3 = os.path.join(EA_path, "P0_90_3")
path_P0_90_4 = os.path.join(EA_path, "P0_90_4")
path_P0_90W_1 = os.path.join(EA_path, "P0_90W_1")
path_P0_90W_3 = os.path.join(EA_path, "P0_90W_3")
path_P45_2 = os.path.join(EA_path, "P45_2")
path_P45_3 = os.path.join(EA_path, "P45_3")
path_P45_4 = os.path.join(EA_path, "P45_4")
path_PQ_1 = os.path.join(EA_path, "PQ_1")
path_PQ_2 = os.path.join(EA_path, "PQ_2")
path_PQ_3 = os.path.join(EA_path, "PQ_3")
path_PQ_4 = os.path.join(EA_path, "PQ_4")

path_P0_A1 = os.path.join(EA_path, "P0_A1")
path_P0_A2 = os.path.join(EA_path, "P0_A2")
path_P0_A3 = os.path.join(EA_path, "P0_A3")
path_P0_A4 = os.path.join(EA_path, "P0_A4")
path_P90_A1 = os.path.join(EA_path, "P90_A1")
path_P90_A2 = os.path.join(EA_path, "P90_A2")
path_P90_A3 = os.path.join(EA_path, "P90_A3")
path_P90_A4 = os.path.join(EA_path, "P90_A4")

# Constants
lower_freq = 0  # Lower frequency limit in kHz
upper_freq = 900  # Upper frequency limit in kHz
sampling_rate = 5.0  # Data acquisition frequency per second in MHz
N_samp = 1024  # Number of samples (data) in each segment
N_seg = 1  # Number of segments (if changed, column names must be changed)
desfase = 200  # The offset ranges from [0 to pretrigger*sampling rate] [0-250]

# =============================================================================
# Object generation
# =============================================================================
# Class initialization
P0_1 = Features(
    path_P0_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0,
)
P0_2 = Features(
    path_P0_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=16.875718,
)
P0_3 = Features(
    path_P0_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=14.416501,
)
P0_4 = Features(
    path_P0_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=11.668197,
)
P0_5 = Features(
    path_P0_5,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.58955476,
)
P0_6 = Features(
    path_P0_6,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=1.548062,
)
P90_1 = Features(
    path_P90_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=4.4939521,
)
P90_2 = Features(
    path_P90_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.61250454,
)
P90_3 = Features(
    path_P90_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=2.3312921,
)
P90_4 = Features(
    path_P90_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.03841427,
)

P0_90_1 = Features(
    path_P0_90_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=6.1713131,
)
P0_90_2 = Features(
    path_P0_90_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.85981781,
)
P0_90_3 = Features(
    path_P0_90_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.065987506,
)
P0_90_4 = Features(
    path_P0_90_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-1.3343409,
)
P0_90W_1 = Features(
    path_P0_90W_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.53398163,
)
P0_90W_3 = Features(
    path_P0_90W_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.15772048,
)
P45_2 = Features(
    path_P45_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-13.833989,
)
P45_3 = Features(
    path_P45_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=7.7640255,
)
P45_4 = Features(
    path_P45_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.191836,
)
PQ_1 = Features(
    path_PQ_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=6.1940507,
)
PQ_2 = Features(
    path_PQ_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=5.8247346,
)
PQ_3 = Features(
    path_PQ_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=7.9953848,
)
PQ_4 = Features(
    path_PQ_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.0063383359,
)

P0_A1 = Features(
    path_P0_A1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.29895222,
)
P0_A2 = Features(
    path_P0_A2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.21784435,
)
P0_A3 = Features(
    path_P0_A3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.12090737,
)
P0_A4 = Features(
    path_P0_A4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.17492944,
)
P90_A1 = Features(
    path_P90_A1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0,
)
P90_A2 = Features(
    path_P90_A2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.4175458,
)
P90_A3 = Features(
    path_P90_A3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.25912817,
)
P90_A4 = Features(
    path_P90_A4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.61885705,
)

# =============================================================================
# Feature extraction
# =============================================================================
# Filtering values
threshold = 36
counts = 5

# =============================================================================
# Feature extraction - Without breakage
# =============================================================================
P0_1_df = P0_1.feature_extr(threshold, counts, clase="[0]", test_id="P0_1", max_trai=66)
P0_2_df = P0_2.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_2", max_trai=111
)
P0_3_df_previo = P0_3.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_3", max_trai=86
)  # Repair file
P0_3_df = P0_3_df_previo[
    50:
]  # File 3 only takes the second test (100 depends on P0_3_previo)
P0_3_df["time"] = P0_3_df["time"] - 375  # Change time by taking only the second test
P0_4_df = P0_4.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_4", max_trai=401
)
P0_5_df = P0_5.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_5", max_trai=385
)
P0_6_df = P0_6.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_6", max_trai=460
)

P90_1_df = P90_1.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_1", max_trai=28
)
P90_2_df = P90_2.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_2", max_trai=56
)
P90_3_df = P90_3.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_3", max_trai=130
)
P90_4_df = P90_4.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_4", max_trai=616
)

P0_90_1_df = P0_90_1.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_1", max_trai=200
)
P0_90_2_df = P0_90_2.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_2", max_trai=440
)
P0_90_3_df = P0_90_3.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_3", max_trai=474
)
P0_90_4_df = P0_90_4.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_4", max_trai=428
)

P0_90W_1_df = P0_90W_1.feature_extr(
    threshold, counts, clase="[0-90W]", test_id="P0_90W_1", max_trai=220
)
P0_90W_3_df = P0_90W_3.feature_extr(
    threshold, counts, clase="[0-90W]", test_id="P0_90W_3", max_trai=270
)

P45_2_df = P45_2.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_2", max_trai=100
)
P45_3_df = P45_3.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_3", max_trai=180
)  # For tsne originally was at 520
P45_4_df = P45_4.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_4", max_trai=304
)

PQ_1_df = PQ_1.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_1", max_trai=211
)
PQ_2_df = PQ_2.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_2", max_trai=109
)
PQ_3_df = PQ_3.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_3", max_trai=166
)
PQ_4_df = PQ_4.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_4", max_trai=318
)

# =============================================================================
# Feature extraction - With breakage
# =============================================================================
P0_1_df = P0_1.feature_extr(threshold, counts, clase="[0]", test_id="P0_1")
P0_2_df = P0_2.feature_extr(threshold, counts, clase="[0]", test_id="P0_2")
P0_3_df_previo = P0_3.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_3", max_trai=187
)  # Repair file
P0_3_df = P0_3_df_previo[
    111:
]  # File 3 only takes the second test (100 depends on P0_3_previo)
P0_3_df["time"] = P0_3_df["time"] - 375  # Change time by taking only the second test
P0_4_df = P0_4.feature_extr(threshold, counts, clase="[0]", test_id="P0_4")
P0_5_df = P0_5.feature_extr(threshold, counts, clase="[0]", test_id="P0_5")
P0_6_df = P0_6.feature_extr(threshold, counts, clase="[0]", test_id="P0_6")

P90_1_df = P90_1.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_1", max_trai=100
)
P90_2_df = P90_2.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_2", max_trai=425
)
P90_3_df = P90_3.feature_extr(threshold, counts, clase="[90]", test_id="P90_3")
P90_4_df = P90_4.feature_extr(threshold, counts, clase="[90]", test_id="P90_4")

P0_90_1_df = P0_90_1.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_1", max_trai=417
)
P0_90_2_df = P0_90_2.feature_extr(threshold, counts, clase="[0-90]", test_id="P0_90_2")
P0_90_3_df = P0_90_3.feature_extr(threshold, counts, clase="[0-90]", test_id="P0_90_3")
P0_90_4_df = P0_90_4.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_4", max_trai=557
)

P0_90W_1_df = P0_90W_1.feature_extr(
    threshold, counts, clase="[0-90W]", test_id="P0_90W_1"
)
P0_90W_3_df = P0_90W_3.feature_extr(
    threshold, counts, clase="[0-90W]", test_id="P0_90W_3"
)

P45_2_df = P45_2.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_2", max_trai=143
)
P45_3_df = P45_3.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_3"
)  # For tsne originally was at 520
P45_4_df = P45_4.feature_extr(threshold, counts, clase="[45]", test_id="P45_4")

PQ_1_df = PQ_1.feature_extr(threshold, counts, clase="[Q]", test_id="PQ_1")
PQ_2_df = PQ_2.feature_extr(threshold, counts, clase="[Q]", test_id="PQ_2")
PQ_3_df = PQ_3.feature_extr(threshold, counts, clase="[Q]", test_id="PQ_3")
PQ_4_df = PQ_4.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_4", max_trai=431
)

# =============================================================================
# Merging dataframes - Dataset of valid tests
# =============================================================================
# Directory to save the datasets
dataset_path = os.path.join(script_dir, "datasets")

file_name = "Dataset_0_90.csv"
df_0_90 = unir_df(
    P0_2_df, P0_3_df, P0_4_df, P0_5_df, P0_6_df, P90_1_df, P90_2_df, P90_3_df
)
df_0_90.to_csv(os.path.join(dataset_path, file_name), index=False)
print(f"File saved as: {file_name}")

file_name = "Dataset_total.csv"
df_total = unir_df(
    df_0_90,
    P0_90_1_df,
    P0_90_2_df,
    P0_90_3_df,
    P0_90_4_df,
    P0_90W_1_df,
    P0_90W_3_df,
    P45_2_df,
    P45_3_df,
    P45_4_df,
    PQ_1_df,
    PQ_2_df,
    PQ_3_df,
    PQ_4_df,
)
df_total.to_csv(os.path.join(dataset_path, file_name), index=False)
print(f"File saved as: {file_name}")

# =============================================================================
# Dataset of force data
# =============================================================================
force_path = os.path.join(script_dir, "Datos_MTS")
force_df = force_data(force_path)
force_df.to_csv(os.path.join(dataset_path, "Datos_MTS.csv"), index=False)
