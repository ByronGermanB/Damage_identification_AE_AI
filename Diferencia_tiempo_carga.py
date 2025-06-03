# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:15:58 2024

@author: bbarmac
"""

# Dataframes
import os

import pandas as pd

# Functions for data splitting
from mis_funciones.analisis_AE import train_test_set

# Current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "datasets")

# Import datasets
EA_dataset = "Dataset_total.csv"
data = pd.read_csv(os.path.join(dataset_dir, EA_dataset))
force = pd.read_csv(os.path.join(dataset_dir, "Datos_MTS.csv"))

X_original, X, y, hits = train_test_set(data, split=False)
test_ids = hits["test_id"].unique()

differences = []

for test_id in test_ids:
    # Condition to filter rows based on the string column
    condition = hits["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    # Filtered DataFrame
    filtered_hits = hits[condition]
    filtered_force = force[condition_force]

    # Time difference
    dif = filtered_hits["time"].iloc[-1] - filtered_force["Time [s]"].iloc[-1]
    differences.append(dif)

# Create a DataFrame
df = pd.DataFrame({"test_id": test_ids, "dif_time": differences})
file_name = "Diferencia_tiempo_carga_agujero.csv"
df.to_csv(os.path.join(dataset_dir, file_name), index=False)
print(f"File saved as: {file_name}")
