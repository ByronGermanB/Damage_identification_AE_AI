# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:31:23 2024

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Directories
import os

# Functions for feature extraction
from utils.analysis_AE import Features

# =============================================================================
# Initial data
# =============================================================================
# Current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory with AE data
EA_path = os.path.join(script_dir, "Datos_EA")
# Directory to save images
figure_path = os.path.join(script_dir, "Figuras")
os.makedirs(figure_path, exist_ok=True)

# Directory for reading AE data
test_id = "MINA_P0_90_2"
data = os.path.join(EA_path, test_id)

# Constants
lower_freq = 0  # Lower frequency limit in kHz
upper_freq = 900  # Upper frequency limit in kHz
sampling_rate = 5.0  # Data acquisition frequency per second in MHz
N_samp = 1024  # Number of samples (data points) in each segment
N_seg = 1  # Number of segments (if changed, column names must be changed)
desfase = 200  # The offset ranges from [0 to pretrigger*sampling rate] [0-250]

# Filtering values
umbral = 36
counts = 5

# =============================================================================
# Feature extraction
# =============================================================================
# Class initialization
trail = Features(
    data, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga=0
)
trail_df = trail.feature_extr(umbral, counts, clase="[trial]", test_id=test_id)

trai = 3
trail.plot_signal(
    trai=trai,
    name_figure="PLB",
    time_graph=260,
    figure_path=figure_path,
    width=90,
    height=70,
    guardar=False,
)
trail.plot_segmentation_1(
    trai=trai,
    name_figure="PLB",
    figure_path=figure_path,
    width=90,
    height=125,
    guardar=True,
)
