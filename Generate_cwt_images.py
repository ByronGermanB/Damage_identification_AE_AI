import os

import numpy as np
from scipy import signal

from utils import EAfunctions as eaf

# =============================================================================
# Directories
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
EA_path = os.path.join(base_dir, "Datos_EA")

figure_path = os.path.join(base_dir, "Figures")
os.makedirs(figure_path, exist_ok=True)

# Create directory to save images
images_set = "train"
images_dir = os.path.join(base_dir, "datasets", "images", images_set)
os.makedirs(images_dir, exist_ok=True)

# Description of wavelet variables
signal_function = signal.ricker
sampling_rate = 2e9  # [MHz]
n_bands = (
    24  # Number of frequency bands to use, in an article with sam_rate=1MHz they used 8
)

# Transient filtering parameters
t_trans = 350
amp_lim = 36
cnts_lim = 5
STD_noise = [1.3e-3, 5e-3]
num_ruido = 0  # Number of noise signals to add
dims = [128, 128]

# =============================================================================
# Load data at 0 orientation
# =============================================================================
pridb_0_1, vae_T0_1 = eaf.abreAE(EA_path, "P0_2")
pridb_0_2, vae_T0_2 = eaf.abreAE(EA_path, "P0_3")
pridb_0_3, vae_T0_3 = eaf.abreAE(EA_path, "P0_4")
pridb_0_4, vae_T0_4 = eaf.abreAE(EA_path, "P0_5")
pridb_0_5, vae_T0_5 = eaf.abreAE(EA_path, "P0_6")

# Data vector at 0º
vae_T0 = [vae_T0_1, vae_T0_2, vae_T0_3, vae_T0_4, vae_T0_5]
vae_pridb0 = [pridb_0_1, pridb_0_2, pridb_0_3, pridb_0_4, pridb_0_5]
max_trai_T0 = [None, 187, None, None, None]

# Specific parameters of the first dataset
t_trai_0 = np.zeros(shape=(len(vae_T0), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T0 = np.concatenate((t_trai_0, t_trai_1), axis=1)

# Apply the CWT calculation function
print("\nCWT 0")
cwt_image0, trais0 = eaf.calcCWT(
    vae_Tarr=vae_T0,
    vae_pridb=vae_pridb0,
    t_trai=t_trai_T0,
    max_trais=max_trai_T0,
    t_trans=t_trans,
    signal_function=signal_function,
    n_bands=n_bands,
    amp_lim=amp_lim,
    cnts_lim=cnts_lim,
    STD_noise=STD_noise,
    num_noisySignals=num_ruido,
    figure_path=figure_path,
)


# =============================================================================
# Load data at 90 orientation
# =============================================================================
pridb_90_1, vae_T90_1 = eaf.abreAE(EA_path, "P90_1")
pridb_90_2, vae_T90_2 = eaf.abreAE(EA_path, "P90_2")
pridb_90_3, vae_T90_3 = eaf.abreAE(EA_path, "P90_3")

# Data vector at 90º
vae_T90 = [vae_T90_1, vae_T90_2, vae_T90_3]
vae_pridb90 = [pridb_90_1, pridb_90_2, pridb_90_3]
max_trai_T90 = [100, 425, None]

# Specific parameters of the first dataset
t_trai_0 = np.zeros(shape=(len(vae_T90), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T90 = np.concatenate((t_trai_0, t_trai_1), axis=1)

# Apply the CWT calculation function
print("\nCWT 90")
cwt_image90, trais90 = eaf.calcCWT(
    vae_Tarr=vae_T90,
    vae_pridb=vae_pridb90,
    t_trai=t_trai_T90,
    max_trais=max_trai_T90,
    t_trans=t_trans,
    signal_function=signal_function,
    n_bands=n_bands,
    amp_lim=amp_lim,
    cnts_lim=cnts_lim,
    STD_noise=STD_noise,
    num_noisySignals=num_ruido,
    figure_path=figure_path,
)

# =============================================================================
# Load data at 0-90-0 orientation
# =============================================================================
pridb_090_1, vae_090T_1 = eaf.abreAE(EA_path, "P0_90_1")
pridb_090_2, vae_090T_2 = eaf.abreAE(EA_path, "P0_90_2")
pridb_090_3, vae_090T_3 = eaf.abreAE(EA_path, "P0_90_3")
pridb_090_4, vae_090T_4 = eaf.abreAE(EA_path, "P0_90_4")

# Data vector
vae_T090 = [vae_090T_1, vae_090T_2, vae_090T_3, vae_090T_4]
vae_pridb090 = [pridb_090_1, pridb_090_2, pridb_090_3, pridb_090_4]
max_trai_T090 = [417, None, None, 557]
# Specific parameters of the dataset
t_trai_0 = np.zeros(shape=(len(vae_T090), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T090 = np.concatenate((t_trai_0, t_trai_1), axis=1)

# Apply the CWT calculation function
print("\nCWT 090")
cwt_image090, trais_090 = eaf.calcCWT(
    vae_Tarr=vae_T090,
    vae_pridb=vae_pridb090,
    t_trai=t_trai_T090,
    max_trais=max_trai_T090,
    t_trans=t_trans,
    signal_function=signal_function,
    n_bands=n_bands,
    amp_lim=amp_lim,
    cnts_lim=cnts_lim,
    STD_noise=STD_noise,
    num_noisySignals=num_ruido,
    figure_path=figure_path,
)

# =============================================================================
# Load data at 0-90W orientation
# =============================================================================
pridb_W_1, vae_WT_1 = eaf.abreAE(EA_path, "P0_90W_1")
pridb_W_3, vae_WT_3 = eaf.abreAE(EA_path, "P0_90W_3")

# Data vector
vae_T090W = [vae_WT_1, vae_WT_3]
vae_pridb090W = [pridb_W_1, pridb_W_3]
max_trai_T090W = [None, None]
# Specific parameters of the dataset
t_trai_0 = np.zeros(shape=(len(vae_T090W), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T090W = np.concatenate((t_trai_0, t_trai_1), axis=1)

# Apply the CWT calculation function
print("\nCWT 090W")
cwt_image090W, trais_090W = eaf.calcCWT(
    vae_Tarr=vae_T090W,
    vae_pridb=vae_pridb090W,
    t_trai=t_trai_T090W,
    max_trais=max_trai_T090W,
    t_trans=t_trans,
    signal_function=signal_function,
    n_bands=n_bands,
    amp_lim=amp_lim,
    cnts_lim=cnts_lim,
    STD_noise=STD_noise,
    num_noisySignals=num_ruido,
    figure_path=figure_path,
)

# =============================================================================
# Load data at +-45 orientation
# =============================================================================
pridb_45_2, vae_45T_2 = eaf.abreAE(EA_path, "P45_2")
pridb_45_3, vae_45T_3 = eaf.abreAE(EA_path, "P45_3")
pridb_45_4, vae_45T_4 = eaf.abreAE(EA_path, "P45_4")

# Data vector
vae_T45 = [vae_45T_2, vae_45T_3, vae_45T_4]
vae_pridb45 = [pridb_45_2, pridb_45_3, pridb_45_4]
max_trai_T45 = [143, None, None]
# Specific parameters of the dataset
t_trai_0 = np.zeros(shape=(len(vae_T45), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T45 = np.concatenate((t_trai_0, t_trai_1), axis=1)

# Apply the CWT calculation function
print("\nCWT 45")
cwt_image45, trais_45 = eaf.calcCWT(
    vae_Tarr=vae_T45,
    vae_pridb=vae_pridb45,
    t_trai=t_trai_T45,
    max_trais=max_trai_T45,
    t_trans=t_trans,
    signal_function=signal_function,
    n_bands=n_bands,
    amp_lim=amp_lim,
    cnts_lim=cnts_lim,
    STD_noise=STD_noise,
    num_noisySignals=num_ruido,
    figure_path=figure_path,
)

# =============================================================================
# Load data at Q orientation
# =============================================================================
pridb_Q_1, vae_QT_1 = eaf.abreAE(EA_path, "PQ_1")
pridb_Q_2, vae_QT_2 = eaf.abreAE(EA_path, "PQ_2")
pridb_Q_3, vae_QT_3 = eaf.abreAE(EA_path, "PQ_3")
pridb_Q_4, vae_QT_4 = eaf.abreAE(EA_path, "PQ_4")

# Data vector
vae_TQ = [vae_QT_1, vae_QT_2, vae_QT_3, vae_QT_4]
vae_pridbQ = [pridb_Q_1, pridb_Q_2, pridb_Q_3, pridb_Q_4]
max_trai_TQ = [None, None, None, 431]
# Specific parameters of the dataset
t_trai_0 = np.zeros(shape=(len(vae_TQ), 1))
t_trai_1 = t_trai_0 - 1
t_trai_TQ = np.concatenate((t_trai_0, t_trai_1), axis=1)

# Apply the CWT calculation function
print("\nCWT Q")
cwt_imageQ, trais_Q = eaf.calcCWT(
    vae_Tarr=vae_TQ,
    vae_pridb=vae_pridbQ,
    t_trai=t_trai_TQ,
    max_trais=max_trai_TQ,
    t_trans=t_trans,
    signal_function=signal_function,
    n_bands=n_bands,
    amp_lim=amp_lim,
    cnts_lim=cnts_lim,
    STD_noise=STD_noise,
    num_noisySignals=num_ruido,
    figure_path=figure_path,
)

# =============================================================================
# Save images
# =============================================================================
eaf.save_image_pred(cwt_image0, dims, "1_T0", images_dir)
eaf.save_image_pred(cwt_image90, dims, "2_T90", images_dir)
eaf.save_image_pred(cwt_image090, dims, "3_T090", images_dir)
eaf.save_image_pred(cwt_image090W, dims, "4_T090W", images_dir)
eaf.save_image_pred(cwt_image45, dims, "5_T45", images_dir)
eaf.save_image_pred(cwt_imageQ, dims, "6_TQ", images_dir)
