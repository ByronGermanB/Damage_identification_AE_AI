from mis_funciones import EAfunctions as eaf
from scipy import signal
import numpy as np
import os

# =============================================================================
# Directories
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
EA_path = os.path.join(base_dir, 'Datos_EA')

figure_path = os.path.join(base_dir, 'Figuras')
os.makedirs(figure_path, exist_ok=True)

# Creación de directorio para guardar las imágenes
images_set = 'train_3'
images_dir = os.path.join(base_dir, 'datasets', 'images', images_set)
os.makedirs(images_dir, exist_ok=True)

# Descripción de las variables de la wavelet
signal_function = signal.ricker   
sampling_rate = 2e9                     # [MHz]
n_bands = 24                            # Nº bandas de frecuencias que se desean emplear, en un artículo con sam_rate=1MHz usaban 8

# Parámetros de filtrado de transitorios
t_trans = 350
amp_lim = 36
cnts_lim = 5
STD_noise = [1.3e-3, 5e-3]
num_ruido = 0                           # Nº de señales de ruido a añadir   
dims = [128, 128]

# =============================================================================
# Carga de datos a orientación 0
# =============================================================================
pridb_0_1, vae_T0_1 = eaf.abreAE(EA_path, 'P0_2')
pridb_0_2, vae_T0_2 = eaf.abreAE(EA_path, 'P0_3')
pridb_0_3, vae_T0_3 = eaf.abreAE(EA_path, 'P0_4')
pridb_0_4, vae_T0_4 = eaf.abreAE(EA_path, 'P0_5')
pridb_0_5, vae_T0_5 = eaf.abreAE(EA_path, 'P0_6')

# Vector de datos a 0º
vae_T0 = [vae_T0_1, vae_T0_2, vae_T0_3, vae_T0_4, vae_T0_5]
vae_pridb0 = [pridb_0_1, pridb_0_2, pridb_0_3, pridb_0_4, pridb_0_5]
max_trai_T0 = [None, 187, None, None, None]

# Parámetros específicos del primer conjunto de datos
t_trai_0 = np.zeros(shape = (len(vae_T0), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T0 = np.concatenate((t_trai_0, t_trai_1), axis = 1)

# Aplicación de la función de cálculo de la CWT
print('\nCWT 0')
cwt_image0, trais0 = eaf.calcCWT(vae_Tarr = vae_T0, vae_pridb = vae_pridb0, t_trai = t_trai_T0, max_trais = max_trai_T0, t_trans = t_trans, signal_function = signal_function, 
                                 n_bands = n_bands, amp_lim = amp_lim, cnts_lim = cnts_lim, STD_noise = STD_noise, num_noisySignals = num_ruido, figure_path = figure_path)


# =============================================================================
# Carga de datos a orientación 90
# =============================================================================
pridb_90_1, vae_T90_1 = eaf.abreAE(EA_path, 'P90_1')
pridb_90_2, vae_T90_2 = eaf.abreAE(EA_path, 'P90_2')
pridb_90_3, vae_T90_3 = eaf.abreAE(EA_path, 'P90_3')

# Vector de datos a 90º
vae_T90 = [vae_T90_1, vae_T90_2, vae_T90_3]
vae_pridb90 = [pridb_90_1, pridb_90_2, pridb_90_3]
max_trai_T90 = [100, 425, None]

# Parámetros específicos del primer conjunto de datos
t_trai_0 = np.zeros(shape = (len(vae_T90), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T90 = np.concatenate((t_trai_0, t_trai_1), axis = 1)

# Aplicación de la función de cálculo de la CWT
print('\nCWT 90')
cwt_image90, trais90 = eaf.calcCWT(vae_Tarr = vae_T90, vae_pridb = vae_pridb90, t_trai = t_trai_T90, max_trais = max_trai_T90, t_trans = t_trans, signal_function = signal_function, 
                                   n_bands = n_bands, amp_lim = amp_lim, cnts_lim = cnts_lim, STD_noise = STD_noise, num_noisySignals = num_ruido, figure_path = figure_path)

# =============================================================================
# Carga de datos a orientación 0-90-0
# =============================================================================
pridb_090_1, vae_090T_1 = eaf.abreAE(EA_path, 'P0_90_1')
pridb_090_2, vae_090T_2 = eaf.abreAE(EA_path, 'P0_90_2')
pridb_090_3, vae_090T_3 = eaf.abreAE(EA_path, 'P0_90_3')
pridb_090_4, vae_090T_4 = eaf.abreAE(EA_path, 'P0_90_4')

# Vector de datos
vae_T090 = [vae_090T_1, vae_090T_2, vae_090T_3, vae_090T_4]
vae_pridb090 = [pridb_090_1, pridb_090_2, pridb_090_3, pridb_090_4]
max_trai_T090 = [417, None, None, 557]
# Parámetros específicos del conjunto de datos
t_trai_0 = np.zeros(shape = (len(vae_T090), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T090 = np.concatenate((t_trai_0, t_trai_1), axis = 1)

# Aplicación de la transformada CWT
print('\nCWT 090')
cwt_image090, trais_090 = eaf.calcCWT(vae_Tarr = vae_T090, vae_pridb = vae_pridb090, t_trai = t_trai_T090, max_trais = max_trai_T090, t_trans = t_trans, signal_function = signal_function, 
                                      n_bands = n_bands, amp_lim = amp_lim, cnts_lim = cnts_lim, STD_noise = STD_noise, num_noisySignals = num_ruido, figure_path=figure_path)

# =============================================================================
# Carga de datos a orientación 0-90W
# =============================================================================
pridb_W_1, vae_WT_1 = eaf.abreAE(EA_path, 'P0_90W_1')
pridb_W_3, vae_WT_3 = eaf.abreAE(EA_path, 'P0_90W_3')

# Vector de datos
vae_T090W = [vae_WT_1, vae_WT_3]
vae_pridb090W = [pridb_W_1, pridb_W_3]
max_trai_T090W = [None, None]
# Parámetros específicos del conjunto de datos
t_trai_0 = np.zeros(shape = (len(vae_T090W), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T090W = np.concatenate((t_trai_0, t_trai_1), axis = 1)

# Aplicación de la transformada CWT
print('\nCWT 090W')
cwt_image090W, trais_090W = eaf.calcCWT(vae_Tarr = vae_T090W, vae_pridb = vae_pridb090W, t_trai = t_trai_T090W, max_trais=max_trai_T090W, t_trans = t_trans, signal_function = signal_function, 
                                        n_bands = n_bands, amp_lim = amp_lim, cnts_lim = cnts_lim, STD_noise = STD_noise, num_noisySignals = num_ruido, figure_path=figure_path)

# =============================================================================
# Carga de datos a orientación +-45
# =============================================================================
pridb_45_2, vae_45T_2 = eaf.abreAE(EA_path, 'P45_2')
pridb_45_3, vae_45T_3 = eaf.abreAE(EA_path, 'P45_3')
pridb_45_4, vae_45T_4 = eaf.abreAE(EA_path, 'P45_4')

# Vector de datos
vae_T45 = [vae_45T_2, vae_45T_3, vae_45T_4]
vae_pridb45 = [pridb_45_2, pridb_45_3, pridb_45_4]
max_trai_T45 = [143, None, None]
# Parámetros específicos del conjunto de datos
t_trai_0 = np.zeros(shape = (len(vae_T45), 1))
t_trai_1 = t_trai_0 - 1
t_trai_T45 = np.concatenate((t_trai_0, t_trai_1), axis = 1)

# Aplicación de la transformada CWT
print('\nCWT 45')
cwt_image45, trais_45 = eaf.calcCWT(vae_Tarr = vae_T45, vae_pridb = vae_pridb45, t_trai = t_trai_T45, max_trais=max_trai_T45, t_trans = t_trans, signal_function = signal_function, 
                                    n_bands = n_bands, amp_lim = amp_lim, cnts_lim = cnts_lim, STD_noise = STD_noise, num_noisySignals = num_ruido, figure_path=figure_path)

# =============================================================================
# Carga de datos a orientación Q
# =============================================================================
pridb_Q_1, vae_QT_1 = eaf.abreAE(EA_path, 'PQ_1')
pridb_Q_2, vae_QT_2 = eaf.abreAE(EA_path, 'PQ_2')
pridb_Q_3, vae_QT_3 = eaf.abreAE(EA_path, 'PQ_3')
pridb_Q_4, vae_QT_4 = eaf.abreAE(EA_path, 'PQ_4')

# Vector de datos
vae_TQ = [vae_QT_1, vae_QT_2, vae_QT_3, vae_QT_4]
vae_pridbQ = [pridb_Q_1, pridb_Q_2, pridb_Q_3, pridb_Q_4]
max_trai_TQ = [None, None, None, 431]
# Parámetros específicos del conjunto de datos
t_trai_0 = np.zeros(shape = (len(vae_TQ), 1))
t_trai_1 = t_trai_0 - 1
t_trai_TQ = np.concatenate((t_trai_0, t_trai_1), axis = 1)

# Aplicación de la transformada CWT
print('\nCWT Q')
cwt_imageQ, trais_Q = eaf.calcCWT(vae_Tarr = vae_TQ, vae_pridb = vae_pridbQ, t_trai = t_trai_TQ, max_trais=max_trai_TQ, t_trans = t_trans, signal_function = signal_function, 
                                  n_bands = n_bands, amp_lim = amp_lim, cnts_lim = cnts_lim, STD_noise = STD_noise, num_noisySignals = num_ruido, figure_path=figure_path)

# =============================================================================
# Save images
# =============================================================================
eaf.save_image_pred(cwt_image0, dims, 'T0', images_dir)
eaf.save_image_pred(cwt_image90, dims, 'T90', images_dir)
eaf.save_image_pred(cwt_image090, dims, 'T090', images_dir)
eaf.save_image_pred(cwt_image090W, dims, 'T090W', images_dir)
eaf.save_image_pred(cwt_image45, dims, 'T45', images_dir)
eaf.save_image_pred(cwt_imageQ, dims, 'TQ', images_dir)

