# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:24:10 2023

@author: bbarmac
"""

# =============================================================================
# Importamos las librerias necesarias
# =============================================================================

# Librerias principales
import os 
import numpy as np
import pandas as pd
import vallenae as vae
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import string

# Librerias para el tratamiento de senales
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hamming

# Tratamiento de datos y normalizacion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Entrenamiento del modelo
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Librerias para optimizar los hiperparametros
from sklearn.model_selection import RandomizedSearchCV

# Librerias para metricas y evaluacion del modelo
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import  ConfusionMatrixDisplay, f1_score, accuracy_score, make_scorer

# Visualizacion de features
from sklearn.manifold import TSNE

# Set font to Times New Roman and size to 8
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['figure.dpi'] = 150


# =============================================================================
# Feature extraction
# =============================================================================

class Features:
    def __init__(self, path, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga):
        '''        
        Parameters
        ----------
        path : str
            Directory where the pri and tra files are located.
        lower_freq : int, float
            Lower frequency limit in kHz.
        upper_freq : int, float
            Upper frequency limit in kHz.
        sampling_rate : int, float
            Data acquisition frequency per second in MHz.
        N_samp : int
            Number of samples (data points) in each segment for Fourier analysis (preferably a power of 2).
        N_seg : int
            Total number of segments to analyze.
        desfase : int
            Index to start the signal from the beginning or with a positive offset, depending on the pretrigger.
        desfase_carga : int
            Offset of the time taken in AE and in MTS.

        Returns
        -------
        Object initialized with values to apply extraction and plotting functions.

        '''
        self.path = path
        self.lower_freq = lower_freq    # Lower frequency limit in kHz
        self.upper_freq = upper_freq    # Upper frequency limit in kHz
        self.sampling_rate = sampling_rate # Data acquisition frequency per second in MHz
        self.N_samp = N_samp               # Number of samples (data points) in each segment
        self.N_seg = N_seg                 # Number of segments
        self.desfase = desfase             # Depends on the pretrigger time, allows starting from the beginning of the signal or with a positive offset
        self.desfase_carga = desfase_carga # Offset of the time taken in AE and in MTS
        
        # Open pri and tra files
        files =  os.listdir(self.path)
        for item in files:
            if item.endswith(".tradb"):
                self.tradb_path = os.path.join(self.path, item)
            
            elif item.endswith(".pridb"):
                self.pridb_path = os.path.join(self.path, item)
        
        # Calculate values 
        self.res_freq = sampling_rate * 1000 / N_samp # Frequency resolution in kHz
        self.time_samp = 1 / sampling_rate   # Sampling time in µs (microseconds)
        self.time_seg = self.time_samp * N_samp   # Time of each segment in µs
        
        # Calculate frequency range indices
        self.lower_freq_index = int(self.lower_freq // self.res_freq) # Lower frequency index
        self.upper_freq_index = int(self.upper_freq // self.res_freq +1) # Upper frequency index
        self.N_feat_seg = 9  # Number of features per segment
        
        # Values for FFT that are calculated only once
        self.window = hamming(self.N_samp)  # Create the Hamming window
        self.freq_fft = (fftfreq(self.N_samp, self.time_samp*1e-6)/1000)[:self.N_samp//2]  # Positive frequency values of the FFT in kHz
        self.freq_fft_rango = self.freq_fft[self.lower_freq_index:self.upper_freq_index + 1]   # Section the frequency values
                  
    def feature_extr(self, umbral, counts, clase, test_id, max_trai=None, min_trai=1):
        '''

        Parameters
        ----------
        umbral : int
            Threshold amplitude value for filtering.
        counts : int
            Minimum counts value for filtering.
        clase : int or str
            Class to which the signal belongs.
        test_id : str
            Test code for later identification.
        max_trai : int, optional
            Maximum trai index to consider, if you want to exclude the break. The default is None.
        min_trai : int, optional
            Minimum trai index to consider, if you want to exclude some erroneous initial data. The default is 1.
        
        Returns
        -------
        df_total : DataFrame 
            DataFrame containing frequency features + rise time, classes, and trai number.

        '''
  
        # Read files 
        pridb = vae.io.PriDatabase(self.pridb_path)
        tradb = vae.io.TraDatabase(self.tradb_path)
        df_hits = pridb.read_hits() # Read the hits that have occurred

        # Signal filtering (trai threshold and counts)
        umbral_V = 10 ** (umbral/20 - 6)  # Threshold in Volts
        no_saturacion = 10 ** (94/20 - 6)
                
        if max_trai is not None:
            df_hits_filtro = df_hits[(df_hits["channel"] >= 1) & (df_hits["amplitude"] >= umbral_V) & (df_hits["amplitude"] <= no_saturacion) & (df_hits["trai"] >= min_trai) & (df_hits["trai"] <= max_trai) & (df_hits["counts"] >= counts)]  # Select only the values that meet the conditions
        else:
            df_hits_filtro = df_hits[(df_hits["channel"] >= 1) & (df_hits["amplitude"] >= umbral_V) & (df_hits["amplitude"] <= no_saturacion) & (df_hits["trai"] >= min_trai) & (df_hits["counts"] >= counts)]  # Select only the values that meet the conditions
        
        trai = df_hits_filtro["trai"].to_numpy() # Extract the column with the TRAI values (transient indices)
        N_trai = trai.size  # Total number of transients
        
        # Create matrix to store features  
        v_features_total = np.zeros((N_trai , self.N_seg * self.N_feat_seg))  # Array that will store the features in each iteration
        
        x = np.arange(0, self.N_seg)  # Vector with factors for segmentation

        # Main for loop for each transient
        N_iter = 0 # This will count the iterations if for some reason the trai do not follow the sequence of 1 in 1 (e.g. use of filter)
        for trans in trai:
            amp, tiempo = tradb.read_wave(trans, time_axis=False) # Read the i-th transient
            
            amp *= 1e3     # in mV   # The operator *= is updating the amplitude value -- amplitude = amplitude * 1e3
                                # The signal was in volts and is represented in mV
                                # This is the amplitude
            # time = tiempo * 1e6  # for µs  # Similarly, the signal is stored in seconds and represented in µs
                           # This is the time
            
            
            # Create the segments
            seg_total = np.zeros((self.N_seg, self.N_samp))  # Array that will store the amplitude values in each segment
        
            # For loop to apply the Fourier transform
            for i in x:
                seg_total[i] = amp[int(i * self.N_samp / 2) + self.desfase : int((i/2 + 1) * self.N_samp) + self.desfase]
                
            # Apply Hamming windows to each segment
            seg_t_window = seg_total * self.window  # Apply the window to the segment matrix
            
            # Calculate FFT of the segment matrix
            amp_fft = fft(seg_t_window)   # Calculate the Fourier transform of the segment matrix
            amp_fft = np.abs(amp_fft)    # Calculate the amplitude modulus to not represent with its phase (Real and imaginary part) 
            
            # Select the range of interest frequencies
            amp_fft_rango = amp_fft[:,self.lower_freq_index:self.upper_freq_index + 1]   # Section the amplitude values in the established frequency range
            
            # Initialize frequency feature vectors for each segment
            peak_freq = np.zeros(self.N_seg) # Store the peak frequencies of each segment
            centroid_freq = np.zeros(self.N_seg) # Store the centroid frequencies of each segment
            part_power_1 = np.zeros(self.N_seg) # Store the total partial power 1 of each segment
            part_power_2 = np.zeros(self.N_seg) # Store the total partial power 2 of each segment
            part_power_3 = np.zeros(self.N_seg) # Store the total partial power 3 of each segment
            part_power_4 = np.zeros(self.N_seg) # Store the total partial power 4 of each segment
            part_power_5 = np.zeros(self.N_seg) # Store the total partial power 5 of each segment
            part_power_6 = np.zeros(self.N_seg) # Store the total partial power 6 of each segment
        
            # For loop to calculate features
            for i in x:
                # Peak frequency
                peak_index = np.argmax(amp_fft_rango[i])
                peak_freq[i] = self.freq_fft_rango[peak_index]
                
                # Centroid frequency
                centroid_freq[i] = np.sum(amp_fft_rango[i] * self.freq_fft_rango) / np.sum(amp_fft_rango[i])
                
                # Partial Power 1 - 6
                denominator = np.sum((amp_fft_rango[i]) ** 2)
                part_power_1[i] = np.sum((amp_fft_rango[i, 0:int(len(self.freq_fft_rango)*1/6)]) ** 2) * 100 / denominator
                part_power_2[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*1/6):int(len(self.freq_fft_rango)*2/6)]) ** 2) *100 / denominator
                part_power_3[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*2/6):int(len(self.freq_fft_rango)*3/6)]) ** 2) *100 / denominator
                part_power_4[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*3/6):int(len(self.freq_fft_rango)*4/6)]) ** 2) *100 / denominator
                part_power_5[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*4/6):int(len(self.freq_fft_rango)*5/6)]) ** 2) *100 / denominator
                part_power_6[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*5/6):]) ** 2) *100 / denominator
                
            w_peak_freq = (peak_freq * centroid_freq) ** 0.5
            
            # Feature vector
            v_features = np.hstack((peak_freq, w_peak_freq, centroid_freq, part_power_1, part_power_2, part_power_3, part_power_4, part_power_5, part_power_6)) # Group the features
            v_features = v_features.reshape(1, self.N_feat_seg * self.N_seg) # Reshape the feature matrix into a single row vector
            v_features_total[N_iter] = v_features # Chain the values to each row of the matrix to create the feature matrix

            N_iter +=  1  # Update the counter

        # Feature names

        if self.N_seg == 3:
            name_feature = ['peak_freq_1', 'peak_freq_2', 'peak_freq_3', 
                            'w_peak_freq_1', 'w_peak_freq_2', 'w_peak_freq_3', 
                            'centroid_freq_1', 'centroid_freq_2', 'centroid_freq_3',
                            'p_power_1_1', 'p_power_1_2', 'p_power_1_3',
                            'p_power_2_1', 'p_power_2_2', 'p_power_2_3',
                            'p_power_3_1', 'p_power_3_2', 'p_power_3_3',
                            'p_power_4_1', 'p_power_4_2', 'p_power_4_3',
                            'p_power_5_1', 'p_power_5_2', 'p_power_5_3',
                            'p_power_6_1', 'p_power_6_2', 'p_power_6_3']  # Array that will store the feature names per segment  
        
        elif self.N_seg == 1:
            name_feature = ['peak_freq', 
                            'w_peak_freq',  
                            'centroid_freq', 
                            'p_power_1', 
                            'p_power_2',
                            'p_power_3',
                            'p_power_4',
                            'p_power_5',
                            'p_power_6']  # Array that will store the feature names per segment  

        # Feature DataFrame
        df_features = pd.DataFrame(v_features_total,
                                  columns = name_feature)
        
        # Trai DataFrame
        df_trai = pd.DataFrame(trai, columns = ['trai'])

        # Time and rise time DataFrame 
        df_time = df_hits_filtro[['rise_time', 'time', 'amplitude', 'energy', 'counts']].reset_index().drop(['set_id'], axis=1)
        df_time['amplitude'] = 20 * (np.log10(df_time['amplitude']) + 6)
        df_time['time'] = df_time['time'] - self.desfase_carga
        df_time['time_norm'] = df_time['time'] / max(df_time['time'])
        
        # Class DataFrame
        df_clase = pd.DataFrame({'Clase' : [clase] * N_trai})
        
        
        # Total DataFrame
        df_total = pd.concat([df_features, df_time, df_clase, df_trai], axis=1)   
        df_total['test_id'] = test_id
        
        # The function returns the total DataFrame
        return (df_total)
        
    def plot_signal(self, trai, name_figure, figure_path, time_graph=None, title=None, x_label='Time [µs]', 
                    y_label='Amplitude [mV]', width=90, height=60, guardar=False):
        '''        
        Parameters
        ----------
        trai : int
            Transient number to plot.
        name_figure : str
            Name of the class that appears in the figure title.
            
            e.g. Transient Wave Plot: {name_figure} - Transient: {trai}
        figure_path : str
            Directory to save the figure.
        time_graph : int or float, optional
            Time in µs up to which the graph is desired.
        title : str, optional
            Title of the figure. The default is None.
        x_label : str, optional
            Label for the x-axis. The default is 'Time [µs]'.
        y_label : str, optional
            Label for the y-axis. The default is 'Amplitude [mV]'.
        width : int or float, optional
            Width of the figure in millimeters. The default is 90.
        height : int or float, optional
            Height of the figure in millimeters. The default is 60.
        guardar : boolean, optional
            "True" to save the image. The default is False.
            
            e.g. Transient_{name_figure}_{trai}.pdf
        Returns
        -------
        Plots and (saves) the transient signal in pdf format.

        '''  
        # Open tradb files
        tradb = vae.io.TraDatabase(self.tradb_path)

        # Plot the transient
        amp, tiempo = tradb.read_wave(trai, time_axis=True)
        amp *= 1e3  # in mV   # The operator *= is updating the amplitude value -- amp = amp * 1e3
                            # The signal was in volts and is represented in mV
                            # This is the amplitude
        time = tiempo * 1e6  # for µs  # Similarly, the signal is stored in seconds and represented in µs
                       # This is the time
        
        figsize_inches = (width / 25.4, height / 25.4)
        plt.figure(figsize=figsize_inches, dpi=300, tight_layout=True)
        
        if time_graph is None:
            plt.plot(time, amp, linewidth=0.6)
        else:
            points = int(time_graph * self.sampling_rate + 1) 
            plt.plot(time[self.desfase:points], amp[self.desfase:points], linewidth=0.6)
            plt.ylim(bottom=min(amp[self.desfase:points]) * 1.6, top=max(amp[self.desfase:points]) * 1.6)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(linewidth=0.3, color='.25')
        if title is None:
            plt.title(f"Transient Wave Plot: {name_figure} - Transient: {trai}")
        else:
            plt.title(title)
        
        # Save the image in figure_path
        if guardar:
            figure_filename = f"Transient_{name_figure}_{trai}.pdf"
            figure_path_name = os.path.join(figure_path, figure_filename)
            plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

        plt.show()

    def plot_segmentation(self, trai, name_figure, figure_path, guardar=False):
        '''
        Parameters
        ----------
        trai : int
            Transient number to plot.
        name_figure : str
            Name of the class that appears in the figure title.
            
            e.g. Segmentation: {name_figure} - Transient: {trai}
        figure_path : str
            Directory to save the figure.
        guardar : boolean, optional
            "True" to save the image. The default is False.
            
            e.g. Segmentation_{name_figure}_{trai}.pdf
        Returns
        -------
        Plots and (saves) the segmentation and FFT in pdf format.

        '''   
        # Open tradb files
        tradb = vae.io.TraDatabase(self.tradb_path)

        # Read amplitude and time
        amp, tiempo = tradb.read_wave(trai, time_axis=True)
        amp *= 1e3  # in mV   # The operator *= is updating the amplitude value -- amp = amp * 1e3
                            # The signal was in volts and is represented in mV
                            # This is the amplitude
        time = tiempo * 1e6  # for µs  # Similarly, the signal is stored in seconds and represented in µs
                       # This is the time       
       
        # Fourier transform calculation
        x = np.arange(0, self.N_seg)  # Vector with factors for segmentation
        
        # Create segments
        seg_total = np.zeros((self.N_seg, self.N_samp))  # Array that will store the amplitude values in each segment
                
        for i in x:
            seg_total[i] = amp[int(i * self.N_samp / 2) + self.desfase : int((i / 2 + 1) * self.N_samp) + self.desfase]
            
        # Apply Hamming windows to each segment
        seg_t_window = seg_total * self.window  # Apply the window to the segment matrix
        
        # Calculate FFT of the segment matrix
        amp_fft = fft(seg_t_window)   # Calculate the Fourier transform of the segment matrix
        amp_fft = np.abs(amp_fft)    # Calculate the amplitude modulus to not represent with its phase (Real and imaginary part) 
        
        # Select the range of interest frequencies
        amp_fft_rango = amp_fft[:, self.lower_freq_index:self.upper_freq_index + 1]   # Section the amplitude values in the established frequency range
              
        # Multiple plots in one figure
        fig, axs = plt.subplots(2, self.N_seg, figsize=(15, 7.5))
        plt.suptitle(f'Segmentation: {name_figure} - Transient: {trai}', fontsize=14)
        plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9)

        # Maximum and minimum for axes
        seg_t_window_max = np.amax(abs(seg_t_window))
        amp_fft_max = np.amax(amp_fft_rango)
            
        for j in x:
            # Time domain
            axs[0, j].grid(linewidth=0.3, color='.25')
            axs[0, j].set_ylim(-1.1 * seg_t_window_max, seg_t_window_max * 1.1)
            axs[0, j].plot(time[int(j * self.N_samp / 2) + self.desfase : int((j / 2 + 1) * self.N_samp) + self.desfase], seg_t_window[j])
            axs[0, j].set_xlabel("Time [µs]")
            axs[0, 0].set_ylabel("Amplitude [mV]")
            axs[0, j].set_title(f"Segment: {j + 1}")
            
            # FFT
            axs[1, j].grid(linewidth=0.3, color='.25')
            axs[1, j].set_ylim(0, amp_fft_max * 1.1)
            axs[1, j].plot(self.freq_fft_rango, amp_fft_rango[j][:self.N_samp // 2])
            
            # Vertical line in the peak frequency
            peak_index = np.argmax(amp_fft_rango[j])
            peak_freq = self.freq_fft_rango[peak_index]
            y_max = np.amax(amp_fft_rango[j]) * 2 / (amp_fft_max * 1.1)    
            axs[1, j].axvline(x=peak_freq, ymax=y_max, color='red', linestyle="--", label='f_peak')
                    
            # Vertical line in the Centroid frequency
            centroid_freq = np.sum(amp_fft_rango[j] * self.freq_fft_rango) / np.sum(amp_fft_rango[j])
            y_max_c = 0.8 * y_max    
            axs[1, j].axvline(x=centroid_freq, ymax=y_max_c, color='green', linestyle=":", label='f_centroid')
            
            axs[1, j].set_xlabel("Frequency [kHz]")
            axs[1, 0].set_ylabel("FFT-Magnitude [mV]")
            axs[1, j].legend()

        # Add a single border around all subplots
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(1)

        # Save the image in figure_path
        if guardar:
            figure_filename = f"Segmentation_{name_figure}_{trai}.pdf"
            figure_path_name = os.path.join(figure_path, figure_filename)
            plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

        plt.show()
        
    def plot_segmentation_1(self, trai, name_figure, figure_path, title=None, width=90, height=120, guardar=False):
        '''
        Parameters
        ----------
        trai : int
            Transient number to plot.
        name_figure : str
            Name of the class that appears in the figure title 
            
            e.g. Segmentation: {name_figure} - Transient: {trai}
        figure_path : str
            Directory to save the figure. 
        title : str, optional
            Title of the figure, if left as default it will be Segmentation: {name_figure} - Transient: {trai}. The default is None    
        width : int or float, optional
            Width of the figure in millimeters. The default is 90    
        height : int or float, optional
            Height of the figure in millimeters. The default is 120
        guardar : boolean, optional
            "True" to save the image. The default is False.
            
            e.g. Segmentation_{name_figure}_{trai}.pdf
        Returns
        -------
        Plots and (saves) the segmentation (with a single segment) and FFT in pdf format.

        '''   
        # Open tradb files
        tradb = vae.io.TraDatabase(self.tradb_path)

        # Read amplitude and time
        amp, tiempo = tradb.read_wave(trai, time_axis=True)
        amp *= 1e3  # in mV   # The operator *= is updating the amplitude value -- amp = amp * 1e3
                            # The signal was in volts and is represented in mV
                            # This is the amplitude
        time = tiempo * 1e6  # for µs  # Similarly, the signal is stored in seconds and represented in µs
                       # This is the time       
        
        # Fourier transform calculation
        x = np.arange(0, self.N_seg)  # Vector with factors for segmentation
        
        # Create segments
        seg_total = np.zeros((self.N_seg, self.N_samp))  # Array that will store the amplitude values in each segment
                
        for i in x:
            seg_total[i] = amp[int(i * self.N_samp / 2) + self.desfase : int((i/2 + 1) * self.N_samp) + self.desfase]
            
        # Apply Hamming windows to each segment
        seg_t_window = seg_total * self.window  # Apply the window to the segment matrix
        
        # Calculate FFT of the segment matrix
        amp_fft = fft(seg_t_window)   # Calculate the Fourier transform of the segment matrix
        amp_fft = np.abs(amp_fft)    # Calculate the amplitude modulus to not represent with its phase (Real and imaginary part) 
        
        # Select the range of interest frequencies
        amp_fft_rango = amp_fft[:,self.lower_freq_index:self.upper_freq_index + 1]   # Section the amplitude values in the established frequency range
              

        # Multiple plots in one figure
        figsize_inches = (width / 25.4, height/ 25.4)
        fig,axs = plt.subplots(2, self.N_seg, figsize=figsize_inches, dpi=300, tight_layout=True)
        if title is None:
            plt.suptitle(f'Segmentation: {name_figure} - Transient: {trai}', fontsize=10)
        else:
            plt.suptitle(title, fontsize=10)
        plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9)

            
        # Maximum and minimum for axes
        seg_t_window_max = np.amax(abs(seg_t_window))
        amp_fft_max = np.amax(amp_fft_rango)
        
        
        # Time domain
        axs[0].grid(linewidth=0.3, color='.25')
        axs[0].set_ylim(-1.1 * seg_t_window_max, seg_t_window_max * 1.1)
        axs[0].plot(time[int(0 * self.N_samp / 2) + self.desfase : int((0/2 + 1) * self.N_samp) + self.desfase], seg_t_window[0], linewidth=1)
        axs[0].set_xlabel("Time [µs]")
        axs[0].set_ylabel("Amplitude [mV]")
        axs[0].set_title(f"Segment: {0+1}")
        
        # FFT
        axs[1].grid(linewidth=0.3, color='.25')
        axs[1].set_ylim(0, amp_fft_max*1.1)
        axs[1].plot(self.freq_fft_rango, amp_fft_rango[0][:self.N_samp//2], linewidth=1)
        
        # Vertical line in the peak frequency
        peak_index = np.argmax(amp_fft_rango[0])
        peak_freq = self.freq_fft_rango[peak_index]
        y_max = np.amax(amp_fft_rango[0]) *2 / (amp_fft_max*1.1)    
        axs[1].axvline(x = peak_freq, ymax = y_max, color ='red', linestyle="--", label='f_peak')
                
        # Vertical line in the Centroid frequency
        centroid_freq = np.sum(amp_fft_rango[0] * self.freq_fft_rango) / np.sum(amp_fft_rango[0])
        y_max_c = 0.8 * y_max    
        axs[1].axvline(x = centroid_freq, ymax = y_max_c, color ='green', linestyle=":", label='f_centroid')
        
        
        axs[1].set_xlabel("Frequency [kHz]")
        axs[1].set_ylabel("FFT-Magnitude [mV]")
        axs[1].legend()

        # Save the image in figure_path
        if guardar:
            figure_filename = f"Segmentation_{name_figure}_{trai}.pdf"
            figure_path_name = os.path.join(figure_path, figure_filename)
            plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

        plt.show()
        
def unir_df(*dataframes) :
    '''
    Parameters
    ----------
    *dataframes : DataFrame
        DataFrames to concatenate.

    Returns
    -------
    concatenated_df : DataFrame
        Returns a single concatenated DataFrame.

    ''' 
    # Concatenate DataFrames vertically
    concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)
    return concatenated_df

def train_test_set(data, normalization=None, columns_to_transform=None, split=True, test_size=0.2):
    '''
    Parameters
    ----------
    data : DataFrame
        DataFrame containing features, classes, and trais.
    normalization : str, optional
        Normalization for the features.
        'log': logarithmic.
        'std': standard. 
        'min_max': min-max. 
        'log-std': Logarithmic on columns_to_transform and standard on all. The default is None.
    columns_to_transform : list, optional
        List of columns for transformation. 
        If not set, the entire DataFrame is transformed. The default is None.
    split : boolean, optional
        True: To split into train and test sets. The default is True.
    test_size : float, optional
        Percentage of the test set. The default is 0.2.

    Returns
    -------
    X_train : DataFrame
        Features for training.
    X_test : DataFrame
        Features for evaluation.
    y_train : Series
        Classes for training.
    y_test : Series
        Classes for evaluation.
    
    If split=False, only X_original, X_train, y_train, and hits (DataFrame for cumulative hits plot) are returned.
    '''      
    # Separate features from classes
    X = data.drop(["Clase", 'trai', 'time', 'test_id', 'amplitude', 'time_norm' , 'counts', 'rise_time', 'p_power_4', 'p_power_5', 'p_power_6'], axis=1) # Remove class and trais columns (also remove counts and time_norm)
    y = data["Clase"].copy() # Create a copy of the class column
    
    X_original = X.copy()
       
    # Feature names
    feature_names = X.columns.tolist()
    
    # If columns to transform are not defined, select all
    if columns_to_transform is None:
        columns_to_transform = feature_names
        
    # Normalization options
    if normalization == 'log':
        log_transformer = FunctionTransformer(np.log)
        X[columns_to_transform] = log_transformer.transform(X[columns_to_transform]) # Apply logarithm to features
        
    if normalization == 'std':
        standardize = StandardScaler()
        X[columns_to_transform] = standardize.fit_transform(X[columns_to_transform]) # Apply standardization to features
    
    if normalization == 'min_max':
        min_max = MinMaxScaler()
        X[columns_to_transform] = min_max.fit_transform(X[columns_to_transform]) # Apply min-max scaling to features
    
    if normalization == 'log-std':
        log_transformer = FunctionTransformer(np.log)
        X[columns_to_transform] = log_transformer.transform(X[columns_to_transform]) # Apply logarithm to features
        standardize = StandardScaler()
        X = standardize.fit_transform(X) # Apply standardization to features
    
    # The DataFrame returned by the function depends on whether it is split or not
    if split:
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=data["Clase"], random_state=0, shuffle=True)  
        
        # The function returns 4 DataFrames
        return X_train, X_test, y_train, y_test
        
    else:
        X_train = X
        y_train = y
        
        # DataFrame for cumulative hits
        hits = data[['test_id', 'trai', 'Clase', 'time', 'amplitude', 'energy', 'time_norm' , 'counts']].copy()
        hits['Count'] = 1
        
        return X_original, X_train, y_train, hits
   
def backward_elimination(X_train, y_train):
    '''
    Parameters
    ----------
    X_train : DataFrame
        Feature set for training.
    y_train : DataFrame
        Feature set for evaluation.

    Returns
    -------
    selected_feature_names : list
        List of selected feature names.

    '''
    # Classifier
    model = RandomForestClassifier(random_state=11)
    
    # Perform backward elimination with cross-validation
    selector = RFECV(estimator=model, step=1, cv=5, scoring='f1_macro')
    selector.fit(X_train, y_train)

    # Get the selected feature names
    selected_feature_names = selector.get_feature_names_out().tolist()

    # Get the best score
    best_f1_score = selector.cv_results_['mean_test_score'][selector.n_features_ - 1]
    
    # Print the best feature subset and its performance
    print('','Backward Elimination', sep='\n')
    print("Best Feature Subset:", selected_feature_names)
    print(f"Best F1 Score: {best_f1_score:.1%}")

    return selected_feature_names
   
def forest_clf(X_train, X_test, y_train, y_test, selected_features=None):
    ''' 
    Parameters
    ----------
    X_train : dataframe, array
        Feature set for training.
    X_test : dataframe, array
        Feature set for evaluation.
    y_train : series, array
        Classes for training.
    y_test : series, array
        Classes for evaluation.
    selected_features : list, optional
        Names of the features to train the model. The default is None.
    
    Returns
    -------
    y_train_pred : array
        Predicted values for the training set.
    y_test_pred : array
        Predicted values for the evaluation set.
    best_features : dataframe
        DataFrame with features sorted by importance.
    model : ensemble._forest
        Machine learning model that can be exported for predictions.

    '''
    # Feature selection (if backward elimination was used)
    if selected_features is not None:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
    
    # Train the model
    model = RandomForestClassifier(random_state=11)  # Model name
    model.fit(X_train, y_train)  # Train the model
    
    # Model accuracy on the test set
    baseline_accuracy = model.score(X_test, y_test) 
    
    # F1 score of the model on the test set
    y_test_pred = model.predict(X_test)
    f1_score_test = f1_score(y_test, y_test_pred, average="macro")  # Calculate the F1 score
    print('', 'Precision and F1 score on the test set', sep='\n')
    print(f"Baseline Accuracy: {baseline_accuracy:.1%}")  # Print the model accuracy in % format with 1 decimal
    print(f"F1 score - test = {f1_score_test:.1%}")
    
    # Cross-validation that returns the model accuracy (scores) and predicted values (predict)
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)  # Predicted values on the training set with cross-validation
    cross_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy
    cross_f1_score = f1_score(y_train, y_train_pred, average="macro")  # Calculate the F1 score
    print('', 'Cross-validation to check for overfitting', sep='\n')
    print(f"Cross-val Accuracy = {cross_accuracy:.1%}")
    print(f"Cross-val F1 score = {cross_f1_score:.1%}")
    
    # Calculate the importance of each feature for classification
    feature_importances = (model.feature_importances_ * 100).round(2)
    best_features = pd.DataFrame(
        sorted(zip(X_train.columns.tolist(), feature_importances), key=lambda x: x[1], reverse=True),
        columns=["Features", "Importances (%)"])    
    
    # Print the sorted DataFrame
    print('', 'Feature Importance', sep='\n')
    print(best_features.head(5))
    
    return y_train_pred, y_test_pred, best_features, model

def forest_clf_hyper(X_train, X_test, y_train, y_test, selected_features=None):
    '''
    Parameters
    ----------
    X_train : dataframe, array
        Feature set for training.
    X_test : dataframe, array
        Feature set for evaluation.
    y_train : series, array
        Classes for training.
    y_test : series, array
        Classes for evaluation.
    selected_features : list, optional
        Names of the features to train the model. The default is None.

    Returns
    -------
    y_train_pred : array
        Predicted values for the training set.
    y_test_pred : array
        Predicted values for the evaluation set.
    best_features : dataframe
        DataFrame with features sorted by importance.
    best_model : ensemble._forest
        Machine learning model that can be exported for predictions.

    '''
    param_grid = {'max_features': ['sqrt', 'log2'],  # Maximum number of features to consider at each split
                  'n_estimators': np.arange(100, 1000, 100),  # Number of trees in the forest
                  'max_depth': [None] + list(np.arange(5, 30, 5)),  # Maximum depth of each tree
                  'min_samples_split': np.arange(2, 11),  # Minimum number of samples required to split an internal node
                  'min_samples_leaf': np.arange(1, 5),  # Minimum number of samples required to be at a leaf node
                  }

    # Define the scoring metric
    scorer = make_scorer(f1_score, average="macro")
    
    model = RandomForestClassifier(random_state=42)  # Model name
    
    # Perform randomized search
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, scoring=scorer, cv=5, n_iter=10, random_state=42)  # Optimizer
    random_search.fit(X_train, y_train)  # Train the optimizer    
    best_params = random_search.best_params_  # Get the optimized parameters 
    best_score = random_search.best_score_  # Get the F1 score from the optimizer training 
    
    print('', 'Hyperparameter tuning', sep='\n')
    print(f"Best parameters: {best_params}")
    print(f"Optimized F1 score: {best_score:.1%}")

    # Train the model with the best estimator
    best_model = random_search.best_estimator_  # Store the best estimator (best model)
    best_model.fit(X_train, y_train)  # Train the machine with the best estimator
    y_test_pred = best_model.predict(X_test)  # Predictions made with the optimized model on the test set
    
    # Model accuracy on the test set
    baseline_accuracy = accuracy_score(y_test, y_test_pred)
    
    # F1 score of the model on the test set
    f1_score_test = f1_score(y_test, y_test_pred, average="macro")  # Calculate the F1 score
    print('', 'Precision and F1 score on the test set', sep='\n')
    print(f"Baseline Accuracy: {baseline_accuracy:.1%}")  # Print the model accuracy in % format with 1 decimal
    print(f"F1 score - test = {f1_score_test:.1%}")

    # Cross-validation that returns the model accuracy (scores) and predicted values (predict)
    y_train_pred = cross_val_predict(best_model, X_train, y_train, cv=5)  # Predicted values on the training set with cross-validation
    cross_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy
    cross_f1_score = f1_score(y_train, y_train_pred, average="macro")  # Calculate the F1 score
    print('', 'Cross-validation to check for overfitting', sep='\n')
    print(f"Cross-val Accuracy = {cross_accuracy:.1%}")
    print(f"Cross-val F1 score = {cross_f1_score:.1%}")

    # Calculate the importance of each feature for classification 
    feature_importances = (best_model.feature_importances_ * 100).round(2)
    best_features = pd.DataFrame(
        sorted(zip(X_train.columns.tolist(), feature_importances), key=lambda x: x[1], reverse=True),
        columns=["Features", "Importances (%)"])    
    
    # Print the sorted DataFrame
    print('', 'Feature Importance', sep='\n')
    print(best_features.head(5))

    return y_train_pred, y_test_pred, best_features, best_model

def conf_matrix(y, y_pred, name_figure, figure_path, width=90, height=60, guardar=False):
    '''
    Parameters
    ----------
    y : series or array
        True classes of the dataset.
    y_pred : series or array
        Predicted classes by the model.
    name_figure : str
        Name of the class that appears in the figure title.
        
        e.g. Confusion matrix - {name_figure}
    figure_path : str
        Directory to save the figure.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90  
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    guardar : boolean, optional
        "True" to save the image. The default is False.
        
        e.g. Confusion matrix {name_figure}.pdf
    Returns
    -------
    Plots and (saves) the confusion matrix in pdf format.

    '''
    # Plot the confusion matrix    
    figsize_inches = (width / 25.4, height / 25.4)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize_inches, dpi=300, tight_layout=True)
    plt.subplots_adjust(wspace=0.4)
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=axs[0])
    axs[0].set_title("By element")
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=axs[1],
                                            normalize="true", values_format=".0%")
    axs[1].set_title("Normalized by row")
    plt.suptitle(f"Confusion matrix - {name_figure}", fontsize=14)
   
    # Save the image in figure_path
    if guardar:
        figure_filename = f"Confusion matrix {name_figure}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

    plt.show()

def plot_tsne(X, y, figure_path, title='t-SNE Scatter Plot', x_label='t-SNE Dimension 1', 
              y_label='t-SNE Dimension 2', width=90, height=60, selected_features=None, guardar=False):
    '''
    Parameters
    ----------
    X : DataFrame, array
        Feature set.
    y : Series, array
        Classes of the dataset.
    figure_path : str
        Directory to save the figure.
    title : str, optional
        Title of the figure. The default is 't-SNE Scatter Plot'.
    x_label : str, optional
        Label for the x-axis. The default is 't-SNE Dimension 1'.
    y_label : str, optional
        Label for the y-axis. The default is 't-SNE Dimension 2'.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90.
    height : int or float, optional
        Height of the figure in millimeters. The default is 60.
    selected_features : list, optional
        Names of the features to train the model. The default is None.
    guardar : boolean, optional
        "True" to save the image. The default is False.
        
        e.g. t-SNE_{name_figure}.pdf

    Returns
    -------
    Plots and (saves) the dimensionality reduction with class labels and markers in pdf format.

    '''
    # Feature selection (if backward elimination was used)
    if selected_features is not None:
        X = X[selected_features]
            
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X)
    
    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f't-SNE_{i+1}' for i in range(2)])    
    
    y = pd.Series(y, name='Labels')
    y.index = tsne_df.index
    tsne_df['Labels'] = y
    
    # Define custom markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Customize the markers as desired
        
    # Create a figure
    figsize_inches = (width / 25.4, height / 25.4)
    plt.figure(figsize=figsize_inches, dpi=300, tight_layout=True)
    
    # Use Seaborn scatter plot with hue and style parameters
    sns.scatterplot(x='t-SNE_1', y='t-SNE_2', hue='Labels', style='Labels', data=tsne_df, palette=sns.color_palette(), markers=markers)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points    
    
    # Save the image in figure_path
    if guardar:
        figure_filename = f"{title}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

    plt.show()   

def plot_histogram(X, y, figure_path, width=90, height=60, bins=30, feature_to_save='', guardar=False):
    '''
    Parameters
    ----------
    X : DataFrame, array
        Feature set.
    y : Series, array
        Classes of the dataset.
    figure_path : str
        Directory to save the figure.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90.
    height : int or float, optional
        Height of the figure in millimeters. The default is 60.
    bins : int, optional
        Number of bins for the histogram. The default is 30.
    feature_to_save : str, optional
        Feature to save its histogram. The default is ''.
    guardar : boolean, optional
        "True" to save the image. The default is False.
        
        e.g. Histogram_{feature_to_save}.pdf

    Returns
    -------
    Plots all feature histograms and (saves) only 'feature_to_save' in pdf format.

    '''
    # Directory and filename to save the image
    figure_filename = f"Histogram_{feature_to_save}.pdf"
    figure_path_name = os.path.join(figure_path, figure_filename)
    
    # Iterate over each column
    for feature in X.columns:
    
        # Set up the figure and axis
        figsize_inches = (width / 25.4, height / 25.4)
        fig, ax = plt.subplots(figsize=figsize_inches, dpi=300, tight_layout=True)
        # Set the number of bins for the histogram
        bins = 30
        
        # Plot the histogram for each class
        class_labels = y.unique()
        for class_label in class_labels:
            feat_to_plot = X[y == class_label][feature]
            ax.hist(feat_to_plot, bins=bins, alpha=0.6, label=class_label, edgecolor="black")
    
        # Add labels and a title
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram - {feature}')
    
        # Add a legend
        ax.legend()
        
        # Save the image in figure_path
        if guardar and (feature == feature_to_save):
            plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
        
        # Display the plot
        plt.show()
    
def plot_feat_vs_feat(X, y, feat_1, feat_2, figure_path, title=None, subtitle=None,
                      x_label=None, y_label=None, width=90, height=60, ax=None, 
                      i=1, n_col=1, n_row=1, guardar=False):
    '''
    Parameters
    ----------
    X : DataFrame, array
        Feature set.
    y : Series, array
        Classes of the dataset.
    feat_1 : str
        Name of feature 1 to plot.
    feat_2 : str
        Name of feature 2 to plot.
    figure_path : str
        Directory to save the figure.
    title : str or list, optional
        Title of the figure, if left as default it will be {feat_1} vs {feat_2}. The default is None.
    subtitle : str or list, optional
        Subtitle of the figure, if left as default it will be {feat_1} vs {feat_2}. The default is None.
    x_label : str, optional
        Label for the x-axis. The default is None.
    y_label : str, optional
        Label for the y-axis. The default is None.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90.
    height : int or float, optional
        Height of the figure in millimeters. The default is 60.
    ax : axes, optional
        Axes subplot. The default is None.
    i : int, optional
        Counter to plot each subplot. The default is 1.
    n_col : int, optional
        Number of columns. The default is 1.
    n_row : int, optional
        Number of rows. The default is 1.
    guardar : boolean, optional
        "True" to save the image. The default is False.
        
        e.g. Feat_vs_Feat_{feat_1}_{feat_2}.pdf

    Returns
    -------
    Plots and (saves) a scatter plot of two features with labels and markers in pdf format.

    '''
    y = pd.Series(y, name='Labels')
    # Rename 'Labels' column in y if it already exists in X
    if 'Labels' in X.columns:
        X = X.drop(['Labels'], axis=1)    
    
    y.index = X.index
    datos = pd.concat([X, y], axis=1)
      
    # Define custom markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Customize the markers as desired
    
    # Create a scatter plot
    figsize_inches = (width / 25.4, height / 25.4)
    
    # If there is just one plot and set the title
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_inches, dpi=300, tight_layout=True)
        if title is None:
            ax.set_title(f'{feat_1} vs {feat_2}')
        else:
            ax.set_title(title)
        if x_label is None:
            ax.set_xlabel(feat_1)
        else:
            ax.set_xlabel(x_label)
          
    else:
        if x_label is None:
            ax.set_xlabel(f'{feat_1}\n(' + string.ascii_lowercase[i-1] + ')')
        else:
            ax.set_xlabel(x_label + '\n(' + string.ascii_lowercase[i-1] + ')')
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(f'{feat_1} vs {feat_2}')
    
    # Create the plot
    sns.scatterplot(x=feat_1, y=feat_2, hue='Labels', style='Labels', data=datos, markers=markers, palette=sns.color_palette(), ax=ax)
    
    # Set y-axis label only for the first plot of each row
    if (i-1) % n_col == 0:
        if y_label is None:
            ax.set_ylabel(feat_2)
        else:
            ax.set_ylabel(y_label)
    else:
        ax.set_ylabel('')   
    
    # Logarithmic scale if you choose energy
    if feat_1 == 'energy': 
        ax.set_xscale('log')
    
    if feat_2 == 'energy':
        ax.set_yscale('log')
    
    # Activate the grid
    ax.grid()
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if guardar:
        if ax is None:
            figure_filename = f"Feat_vs_Feat_{feat_1}_{feat_2}.pdf"
        else:
            figure_filename = 'Feat_vs_Feat_plot.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

    return ax