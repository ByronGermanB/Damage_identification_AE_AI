# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:24:10 2023

@author: bbarmac
"""

#%%
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
from scipy.signal import hamming

# Tratamiento de datos y normalizacion
# from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
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
matplotlib.rcParams['figure.dpi'] = 300

#%%
# =============================================================================
# Extraccion de features
# =============================================================================

class Features:
    def __init__(self, path, lower_freq, upper_freq, sampling_rate, N_samp, N_seg, desfase, desfase_carga):
        '''        
        Parameters
        ----------
        path : str
            Directorio donde estan los archivos pri y tra.
        lower_freq : int, float
            Limite inferior de frecuencia en kHz.
        upper_freq : int, float
            Limite superior de frecuencia en kHz.
        sampling_rate : int, float
            Frecuencia de adquisicion de datos por segundo en MHz.
        N_samp : int
            Numero de muestras (datos) en cada segmento para analisis de Fourier (Preferiblemente potencia de 2).
        N_seg : int
            Numero total de segmentos a analizar.
        desfase : int
            Indice para tomar la senal desde el inicio o con un desfase positivo, depende del pretrigger.

        Returns
        -------
        Objeto con inicializacion de valores para aplicar funciones de extraccion y graficas.

        '''
        self.path = path
        self.lower_freq = lower_freq    # Limite inferior de frecuencia en kHz
        self.upper_freq = upper_freq    # Limite superior de frecuencia en kHz
        self.sampling_rate = sampling_rate # Frecuencia de adquisicion de datos por segundo en MHz
        self.N_samp = N_samp               # Numero de muestras (datos) en cada segmento
        self.N_seg = N_seg                 # Numero de segmentos
        self.desfase = desfase             # Depende del tiempo de pretrigger permite empezar desde el inicio de la senal o con un desfase positivo
        self.desfase_carga = desfase_carga # Desfase del tiempo tomado en AE y en MTS
        
        # Abrir archivos pri y tra
        files =  os.listdir(self.path)
        for item in files:
            if item.endswith(".tradb"):
                self.tradb_path = os.path.join(self.path, item)
            
            elif item.endswith(".pridb"):
                self.pridb_path = os.path.join(self.path, item)
        
        # Calculo de valores 
        self.res_freq = sampling_rate * 1000 / N_samp # Resolucion en frecuencia en kHz
        self.time_samp = 1 / sampling_rate   # Tiempo de muestreo en us (microsegundos)
        self.time_seg = self.time_samp * N_samp   # Tiempo de cada segmento en us
        
        # Calculo de indices del rango de frecuencias
        self.lower_freq_index = int(self.lower_freq // self.res_freq) # Indice de la frecuencia inferior
        self.upper_freq_index = int(self.upper_freq // self.res_freq +1) # Indice de la frecuencia superior
        self.N_feat_seg = 9  # Numero de features por segmento
        
        # Valores para la FFT que solo se calculan una vez
        self.window = hamming(self.N_samp)  # Creamos la ventana Hamming
        self.freq_fft = (fftfreq(self.N_samp, self.time_samp*1e-6)/1000)[:self.N_samp//2]  # Valores positivos de frecuencias de la FFT en kHz
        self.freq_fft_rango = self.freq_fft[self.lower_freq_index:self.upper_freq_index + 1]   # Secciona los valores de frecuencia
                  
    def feature_extr(self, umbral, counts, clase, test_id, max_trai=None, min_trai=1):
        '''

        Parameters
        ----------
        umbral : int
            Valor umbral de amplitud para filtrado.
        counts : int
            Valor minimo de counts para filtrado.
        clase : int or str
            Clase a la que pertenece la senal.
        test_id : str
            Codigo del ensayo para luego identificarlo
        max_trai : int, optional
            Indice del trai maximo a considerar, si se quiere excluir la rotura. The default is None.
        min_trai : int, optional
            Indice del trai minimo a considerar, si se quiere excluir algún dato de inicio erróneo. The default is 1.
        
        Returns
        -------
        df_total : Dataframe 
            Dataframe que contiene features de frecuencia + rise time, clases y numero de trai.

        '''
  
        # Lectura de archivos 
        pridb = vae.io.PriDatabase(self.pridb_path)
        tradb = vae.io.TraDatabase(self.tradb_path)
        df_hits = pridb.read_hits() # Leemos los hits que se han producido

        # Filtrado de la senal (trai umbral y counts)
        umbral_V = 10 ** (umbral/20 - 6)  # Umbral en Volvtios
        no_saturacion = 10 ** (94/20 - 6)
                
        if max_trai is not None:
            df_hits_filtro = df_hits[(df_hits["channel"] >= 1) & (df_hits["amplitude"] >= umbral_V) & (df_hits["amplitude"] <= no_saturacion) & (df_hits["trai"] >= min_trai) & (df_hits["trai"] <= max_trai) & (df_hits["counts"] >= counts)]  # Seleccionamos solo los valores cumplen las condiciones
        else:
            df_hits_filtro = df_hits[(df_hits["channel"] >= 1) & (df_hits["amplitude"] >= umbral_V) & (df_hits["amplitude"] <= no_saturacion) & (df_hits["trai"] >= min_trai) & (df_hits["counts"] >= counts)]  # Seleccionamos solo los valores cumplen las condiciones
        
        trai = df_hits_filtro["trai"].to_numpy() # Extraemos la columna con los valores TRAI (indices de transitorio)
        N_trai = trai.size  # Numero de transitorios totales
        
        # Creaccion de matriz para almacenar features  
        v_features_total = np.zeros((N_trai , self.N_seg * self.N_feat_seg))  # Array que almacenara los features en cada iteracion
        
        x = np.arange(0, self.N_seg)  # Vector con factores para la segmentacion

        # Ciclo for principal para cada transitorio
        N_iter = 0 # Esto va a contar las iteraciones si por alguna razon los trai no siguen la secuencia de 1 en 1 (e.g uso de filtro)
        for trans in trai:
            amp, tiempo = tradb.read_wave(trans, time_axis=False) # Leemos el transitorio i-esimo
            
            amp *= 1e3     # in mV   # El operador *= esta actualizando el valor de amplitud -- amplitud = amplitud * 1e3
                                # La senal estaba en voltios y la representa en mV
                                # Esta es la amplitud
            # time = tiempo * 1e6  # for us  # Lo mismo la senal se almacena en segundos y se representa en us
                           # Este es el tiempo
            
            
            # Creacion de los segmentos
            seg_total = np.zeros((self.N_seg, self.N_samp))  # Array que almacenara los valores de las amplitudes en cada segmento
        
            # Ciclo for  para aplicar la transformada de Fourier
            for i in x:
                seg_total[i] = amp[int(i * self.N_samp / 2) + self.desfase : int((i/2 + 1) * self.N_samp) + self.desfase]
                
            # Aplicar ventanas Hamming a cada segmento
            seg_t_window = seg_total * self.window  # Aplicamos la ventana a la matriz de segmentos
            
            # Calcular FFT la matriz de segmento
            amp_fft = fft(seg_t_window)   # Calcula la transformada de Fourier de la matriz de segmentos
            amp_fft = np.abs(amp_fft)    # Calcula el modulo de la amplitud para no representar con su fase (Parte real e imaginaria) 
            
            # Escoger el rango de frecuencias de interes
            amp_fft_rango = amp_fft[:,self.lower_freq_index:self.upper_freq_index + 1]   # Secciona los valores de amplitud en el rango de frecuencias establecido
            
            # Inicializacion de vectores de features en frecuencia para cada segmento
            peak_freq = np.zeros(self.N_seg) # Almacena los peak freq de cada segmento
            centroid_freq = np.zeros(self.N_seg) # Almacena el centroide de frecuencia de cada segmento
            part_power_1 = np.zeros(self.N_seg) # Almacena la potencia parcial 1 total de cada segmento
            part_power_2 = np.zeros(self.N_seg) # Almacena la potencia parcial 2 total de cada segmento
            part_power_3 = np.zeros(self.N_seg) # Almacena la potencia parcial 3 total de cada segmento
            part_power_4 = np.zeros(self.N_seg) # Almacena la potencia parcial 4 total de cada segmento
            part_power_5 = np.zeros(self.N_seg) # Almacena la potencia parcial 5 total de cada segmento
            part_power_6 = np.zeros(self.N_seg) # Almacena la potencia parcial 6 total de cada segmento
        
            # Ciclo for para calculo de features
            for i in x:
                # Peak frequency
                peak_index = np.argmax(amp_fft_rango[i])
                peak_freq[i] = self.freq_fft_rango[peak_index]
                
                # Centroid frequency
                centroid_freq[i] = np.sum(amp_fft_rango[i] * self.freq_fft_rango) / np.sum(amp_fft_rango[i])
                
                # Partial Power 1 - 6
                denominador = np.sum((amp_fft_rango[i]) ** 2)
                part_power_1[i] = np.sum((amp_fft_rango[i, 0:int(len(self.freq_fft_rango)*1/6)]) ** 2) * 100 / denominador
                part_power_2[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*1/6):int(len(self.freq_fft_rango)*2/6)]) ** 2) *100 / denominador
                part_power_3[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*2/6):int(len(self.freq_fft_rango)*3/6)]) ** 2) *100 / denominador
                part_power_4[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*3/6):int(len(self.freq_fft_rango)*4/6)]) ** 2) *100 / denominador
                part_power_5[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*4/6):int(len(self.freq_fft_rango)*5/6)]) ** 2) *100 / denominador
                part_power_6[i] = np.sum((amp_fft_rango[i, int(len(self.freq_fft_rango)*5/6):]) ** 2) *100 / denominador
                
            w_peak_freq = (peak_freq * centroid_freq) ** 0.5
            
            # Vector de features
            v_features = np.hstack((peak_freq, w_peak_freq, centroid_freq, part_power_1, part_power_2, part_power_3, part_power_4, part_power_5, part_power_6)) # Agrupamos los features
            v_features = v_features.reshape(1, self.N_feat_seg * self.N_seg) # Reshape de la matriz de features a un vector de una sola fila
            v_features_total[N_iter] = v_features # Encadenamos los valores a cada fila de la matriz para crear la matriz de features

            N_iter +=  1  # Actualizamos el contador

        # Nombre de los features

        if self.N_seg == 3:
            name_feature = ['peak_freq_1', 'peak_freq_2', 'peak_freq_3', 
                            'w_peak_freq_1', 'w_peak_freq_2', 'w_peak_freq_3', 
                            'centroid_freq_1', 'centroid_freq_2', 'centroid_freq_3',
                            'p_power_1_1', 'p_power_1_2', 'p_power_1_3',
                            'p_power_2_1', 'p_power_2_2', 'p_power_2_3',
                            'p_power_3_1', 'p_power_3_2', 'p_power_3_3',
                            'p_power_4_1', 'p_power_4_2', 'p_power_4_3',
                            'p_power_5_1', 'p_power_5_2', 'p_power_5_3',
                            'p_power_6_1', 'p_power_6_2', 'p_power_6_3']  # Array que almacenara los nombres de los features por segmento  
        
        elif self.N_seg == 1:
            name_feature = ['peak_freq', 
                            'w_peak_freq',  
                            'centroid_freq', 
                            'p_power_1', 
                            'p_power_2',
                            'p_power_3',
                            'p_power_4',
                            'p_power_5',
                            'p_power_6']  # Array que almacenara los nombres de los features por segmento  

        # Dataframe de features
        df_features = pd.DataFrame(v_features_total,
                                  columns = name_feature)
        
        # Dataframe de trais
        df_trai = pd.DataFrame(trai, columns = ['trai'])

        # Dataframe de time  y rise time 
        df_time = df_hits_filtro[['rise_time', 'time', 'amplitude', 'energy', 'counts']].reset_index().drop(['set_id'], axis=1)
        df_time['amplitude'] = 20 * (np.log10(df_time['amplitude']) + 6)
        df_time['time'] = df_time['time'] - self.desfase_carga
        df_time['time_norm'] = df_time['time'] / max(df_time['time'])
        
        # Dataframe de clases
        df_clase = pd.DataFrame({'Clase' : [clase] * N_trai})
        
        
        # DataFrame total
        df_total = pd.concat([df_features, df_time, df_clase, df_trai], axis=1)   
        df_total['test_id'] = test_id
        
        # La funcion regresa el DataFrame total
        return (df_total)
        
    def plot_signal(self, trai, name_figure, figure_path, time_graph = None, title = None, x_label = 'Time [µs]', 
                    y_label = 'Amplitude [mV]' , width=90, height=60, guardar=False):
        '''        
        Parameters
        ----------
        trai : int
            Numero de transitorio a graficar.
        name_figure : str
            Nombre de la clase que aparece en titulo de la figura.
            
            e.g. Transient Wave Plot: {name_figure} - Transient: {trai}
        figure_path : str
            Directorio para guardar la figura.
        time_graph : int or float, optional
            Tiempo en us para hasta donde se quiere la grafica
        title : str, optional
            titulo de la figura. The default is None
        x_label : str, optional
            leyenda del eje x. The default is 'Time [µs]'
        y_label : str, optional
            leyenda del eje y. The default is 'Amplitude [mV]'
        width : int or float, optional
            Ancho de la figura en milimetros. The default is 90    
        height : int or float, optional
            Alto de la figura en milimetros. The default is 60    
        guardar : boolean, optional
            "True" para guardar la imagen. The default is False.
            
            e.g. Transitorio_{name_figure}_{trai}.pdf
        Returns
        -------
        Grafica y (guarda) la senal transitoria en formato pdf

        '''  
        # Abrir archivos tradb
        tradb = vae.io.TraDatabase(self.tradb_path)

        # Graficamos del transitorio
        amp, tiempo = tradb.read_wave(trai, time_axis=True)
        amp *= 1e3  # in mV   # El operador *= esta actualizando el valor de amplitud -- amp = amp * 1e3
                            # La senal estaba en voltios y la representa en mV
                            # Esta es la amp
        time = tiempo * 1e6  # for µs  # Lo mismo la senal se almacena en segundos y se representa en us
                       # Este es el tiempo
        
        figsize_inches = (width / 25.4, height/ 25.4)
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
        
        # Guardar la imagen en figure_path
        if guardar:
            figure_filename = f"Transitorio_{name_figure}_{trai}.pdf"
            figure_path_name = os.path.join(figure_path, figure_filename)
            plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

        plt.show()

    def plot_segmentation(self, trai, name_figure, figure_path, guardar=False):
        '''
        Parameters
        ----------
        trai : int
            Numero de transitorio a graficar.
        name_figure : str
            Nombre de la clase que aparece en titulo de la figura 
            
            e.g. Segementation: {name_figure} - Transient: {trai}
        figure_path : str
            Directorio para guardar la figura.          
        guardar : boolean, optional
            "True" para guardar la imagen. The default is False.
            
            e.g. Segmentation_{name_figure}_{trai}.pdf
        Returns
        -------
        Grafica y (guarda) la segmentacion y FFT en formato pdf.

        '''   
        # Abrir archivos tradb
        tradb = vae.io.TraDatabase(self.tradb_path)

        # Leemos la amplitud y el tiempo
        amp, tiempo = tradb.read_wave(trai, time_axis=True)
        amp *= 1e3  # in mV   # El operador *= esta actualizando el valor de amplitud -- amp = amp * 1e3
                            # La senal estaba en voltios y la representa en mV
                            # Esta es la amp
        time = tiempo * 1e6  # for µs  # Lo mismo la senal se almacena en segundos y se representa en us
                       # Este es el tiempo       
        
        # Calculo de la trasnformada de Fourier
        x = np.arange(0, self.N_seg)  # Vector con factores para la segmentacion
        
        # Creacion de los segmentos
        seg_total = np.zeros((self.N_seg, self.N_samp))  # Array que almacenara los valores de las amplitudes en cada segmento
                
        for i in x:
            seg_total[i] = amp[int(i * self.N_samp / 2) + self.desfase : int((i/2 + 1) * self.N_samp) + self.desfase]
            
        # Aplicar ventanas Hamming a cada segmento
        seg_t_window = seg_total * self.window  # Aplicamos la ventana a la matriz de segmentos
        
        # Calcular FFT la matriz de segmento
        amp_fft = fft(seg_t_window)   # Calcula la transformada de Fourier de la matriz de segmentos
        amp_fft = np.abs(amp_fft)    # Calcula el modulo de la amplitud para no representar con su fase (Parte real e imaginaria) 
        
        # Escoger el rango de frecuencias de interes
        amp_fft_rango = amp_fft[:,self.lower_freq_index:self.upper_freq_index + 1]   # Secciona los valores de amplitud en el rango de frecuencias establecido
              

        # Varias gráficas en una figura
        fig,axs = plt.subplots(2, self.N_seg,figsize=(15, 7.5))
        plt.suptitle(f'Segementation: {name_figure} - Transient: {trai}', fontsize=14)
        plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9)

            
        # Maximos y minimos para ejes
        seg_t_window_max = np.amax(abs(seg_t_window))
        amp_fft_max = np.amax(amp_fft_rango)
            
        for j in x:
            
            # Time domain
            axs[0,j].grid(linewidth=0.3, color='.25')
            axs[0,j].set_ylim(-1.1 * seg_t_window_max, seg_t_window_max * 1.1)
            axs[0,j].plot(time[int(j * self.N_samp / 2) + self.desfase : int((j/2 + 1) * self.N_samp) + self.desfase], seg_t_window[j])
            axs[0,j].set_xlabel("Time [µs]")
            axs[0,0].set_ylabel("Amplitude [mV]")
            axs[0,j].set_title(f"Segment: {j+1}")
            
            # FFT
            axs[1,j].grid(linewidth=0.3, color='.25')
            axs[1,j].set_ylim(0, amp_fft_max*1.1)
            axs[1,j].plot(self.freq_fft_rango, amp_fft_rango[j][:self.N_samp//2])
            
            # Vertical line in the peak frequency
            peak_index = np.argmax(amp_fft_rango[j])
            peak_freq = self.freq_fft_rango[peak_index]
            y_max = np.amax(amp_fft_rango[j]) *2 / (amp_fft_max*1.1)    
            axs[1,j].axvline(x = peak_freq, ymax = y_max, color ='red', linestyle="--", label='f_peak')
                    
            # Vertical line in the Centroid frequency
            centroid_freq = np.sum(amp_fft_rango[j] * self.freq_fft_rango) / np.sum(amp_fft_rango[j])
            y_max_c = 0.8 * y_max    
            axs[1,j].axvline(x = centroid_freq, ymax = y_max_c, color ='green', linestyle=":", label='f_centroid')
            
            
            axs[1,j].set_xlabel("Frequency [kHz]")
            axs[1,0].set_ylabel("FFT-Magnitude [mV]")
            axs[1,j].legend()

        
        # Add a single border around all subplots
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(1)

        # Guardar la imagen en figure_path
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
            Numero de transitorio a graficar.
        name_figure : str
            Nombre de la clase que aparece en titulo de la figura 
            
            e.g. Segementation: {name_figure} - Transient: {trai}
        figure_path : str
            Directorio para guardar la figura. 
        title : str, optional
            Titulo de la figura, si se deja en default sera Segementation: {name_figure} - Transient: {trai}. The default is None    
        width : int or float, optional
            Ancho de la figura en milimetros. The default is 90    
        height : int or float, optional
            Alto de la figura en milimetros. The default is 60
        guardar : boolean, optional
            "True" para guardar la imagen. The default is False.
            
            e.g. Segmentation_{name_figure}_{trai}.pdf
        Returns
        -------
        Grafica y (guarda) la segmentacion (con un solo segmento) y FFT en formato pdf.

        '''   
        # Abrir archivos tradb
        tradb = vae.io.TraDatabase(self.tradb_path)

        # Leemos la amplitud y el tiempo
        amp, tiempo = tradb.read_wave(trai, time_axis=True)
        amp *= 1e3  # in mV   # El operador *= esta actualizando el valor de amplitud -- amp = amp * 1e3
                            # La senal estaba en voltios y la representa en mV
                            # Esta es la amp
        time = tiempo * 1e6  # for µs  # Lo mismo la senal se almacena en segundos y se representa en us
                       # Este es el tiempo       
        
        # Calculo de la trasnformada de Fourier
        x = np.arange(0, self.N_seg)  # Vector con factores para la segmentacion
        
        # Creacion de los segmentos
        seg_total = np.zeros((self.N_seg, self.N_samp))  # Array que almacenara los valores de las amplitudes en cada segmento
                
        for i in x:
            seg_total[i] = amp[int(i * self.N_samp / 2) + self.desfase : int((i/2 + 1) * self.N_samp) + self.desfase]
            
        # Aplicar ventanas Hamming a cada segmento
        seg_t_window = seg_total * self.window  # Aplicamos la ventana a la matriz de segmentos
        
        # Calcular FFT la matriz de segmento
        amp_fft = fft(seg_t_window)   # Calcula la transformada de Fourier de la matriz de segmentos
        amp_fft = np.abs(amp_fft)    # Calcula el modulo de la amplitud para no representar con su fase (Parte real e imaginaria) 
        
        # Escoger el rango de frecuencias de interes
        amp_fft_rango = amp_fft[:,self.lower_freq_index:self.upper_freq_index + 1]   # Secciona los valores de amplitud en el rango de frecuencias establecido
              

        # Varias gráficas en una figura
        figsize_inches = (width / 25.4, height/ 25.4)
        fig,axs = plt.subplots(2, self.N_seg, figsize=figsize_inches, dpi=300, tight_layout=True)
        if title is None:
            plt.suptitle(f'Segementation: {name_figure} - Transient: {trai}', fontsize=10)
        else:
            plt.suptitle(title, fontsize=10)
        plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9)

            
        # Maximos y minimos para ejes
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

        # Guardar la imagen en figure_path
        if guardar:
            figure_filename = f"Segmentation_{name_figure}_{trai}.pdf"
            figure_path_name = os.path.join(figure_path, figure_filename)
            plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

        plt.show()
        
def unir_df(*dataframes) :
    '''
    Parameters
    ----------
    *dataframes : Dataframe
        Dataframes para concatenar.

    Returns
    -------
    concatenated_df : Dataframe
        Regresa un solo dataframe unido

    ''' 
    # Concatena dataframes verticalmente
    concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)
    return (concatenated_df)

def train_test_set(data, normalization=None, columns_to_transform=None, split=True, test_size=0.2):
    '''
    Parameters
    ----------
    data : Dataframe
        Dataframe que contiene los features, clases y trais.
    normalization : str, optional
        Normalizacion para los features.
        'log': logaritmica.
        'std': estandar. 
        'min_max': minimo y maximo. 
        'log-std' Logaritmica en columns_to_transform y estandar en todo. The default is None.
    columns_to_transform : list, optional
        Lista de columnas para la transformacion. 
        Si no se establece se transforma todo el dataframe. The default is None.
    split : boolean, optional
        True: Para dividir en train y test set. The default is True.
    test_size : float, optional
        Porcentaje del test set. The default is 0.2.

    Returns
    -------
    X_train : Dataframe
        Features para entrenamiento.
    X_test : Dataframe
        Features para evaluacion.
    y_train : Series
        Clases para entrenamiento.
    y_test : Series
        Clases para evaluacion.
    
    Si split=False solo se devuelve X_original, X_train, y_train, y hits(dataframe para grafica de hits acumulados)
    '''      
    # Separamos los features de las clases
    X = data.drop(["Clase", 'trai', 'time', 'test_id', 'amplitude', 'time_norm' , 'counts', 'rise_time', 'p_power_4', 'p_power_5', 'p_power_6'], axis=1) # Elimina la columna de clases y trais (eliminar, tambien counts y time_norm)
    y = data["Clase"].copy() # Cremos una copia solo de la columna clase
    
    X_original = X.copy()
       
    # Feature names
    feature_names = X.columns.tolist()
    
    # Si no se ha definido las columnas a trasnformar se seleccionan todas
    if columns_to_transform is None:
        columns_to_transform = feature_names
        
    # Opciones de nomralizacion
    if normalization == 'log':
        log_transformer = FunctionTransformer(np.log, feature_names_out="one-to-one")
        X[columns_to_transform] = log_transformer.transform(X[columns_to_transform]) # Aplicamos el logaritmo a los features
        
    if normalization == 'std':
        standardize = StandardScaler()
        X[columns_to_transform] = standardize.fit_transform(X[columns_to_transform]) # Aplicamos la estandarizacion a los features
    
    if normalization == 'min_max':
        min_max = MinMaxScaler()
        X[columns_to_transform] = min_max.fit_transform(X[columns_to_transform]) # Aplicamos la estandarizacion a los features
    
    if normalization == 'log-std':
        log_transformer = FunctionTransformer(np.log, feature_names_out="one-to-one")
        X[columns_to_transform] = log_transformer.transform(X[columns_to_transform]) # Aplicamos el logaritmo a los features
        standardize = StandardScaler()
        X = standardize.fit_transform(X) # Aplicamos la estandarizacion a los features
    
    # El dataframe que devuele la funcion depende si se particiona o no
    if split:
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=data["Clase"], random_state=0, shuffle=True)  
        
        # La funcion retorna 4 dataframes
        return X_train, X_test, y_train, y_test
        
    else:
        X_train = X
        y_train = y
        
        # Dataframe para hits acumulados
        hits = data[['test_id', 'trai', 'Clase', 'time', 'amplitude', 'energy', 'time_norm' , 'counts']].copy()
        hits['Count'] = 1
        # hits['Cumulative'] = hits['Count'].cumsum()
        
        return X_original, X_train, y_train, hits
   
def backward_elimantion(X_train, y_train):
    '''
    Parameters
    ----------
    X_train : dataframe
        Set de features para entrenamiento.
    y_train : dataframe
        Set de features para evaluacion.

    Returns
    -------
    selected_feature_names : list
        Lista con los nombres de los features seleccionados.

    '''
    # Clasificador
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
        Set de features para entrenamiento.
    X_test : dataframe, array
        Set de features para evaluacion.
    y_train : series, array
        Clases para entrenamiento.
    y_test : series, array
        Clases para evaluacion.
    selected_features : list, optional
        Nombre de los features para entrenar el modelo. The default is None.
    
    Returns
    -------
    y_train_pred : array
        Valores predicios para el set de entrenamiento.
    y_test_pred : array
        Valores predicios para el set de evaluacion.
    best_features : dataframe
        Dataframe con los features ordenados por orden de importancia.
    model : esemble._forest
        Modelo de machine learning que se puede exportar para predicciones.

    '''
    # Seleccion de features (si se utilizo backward elimination)
    if selected_features is not None:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
    
    # Entrenar el modelo 
    model = RandomForestClassifier(random_state=11)  # Nombre del modelo
    model.fit(X_train, y_train)       # Entrenamiento del modelo
    
    # Accuracy del modelo en el test_set
    baseline_accuracy = model.score(X_test, y_test) 
    
    # F1 score del modelo en el test_set
    y_test_pred = model.predict(X_test)
    f1_score_test = f1_score(y_test, y_test_pred, average="macro") # Calcula el F1 score
    print('','Precision y F1 score en el test set', sep='\n')
    print(f"Baseline Accuracy: {baseline_accuracy:.1%}") # Imprime el accuracy del modelo en formato % con 1 decimal
    print(f"F1 score - test = {f1_score_test:.1%}")
    
    # Cross-validation que nos devuelve la precision del modelo (scores) y los valores predichos (predict)
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)  # Valores predichos en el training set con validacion cruzada
    cross_accuracy = accuracy_score(y_train, y_train_pred)  # Calcula accuracy
    cross_f1_score = f1_score(y_train, y_train_pred, average="macro")     # Calcula el F1 score
    print('','Validacion cruzada para verificar overfitting', sep='\n')
    print(f"Cross-val Accuracy = {cross_accuracy:.1%}")
    print(f"Cross-val F1 score = {cross_f1_score:.1%}")
    
    # Calculamos la importancia de cada features para la clasificacion 
    feature_importances = (model.feature_importances_ * 100).round(2)
    best_features = pd.DataFrame(
        sorted(zip(X_train.columns.tolist(), feature_importances), key=lambda x: x[1], reverse=True),
        columns = ["Features", "Importances (%)"])    
    
    # Print the sorted DataFrame
    print('','Importancia de Features', sep='\n')
    print(best_features.head(5))
    
    return y_train_pred, y_test_pred, best_features, model

def forest_clf_hyper(X_train, X_test, y_train, y_test, selected_features=None):
    '''
    Parameters
    ----------
    X_train : dataframe, array
        Set de features para entrenamiento.
    X_test : dataframe, array
        Set de features para evaluacion.
    y_train : series, array
        Clases para entrenamiento.
    y_test : series, array
        Clases para evaluacion.
    selected_features : list, optional
        Nombre de los features para entrenar el modelo. The default is None.

    Returns
    -------
    y_train_pred : array
        Valores predicios para el set de entrenamiento.
    y_test_pred : array
        Valores predicios para el set de evaluacion.
    best_features : dataframe
        Dataframe con los features ordenados por orden de importancia.
    best_model : esemble._forest
        Modelo de machine learning que se puede exportar para predicciones.

    '''
    param_grid = {'max_features': ['sqrt', 'log2'],  # Maximum number of features to consider at each split
                'n_estimators': np.arange(100, 1000, 100),  # Number of trees in the forest
                'max_depth': [None] + list(np.arange(5, 30, 5)),  # Maximum depth of each tree
                'min_samples_split': np.arange(2, 11),  # Minimum number of samples required to split an internal node
                'min_samples_leaf': np.arange(1, 5),  # Minimum number of samples required to be at a leaf node
                }

    # Define the scoring metric
    scorer = make_scorer(f1_score, average="macro")
    
    model = RandomForestClassifier(random_state=42)  # Nombre del modelo
    
    # Perform randomized search
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, scoring=scorer, cv=5, n_iter=10, random_state=42) # Optimizador
    random_search.fit(X_train, y_train) # Entrenamiento del optimizador    
    best_params = random_search.best_params_  # Obtiene los parametros optimizados 
    best_score = random_search.best_score_    # Obtiene el f1 score del entrenamiento del optimizador 
    
    print('','Ajuste de hiperparametros', sep='\n')
    print(f"Best parameters: {best_params}")
    print(f"Optimized F1 score: {best_score:.1%}")

    # Entrenando el modelo con el mejor estimador
    best_model = random_search.best_estimator_     # Almacena el mejor estimador (mejor modelo)
    best_model.fit(X_train, y_train)                 # Entrena la maquina con el mejor estimador
    y_test_pred = best_model.predict(X_test)     # Predicciones realizadas con el modelo optimizado sobre el test set
    
    # Accuracy del modelo en el test_set
    baseline_accuracy = accuracy_score(y_test, y_test_pred)
    
    # F1 score del modelo en el test_set
    f1_score_test = f1_score(y_test, y_test_pred, average="macro") # Calcula el F1 score
    print('','Precision y F1 score en el test set', sep='\n')
    print(f"Baseline Accuracy: {baseline_accuracy:.1%}") # Imprime el accuracy del modelo en formato % con 1 decimal
    print(f"F1 score - test = {f1_score_test:.1%}")

    # Cross-validation que nos devuelve la precision del modelo (scores) y los valores predichos (predict)
    y_train_pred = cross_val_predict(best_model, X_train, y_train, cv=5)  # Valores predichos en el training set con validacion cruzada
    cross_accuracy = accuracy_score(y_train, y_train_pred)  # Calcula accuracy
    cross_f1_score = f1_score(y_train, y_train_pred, average="macro")     # Calcula el F1 score
    print('','Validacion cruzada para verificar overfitting', sep='\n')
    print(f"Cross-val Accuracy = {cross_accuracy:.1%}")
    print(f"Cross-val F1 score = {cross_f1_score:.1%}")

    # Calculamos la importancia de cada feature para la clasificacion 
    feature_importances = (best_model.feature_importances_ * 100).round(2)
    best_features = pd.DataFrame(
        sorted(zip(X_train.columns.tolist(), feature_importances), key=lambda x: x[1], reverse=True),
        columns = ["Features", "Importances (%)"])    
    
    # Print the sorted DataFrame
    print('','Importancia de Features', sep='\n')
    print(best_features.head(5))

    return y_train_pred, y_test_pred, best_features, best_model

def conf_matrix(y, y_pred, name_figure, figure_path, width=90, height=60, guardar=False):
    '''
    Parameters
    ----------
    y : series or array
        Clases reales del dataset.
    y_pred : series or array
        Clases predichas por el modelo.
    name_figure : str
        Nombre de la clase que aparece en titulo de la figura.
        
        e.g. Confusion matrix - {name_figure}
    figure_path : str
        Directorio para guardar la figura.
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90  
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. Confusion matrix {name_figure}.pdf
    Returns
    -------
    Grafica y (guarda) la matriz de confusion en formato pdf.

    '''
    # Grafica la matriz de confusion    
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
   
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = f"Confusion matrix {name_figure}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

    plt.show()
    
def plot_tsne(X, y, figure_path, title = 't-SNE Scatter Plot', x_label='t-SNE Dimension 1', 
              y_label='t-SNE Dimension 2', width=90, height=60, selected_features=None, guardar=False):
    '''
    Parameters
    ----------
    X : dataframe, array
        Conjunto de features.
    y : series, array
        Clases del dataset.
    figure_path : str
        Directorio para guardar la figura.
    title : str, optional
        titulo de la figura. The default is 't-SNE Scatter Plot'
    x_label : str, optional
        leyenda del eje x. The default is 't-SNE Dimension 1'
    y_label : str, optional
        leyenda del eje y. The default is 't-SNE Dimension 2'
    width : int or float, optional
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90  
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    selected_features : list, optional
       Nombre de los features para entrenar el modelo. The default is None.
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. t-SNE_{name_figure}.pdf

    Returns
    -------
    Grafica y (guarda) la reduccion de dimensionalidad con etiqueta y marcadores de clase en formato pdf.

    '''
    # Seleccion de features (si se utilizo backward elimination)
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
    
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = f"{title}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

    plt.show()   

def plot_histogram(X, y, figure_path, width=90, height=60, bins=30, feature_to_save='', guardar=False):
    '''
    Parameters
    ----------
    X : dataframe, array
        Conjunto de features.
    y : series, array
        Clases del dataset.
    figure_path : str
        Directorio para guardar la figura.
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90  
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    bins : int, optional
        Define el numero de barras (de igual ancho) del histograma. The default is 30.
    feature_to_save : str, optional
        Feature para guardar su histograma. The default is ''.
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. Histogram_{feature_to_save}.pdf

    Returns
    -------
    Grafica todas los histogramas de features y (guarda) solo 'feature_to_save' en formato pdf.

    '''
    # Directorio y nombre de archivo para guardar imagen
    figure_filename = f"Histogram_{feature_to_save}.pdf"
    figure_path_name = os.path.join(figure_path, figure_filename)
    
    # Iterate over each column
    for feature in X.columns:
    
        # Set up the figure and axis
        figsize_inches = (width / 25.4, height / 25.4)
        fig, ax = plt.subplots(figsize=figsize_inches, dpi=300, tight_layout=True)
        # Set the number of bins for the histogram
        bins = 30
        
        # Compute the range for the bins
        # min_val = X[feature].min()
        # max_val = X[feature].max()
        # bin_width = 0.5
        # bins = np.arange(min_val, max_val + bin_width, bin_width)
        
        # Plot the histogram for each class
        class_labels = y.unique()
        for class_label in class_labels:
            feat_to_plot = X[y == class_label][feature]
            ax.hist(feat_to_plot, bins=bins, alpha=0.6, label=class_label, edgecolor = "black")
    
        # Add labels and a title
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram - {feature} ')
    
        # Add a legend
        ax.legend()
        
        # Guardar la imagen en figure_path
        if guardar and (feature == feature_to_save):
            plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
        
        # Display the plot
        plt.show()
    
def plot_feat_vs_feat(X, y, feat_1, feat_2, figure_path, title = None, subtitle=None,
                      x_label = None, y_label = None, width=90, height=60, ax=None, 
                      i=1, n_col=1, n_row=1, guardar=False):
    '''
    Parameters
    ----------
    X : dataframe, array
        Conjunto de features.
    y : series, array
        Clases del dataset.
    feat_1 : str
        Nombre del feature 1 para graficar.
    feat_2 : str
        Nombre del feature 2 para graficar.
    figure_path : str
        Directorio para guardar la figura.
    title : str or list, optional
        titulo de la figura, si se deja en default sera {feat_1} vs {feat_2}. The default is None
    subtitle : str or list, optional
        subtitulo de la figura, si se deja en default sera {feat_1} vs {feat_2}. The default is None
    x_label : str, optional
        leyenda del eje x. The default is None
    y_label : str, optional
        leyenda del eje y. The default is None
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90  
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    ax : axes, optional
        axes subplot. The default is None.
    i : int, optional
        contador para graficar cada subplot. The default is 1.
    n_col : int, optional
        numero de columnas. The default is 1.
    n_row : int, optional
        numero de filas. The default is 1.
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. Feat_vs_Feat_{feat_1}_{feat_2}.pdf

    Returns
    -------
    Grafica y (guarda) un diagrama de dispersion de dos features con etiquetas y marcadores en formato pdf.

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
    sns.scatterplot(x=feat_1, y=feat_2, hue='Labels',style='Labels', data=datos, markers=markers, palette=sns.color_palette(), ax=ax)
    
    # Set y-axis label only for the first plot of each row
    if (i-1) % n_col == 0:
        if y_label is None:
            ax.set_ylabel(feat_2)
        else:
            ax.set_ylabel(y_label)
    else:
        ax.set_ylabel('')   
    
    # Logaritmic scale if you choose energy
    if feat_1 == 'energy': 
        ax.set_xscale('log')
    
    if feat_2 == 'energy':
        ax.set_yscale('log')
    
    # Activate the grid
    ax.grid()
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Guardar la imagen en figure_path
    if guardar:
        if ax is None:
            figure_filename = f"Feat_vs_Feat_{feat_1}_{feat_2}.pdf"
        else:
            figure_filename = 'Feat_vs_Feat_plot.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')

    return ax

