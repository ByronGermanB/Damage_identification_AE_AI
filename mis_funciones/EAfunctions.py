# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:34:48 2023

@author: ahercas1
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import os 
import pandas as pd
from scipy import ndimage, signal
from skimage.transform import resize
import vallenae as vae
import matplotlib
import matplotlib.image as mpimg
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['figure.dpi'] = 300

# Carga archivos tanto .pridb como .tradb pero no los hits o las waves
def abreAE(path, filename):
    '''
    Description
    -----------
    Carga los archivos .pridb y .tradb sin llegar a leer los datos internos de los archivos, por lo que devuelve el propio archivo.
    
    Parameters
    ----------
    path : str
        Directorio de almacenamiento de los datos de Emisiones Acústicas.
    filename : str
        Nombre del archivo que se desea cargar.

    Returns
    -------
    vae_pridb : PriDatabase
        Archivo .pridb con el nombre 'filename'.
    vae_tradb : TraDatabase
        Archivo .tradb con el nombre 'filename'.

    '''
    pridb_filename = str(filename + '.pridb')
    tradb_filename = str(filename + '.tradb')
    
    directorio = os.path.dirname(path  + '\\' +  filename + '\\' + pridb_filename)
    
    pridb_file = os.path.join(directorio, pridb_filename)
     # Debugging statements
    print(f"pridb_file: {pridb_file}")
    print(f"Directory exists: {os.path.exists(directorio)}")
    print(f"pridb_file exists: {os.path.exists(pridb_file)}")
    vae_pridb = vae.io.PriDatabase(pridb_file)
    
    tradb_file = os.path.join(directorio, tradb_filename)
    vae_tradb = vae.io.TraDatabase(tradb_file)
    
    return vae_pridb, vae_tradb


# Carga el archivo .csv separado por , y lo convierte a array()
def abreCSV(path, filename, val = int()):
    '''
    Description
    -----------
    Carga un archivo .csv separado por ',' y lo convierte en array eliminando las iniciales hasta la número 'val'.
    
    Parameters
    ----------
    path : str
        Directorio de almacenamiento del archivo tipo csv que desea cargarse.
    filename : str
        Nombre dle archivo tipo csv que se desea cargar.
    val : int, optional
        Número de fila a partir de la cuál se vuelcan los datos. 
        Esto se hace porque las primeras filas del archivo se emplean para almacenar información sobre el mismo en formato texto y no es relevante.

    Returns
    -------
    cont_array : float
        Array multidimensional o matriz de los datos del csv cargado.

    '''
    directorio = path + '\\' + filename
    with open(directorio, "r") as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        contenido = list(lector_csv)
    cont_array = np.array(contenido[val:]).astype(float)
    return cont_array
    

# Carga el archivo .csv separado por ; y lo convierte a array()
def abreCSVdot(path, filename, val = int()):
    '''
    Description
    -----------
    Carga un archivo .csv separado por ';' y lo convierte en array eliminando las iniciales hasta la número 'val'.

    Parameters
    ----------
    path : str
        Directorio de almacenamiento del archivo tipo csv que desea cargarse.
    filename : str
        Nombre dle archivo tipo csv que se desea cargar.
    val : int, optional
        Número de fila a partir de la cuál se vuelcan los datos. 
        Esto se hace porque las primeras filas del archivo se emplean para almacenar información sobre el mismo en formato texto y no es relevante.

    Returns
    -------
    cont_array : array(float)
        Array multidimensional o matriz de los datos del csv cargado.

    '''
    directorio = path + '\\' + filename
    drops = np.arange(0, val, 1)
    contenido = pd.read_csv(directorio, sep=';').drop(drops)
    cont_array = contenido.to_numpy().astype(float)
    return cont_array


# Función de extracción del transitorio y aplicación de la transformada        
def calcCWT(vae_Tarr, vae_pridb, t_trai, max_trais, t_trans, signal_function, n_bands, amp_lim, cnts_lim, STD_noise, num_noisySignals, figure_path, plot = False):
    '''
    Description
    -----------
    Función principal para el procesado de los datos EA para el entrenamiento del modelo predictivo. 
    Se cargan los datos pridb y tradb, pudiendo recortar la cantidad de datos a gusto, bien sea para eliminar el periodo de rotura de la probeta o cualquier otro evento que haya surgido en el ensayo.
    Adicionalmente se filtran los datos en función del número de counts y la amplitud de cada hit.
    También es posible recortar la longitud de los transitorios en caso de que sean de una duración excesiva.
    Una vez cargado los archivos y preprocesados, se les aplica la Continuous Wavelet Transform (CWT) para obtener imágenes en 2D normalizadas, se aplican métodos de Data Augmentation y se permite recortar las dimensiones de las imágenes obtenidas. 

    Parameters
    ----------
    vae_Tarr : list(tradb)
        Lista de los datos tradb cargados para el procesado.
        Ejemplo: vae_Tarr = [vae_T01, vae_T02, vae_T03, vae_T04]
    vae_pridb : list(DataFrames)
        Lista de los datos pridb cargados para el procesado.
        Ejemplo: vae_pridb = [pridb_01, pridb_02, pridb_03, pridb_04]
    t_trai : array(int)
        Vector de los rangos temporales de los datos de cada ensayo que se desean procesar.
    t_trans : int
        Instante de tiempo a partir del cual la señal temporal de los transitorios no aportan información relevante.
    signal_function : function
        Tipo de Mother Wavelet aplicada en la transformación por CWT.
    n_bands : int
        Cantidad de bandas de frecuencia tenidas en cuenta para la aplicación de las CWT.
    amp_lim : int
        Valor de threshold a partir del cuál se consideran válidos los hits almacenados.
    cnts_lim : int
        Cantidad de counts mínimos que debe tener un hit para ser considerado válido.
    STD_noise : list(float)
        Lista de los valores de desviación estándar para la aplicación de Data Augmentation.
    num_noisySignals : int
        Cantidad de señales de ruido que se generan con cada desviación estándar para la aplicación de Data Augmentation.

    Returns
    -------
    cwt_image : array(float)
        Matriz de datos (realmente es un array de imágenes) compuesta por las imágenes obtenidas tras la CWT.
    cwt_trai : list(int)
        Lista de los array de transitorios almacenados tras el preprocesado de los datos.

    '''
    # Parámetro empleado para la configuración de la calidad del visualizado de las gráficas
    dpi = 260
    
    # Definicón de las listas en las que se van a a almacenar las imágenes y los valores de transitorios
    cwt_trai = []
    cwt_image = []

    # Inicio del procesado de los datos para la aplicación de la CWT
    for vae, (vae_T, max_trai) in enumerate(zip(vae_Tarr, max_trais)):
        # Carga de datos
        df_T = vae_T.read()
        
                
        # Definición de los límites temporales del ensayo
        limite_inf = float(t_trai[vae][0])
        limite_sup = float(t_trai[vae][1])
        
        trai_lims = limites(df_T, limite_inf, limite_sup)
        TRAI_arr = np.arange(trai_lims[0] + 1, trai_lims[1])
        trai_lims[1] = max_trai 
        #print(trai_lims, TRAI_arr)

        # Selección aleatoria de los transitorios a mostrar de cada conjunto de datos 
        trai_values = np.random.choice(TRAI_arr, 5) 
        
        # Recortado temporal de las señales de los transitorios al valor dado por 't_trans'
        if t_trans:
            vline = int(t_trans)                      # [us]
            for trai in trai_values:
                # Carga de transitorios
                tra_signal, tra_time = vae_T.read_wave(trai)

                # Arreglo de unidades de los datos cargados
                time = tra_time * 1e6
                signal_amp = tra_signal * 1e3
                
                if plot:
                    # Visualizado de la señal temporal con el valor de recorte marcado con una recta vertical 
                    plt.figure(figsize=(10, 6), dpi=dpi)
                    plt.plot(time, signal_amp)
                    plt.xlabel('Time [us]')
                    plt.ylabel('Amplitude [mV]')
                    plt.title('Trai nº %i' % trai)
                    plt.axvline(x=vline, linestyle='--', linewidth=0.75, color='red', label='Corte')
                    plt.grid(True)
                    plt.legend(loc='upper right')
                    plt.show()

        # Definición de los parámetros de la wavelet
        widths = np.arange(0, n_bands, 0.5) + 1                       # Vector n_bands
        
        # Carga de datos .pridb para el filtrado de los transitorios válidos
        pridb = vae_pridb[vae].read_hits().reset_index(drop=True)

        # Transformación de las unidades de la columna de la amplitud de la señal temporal
        pridb['amplitude'] = 20 * np.log10(pridb['amplitude'] / 1e-6)
        no_saturacion = 94

        if max_trai is not None:
            df_hits_filtro = pridb[(pridb["channel"] >= 1) & (pridb["amplitude"] >= amp_lim) & (pridb["amplitude"] <= no_saturacion) & (pridb["trai"] <= max_trai) & (pridb["counts"] >= cnts_lim)]  # Seleccionamos solo los valores cumplen las condiciones
        else:
            df_hits_filtro = pridb[(pridb["channel"] >= 1) & (pridb["amplitude"] >= amp_lim) & (pridb["amplitude"] <= no_saturacion) & (pridb["counts"] >= cnts_lim)]  # Seleccionamos solo los valores cumplen las condiciones

        trais = df_hits_filtro["trai"].to_numpy() # Extraemos la columna con los valores TRAI (indices de transitorio)  

        # Find the index where the values start over
        if (np.diff(trais) < 0).any():
            start_over_index = np.where(np.diff(trais) < 0)[0][0] + 1
        else:
            start_over_index = 0

        # Slice the array from that index to the end
        trais = trais[start_over_index:]

        # Carga de los transitorios válidos y adecuado de los datos                        
        signals = []
        for trai in trais:
            # Carga de los transitorios válidos
            tra_signal, tra_time = vae_T.read_wave(trai)   # [V], [seg]
            
            # Arreglo de los datos
            time_arr = tra_time * 1e6                         # [us]
            signal_arr = tra_signal * 1e3                     # [mV]
            
            # Almacenamiento de la señal arreglada
            time = time_arr[np.where(time_arr <= vline)]
            signal_amp = signal_arr[:len(time)]
            signals.append(signal_amp)
        signals = np.array(signals)
        
        # Aplicación del método de DataAugmentation en caso de que 'num_noisySignals =! 0'
        if int(num_noisySignals) == 0:
            more_signals = signals
        else:
            more_signals = moreSignals(STD_noise, signals, time, figure_path, num_noisySignals)
        
        # Aplicación de la CWT a las señales almacenadas {originales + aumentadas (en caso de que hubiera)}
        for sig in more_signals:
            # Cálculo de la transformada
            cwt_signal = signal.cwt(sig, signal_function, widths)
            cwt_signal_abs = np.abs(cwt_signal).astype(np.float32)

            # Normalización de la señal
            #images = normalize(cwt_signal_abs, [128, 128])
            
            # Almacenamiento de las imágenes obtenidas 
            cwt_image.append(cwt_signal_abs)
        
        # Almacenamiento de los trais empleados en la carga de los transitorios
        cwt_trai.append(trais)    
    return cwt_image, cwt_trai      


# Similar a la función multiplot pero para conjuntos de matrices de pandas
def df_multiplot(conjunto, legend, labels, axis):
    '''
    Description
    -----------
    Graficado de conjuntos de datos almacenados en una lista de DataFrames. 

    Parameters
    ----------
    conjunto : list(DataFrames)
        Lista compuesta por DataFrames de diferentes longitudes.
    legend : list(str)
        Leyenda que aparece en el gráfico para los datos proporcionados en el 'conjunto'.
    labels : lista(str)
        Lista de los títulos que desean ponerse en cada eje.
        Ejemplo:
            labels = ['Eje x', 'Eje y']
    axis : list(int)
        Lista de las columnas que van a graficarse de los vectores del 'conjunto'.
        Ejemplo:
            axis = [[0, 1], [2, 1]]
            Del primer vector en el 'conjunto' se grafica la columna 0 como eje horizontal y la columna 1 como eje vertical.
            Del segunto vector se grafica la columna 2 como eje horizontal y la columna 1 como eje vertical.

    Returns
    -------
    Graficado de los datos proporcionados.

    '''
    plt.figure(figsize=(10, 6), dpi=260)
    for vector in conjunto:
        plt.plot(vector[axis[0]], vector[axis[1]])
    plt.grid(True)
    plt.legend(legend, loc='best')
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    return plt.show()


# Recorte y adecuado de los datos de EA en función del tiempo de ensayo 
def HrecortePRIDB(df_data, time, recorte, vals_limite):
    '''
    Description
    -----------
    Eliminación de datos del pridb a partir de un instante de tiempo del ensayo, siendo estos datos no válidos para su procesado.
    También se ajustan los tiempos de medida del 'linwave' y de la MTS.

    Parameters
    ----------
    df_data : DataFrame
        Matriz de datos del ensayo.
    time : int
        Instante de tiempo por encima del cual se considera que las medidas del 'linwave' son válidas.
    recorte : int
        Desfase temporal entre el inicio de las medidas de 'linwave' y los datos de la MTS.
    vals_limite : int
        Rango de datos válido del df_data cargado.

    Returns
    -------
    df_data_rec : DataFrame
        Matriz de datos tras el proceso de recorte.
    df_data : DataFrame
        Misma matriz de datos que la introducida en la función pero con el tiempo de ensayo adecuado.

    '''
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(df_data['time'], df_data['amplitude'])
    plt.show()
    
    if vals_limite:
        limite_inf, limite_sup = vals_limite
        limite_inf2 = df_data['time'].iloc[limite_inf]
        limite_sup2 = df_data['time'].iloc[limite_sup]
    else:
        print('\n¬øL√≠mite inferior de recorte?')
        limite_inf = int(input())
        print('\n¬øL√≠mite superior de recorte?')
        limite_sup = int(input())
    
    if limite_inf == 0:
        limite_inf2 = df_data['time'].iloc[limite_inf]
    if limite_sup == -1:
        limite_sup2 = df_data['time'].iloc[limite_sup]

    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(df_data['time'], df_data['amplitude'])
    plt.axvline(limite_inf2, linestyle='--', color='red')
    plt.axvline(limite_sup2, linestyle='--', color='red')
    plt.show()
    
    # Obtención de las posiciones límite del vector
    data_lim = limites(df_data, limite_inf, limite_sup)
    print(data_lim)
    # Se limita el vector de datos para quedarnos con los datos de tiempo válido
    df_data_lim = df_data[data_lim[0]:data_lim[1]]
    
    # Se deshechan los datos tomados por encima del tiempo impuesto por el parámetro 'time'
    df_drop = np.where(df_data_lim['time'] >= float(time))[0]
    
    df_data_rec = df_data_lim[:df_drop[0]]
    
    # Se arregla el tiempo de las medidas AE para que coincida con las medidas de la MTS 
    if limite_inf != 0 or limite_sup != -1:
        df_data['time'] = df_data_rec['time'] - recorte
    
    return df_data_rec, df_data


# Elimina los datos temporales fuera de los límites estipulados
def limites(vector, limite_inf, limite_sup):
    '''
    Description
    -----------
    Obtención de los límites del rango temporal válido del ensayo.

    Parameters
    ----------
    vector : DataFrame
        Matriz de datos del ensayo ordenados temporalmente.
    limite_inf : int
        Valor de la fila de la matriz a partir del cual se consideran válidos los datos de la misma.
    limite_sup : int
        Valor de la fila de la matriz a partir del cual ya no se consideran válidos los datos de la misma.

    Returns
    -------
    vec_lims : list(int)
        Lista en la que se almacenan los límites del rango de datos.

    '''
    limite_inf = int(limite_inf)
    limite_sup = int(limite_sup)
        
    if limite_inf != 0:
        inf_pos = np.where(vector['time'] <= limite_inf)[0] + 1
        inf_lim = inf_pos[-1]
        
    if limite_inf == 0:
        inf_lim = limite_inf
        
    if limite_sup != -1:
        sup_pos = np.where(vector['time'] <= limite_sup)[0] + 1
        sup_lim = sup_pos[-1]
        
    if limite_sup == -1:
        sup_lim = vector.shape[0]
               
    vec_lims = [inf_lim, sup_lim]
    return vec_lims


# Aplica modificaciones a las imágenes originales para generar datos sintéticos 
def masIm(vector):
    '''
    Description
    -----------
    Aplicación de la metodología DataAugmentation para generar más imágenes

    Parameters
    ----------
    vector : array(float)
        Vector de imágenes originales.

    Returns
    -------
    new_vector : array(float)
        Vector de imágenes originales más las generadas por DataAugmentation.

    '''
    new_vector = []
    for image in vector:
        flipped_ud = np.flipud(image)
                
        rotated_noreshape = ndimage.rotate(image, randint(0, 360, size=1)[0], reshape=False)
        
        lx, ly, lz = image.shape
        div = randint(4, 10, size=1)[0]
        lxx, lyy = int(lx / div), int(ly / div)
        crop_image = resize(image[lxx:-lxx, lyy:-lyy], (lx, ly))
        
        sigma = randint(0, 5, size=1)[0]
        blurred_image = ndimage.gaussian_filter(image, sigma=sigma)
        local_mean = ndimage.uniform_filter(image, size=11)
        
        image_add = [image, flipped_ud, rotated_noreshape, crop_image, blurred_image, local_mean]
        for addin in image_add:
            new_vector.append(addin)
    values = np.random.choice(np.arange(0, len(new_vector)), 10)
    for im in values:
        plt.imshow(new_vector[im])
        plt.axis('off')
        plt.show()
    return new_vector


# Genera señales temporales como ruido de fondo para modificar las señales originales y obtener una mayor cantidad de datos
def moreSignals(STD_noise, signals, time, figure_path, num_noisySignals = 5, plot = False, save_noised = False):
    '''
    Description
    -----------
    Aplicación de la metodología DataAugmnetation a señales temporales, de manera que se aplican señales de ruido blanco que modifiquen el contenido en frecuencia de la señal original.
    Se generan señales de ruido blanco según una distribución normal con ciertos niveles de desviación típica.

    Parameters
    ----------
    STD_noise : list(float)
        Vector de los valores de desviación estándar.
    signals : array(float)
        Vector de la amplitud de la señal temporal.
    time : array(float)
        Vector de los valores temporales de cada punto de la señal.
    num_noisySignals : int, optional
        Cantidad de señales de ruido que se generan con cada desviación estándar.

    Returns
    -------
    more_signals : array(float)
        Vector de señales temporales concatenadas siendo la primera la señal original y el resto las sintéticas.

    '''
    more_signals = []
    length = signals[0].shape[0]
    noise_signals_1 = []
    noise_signals_2 = []
    # Bucle para la generación de 5 señales de ruido blanco distintas para cada desviación estándar
    for n in range(num_noisySignals):
        noise_1 = np.random.normal(0, STD_noise[0], length)        
        noise_2 = np.random.normal(0, STD_noise[1], length)
        noise_signals_1.append(noise_1)
        noise_signals_2.append(noise_2)
    
    if plot:
        # Graficado del ruido blanco generado
        noise_signals_1 = np.array(noise_signals_1)
        plt.figure(figsize = (10, 8), dpi = 130)
        plt.title('Noise 1')
        plt.plot(noise_signals_1[-1])
        plt.xlabel('Time [us]')
        plt.ylabel('Amplitude [mV]')
        plt.axhline(y = 0, linestyle = '--', color = 'black')
        plt.grid(True)
        plt.show()
        
        noise_signals_2 = np.array(noise_signals_2)
        plt.figure(figsize = (10, 8), dpi = 130)
        plt.title('Noise 2')
        plt.plot(noise_signals_2[-1])
        plt.xlabel('Time [us]')
        plt.ylabel('Amplitude [mV]')
        plt.axhline(y = 0, linestyle = '--', color = 'black')
        plt.grid(True)
        plt.show()
    
    noise_choices = np.arange(0, num_noisySignals, 1)
    
    # Aplicación del ruido blanco
    # Se aplica de manera aleatoria una de las 5 opciones de ruido blanco para cada desviación estándar, dando lugar a 2 señales temporales adicionales a la original
    for sig in signals:
        noise = np.random.choice(noise_choices)
        noised_signal_1 = sig + noise_signals_1[noise]
        noised_signal_2 = sig + noise_signals_2[noise]
        
        more_signals.append(sig)
        more_signals.append(noised_signal_1)
        more_signals.append(noised_signal_2)
    
    # Se obtiene el triple de señales temporales tras el proceso 
    more_signals = np.array(more_signals)
    
    if plot:
        # Graficado solapado de la última señal a la que se aplica el ruido, pudiendo ver la original y la modificada para cada desviación estándar
        plt.figure(figsize = (10, 8), dpi = 130)
        plt.plot(time, sig)
        plt.scatter(time, noised_signal_1, marker = '.', color = 'orange')
        plt.xlabel('Time [us]')
        plt.ylabel('Amplitude [mV]')
        plt.legend(['Señal sin ruido', 'Señal con ruido'])
        plt.grid(True)
        plt.show() 
    
    figsize = (80 / 25.4, 60 / 25.4)

    if save_noised:
        # Guardado de la señal original y la señal con ruido
        plt.figure(figsize = figsize)
        #plt.title('Superimposition of an original signal and its augmented signal')
        plt.plot(time, sig, linewidth = 1)
        plt.scatter(time, noised_signal_2, marker = '.', color = 'orange', linewidths = 0.25)
        plt.xlabel('Time [μs]')
        plt.ylabel('Amplitude [mV]')
        plt.legend(['Original signal', 'Noised signal'])
        plt.grid(True)
        figure_path_name = os.path.join(figure_path, 'Noised_signal.pdf')
        plt.savefig(figure_path_name, bbox_inches = 'tight')
        plt.show()
    
    return more_signals 


# Genera una gráfica completa de varias funciones
def multiplot(conjunto, legend, plt_title, labels, axis):
    '''
    Description
    -----------
    Graficado de un conjunto de datos, superponiendo las rectas en el mismo gráfico

    Parameters
    ----------
    conjunto : list(DataFrames)
        Lista compuesta por DataFrames de diferentes longitudes.
    legend : list(str)
        Leyenda que aparece en el gráfico para los datos proporcionados en el 'conjunto'.
    plt_title : list(str)
        Lista con el título de a gráfica.
    labels : lista(str)
        Lista de los títulos que desean ponerse en cada eje.
        Ejemplo:
            labels = ['Eje x', 'Eje y']
    axis : list(int)
        Lista de las columnas que van a graficarse de los vectores del 'conjunto'.
        Ejemplo:
            axis = [0, 1]
            
    Returns
    -------
    Graficado de los datos proporcionados.
    
    '''
    plt.rcParams['figure.constrained_layout.use'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize = (10, 6), layout = 'tight')
    for vector in conjunto:
        if type(vector) == type(pd.DataFrame()):
            vector = vector.to_numpy()
        plt.plot(vector[:, axis[0]], vector[:, axis[1]])
    plt.grid(True)
    plt.legend(legend, loc='best')
    if plt_title:
        plt.title(plt_title[0])
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.show()
    return print('Done')
    

def normalize(x_vec):
    '''
    Description
    -----------
    Normaliza el vector introducido.
    
    Parameters
    ----------
    x_vec : array(float)
        Vector de datos.
    Returns
    -------
    x_norm : array(float)
        Vector normalizado.

    '''
    # Normalización del vector de imágenes
    max_amp = x_vec.max()
    min_amp = x_vec.min()
    x_norm = (x_vec - min_amp) / (max_amp - min_amp)
    return x_norm, max_amp


def redim(x_vec, dims):
    '''
    Description
    -----------
    Redimensiona el vector introducido.
    
    Parameters
    ----------
    x_vec : array(float)
        Vector de datos.
    dims : list(int)
        Dimensiones en las que se desean los datos proporcionados por 'vector'.
    Returns
    -------
    x_norm : array(float)
        Vector redimensionado.
    '''    
    # Redimensionado de las imágenes
    resized_images = []
    for im in x_vec:
        resized_images.append(resize(im, (dims[0], dims[1])))
    x_resized = np.array(resized_images)
    return x_resized


# Recorta los datos del conjunto introducido
def recorte(conjunto, val_rec):
    '''
    Description
    -----------
    Recorte sobre el eje vertical de los conjuntos de imágenes.

    Parameters
    ----------
    conjunto : list(list)
        Conjunto de listas de imágenes.
    val_rec : int
        Valor de la fila a partir de la cual la información de la imagen es irrelevante.

    Returns
    -------
    conjunto_rec : list(list)
        Conjunto de listas de las imágenes ya recortadas.

    '''
    conjunto_rec = []
    for cwt in conjunto:
        vector_rec = []
        choices = np.random.choice(np.arange(len(cwt)), 5)
        for choice in choices:
                image_rec = cwt[choice][:val_rec]
                plt.title(f'Recortada nº{choice}')
                plt.imshow(image_rec, cmap='gray')
                plt.axis('on')
                plt.show()
        for index in range(len(cwt)):
            vector_rec.append(cwt[index][:val_rec])
        conjunto_rec.append(vector_rec)
    return conjunto_rec


# Permite truncar los datos del archivo .csv visualizando su representación gráfica
def recorteCSV(conjunto, legend, time_rotura):
    '''
    Description
    -----------
    Permite graficar datos de un conjunto de datos y eliminar los datos presentados a partir de una cierta fila.

    Parameters
    ----------
    conjunto : list(DataFrame)
        Lista de matrices de datos.
    legend : list(str)
        Lista de los nombres a aplicar en la leyenda de las gráficas.
    time_rotura : 


    Returns
    -------
    conjunto_rec : list(numpy)
        Lista de los vectores con los datos ya eliminados.
    time : list(float)
        Lista de los instantes de tiempo empleados en el arreglo de los datos.

    '''
    conjunto_rec = []
    time = []
    for vector in conjunto:    
        plt.figure(figsize=(10, 6), dpi=260)
        plt.plot(vector[:, 2], vector[:, 1])
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Carga [kN]')
        plt.grid(True)
        plt.show()
        
        if time_rotura:
            time_rotura = int(time_rotura)

        else:
            response = 'No'
            while response == 'No' or response == 'no':
                print('\n¬øSobre qu√© tiempo recortamos para quitar la rotura?')
                time_rotura = int(input())
                
                if time_rotura == -1:
                    time_rotura = vector[-1, 2] 
                    
                plt.figure(figsize=(10, 6), dpi=260)
                plt.plot(vector[:, 2], vector[:, 1])
                plt.axvline(time_rotura, linestyle='--', color='red')
                plt.xlabel('Tiempo [s]')
                plt.ylabel('Carga [kN]')
                plt.grid(True)
                plt.show()
                
                print('\n¬øConforme?')
                response = input()
            
        if time_rotura == -1:
            vector_rec = vector
        else:
            pos_rotura = np.where(vector[:, 2] >= time_rotura)[0][0]
            vector_rec = vector[:pos_rotura, :]
        conjunto_rec.append(vector_rec)
        time.append(time_rotura)
    multiplot(conjunto, legend, ['Conjunto sin recortar'], ['Tiempo [s]', 'Fuerza [kN]'], [2, 1])
    multiplot(conjunto_rec, legend, ['Conjunto recortado'], ['Tiempo [s]', 'Fuerza [kN]'], [2, 1])
    return conjunto_rec, time


# Arregla el archivo .pridb en función del recorte generado en el .csv
def recortePRIDB(df_data, time):
    '''
    Description
    -----------
    Recorte de los datos del pridb a partir de un instante de tiempo del ensayo.

    Parameters
    ----------
    df_data : DataFrame
        Matriz de datos del pridb.
    time : int
        Tiempo a partir del cual el ensayo no se considera válido.

    Returns
    -------
    df_data_rec : DataFrame
        Matriz de datos válidos.

    '''
    if time == -1:
        time = df_data['time'].iloc[-1]
    df_drop = np.where(df_data['time'] >= time)[0]
    df_data_rec = df_data[:df_drop[0]]
    return df_data_rec


# Elimina los trais que no cumplan las condiciones estipuladas
def TRAIdelete(df_data, amp_lim, cnts_lim):
    '''
    Description
    -----------
    Eliminación de los datos que no cumplan con las condiciones de umbral mínimo.

    Parameters
    ----------
    df_data : DataFrame
        Matriz de datos del tradb.
    amp_lim : int
        Valor mínimo a partir del cual los datos son válidos.
    cnts_lim : int
        Valor mínimo para que un hit sea válido.

    Returns
    -------
    df_data : DataFrame
        Matriz de datos que sí cumplen las condiciones.

    '''
    amp_lim = int(amp_lim)
    cnts_lim = int(cnts_lim)
    deleted = []
    for row in range(df_data.shape[0]):
        if np.array(df_data['amplitude'])[row] < amp_lim or np.array(df_data['counts'])[row] < cnts_lim:
            deleted.append(row)
    df_data = df_data.reset_index(drop=True).drop(deleted, axis=0)
    return df_data


def save_image_pred(image_list, dims, test_name, images_dir):
    '''
    Description
    -----------
    Guardado de las imágenes generadas por el modelo predictivo.

    Parameters
    ----------
    image_list : list(array(float))
        Lista de las imágenes generadas por el modelo.
    dims : list(int)
        Dimensiones de las imágenes.
    path : str
        Directorio de almacenamiento de las imágenes.
    name : str
        Nombre del archivo.

    Returns
    -------
    Guardado de las imágenes en el directorio especificado.

    '''
    path = os.path.join(images_dir, test_name)
    os.makedirs(path, exist_ok=True)
    image_redim = redim(image_list, dims)
    for i, image in enumerate(image_redim):
        image_name = f'CWT_{test_name}_{i:04d}.png'
        image_path = os.path.join(path, image_name)
        plt.imsave(image_path, image, cmap='gray')
    return print(f'Images of test {test_name} for prediction saved')    