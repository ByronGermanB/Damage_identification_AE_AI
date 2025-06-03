# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:34:48 2023

@author: ahercas1
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vallenae as vae
from numpy.random import randint
from scipy import ndimage, signal
from skimage.transform import resize

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["figure.dpi"] = 300


# Loads both .pridb and .tradb files but not hits or waves
def abreAE(path, filename):
    """
    Description
    -----------
    Loads .pridb and .tradb files without reading their internal data, so it returns the file itself.

    Parameters
    ----------
    path : str
        Directory where the Acoustic Emission data is stored.
    filename : str
        Name of the file to load.

    Returns
    -------
    vae_pridb : PriDatabase
        .pridb file with the name 'filename'.
    vae_tradb : TraDatabase
        .tradb file with the name 'filename'.

    """
    pridb_filename = str(filename + ".pridb")
    tradb_filename = str(filename + ".tradb")

    directorio = os.path.dirname(path + "\\" + filename + "\\" + pridb_filename)

    pridb_file = os.path.join(directorio, pridb_filename)
    vae_pridb = vae.io.PriDatabase(pridb_file)

    tradb_file = os.path.join(directorio, tradb_filename)
    vae_tradb = vae.io.TraDatabase(tradb_file)

    return vae_pridb, vae_tradb


# Loads a .csv file separated by , and converts it to array()
def abreCSV(path, filename, val=int()):
    """
    Description
    -----------
    Loads a .csv file separated by ',' and converts it to an array, removing the initial
    rows up to number 'val'.

    Parameters
    ----------
    path : str
        Directory where the csv file to be loaded is stored.
    filename : str
        Name of the csv file to load.
    val : int, optional
        Row number from which the data is loaded.
        This is done because the first rows of the file are used to store information
          about the file in text format and are not relevant.

    Returns
    -------
    cont_array : float
        Multidimensional array or matrix of the loaded csv data.

    """
    directorio = path + "\\" + filename
    with open(directorio, "r") as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        contenido = list(lector_csv)
    cont_array = np.array(contenido[val:]).astype(float)
    return cont_array


# Loads a .csv file separated by ; and converts it to array()
def abreCSVdot(path, filename, val=int()):
    """
    Description
    -----------
    Loads a .csv file separated by ';' and converts it to an array, removing the initial 
    rows up to number 'val'.

    Parameters
    ----------
    path : str
        Directory where the csv file to be loaded is stored.
    filename : str
        Name of the csv file to load.
    val : int, optional
        Row number from which the data is loaded.
        This is done because the first rows of the file are used to store information 
        about the file in text format and are not relevant.

    Returns
    -------
    cont_array : array(float)
        Multidimensional array or matrix of the loaded csv data.

    """
    directorio = path + "\\" + filename
    drops = np.arange(0, val, 1)
    contenido = pd.read_csv(directorio, sep=";").drop(drops)
    cont_array = contenido.to_numpy().astype(float)
    return cont_array


# Function to extract the transient and apply the transform
def calcCWT(
    vae_Tarr,
    vae_pridb,
    t_trai,
    max_trais,
    t_trans,
    signal_function,
    n_bands,
    amp_lim,
    cnts_lim,
    STD_noise,
    num_noisySignals,
    figure_path,
    plot=False,
):
    """
    Description
    -----------
    Main function for processing AE data for training the predictive model.
    Loads pridb and tradb data, allowing you to trim the amount of data as desired, 
    either to remove the specimen failure period or any other event that occurred during t
    he test.
    Additionally, data is filtered based on the number of counts and amplitude of each hit.
    It is also possible to trim the length of the transients if they are excessively long.
    Once the files are loaded and preprocessed, the Continuous Wavelet Transform (CWT) 
    is applied to obtain normalized 2D images, Data Augmentation methods are applied, and 
    the dimensions of the obtained images can be trimmed.

    Parameters
    ----------
    vae_Tarr : list(tradb)
        List of loaded tradb data for processing.
        Example: vae_Tarr = [vae_T01, vae_T02, vae_T03, vae_T04]
    vae_pridb : list(DataFrames)
        List of loaded pridb data for processing.
        Example: vae_pridb = [pridb_01, pridb_02, pridb_03, pridb_04]
    t_trai : array(int)
        Vector of time ranges of the data from each test to be processed.
    t_trans : int
        Time instant after which the transient signal does not provide relevant information.
    signal_function : function
        Type of Mother Wavelet applied in the CWT.
    n_bands : int
        Number of frequency bands considered for the CWT.
    amp_lim : int
        Threshold value above which stored hits are considered valid.
    cnts_lim : int
        Minimum number of counts a hit must have to be considered valid.
    STD_noise : list(float)
        List of standard deviation values for Data Augmentation.
    num_noisySignals : int
        Number of noise signals generated with each standard deviation for Data Augmentation.

    Returns
    -------
    cwt_image : array(float)
        Data matrix (actually an array of images) composed of the images obtained after the CWT.
    cwt_trai : list(int)
        List of arrays of transients stored after data preprocessing.

    """
    # Parameter used to configure the quality of the plot visualization
    dpi = 260

    # Define lists to store images and transient values
    cwt_trai = []
    cwt_image = []

    # Start processing data for CWT application
    for vae_data, (vae_T, max_trai) in enumerate(zip(vae_Tarr, max_trais)):
        # Load data
        df_T = vae_T.read()

        # Define the time limits of the test
        limite_inf = float(t_trai[vae_data][0])
        limite_sup = float(t_trai[vae_data][1])

        trai_lims = limites(df_T, limite_inf, limite_sup)
        TRAI_arr = np.arange(trai_lims[0] + 1, trai_lims[1])
        trai_lims[1] = max_trai

        # Random selection of transients to display from each data set
        trai_values = np.random.choice(TRAI_arr, 5)

        # Temporally trim the transient signals to the value given by 't_trans'
        if t_trans:
            vline = int(t_trans)  # [us]
            for trai in trai_values:
                # Load transients
                tra_signal, tra_time = vae_T.read_wave(trai)

                # Adjust units of the loaded data
                time = tra_time * 1e6
                signal_amp = tra_signal * 1e3

                if plot:
                    # Visualize the time signal with the trim value marked by a vertical line
                    plt.figure(figsize=(10, 6), dpi=dpi)
                    plt.plot(time, signal_amp)
                    plt.xlabel("Time [us]")
                    plt.ylabel("Amplitude [mV]")
                    plt.title("Trai nº %i" % trai)
                    plt.axvline(
                        x=vline,
                        linestyle="--",
                        linewidth=0.75,
                        color="red",
                        label="Trim",
                    )
                    plt.grid(True)
                    plt.legend(loc="upper right")
                    plt.show()

        # Define wavelet parameters
        widths = np.arange(0, n_bands, 0.5) + 1  # n_bands vector

        # Load .pridb data for filtering valid transients
        pridb = vae_pridb[vae_data].read_hits().reset_index(drop=True)

        # Convert the amplitude column units of the time signal
        pridb["amplitude"] = 20 * np.log10(pridb["amplitude"] / 1e-6)
        no_saturation = 94

        if max_trai is not None:
            df_hits_filtro = pridb[
                (pridb["channel"] >= 1)
                & (pridb["amplitude"] >= amp_lim)
                & (pridb["amplitude"] <= no_saturation)
                & (pridb["trai"] <= max_trai)
                & (pridb["counts"] >= cnts_lim)
            ]  # Select only values that meet the conditions
        else:
            df_hits_filtro = pridb[
                (pridb["channel"] >= 1)
                & (pridb["amplitude"] >= amp_lim)
                & (pridb["amplitude"] <= no_saturation)
                & (pridb["counts"] >= cnts_lim)
            ]  # Select only values that meet the conditions

        trais = (
            df_hits_filtro["trai"].to_numpy()
        )  # Extract the column with TRAI values (transient indices)

        # Find the index where the values start over
        if (np.diff(trais) < 0).any():
            start_over_index = np.where(np.diff(trais) < 0)[0][0] + 1
        else:
            start_over_index = 0

        # Slice the array from that index to the end
        trais = trais[start_over_index:]

        # Load valid transients and adjust data
        signals = []
        for trai in trais:
            # Load valid transients
            tra_signal, tra_time = vae_T.read_wave(trai)  # [V], [s]

            # Adjust data
            time_arr = tra_time * 1e6  # [us]
            signal_arr = tra_signal * 1e3  # [mV]

            # Store the adjusted signal
            time = time_arr[np.where(time_arr <= vline)]
            signal_amp = signal_arr[: len(time)]
            signals.append(signal_amp)
        signals = np.array(signals)

        # Apply Data Augmentation method if 'num_noisySignals != 0'
        if int(num_noisySignals) == 0:
            more_signals = signals
        else:
            more_signals = moreSignals(
                STD_noise, signals, time, figure_path, num_noisySignals
            )

        # Apply CWT to the stored signals {original + augmented (if any)}
        for sig in more_signals:
            # Compute the transform
            cwt_signal = signal.cwt(sig, signal_function, widths)
            cwt_signal_abs = np.abs(cwt_signal).astype(np.float32)

            # Store the obtained images
            cwt_image.append(cwt_signal_abs)

        # Store the trais used in loading the transients
        cwt_trai.append(trais)
    return cwt_image, cwt_trai


# Similar to the multiplot function but for sets of pandas DataFrames
def df_multiplot(conjunto, legend, labels, axis):
    """
    Plots multiple datasets stored in a list of DataFrames.
    conjunto : list of pandas.DataFrame
        List containing DataFrames of different lengths to be plotted.
    legend : list of str
        List of legend labels corresponding to each DataFrame in 'conjunto'.
    labels : list of str
        List of axis titles to be set for the plot.
        Example:
            labels = ['X axis', 'Y axis']
    axis : list of int
        List specifying the columns to be plotted from each DataFrame in 'conjunto'.
        Example:
            For the first DataFrame in 'conjunto', column 0 is plotted on the x-axis and
            column 1 on the y-axis.
            For the second DataFrame, column 2 is plotted on the x-axis and column 1 on 
            the y-axis.
    None
        Displays the plot of the provided data.
    """
    plt.figure(figsize=(10, 6), dpi=260)
    for vector in conjunto:
        plt.plot(vector[axis[0]], vector[axis[1]])
    plt.grid(True)
    plt.legend(legend, loc="best")
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    return plt.show()


# Cropping and adjustment of AE data based on test time
def HrecortePRIDB(df_data, time, recorte, vals_limite):
    """
    Description
    -----------
    Removes data from the pridb after a certain time instant of the test, as these data 
    are not valid for processing.
    Also adjusts the measurement times of 'linwave' and the MTS.

    Parameters
    ----------
    df_data : DataFrame
        Data matrix of the test.
    time : int
        Time instant above which the 'linwave' measurements are considered valid.
    recorte : int
        Time offset between the start of 'linwave' measurements and the MTS data.
    vals_limite : int
        Valid data range of the loaded df_data.

    Returns
    -------
    df_data_rec : DataFrame
        Data matrix after the cropping process.
    df_data : DataFrame
        Same data matrix as introduced in the function but with the test time adjusted.

    """
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(df_data["time"], df_data["amplitude"])
    plt.show()

    if vals_limite:
        limite_inf, limite_sup = vals_limite
        limite_inf2 = df_data["time"].iloc[limite_inf]
        limite_sup2 = df_data["time"].iloc[limite_sup]
    else:
        print("\n¬øL√≠mite inferior de recorte?")
        limite_inf = int(input())
        print("\n¬øL√≠mite superior de recorte?")
        limite_sup = int(input())

    if limite_inf == 0:
        limite_inf2 = df_data["time"].iloc[limite_inf]
    if limite_sup == -1:
        limite_sup2 = df_data["time"].iloc[limite_sup]

    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(df_data["time"], df_data["amplitude"])
    plt.axvline(limite_inf2, linestyle="--", color="red")
    plt.axvline(limite_sup2, linestyle="--", color="red")
    plt.show()

    # Get the limit positions of the vector
    data_lim = limites(df_data, limite_inf, limite_sup)
    print(data_lim)
    # Limit the data vector to keep only valid time data
    df_data_lim = df_data[data_lim[0] : data_lim[1]]

    # Discard data taken above the time imposed by the 'time' parameter
    df_drop = np.where(df_data_lim["time"] >= float(time))[0]

    df_data_rec = df_data_lim[: df_drop[0]]

    # Adjust the AE measurement times to match the MTS measurements
    if limite_inf != 0 or limite_sup != -1:
        df_data["time"] = df_data_rec["time"] - recorte

    return df_data_rec, df_data


# Removes temporal data outside the specified limits
def limites(vector, limite_inf, limite_sup):
    """
    Description
    -----------
    Gets the limits of the valid time range of the test.

    Parameters
    ----------
    vector : DataFrame
        Data matrix of the test ordered by time.
    limite_inf : int
        Row value of the matrix from which the data is considered valid.
    limite_sup : int
        Row value of the matrix from which the data is no longer considered valid.

    Returns
    -------
    vec_lims : list(int)
        List storing the limits of the data range.

    """
    limite_inf = int(limite_inf)
    limite_sup = int(limite_sup)

    if limite_inf != 0:
        inf_pos = np.where(vector["time"] <= limite_inf)[0] + 1
        inf_lim = inf_pos[-1]

    if limite_inf == 0:
        inf_lim = limite_inf

    if limite_sup != -1:
        sup_pos = np.where(vector["time"] <= limite_sup)[0] + 1
        sup_lim = sup_pos[-1]

    if limite_sup == -1:
        sup_lim = vector.shape[0]

    vec_lims = [inf_lim, sup_lim]
    return vec_lims


# Applies modifications to the original images to generate synthetic data
def masIm(vector):
    """
    Description
    -----------
    Applies Data Augmentation methodology to generate more images.

    Parameters
    ----------
    vector : array(float)
        Array of original images.

    Returns
    -------
    new_vector : array(float)
        Array of original images plus those generated by Data Augmentation.

    """
    new_vector = []
    for image in vector:
        flipped_ud = np.flipud(image)

        rotated_noreshape = ndimage.rotate(
            image, randint(0, 360, size=1)[0], reshape=False
        )

        lx, ly, lz = image.shape
        div = randint(4, 10, size=1)[0]
        lxx, lyy = int(lx / div), int(ly / div)
        crop_image = resize(image[lxx:-lxx, lyy:-lyy], (lx, ly))

        sigma = randint(0, 5, size=1)[0]
        blurred_image = ndimage.gaussian_filter(image, sigma=sigma)
        local_mean = ndimage.uniform_filter(image, size=11)

        image_add = [
            image,
            flipped_ud,
            rotated_noreshape,
            crop_image,
            blurred_image,
            local_mean,
        ]
        for addin in image_add:
            new_vector.append(addin)
    values = np.random.choice(np.arange(0, len(new_vector)), 10)
    for im in values:
        plt.imshow(new_vector[im])
        plt.axis("off")
        plt.show()
    return new_vector


# Generates time signals as background noise to modify the original signals and obtain 
# a larger amount of data

def moreSignals(
    STD_noise,
    signals,
    time,
    figure_path,
    num_noisySignals=5,
    plot=False,
    save_noised=False,
):
    """
    Description
    -----------
    Applies Data Augmentation methodology to time signals by adding white noise, 
    modifying the frequency content of the original signal.
    White noise signals are generated according to a normal distribution with specified
      standard deviation levels.

    Parameters
    ----------
    STD_noise : list(float)
        List of standard deviation values.
    signals : array(float)
        Array of the amplitude of the time signal.
    time : array(float)
        Array of the time values for each point of the signal.
    num_noisySignals : int, optional
        Number of noise signals generated for each standard deviation.

    Returns
    -------
    more_signals : array(float)
        Array of concatenated time signals, with the first being the original signal and 
        the rest being synthetic.

    """
    more_signals = []
    length = signals[0].shape[0]
    noise_signals_1 = []
    noise_signals_2 = []
    # Loop for generating 5 different white noise signals for each standard deviation
    for n in range(num_noisySignals):
        noise_1 = np.random.normal(0, STD_noise[0], length)
        noise_2 = np.random.normal(0, STD_noise[1], length)
        noise_signals_1.append(noise_1)
        noise_signals_2.append(noise_2)

    if plot:
        # Plot the generated white noise
        noise_signals_1 = np.array(noise_signals_1)
        plt.figure(figsize=(10, 8), dpi=130)
        plt.title("Noise 1")
        plt.plot(noise_signals_1[-1])
        plt.xlabel("Time [us]")
        plt.ylabel("Amplitude [mV]")
        plt.axhline(y=0, linestyle="--", color="black")
        plt.grid(True)
        plt.show()

        noise_signals_2 = np.array(noise_signals_2)
        plt.figure(figsize=(10, 8), dpi=130)
        plt.title("Noise 2")
        plt.plot(noise_signals_2[-1])
        plt.xlabel("Time [us]")
        plt.ylabel("Amplitude [mV]")
        plt.axhline(y=0, linestyle="--", color="black")
        plt.grid(True)
        plt.show()

    noise_choices = np.arange(0, num_noisySignals, 1)

    # Application of white noise
    # Randomly applies one of the 5 white noise options for each standard deviation,
    # resulting in 2 additional time signals per original signal
    for sig in signals:
        noise = np.random.choice(noise_choices)
        noised_signal_1 = sig + noise_signals_1[noise]
        noised_signal_2 = sig + noise_signals_2[noise]

        more_signals.append(sig)
        more_signals.append(noised_signal_1)
        more_signals.append(noised_signal_2)

    # Triple the number of time signals after the process
    more_signals = np.array(more_signals)

    if plot:
        # Overlapped plot of the last signal with noise applied, showing
        # the original and the modified signal for each standard deviation
        plt.figure(figsize=(10, 8), dpi=130)
        plt.plot(time, sig)
        plt.scatter(time, noised_signal_1, marker=".", color="orange")
        plt.xlabel("Time [us]")
        plt.ylabel("Amplitude [mV]")
        plt.legend(["Signal without noise", "Signal with noise"])
        plt.grid(True)
        plt.show()

    figsize = (80 / 25.4, 60 / 25.4)

    if save_noised:
        # Save the original signal and the signal with noise
        plt.figure(figsize=figsize)
        # plt.title('Superimposition of an original signal and its augmented signal')
        plt.plot(time, sig, linewidth=1)
        plt.scatter(time, noised_signal_2, marker=".", color="orange", linewidths=0.25)
        plt.xlabel("Time [μs]")
        plt.ylabel("Amplitude [mV]")
        plt.legend(["Original signal", "Noised signal"])
        plt.grid(True)
        figure_path_name = os.path.join(figure_path, "Noised_signal.pdf")
        plt.savefig(figure_path_name, bbox_inches="tight")
        plt.show()

    return more_signals
def multiplot(conjunto, legend, plt_title, labels, axis):
    """
    Description
    -----------
    Plots a set of data, overlaying the lines in the same graph.

    Parameters
    ----------
    conjunto : list(DataFrames)
        List composed of DataFrames of different lengths.
    legend : list(str)
        Legend that appears in the graph for the data provided in 'conjunto'.
    plt_title : list(str)
        List with the title of the graph.
    labels : list(str)
        List of the titles to be set on each axis.
        Example:
            labels = ['X axis', 'Y axis']
    axis : list(int)
        List of the columns to be plotted from the vectors in 'conjunto'.
        Example:
            axis = [0, 1]

    Returns
    -------
    None
        Displays the plot of the provided data.

    """
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.figure(figsize=(10, 6), layout="tight")
    for vector in conjunto:
        if isinstance(vector, pd.DataFrame):
            vector = vector.to_numpy()
        plt.plot(vector[:, axis[0]], vector[:, axis[1]])
    plt.grid(True)
    plt.legend(legend, loc="best")
    if plt_title:
        plt.title(plt_title[0])
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.show()
    return print("Done")


def normalize(x_vec):
    """
    Description
    -----------
    Normalizes the input vector.

    Parameters
    ----------
    x_vec : array(float)
        Data vector.
    Returns
    -------
    x_norm : array(float)
        Normalized vector.

    """
    # Image vector normalization
    max_amp = x_vec.max()
    min_amp = x_vec.min()
    x_norm = (x_vec - min_amp) / (max_amp - min_amp)
    return x_norm, max_amp


def redim(x_vec, dims):
    """
    Description
    -----------
    Resizes the input vector.

    Parameters
    ----------
    x_vec : array(float)
        Data vector.
    dims : list(int)
        Dimensions to which the data in 'vector' should be resized.
    Returns
    -------
    x_norm : array(float)
        Resized vector.
    """
    # Image resizing
    resized_images = []
    for im in x_vec:
        resized_images.append(resize(im, (dims[0], dims[1])))
    x_resized = np.array(resized_images)
    return x_resized


# Crops the data of the input set
def recorte(conjunto, val_rec):
    """
    Description
    -----------
    Crop along the vertical axis of the image sets.

    Parameters
    ----------
    conjunto : list(list)
        Set of lists of images.
    val_rec : int
        Row value from which the image information is irrelevant.

    Returns
    -------
    conjunto_rec : list(list)
        Set of lists of already cropped images.

    """
    conjunto_rec = []
    for cwt in conjunto:
        vector_rec = []
        choices = np.random.choice(np.arange(len(cwt)), 5)
        for choice in choices:
            image_rec = cwt[choice][:val_rec]
            plt.title(f"Cropped nº{choice}")
            plt.imshow(image_rec, cmap="gray")
            plt.axis("on")
            plt.show()
        for index in range(len(cwt)):
            vector_rec.append(cwt[index][:val_rec])
        conjunto_rec.append(vector_rec)
    return conjunto_rec


# Allows truncating the data of the .csv file by visualizing its graphical representation
def recorteCSV(conjunto, legend, time_rotura):
    """
    Description
    -----------
    Allows plotting data from a dataset and removing data presented from a certain row onwards.

    Parameters
    ----------
    conjunto : list(DataFrame)
        List of data matrices.
    legend : list(str)
        List of names to use in the plot legends.
    time_rotura :


    Returns
    -------
    conjunto_rec : list(numpy)
        List of vectors with the data already removed.
    time : list(float)
        List of time instants used in the data arrangement.

    """
    conjunto_rec = []
    time = []
    for vector in conjunto:
        plt.figure(figsize=(10, 6), dpi=260)
        plt.plot(vector[:, 2], vector[:, 1])
        plt.xlabel("Time [s]")
        plt.ylabel("Load [kN]")
        plt.grid(True)
        plt.show()

        if time_rotura:
            time_rotura = int(time_rotura)

        else:
            response = "No"
            while response == "No" or response == "no":
                print("\nAt what time should we crop to remove the failure?")
                time_rotura = int(input())

                if time_rotura == -1:
                    time_rotura = vector[-1, 2]

                plt.figure(figsize=(10, 6), dpi=260)
                plt.plot(vector[:, 2], vector[:, 1])
                plt.axvline(time_rotura, linestyle="--", color="red")
                plt.xlabel("Time [s]")
                plt.ylabel("Load [kN]")
                plt.grid(True)
                plt.show()

                print("\nSatisfied?")
                response = input()

        if time_rotura == -1:
            vector_rec = vector
        else:
            pos_rotura = np.where(vector[:, 2] >= time_rotura)[0][0]
            vector_rec = vector[:pos_rotura, :]
        conjunto_rec.append(vector_rec)
        time.append(time_rotura)
    multiplot(
        conjunto,
        legend,
        ["Uncropped set"],
        ["Time [s]", "Force [kN]"],
        [2, 1],
    )
    multiplot(
        conjunto_rec,
        legend,
        ["Cropped set"],
        ["Time [s]", "Force [kN]"],
        [2, 1],
    )
    return conjunto_rec, time


# Adjusts the .pridb file based on the cropping generated in the .csv
def recortePRIDB(df_data, time):
    """
    Description
    -----------
    Crops the pridb data from a certain test time instant.

    Parameters
    ----------
    df_data : DataFrame
        pridb data matrix.
    time : int
        Time after which the test is not considered valid.

    Returns
    -------
    df_data_rec : DataFrame
        Matrix of valid data.

    """
    if time == -1:
        time = df_data["time"].iloc[-1]
    df_drop = np.where(df_data["time"] >= time)[0]
    df_data_rec = df_data[: df_drop[0]]
    return df_data_rec


# Removes trais that do not meet the stipulated conditions
def TRAIdelete(df_data, amp_lim, cnts_lim):
    """
    Description
    -----------
    Removes data that do not meet the minimum threshold conditions.

    Parameters
    ----------
    df_data : DataFrame
        tradb data matrix.
    amp_lim : int
        Minimum value from which the data are valid.
    cnts_lim : int
        Minimum value for a hit to be valid.

    Returns
    -------
    df_data : DataFrame
        Data matrix that meets the conditions.

    """
    amp_lim = int(amp_lim)
    cnts_lim = int(cnts_lim)
    deleted = []
    for row in range(df_data.shape[0]):
        if (
            np.array(df_data["amplitude"])[row] < amp_lim
            or np.array(df_data["counts"])[row] < cnts_lim
        ):
            deleted.append(row)
    df_data = df_data.reset_index(drop=True).drop(deleted, axis=0)
    return df_data


def save_image_pred(image_list, dims, test_name, images_dir):
    """
    Description
    -----------
    Saves the images generated by the predictive model.

    Parameters
    ----------
    image_list : list(array(float))
        List of images generated by the model.
    dims : list(int)
        Dimensions of the images.
    path : str
        Directory to store the images.
    name : str
        Name of the file.

    Returns
    -------
    None
        Saves the images in the specified directory.

    """
    path = os.path.join(images_dir, test_name)
    os.makedirs(path, exist_ok=True)
    image_redim = redim(image_list, dims)
    for i, image in enumerate(image_redim):
        image_name = f"CWT_{test_name}_{i:04d}.png"
        image_path = os.path.join(path, image_name)
        plt.imsave(image_path, image, cmap="gray")
    return print(f"Images of test {test_name} for prediction saved")
