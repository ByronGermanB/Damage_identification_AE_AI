# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:09:45 2024

@author: bbarmac
"""

#%%
# =============================================================================
# Importamos las librerias necesarias
# =============================================================================
# Dataframes
import pandas as pd
import os
import numpy as np

# Figures
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import string
from matplotlib.ticker import MultipleLocator
matplotlib.rcParams['font.family'] = 'Times New Roman'

# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['figure.dpi'] = 300

#%%

def force_data(path, section=[22, 2.4]):
    '''
    
    Parameters
    ----------
    path : str
        Directorio donde se encuentran almacenados los csv de fuerza mts.
    section : list or array
        Medidas de ancho y espesor de la probeta en mm. The default is [22, 2.4]

    Returns
    -------
    force_df : Dataframe
        Dataframe que contiene los datos de fuerza, desplazamiento, tiempo y test_id.

    '''
    files =  os.listdir(path)
    forces_list = []
    
    for item in files:
        # Verificar si los archivos son .csv
        if item.endswith(".csv"):
            
            # Path de la carpeta unida con el archivo
            force_filename_path = os.path.join(path, item)
            
            # Carga del archivo csv
            force = pd.read_csv(force_filename_path, sep=';')
            
            # Renombrar las columnas y eliminar la fila de unidades
            new_column_names = ['Crosshead [mm]', 'Load [kN]', 'Time [s]']
            force.columns = new_column_names
            force = force.iloc[1:]
            
            # Add una columna de 0 donde incia el ensayo
            new_row = pd.Series(0, index=force.columns)
            force = pd.concat([pd.DataFrame([new_row]), force], ignore_index=True)
            
            # Change all the , for .
            force = force.replace({',':'.'}, regex=True)
            force = force.apply(pd.to_numeric, errors='coerce')
            
            # Stress
            area = np.prod(section)
            force['Stress [MPa]'] = force['Load [kN]'] * 1000 / area

            # Add la columna de test id
            test_id = item.split('.')[0]
            force['test_id'] = test_id
            forces_list.append(force)
        
    force_df = pd.concat(forces_list, ignore_index=True)
    return force_df
    
def plot_stress_hits_cluster(labels, hits, force, test_id, figure_path, plot_type='scatter', 
                    title='Stress and Cumulative hits vs Time - Clustering', x='time', y='Cumulative_Label', 
                    x_label='Time [s]', y_label='Cumulative hits', y_label_right='Stress [MPa]',
                    subtitle=None, limits=None, width=90, height=60, ax=None, i=1, n_col=1, n_row=1, guardar=False):
    '''
    Parameters
    ----------
    labels : array
        Labels de los clusters obtenidos por k-means o dbscan.
    hits : Dataframe
        Set que contiene las caracteristicas en el tiempo de los hits y su id.
    force : Dataframe
        Set que contiene los datos de carga desplazamiento, tiempo y su id
    test_id : str
        Id del ensayo que se vaya a graficar.
    figure_path : str
        Directorio para guardar la figura.
    plot_type : str, optional
        Tipo de gráfico a realizar ('scatter' o 'line'). The default is 'scatter'.
    title : str, optional
        Titulo de la imagen para mostrar y guardar. The default is 'Cumulative hits vs Time'.
    x : str, optional
        Caracteristica en el tiempo para el eje x. The default is 'time'.
        
        Otras opciones: 'Cumulative_Label', 'Cumulative', 'amplitude', 'energy'
    y : str, optional
        Caracteristica en el tiempo para el eje y. The default is 'Cumulative_Label'.
        
        Otras opciones: 'time', 'Cumulative', amplitude', 'energy'
    x_label : str, optional
        Leyenda del eje x. The default is 'Time [s]'.
    y_label : str, optional
        Leyenda del eje y. The default is 'Cumulative hits'.
    y_label_rigth : str, optional
        Leyenda del eje y en la derecha. The default is 'Stress [MPa]'.
    subtitle : str or list, optional
        Titulo para cada uno de los subplots, si se deja en default se plotea el test_id. The default is None.
    hits_limit : int or float, optional
         Limite superior para el eje y de hits. The default is None
    stress_limit : int or float, optional
         Limite superior para el eje y de fuerza. The default is None
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90  
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    ax : matplotlib axis object, optional
        Axis to plot on. If not provided, a new axis will be created.
    i : int, optional
        Index for subplot. Default is 1.
    n_col : int, optional
        numero de columnas. The default is 1.
    n_row : int, optional
        numero de filas. The default is 1.
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. Cluster vs Time {test_id}.pdf
    
    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Grafica y (guarda) un diagrama de los hits acumulados y esfuerzo en formato pdf.
    '''
    # Extract the limits
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    # Create a DataFrame with the labels
    labels_df = pd.Series(labels, name='Labels')
    labels_df.index = hits.index
    
    # Concatenate the new DataFrame with the existing one
    hits_labels = pd.concat([hits, labels_df], axis=1)
            
    # Condition to filter rows based on the string column
    condition = (hits_labels['test_id'] == test_id)
    condition_force = (force['test_id'] == test_id)
        
    # Filtered DataFrame
    filtered_hits = hits_labels.loc[condition].copy()
    filtered_hits['Cumulative_Label'] = filtered_hits.groupby('Labels')['Count'].cumsum()
    
    # Force data
    filtered_force = force[condition_force]
    
    # Count the number of unique labels in the DataFrame
    num_labels = filtered_hits['Labels'].nunique()

    # Define custom markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Customize the markers as desired
    palette = palette=sns.color_palette()
    # Customize the markers list to match the number of unique labels
    markers = markers[:num_labels]
    palette = palette[:num_labels]
        
    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_inches, dpi=300, tight_layout=True)
        ax.set_title(f'Stress and Cumulative hits vs Time - {test_id}')
        ax.set_xlabel(x_label)
    
    else:
        ax.set_xlabel(x_label + '\n(' + string.ascii_lowercase[i-1] + ')')     
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(test_id)  
    
    # Create the scatter plot
    if plot_type == 'scatter':
        sns.scatterplot(x=x, y=y, hue='Labels', style='Labels', data=filtered_hits, 
                    markers=markers, palette = palette, linewidth=0.2, s=20, ax=ax)
    elif plot_type == 'line':
        # sns.lineplot(x=x, y=y, hue='Labels', style='Labels', data=filtered_hits, 
        #             markers=markers, palette=sns.color_palette(), markersize=4, markeredgewidth=0.0, ax=ax)
        
        sns.lineplot(x=x, y=y, hue='Labels', style='Labels', data=filtered_hits, palette = palette, linewidth=2, ax=ax)
    
    #Set the x-axis limit
    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    # Set y-axis label only for the first plot of each row
    if (i-1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel('')
       
    # Set the y-axis limit
    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)   
    
    # Create a twin Axes sharing the x-axis
    ax2 = ax.twinx()
    
    # Create the second scatter plot on the twin Axes
    sns.lineplot(x='Time [s]', y='Stress [MPa]', data=filtered_force, ax=ax2, linewidth=1, color='black', label='Stress')
    
    # Set the y-axis limit for the second plot
    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force['Stress [MPa]']) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)
       
    # Set y-axis label only for the last plot of each row
    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel('')
    
    # Combine legends
    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    # Legend in the first plot
    if i == 1:
        # Create a new legend with combined handles and legends
        ax2.legend(handles, legends, loc='upper left')
        
        # Remove the separate legend from ax if it exists
        ax.get_legend().remove()
    else:
        ax.get_legend().remove()
        ax2.get_legend().remove()
        
    # Plot the grid
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)  # Set grid lines behind the data points
    
    # Guardar la imagen en figure_path
    if guardar:
        if ax is None:
            figure_filename = f'{title} - {test_id}.pdf'
        else:
            figure_filename = f'{title}.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
    
    return ax

def plot_stress_hits(hits, force, test_id, figure_path, plot_type='scatter', 
                    title='Stress and Cumulative hits vs Time', x='time', y='Cumulative_Label', 
                    x_label='Time [s]', y_label='Cumulative hits', y_label_right='Stress [MPa]',
                    subtitle=None, limits=None, width=90, height=60, ax=None, i=1, n_col=1, n_row=1, guardar=False):
    '''
    Parameters
    ----------
    hits : Dataframe
        Set que contiene las caracteristicas en el tiempo de los hits y su id.
    force : Dataframe
        Set que contiene los datos de carga desplazamiento, tiempo y su id
    test_id : str
        Id del ensayo que se vaya a graficar.
    figure_path : str
        Directorio para guardar la figura.
    plot_type : str, optional
        Tipo de gráfico a realizar ('scatter' o 'line'). The default is 'scatter'.
    title : str, optional
        Titulo de la imagen para mostrar y guardar. The default is 'Stress and Cumulative hits vs Time'.
    x : str, optional
        Caracteristica en el tiempo para el eje x. The default is 'time'.
        
        Otras opciones: 'Cumulative_Label', 'Cumulative', 'amplitude', 'energy'
    y : str, optional
        Caracteristica en el tiempo para el eje y. The default is 'Cumulative_Label'.
        
        Otras opciones: 'time', 'Cumulative', amplitude', 'energy'
    x_label : str, optional
        Leyenda del eje x. The default is 'Time [s]'.
    y_label : str, optional
        Leyenda del eje y. The default is 'Cumulative hits'.
    y_label_rigth : str, optional
        Leyenda del eje y en la derecha. The default is 'Stress [MPa]'.
    subtitle : str or list, optional
        Titulo para cada uno de los subplots, si se deja en default se plotea el test_id. The default is None.
    hits_limit : int or float, optional
         Limite superior para el eje y de hits. The default is None
    stress_limit : int or float, optional
         Limite superior para el eje y de fuerza. The default is None
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90  
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    ax : matplotlib axis object, optional
        Axis to plot on. If not provided, a new axis will be created.
    i : int, optional
        Index for subplot. Default is 1.
    n_col : int, optional
        numero de columnas. The default is 1.
    n_row : int, optional
        numero de filas. The default is 1.
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. Cluster vs Time {test_id}.pdf
    
    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Grafica y (guarda) un diagrama de los hits acumulados y esfuerzo en formato pdf.
    '''
    # Extract the limits
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    # Condition to filter rows based on the string column
    condition = (hits['test_id'] == test_id)
    condition_force = (force['test_id'] == test_id)
        
    # Filtered DataFrame
    filtered_hits = hits.loc[condition].copy()
    filtered_hits['Cumulative_Label'] = filtered_hits['Count'].cumsum()
    
    # Force data
    filtered_force = force[condition_force]
        
    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_inches, dpi=300, tight_layout=True)
        ax.set_title(f'Stress and Cumulative hits vs Time - {test_id}')
        ax.set_xlabel(x_label)
    
    else:
        ax.set_xlabel(x_label + '\n(' + string.ascii_lowercase[i-1] + ')')     
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(test_id)
    
    # # Set x-axis label only for the bottom column 
    # elif i > n_col*(n_row - 1):
    #     ax.set_xlabel(x_label + '\n(' + string.ascii_lowercase[i-1] + ')')     
    #     # Title for each test id
    #     if subtitle is not None:
    #         ax.set_title(subtitle)
    #     else:
    #         ax.set_title(test_id)
        
    # else:
    #     ax.set_xlabel('(' + string.ascii_lowercase[i-1] + ')')
    #     # Title for each test id
    #     if subtitle is not None:
    #         ax.set_title(subtitle)
    #     else:
    #         ax.set_title(test_id)
    
    
    # Create the scatter plot
    if plot_type == 'scatter':
        sns.scatterplot(x=x, y=y, data=filtered_hits, label='hits', linewidth=0.2, s=20, ax=ax)
    elif plot_type == 'line':
        # sns.lineplot(x=x, y=y, hue='Labels', style='Labels', data=filtered_hits, 
        #             markers=markers, palette=sns.color_palette(), markersize=4, markeredgewidth=0.0, ax=ax)
        
        sns.lineplot(x=x, y=y, label='hits', data=filtered_hits, linewidth=2, ax=ax)

    #Set the x-axis limit
    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    # Set y-axis label only for the first plot of each row
    if (i-1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel('')
       
    # Set the y-axis limit
    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)   
    
    # Create a twin Axes sharing the x-axis
    ax2 = ax.twinx()
    
    # Create the second scatter plot on the twin Axes
    sns.lineplot(x='Time [s]', y='Stress [MPa]', data=filtered_force, ax=ax2, linewidth=1, color='black', label='Stress')
    
    # Set the y-axis limit for the second plot
    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force['Stress [MPa]']) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)
       
    # Set y-axis label only for the last plot of each row
    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel('')
    
    # Combine legends
    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    # Legend in the first plot
    if i == 1:
        # Create a new legend with combined handles and legends
        ax2.legend(handles, legends, loc='upper left')
        
        # Remove the separate legend from ax if it exists
        ax.get_legend().remove()
    else:
        ax.get_legend().remove()
        ax2.get_legend().remove()
    
    # ax2.yaxis.set_major_locator(MultipleLocator(10))
    
    # Plot the grid
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)  # Set grid lines behind the data points
    
    # Guardar la imagen en figure_path
    if guardar:
        if ax is None:
            figure_filename = f'{title} - {test_id}.pdf'
        else:
            figure_filename = f'{title}.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
    
    return ax

def limit_finder(labels, hits, force, test_ids):
    '''
    Parameters
    ----------
    labels : array
        Labels de los clusters obtenidos por k-means o dbscan.
    hits : Dataframe
        Set que contiene las caracteristicas en el tiempo de los hits y su id.
    force : Dataframe
        Set que contiene los datos de carga desplazamiento, tiempo y su id
    test_ids : list
        Id de los ensayos que se toman en cuenta para el calculo.
   
    Returns
    -------
    max_value_hits : float
        valor maximo para hits.
    max_value_stress : TYPE
        valor maximo para esfuerzo.

    '''
    # Create a DataFrame with the labels
    labels_df = pd.Series(labels, name='Labels')
    labels_df.index = hits.index
    
    # Concatenate the new DataFrame with the existing one
    hits_labels = pd.concat([hits, labels_df], axis=1)
    
    # Condition to filter rows based on the string column
    condition = hits_labels['test_id'].isin(test_ids)
    condition_force = force['test_id'].isin(test_ids)
    
    # Filtered DataFrame
    filtered_hits = hits_labels.loc[condition].copy()
    filtered_hits['Cumulative_Label'] = None
    
    for test_id in test_ids:
        single_condition = filtered_hits['test_id'] == test_id
        filtered_hits.loc[single_condition, 'Cumulative_Label'] = filtered_hits[single_condition].groupby('Labels')['Count'].cumsum()
    
    # Force data
    filtered_force = force[condition_force]
    
    # Maximum values of the df
    max_value_time = filtered_hits['time'].max()
    max_value_hits = filtered_hits['Cumulative_Label'].max()
    max_value_stress = filtered_force['Stress [MPa]'].max()
    
    return [max_value_time, max_value_hits, max_value_stress]

def limit_finder_no_label(hits, force, test_ids):
    '''
    Parameters
    ----------
    hits : Dataframe
        Set que contiene las caracteristicas en el tiempo de los hits y su id.
    force : Dataframe
        Set que contiene los datos de carga desplazamiento, tiempo y su id
    test_ids : list
        Id de los ensayos que se toman en cuenta para el calculo.
   
    Returns
    -------
    max_value_hits : float
        valor maximo para hits.
    max_value_force : TYPE
        valor maximo para esfuerzo.

    '''
    # Condition to filter rows based on the string column
    condition = hits['test_id'].isin(test_ids)
    condition_force = force['test_id'].isin(test_ids)
        
    # Filtered DataFrame
    filtered_hits = hits.loc[condition].copy()
    filtered_hits['Cumulative_Label'] = None
    
    for test_id in test_ids:
        single_condition = filtered_hits['test_id'] == test_id
        filtered_hits.loc[single_condition, 'Cumulative_Label'] = filtered_hits[single_condition]['Count'].cumsum()
    
    # Force data
    filtered_force = force[condition_force]
       
    # Maximum values of the df
    max_value_time = filtered_hits['time'].max()
    max_value_hits = filtered_hits['Cumulative_Label'].max()
    max_value_stress = filtered_force['Stress [MPa]'].max()
    
    return [max_value_time, max_value_hits, max_value_stress]
    
def reorder_hits(hits, y, y_pred):
    '''
    Parameters
    ----------
    hits : dataframe
        Dataframe para grafica de hits acumulados.
    y : series or array
        Clases.
    y_pred : TYPE
        Predicciones.

    Returns
    -------
    hits_ordered : dataframe
        Hits ordenados por tiempo ascendente.
    y_pred_ordered : series
        Predicciones ordenadas por tiempo ascendente.

    '''
    # Create a DataFrame with the predictions
    y_pred_df = pd.Series(y_pred, name='Clase')
    
    # Seleccionar solo los hits de training o test set
    hits = hits.loc[y.index]
    hits = hits.reset_index(drop=True)
    
    # Eliminar columna clase
    hits = hits.drop('Clase', axis=1)
    
    # Unir dataframes de hits y predicciones
    hits_and_pred = pd.concat([hits, y_pred_df], axis=1)   
    
    # Ordenar dataframe en base a la columna tiempo
    hits_and_pred = hits_and_pred.sort_values(by='time', ascending=True)
    
    # Separamos hits y predicciones
    hits_ordered = hits_and_pred.drop('Clase', axis=1)
    y_pred_ordered = hits_and_pred['Clase'].copy()
    
    return hits_ordered, y_pred_ordered