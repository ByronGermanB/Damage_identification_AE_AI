# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:54:49 2023

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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import string

# Librerias de algoritmos no supervisados
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

# Librerias de metricas
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import davies_bouldin_score

# Libreria para grafica de silueta
from matplotlib.ticker import FixedLocator, FixedFormatter

# Set font to Times New Roman and size to 8

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['figure.dpi'] = 150

#%%
# =============================================================================
# DBSCAN 
# =============================================================================

def grid_search_dbscan(X, target_clusters, epsilon_values, min_samples_values):
    '''
    Parameters
    ----------
    X : Dataframe, array
        Set de features para entrenamiento.
    target_clusters : int
        Numero objetivo de clusters.
    epsilon_values : array or list
        Valores que puede tomar epsilon para clustering.
    min_samples_values : array or list
        valores que puede tomar el numero minimo de instancias para clusterin.

    Returns
    -------
    best_models : list
        Lista que contiene los modelos DBSCAN que cumplen la condicion de target clusters.
    results : list
        Lista que contiene los valores de epsilon, min_samples, numero de clusters, numero y porcentaje de anomalias de cada modelo DBSCAN.

    '''
    best_models = []
    results = []

    total_samples = len(X)

    for epsilon in epsilon_values:
        for min_samples in min_samples_values:
            # Create a DBSCAN model with current parameter values
            dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples)

            # Fit the model and obtain labels
            labels = dbscan_model.fit_predict(X)

            # Check if the number of clusters matches the target_clusters
            unique_labels = np.unique(labels)
            
            if -1 in unique_labels:
                num_clusters = len(unique_labels) - 1  # Subtract 1 for the noise cluster
            else:
                num_clusters = len(unique_labels)
            
            # Check if the current model meets the target_clusters
            if num_clusters == target_clusters:
                results.append({
                    "Epsilon": epsilon,
                    "min_samples": min_samples,
                    "Number of Clusters": num_clusters,
                    "Number of Anomalies": np.sum(labels == -1),
                    "Percentage of Anomalies": (np.sum(labels == -1) / total_samples) * 100,
                    'Silhouette Score': silhouette_score(X, labels),
                    'DBI Score': davies_bouldin_score(X, labels)
                })
                best_models.append(dbscan_model)
                
    # Print the results
    for i, result in enumerate (results):
        for key, value in result.items():
            print(f"{key}: {value}")
        print("-----")

    return best_models, results

def kmeans_per_k(X, k, selected_features=None):
    '''
    Parameters
    ----------
    X : Dataframe, array
        Set de features para entrenamiento.
    k : int
        Numero maximo de clusters a evaluar.
    selected_features : list, optional
        Nombre de los features para entrenar el modelo. The default is None.

    Returns
    -------
    kmeans_per_k : List
        Lista que contiene los modelos de k-means entrenados desde 2 hasta k.
    silhouette_scores : List
        Valores del silhouette score de cada modelo.
    dbi_scores : List
        Valores del Davies-Bouldin score de cada modelo.

    '''
    # Seleccion de features 
    if selected_features is not None:
        X = X[selected_features]
    
    kmeans_per_k = [KMeans(n_clusters=i, random_state=42).fit(X)
                    for i in range(2, k+1)]
       
    # Calculo de los scores para cada k
    silhouette_scores = [silhouette_score(X, model.labels_)
                         for model in kmeans_per_k]
    
    dbi_scores = [davies_bouldin_score(X, model.labels_)
                         for model in kmeans_per_k]
    
    # Print silhouette scores for each model
    for i, score in enumerate(silhouette_scores, start=2):
        print(f"Silhouette Score for k={i}: {score:.2f}")
    
    for i, score in enumerate(dbi_scores, start=2):  
        print(f"Davies-Bouldin Score for k={i}: {score:.2f}")

    return kmeans_per_k, silhouette_scores, dbi_scores

def tsne(X, figure_path, width=90, height=60, selected_features=None, guardar=False):
    '''
    Parameters
    ----------
    X : Dataframe, array
        Set de features para entrenamiento.
    figure_path : str
        Directorio para guardar la figura.
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90    
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    selected_features : list, optional
        Nombre de los features para entrenar el modelo. The default is None.
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. t-SNE Visualization.pdf
    Returns
    -------
    X_reduced : Array
        Array de 2 columnas con la reduccion de dimensionalidad previa para entrenamiento de clustering.
        
    Grafica y (guarda) un diagrama de dispersion de los datos con reduccion de dimensionalidad en formato pdf.
    
    '''   
    print("Running t-SNE...")
    # Seleccion de features (si se utilizo backward elimination)
    if selected_features is not None:
        X = X[selected_features]
            
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X)
        
    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f't-SNE_{i+1}' for i in range(2)])

    # Create a scatter plot with Seaborn
    figsize_inches = (width / 25.4, height / 25.4)
    plt.figure(figsize=figsize_inches, tight_layout=True)
    sns.scatterplot(x='t-SNE_1', y='t-SNE_2', data=tsne_df, palette='viridis')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid()
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points    
        
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = 't-SNE Visualization.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
        
    plt.show()
    
    return X_reduced

def plot_kmeans_per_k(X, kmeans_per_k, silhouette_scores, figure_path, guardar=False):
    '''
    Parameters
    ----------
    X : Dataframe, array
        Set de features para entrenamiento.
    kmeans_per_k : List
        Lista de modelos k-means.
    silhouette_scores : List
        Valores del silhouette score de cada modelo.
    figure_path : str
        Directorio para guardar la figura.
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. Silhouette Analysis for K-Means Clustering.pdf
    Returns
    -------
    Grafica y (guarda) los graficos de silhouette score en formato pdf.

    '''
    
    plt.figure(figsize=(11, 9))
    
    # Add a super title
    plt.suptitle("Silhouette Analysis for K-Means Clustering", fontsize=16)
    
    for k in (2, 3, 4, 5):
        plt.subplot(2, 2, k - 1)
        
        y_pred = kmeans_per_k[k - 2].labels_
        silhouette_coefficients = silhouette_samples(X, y_pred)
    
        padding = len(X) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()
    
            # color = plt.cm.Spectral(i / k)
            color = plt.cm.nipy_spectral(float(i) / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding
    
        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        if k in (2, 4):
            plt.ylabel("Cluster")        
        
        if k in (4,5):
            plt.xlabel("Silhouette Coefficient")
      
    
        plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
        plt.title(f"$k={k}$")
        # Activate grid
        plt.grid(axis='x', linestyle='--', which='major', color='grey', alpha=0.5)
        plt.gca().set_axisbelow(True)  # Set grid lines behind the data points
    
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = "Silhouette Analysis for K-Means Clustering.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
    
    plt.show()

def plot_silhouette(labels, X, silhouette_score, figure_path, width=90, height=60, guardar=False):
    '''
    Parameters
    ----------
    labels : array
        Labels de los clusters obtenidos por k-means o dbscan.
    X : Dataframe, array
        Set de features para entrenamiento.
    silhouette_score : List
        Valores del silhouette score de cada modelo.
    figure_path : str
        Directorio para guardar la figura.
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90
        
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. Silhouette Analysis for K-Means Clustering.pdf
    Returns
    -------
    Grafica y (guarda) los graficos de silhouette score en formato pdf.

    '''
    
    figsize_inches = (width / 25.4, height / 25.4)
    plt.figure(figsize=figsize_inches, tight_layout=True)
    
    unique_labels = np.unique(labels)
    k = unique_labels.size
           
    silhouette_coefficients = silhouette_samples(X, labels)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[labels == i]
        coeffs.sort()

        # color = plt.cm.Spectral(i / k)
        color = plt.cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    

    plt.ylabel("Cluster")          
    plt.xlabel("Silhouette Coefficient")
  

    plt.axvline(x=silhouette_score, color="red", linestyle="--")
    plt.title('Silhouette Analysis')
    # Activate grid
    plt.grid(axis='x', linestyle='--', which='major', color='grey', alpha=0.5)
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points
    
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = "Silhouette Analysis.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
    
    plt.show()

def tsne_3d(X, figure_path, width=90, height=60, selected_features=None, guardar=False):
    '''
    Parameters
    ----------
    X : Dataframe, array
        Set de features para entrenamiento.
    figure_path : str
        Directorio para guardar la figura.
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90    
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    selected_features : list, optional
        Nombre de los features para entrenar el modelo. The default is None.
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. t-SNE 3D Visualization.pdf
    Returns
    -------
    X_reduced : Array
        Array de 3 columnas con la reduccion de dimensionalidad previa para entrenamiento de clustering.
        
    Grafica y (guarda) un diagrama de dispersion de los datos con reduccion de dimensionalidad en formato pdf.
    
    '''   
    print("Running t-SNE in 3D...")
    # Seleccion de features (si se utilizo backward elimination)
    if selected_features is not None:
        X = X[selected_features]
            
    tsne = TSNE(n_components=3, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X)
        
    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f't-SNE_{i+1}' for i in range(3)])

    # Create a 3D scatter plot
    figsize_inches = (width / 25.4, height / 25.4)
    fig = plt.figure(figsize=figsize_inches, tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(tsne_df['t-SNE_1'], tsne_df['t-SNE_2'], tsne_df['t-SNE_3'], c=tsne_df.index, cmap='viridis')
    ax.set_title('t-SNE 3D Visualization')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = 't-SNE 3D Visualization.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
        
    plt.show()
    
    return X_reduced

def plot_cluster_feat(labels, X, feat_1, feat_2, figure_path, title=None, subtitle=None,
                      x_label = None, y_label = None, width=90, height=60, ax=None, 
                      i=1, n_col=1, n_row=1, guardar=False):
    '''
    Parameters
    ----------
    labels : array
        Labels de los clusters obtenidos por k-means o dbscan.
    X : Dataframe, array
        Set de features para entrenamiento.
    feat_1 : str
        Nombre del feature 1 para graficar.
    feat_2 : str
        Nombre del feature 2 para graficar.
    figure_path : str
        Directorio para guardar la figura.
    title : str or list, optional
        titulo de la figura, si se deja en default sera {feat_1} vs {feat_2} - Clustering. The default is None
    subtitle : str or list, optional
        subtitulo de la figura, si se deja en default sera {feat_1} vs {feat_2} - Clustering. The default is None
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
        
        e.g. Feat_vs_Feat_{feat_1}_{feat_2} - Clustering.pdf
    Returns
    -------
    Grafica y (guarda) un diagrama de dispersion de dos features con etiquetas y marcadores en formato pdf.

    '''    
    labels = pd.Series(labels, name='Labels')
    # Rename 'Labels' column in y if it already exists in X
    if 'Labels' in X.columns:
        X = X.drop(['Labels'], axis=1)    
    
    labels.index = X.index
    data = pd.concat([X, labels], axis=1)
    
    # Mascaras para cambiar de marcadores 
    valid_points = data[data['Labels'] != -1]
    anomalies = data[data['Labels'] == -1]
    
    # Define custom markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Customize the markers as desired
    
    # Create a figure
    figsize_inches = (width / 25.4, height / 25.4)
    
    # If there is just one plot and set the title
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        if title is None:
            ax.set_title(f'{feat_1} vs {feat_2} - Clustering')
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
            
    # Use Seaborn scatter plot with hue and style parameters
    sns.scatterplot(x=valid_points[feat_1], y=valid_points[feat_2], hue='Labels', style='Labels',
                    data=valid_points, palette=sns.color_palette(), markers=markers, ax=ax)
    
    # Scatter points with label '-1' separately as red 'x'
    sns.scatterplot(x=anomalies[feat_1], y=anomalies[feat_2], hue='Labels', style='Labels', 
                    data=anomalies, palette=['r'], markers='x', linewidth=1.5, ax=ax)    

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
            figure_filename = f"Feat_vs_Feat_{feat_1}_{feat_2} - Clustering.pdf"
        
        elif title is not None:
            figure_filename = f'{title}.pdf'
        
        else:
            figure_filename = 'Feat_vs_Feat_clustering.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
    
    return ax    

def plot_cluster_time(labels, hits, test_id, figure_path, width=90, height=60, guardar=False):
    '''
    Parameters
    ----------
    labels : array
        Labels de los clusters obtenidos por k-means o dbscan.
    hits : Dataframe
        Set que contiene las caracteristicas en el tiempo de los hits y su id.
    test_id : str
        Id del ensayo que se vaya a graficar.
    figure_path : str
        Directorio para guardar la figura.
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90  
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. Cluster vs Time - {test_id}.pdf

    Returns
    -------
    Grafica y (guarda) un diagrama de dispersion de las etiquetas de clusters vs tiempo en formato pdf.

    '''
    # Create a DataFrame with the labels
    labels_df = pd.Series(labels, name='Labels')
    labels_df.index = hits.index
    
    # Concatenate the new DataFrame with the existing one
    hits_labels = pd.concat([hits, labels_df], axis=1)
    
    # Condition to filter rows based on the string column
    condition = (hits_labels['test_id'] == test_id)

    # Filtered DataFrame
    filtered_hits = hits_labels[condition]

    # Define custom markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Customize the markers as desired

    # Plotting with Seaborn scatter plot
    figsize_inches = (width / 25.4, height / 25.4)
    plt.figure(figsize=figsize_inches, tight_layout=True)
    sns.scatterplot(x='time', y='Labels', hue='Labels', style='Labels', data=filtered_hits, markers=markers, palette=sns.color_palette(), s=100)
    plt.xlabel('Time [s]')
    plt.ylabel('Labels')
    plt.title(f'Cluster vs Time - {test_id}')
    plt.grid()
    plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=0.5)
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points
        
    # Set y-axis ticks to display only integer values
    plt.yticks(range(int(filtered_hits['Labels'].min()), int(filtered_hits['Labels'].max()) + 1))
    
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = f'Cluster vs Time - {test_id}.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
        
    plt.show()
    
def plot_cluster_hits(labels, hits, test_id, figure_path, width=90, height=60, plot_type='scatter', 
                      x='time', y='Cumulative_Label', x_label='Time [s]', y_label='Cumulative hits', 
                      title='Cumulative hits vs Time', guardar=False):
    '''
    Parameters
    ----------
    labels : array
        Labels de los clusters obtenidos por k-means o dbscan.
    hits : Dataframe
        Set que contiene las caracteristicas en el tiempo de los hits y su id.
    test_id : str
        Id del ensayo que se vaya a graficar.
    figure_path : str
        Directorio para guardar la figura.
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90  
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    plot_type : str, optional
        Tipo de gráfico a realizar ('scatter' o 'line'). The default is 'scatter'.
    x : str, optional
        Caracteristica en el tiempo para el eje x. The default is 'time'.
        
        Otras opciones: 'Cumulative_Label', 'Cumulative', 'amplitude', 'energy'
    y : str, optional
        Caracteristica en el tiempo para el eje y. The default is 'Cumulative_Label'.
        
        Otras opciones: 'time', 'Cumulative', amplitude', 'energy'
    x_label : str, optional
        Nombre del eje x. The default is 'Time [s]'.
    y_label : str, optional
        Nombre del eje y. The default is 'Cumulative hits'.
    title : str, optional
        Titulo de la imagen para mostrar y guardar. The default is 'Cumulative hits vs Time'.
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. Cluster vs Time {test_id}.pdf

    Returns
    -------
    Grafica y (guarda) un diagrama de dispersion de las caracteristicas temporales con etiquetas y marcadores en formato pdf.

    '''
    
    # Create a DataFrame with the labels
    labels_df = pd.Series(labels, name='Labels')
    labels_df.index = hits.index
    
    # Concatenate the new DataFrame with the existing one
    hits_labels = pd.concat([hits, labels_df], axis=1)
            
    # Condition to filter rows based on the string column
    condition = (hits_labels['test_id'] == test_id)
        
    # Filtered DataFrame
    filtered_hits = hits_labels[condition]
    filtered_hits['Cumulative_Label'] = filtered_hits.groupby('Labels')['Count'].cumsum()
    
    # Define custom markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Customize the markers as desired

    # Plotting with Seaborn scatter plot or line plot based on plot_type
    figsize_inches = (width / 25.4, height / 25.4)
    plt.figure(figsize=figsize_inches, tight_layout=True)
    if plot_type == 'scatter':
        sns.scatterplot(x=x, y=y, hue='Labels', style='Labels', data=filtered_hits, markers=markers, palette=sns.color_palette())
    elif plot_type == 'line':
        sns.lineplot(x=x, y=y, hue='Labels', style='Labels', data=filtered_hits, palette=sns.color_palette())
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{title} - {test_id}')
    plt.grid()
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points
    
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = f'{title} - {test_id}.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
    
    plt.show()
    
def plot_cluster_tsne(labels, X_reduced, figure_path, title='t-SNE Clustering', subtitle='', 
                      x_label='t-SNE Dimension 1', y_label='t-SNE Dimension 2',
                      width=90, height=60, ax=None, i=1, n_col=1, n_row=1, guardar=False):
    '''
    Parameters
    ----------
    labels : array
        Labels de los clusters obtenidos por k-means o dbscan.
    X_reduced : array
        Set de reduccion de dimensionalidad obtenido con t-sne y usado para entrenamiento de clustering.
    figure_path : str
        Directorio para guardar la figura.
    title : str, optional
        Titulo de la imagen para mostrar y guardar. The default is 't-SNE Clustering'.
    x_label : str, optional
        Leyenda del eje x. The default is 't-SNE Dimension 1'.
    y_label : str, optional
        Leyenda del eje y. The default is 't-SNE Dimension 2'.
    subtitle : str or list, optional
        Titulo para cada uno de los subplots. The default is ''.
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
        
        e.g. t-SNE - Clustering.pdf

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Grafica y (guarda) un diagrama de dispersion de los datos con reduccion de dimensionalidad con marcadores de cluster en formato pdf.
    '''    
    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f't-SNE_{i+1}' for i in range(2)])    
    tsne_df['Labels'] = labels
    
    # Mascaras para cambiar de marcadores 
    valid_points = tsne_df[tsne_df['Labels'] != -1]
    anomalies = tsne_df[tsne_df['Labels'] == -1]
    
    # Count the number of unique labels in the DataFrame
    num_labels = valid_points['Labels'].nunique()

    # Define custom markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Customize the markers as desired
    palette = palette=sns.color_palette()
    # Customize the markers list to match the number of unique labels
    markers = markers[:num_labels]
    palette = palette[:num_labels]
    
    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_title(title)
        ax.set_xlabel(x_label)
    else:
        ax.set_xlabel(x_label + '\n(' + string.ascii_lowercase[i-1] + ')')
        # Title for each test id
        ax.set_title(subtitle)
        
    # # Set x-axis label only for the bottom column 
    # elif i > n_col*(n_row - 1):
    #     ax.set_xlabel(x_label + '\n(' + string.ascii_lowercase[i-1] + ')')
    #     # Title for each test id
    #     ax.set_title(subtitle)
    # else:
    #     ax.set_xlabel('(' + string.ascii_lowercase[i-1] + ')')
    #     # Title for each test id
    #     ax.set_title(subtitle)

    # Use Seaborn scatter plot with hue and style parameters
    sns.scatterplot(x=valid_points['t-SNE_1'], y=valid_points['t-SNE_2'], hue='Labels', style='Labels',
                    data=valid_points, palette = palette, markers=markers, ax=ax)
    
    # Scatter points with label '-1' separately as red 'x'
    sns.scatterplot(x=anomalies['t-SNE_1'], y=anomalies['t-SNE_2'], hue='Labels', style='Labels', data=anomalies, 
                    palette=['r'], markers='x', linewidth=1.5, ax=ax)    

    
    # Set y-axis label only for the first plot of each row
    if (i-1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel('')
    
    ax.grid()
    ax.set_axisbelow(True)  # Set grid lines behind the data points    
    
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = f'{title}.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
    
    return ax
    
def plot_cluster_tsne_3d(labels, X_reduced, figure_path, title='t-SNE 3D Clustering', subtitle='', 
                            x_label='t-SNE Dimension 1', y_label='t-SNE Dimension 2', z_label='t-SNE Dimension 3',
                            width=90, height=60, ax=None, i=1, n_col=1, n_row=1, guardar=False):
    '''
    Parameters
    ----------
    labels : array
        Labels de los clusters obtenidos por k-means o dbscan.
    X_reduced : array
        Set de reduccion de dimensionalidad obtenido con t-sne y usado para entrenamiento de clustering.
    figure_path : str
        Directorio para guardar la figura.
    title : str, optional
        Titulo de la imagen para mostrar y guardar. The default is 't-SNE 3D Clustering'.
    x_label : str, optional
        Leyenda del eje x. The default is 't-SNE Dimension 1'.
    y_label : str, optional
        Leyenda del eje y. The default is 't-SNE Dimension 2'.
    z_label : str, optional
        Leyenda del eje z. The default is 't-SNE Dimension 3'.
    subtitle : str, optional
        Titulo para cada uno de los subplots. The default is ''.
    width : int or float, optional
        Ancho de la figura en milimetros. The default is 90  
    height : int or float, optional
        Alto de la figura en milimetros. The default is 60
    guardar : boolean, optional
        "True" para guardar la imagen. The default is False.
        
        e.g. t-SNE 3D - Clustering.pdf

    Returns
    -------
    Grafica y (guarda) un diagrama de dispersion de los datos con reduccion de dimensionalidad en 3D con marcadores de cluster en formato pdf.
    '''    
    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f't-SNE_{i+1}' for i in range(3)])    
    tsne_df['Labels'] = labels
    
    # Mascaras para cambiar de marcadores 
    valid_points = tsne_df[tsne_df['Labels'] != -1]
    anomalies = tsne_df[tsne_df['Labels'] == -1]
    
    # Count the number of unique labels in the DataFrame
    num_labels = valid_points['Labels'].nunique()

    # Define custom markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Customize the markers as desired
    palette = sns.color_palette()
    # Customize the markers list to match the number of unique labels
    markers = markers[:num_labels]
    palette = palette[:num_labels]
    
    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        fig, ax = plt.figure(figsize=figsize_inches, tight_layout=True)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.set_xlabel(x_label + '\n(' + string.ascii_lowercase[i-1] + ')')
        # Title for each test id
        ax.set_title(subtitle)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    # Use Seaborn scatter plot with hue and style parameters
    scatter = ax.scatter(valid_points['t-SNE_1'], valid_points['t-SNE_2'], valid_points['t-SNE_3'], 
                            c=valid_points['Labels'], cmap='viridis', marker='o')
    
    # Scatter points with label '-1' separately as red 'x'
    ax.scatter(anomalies['t-SNE_1'], anomalies['t-SNE_2'], anomalies['t-SNE_3'], 
                c='r', marker='x', label='Anomalies')
    
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = f'{title}.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
    
    plt.show()
    
    return ax
    
def plot_dbi(dbi_kmeans, dbi_dbscan, figure_path, k_min=2, k_max=6, width=90, height=60, ax=None, i=1, n_col=1, n_row=1, guardar=False):
    '''
    Parameters
    ----------
    dbi_kmeans : array, list
        dbi scores obtenidos por kmeans.
    dbi_dbscan : array, list
        dbi scores obtenidos por dbscan.
    figure_path : str
        Directorio para guardar la figura.
    k_min : int, optional
        numero minimo de clusters. The default is 2.
    k_max : int, optional
        numero maximo de clusters. The default is 6.
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
    guardar : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    ax : axes
        Subplot de la figura.
        Grafica y (guarda) un diagrama de los dbi scores para kmeans y dbscan en formato pdf.
    '''    
    # Create a DataFrame with the labels  
    x = pd.Series(range(k_min, k_max + 1), name='k')
    
    dbi_kmeans_df = pd.Series(dbi_kmeans, name='DBI')
    dbi_kmeans_df = pd.concat([x, dbi_kmeans_df], axis=1)
    dbi_kmeans_df['Algorithm'] = 'k-means'
    
    dbi_dbscan_df = pd.Series(dbi_dbscan, name='DBI')
    dbi_dbscan_df = pd.concat([x, dbi_dbscan_df], axis=1)
    dbi_dbscan_df['Algorithm'] = 'DBSCAN'
        
    # Concatenate the new DataFrames 
    dbi_scores = pd.concat([dbi_kmeans_df, dbi_dbscan_df], axis=0)
    
    # Define custom markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Customize the markers as desired

    
    # Plotting with Seaborn scatter plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_xlabel('Number of Clusters')
        ax.set_title('Davies-Bouldin Index for Different Cluster Numbers')
    
    elif i > n_col*(n_row - 1):
        ax.set_xlabel('Number of Clusters\n(' + string.ascii_lowercase[i-1] + ')')  
    
    else:
        ax.set_xlabel('(' + string.ascii_lowercase[i-1] + ')')
    
    # Create the plot
    sns.lineplot(x='k', y='DBI', hue='Algorithm', style='Algorithm', data=dbi_scores, markers=markers, palette=sns.color_palette(), ax=ax)    
    
    # Set y-axis label only for the first plot of each row
    if (i-1) % n_col == 0:
        ax.set_ylabel('Davies-Bouldin Index')
    else:
        ax.set_ylabel('')
        
    # Set x-axis ticks to display only integer values
    ax.set_xticks(range(k_min, k_max + 1))
    ax.grid()
    ax.set_axisbelow(True)  # Set grid lines behind the data points
    
    # Guardar la imagen en figure_path
    if guardar:
        figure_filename = 'Davies-Bouldin Index.pdf'
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches='tight')
        
    return ax
  