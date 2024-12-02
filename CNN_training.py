from mis_funciones import EAfunctions as eaf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential # type: ignore
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, UpSampling2D # type: ignore
import os
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
# Directories
# =============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
EA_path = os.path.join(base_dir, 'Datos_EA')
figure_path = os.path.join(base_dir, 'Figuras')
images_set = 'train_2'
images_dir = os.path.join(base_dir, 'datasets', 'images', images_set)
dbscan_labels_path = os.path.join(base_dir, 'Results', 'labels_dbscan.csv')

plot = False

# =============================================================================
# Load images
# =============================================================================
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    images_dir,
    labels=None,
    shuffle=False,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=1
)

# Rescale the images from [0, 255] to [0, 1]
train_dataset = train_dataset.map(lambda x: x / 255.0)
x_images = np.array(list(train_dataset.as_numpy_iterator()))
x_images = x_images.reshape(-1, 128, 128, 1)
# Read the DBSCAN labels
dbscan_labels = pd.read_csv(dbscan_labels_path)

# Define test IDs and repeat counts for each label set
label_sets = {
    'T0': (['P0_2', 'P0_3', 'P0_4', 'P0_5', 'P0_6'], 1),
    'T90': (['P90_1', 'P90_2', 'P90_3'], 3)
}

# Initialize an empty list to store all labels
labels = []

# Loop through each label set and process the labels
for label_name, (test_ids, n_repeat) in label_sets.items():
    dbscan_labels_subset = dbscan_labels[dbscan_labels['test_id'].isin(test_ids)]
    labels_subset = dbscan_labels_subset['DBSCAN label']
    repeated_labels = np.repeat(labels_subset.values, n_repeat)
    labels.append(repeated_labels)

# Concatenate all labels
y_images = np.concatenate(labels)
classes = np.unique(y_images)
num_classes = len(classes)

# =============================================================================
# Bucle de entrenamiento de varios modelos 
# =============================================================================
# Reinicio del nombre de los modelos prediccitivos
prueba = 2
# Definición de las variables de almacenamiento de los datos del modelo
cm_test_prec_list = []
accuracy_arr = []
model_arr = []
history_arr = []
cm_test_arr = []

# Definición de la variable iterativa para el entrenamiento de los modelos
random_state = 0
rs_lim = 5

# Una vez terminada la iteración, se comprueba qué modelo presenta una mayor precisión y se almacena para su posterior uso
while random_state <= rs_lim:
    print('\nInicio iteración: %i\n' % random_state)
    
    # Split para Train, Validation y Test
    x_train_val, x_test, y_train_val, y_test = train_test_split(x_images, y_images, test_size = 0.15, random_state = random_state, stratify = y_images)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size = 0.2, random_state = random_state, stratify = y_train_val) 
    
    # Ensure y_train is a 1-dimensional array
    y_train = np.array(y_train).flatten()
    y_val = np.array(y_val).flatten()

    # Borrado de los parámetros del modelo previo
    tf.keras.backend.clear_session()
    
    # Obtención de las dimensiones de entrada del modelo CNN
    input_shape = x_train[0].shape
    
    # Generación de la arquitectura del modelo CNN
    input_layer = Input(shape = (input_shape[0], input_shape[1], 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
    x = MaxPooling2D((2, 2))(x)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
    x = Flatten()(conv)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.15)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.15)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    
    # Generación del modelo CNN
    model = Model(inputs = input_layer, outputs = output_layer)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Muestreo de la configuración del modelo
    # model.summary()
    
    # Compilación de modelo
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',     # 'binary_crossentropy', 'mse
                  metrics = ['accuracy'])
    
    # Definición de parámetros de entrenamiento
    epocas = 200
    lote = 32
    
    # Assuming y_train contains the class labels
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(enumerate(class_weights))

    # Train the model with class weights
    history = model.fit(x_train, y_train, 
                        epochs=epocas, 
                        batch_size=lote,
                        verbose=1,
                        shuffle=True,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping],
                        class_weight=class_weights)

    # Predicción del modelo
    y_pred = model.predict(x_test)
    
    # Comprobación con matriz de confusión
    y_predn = np.round(y_pred).astype(int)
    
    # Obtención de la matriz de confusión para calcular la precisión del modelo
    cm_test = confusion_matrix(y_test, y_predn).astype(float)
    cm_test_prec = np.zeros((num_classes, num_classes))
    cm_test_sum = np.sum(cm_test, axis=1)
    for row in range(cm_test.shape[0]):
        cm_test_prec[row] = 100 * cm_test[row] / cm_test_sum[row]
    
    precision = 100 * np.sum(np.diag(cm_test) / np.sum(cm_test, axis=1)) / len(cm_test)
    print('Precisión en iteración = %.2f' % precision, '%')
    
    # Almacenado del modelo entrenado para la posterior selección del mejor
    model_arr.append([model])
    history_arr.append([history])
    accuracy_arr.append([precision])
    cm_test_arr.append([cm_test])
    cm_test_prec_list.append([cm_test_prec])
    print('\nFin iteración: %i\n' % random_state)
    
    # Aumento del contador de random_state para que entrene el modelo con un dataset diferente
    random_state += 1
    
# =============================================================================
# Extracción del modelo de mayor precisión para su representación gráfica
# =============================================================================
# Conversión de list a array
np_model = np.array(model_arr)
np_history = np.array(history_arr)
np_acc = np.array(accuracy_arr)
np_cm_test = np.array(cm_test_arr)
np_cm_test_prec = np.array(cm_test_prec_list)

# Obtención de los parámetros del modelo
max_acc = np_acc.max()
max_acc_pos = np.argmax(np_acc)
best_model = np_model[max_acc_pos][0]
history = np_history[max_acc_pos][0]
cm_test = np_cm_test[max_acc_pos][0]
cm_test_prec = np_cm_test_prec[max_acc_pos][0]

# Obtención de la precisión y pérdida del modelo
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# =============================================================================
# Graficado de la evolución temporal del modelo
# =============================================================================
model_name = 'relu_model_' + str(prueba) + '_' + str(round(precision, 2)).replace('.', '-')
if plot:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (12, 8), dpi = 130)
    fig.tight_layout()

    # Precisión
    ax1.plot(accuracy)
    ax1.plot(val_accuracy)
    ax1.set_title("Evolution of the model's accuracy")
    ax1.set_ylabel('Accuracy')
    ax1.legend(['Training', 'Validation'], loc='lower right')
    ax1.grid()
            
    # Pérdida
    ax2.plot(loss)
    ax2.plot(val_loss)
    ax2.set_title("Evolution of the model's loss")
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Training', 'Validation'], loc='upper right')
    ax2.grid()
    ax2.set_xticks(np.arange(0, int(epocas + 1), 20))
    # Guardado de la gráfica generada
    #plt.savefig(r'C:\Users\ahercas1\Desktop\EA\Fab aditiva\Article\Images\Evolución-LossVsEpochs-AccVsEpochs_' + model_name + '.pdf', bbox_inches = 'tight')
    plt.show()

# =============================================================================
# Representación gráfica de las matrices de confusión 'cm1 = cantidad de datos' y 'cm2 = porcentaje' ambas comprendidas entre 0 y valor máximo 
# =============================================================================
    # Configruación del plot
    vertical = 150 / 25.4
    horizontal = 60 / 25.4
    figsize = (vertical, horizontal)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 300

    # Inicio del graficado 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize, dpi = 300)
    fig.tight_layout()
    #plt.suptitle('Confusion Matrix')

    # Graficado de las predicciones por unidad predicha
    heatmap = sns.heatmap(cm_test, annot=True, fmt='.0f', cmap='viridis', vmin=0, vmax=cm_test_sum.max(), linewidth=.5, ax=ax1)

    # Arreglo de la colorbar
    colorbar = heatmap.collections[0].colorbar  # Obtener la colorbar del heatmap
    colorbar.set_ticks(np.arange(0, cm_test_sum.max(), 50))  # Definir los ticks deseados

    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')

    # Graficado de las predicciones en porcentaje
    sns.heatmap(cm_test_prec, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=100, linewidth=.5, ax=ax2)
    ax2.set_xlabel('Predicted label')

    # Guardado de la gráfica generada
    figure_path_name = os.path.join(figure_path, 'Confussion_matrix.pdf')
    plt.savefig(figure_path_name, bbox_inches = 'tight')
    plt.show()

# =============================================================================
# Obtención de la precisión del modelo para el set de test
# =============================================================================
precision = 100 * np.sum(np.diag(cm_test) / np.sum(cm_test, axis=1)) / len(cm_test)
print('Mejor modelo: %i' % max_acc_pos, '\nPrecisión final = %.2f' % precision, '%')

# =============================================================================
# Guardado del modelo de mayor precisión
# =============================================================================
model_path = os.path.join(base_dir, 'models')
os.makedirs(model_path, exist_ok=True)
model_name = f'CNN_model_{prueba}.keras'
model_path_name = os.path.join(model_path, model_name)
best_model.save(model_path_name)

print('Modelo "%s"' % model_name, 'guardado')
