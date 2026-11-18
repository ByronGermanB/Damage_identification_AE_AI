
## File: `CNN_accuracy.py`
```python
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Directory to save results
results_dir = os.path.join(base_dir, "Results")

predicted_csv = "labels_CNN_model_1.csv"
ground_truth_csv = "labels_dbscan.csv"

# Load CSV files
predicted = pd.read_csv(os.path.join(results_dir, predicted_csv))
ground_truth = pd.read_csv(os.path.join(results_dir, ground_truth_csv))

# Test set IDs to include in the evaluation
test_set = [
    "P0_90_1",
    "P0_90_2",
    "P0_90_3",
    "P0_90_4",
    "P0_90W_1",
    "P0_90W_3",
    "P45_2",
    "P45_3",
    "P45_4",
    "PQ_1",
    "PQ_2",
    "PQ_3",
    "PQ_4",
]

# Filter the data based on the test set
predicted = predicted[predicted["test_id"].isin(test_set)]
ground_truth = ground_truth[ground_truth["test_id"].isin(test_set)]

y_pred = predicted["CNN label"].values
y_true = ground_truth["DBSCAN label"].values

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")  # Use 'weighted' for multiclass
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

plt.rcParams.update({"font.size": 8, "font.family": "times new roman"})
figsize_mm = (90, 60)
fig_size = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)  # Convert mm to inches
plt.figure(figsize=(fig_size))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "CNN_model_1_confusion_matrix.pdf"), dpi=300)
plt.show()
plt.close()

# Save results to a text file
results_file = os.path.join(results_dir, "CNN_model_1_accuracy_results.txt")
with open(results_file, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

```

## File: `CNN_predictions.py`
```python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:52:46 2024

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Dataframes
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# Functions for dataset splitting
from utils.analysis_AE import train_test_set
from utils.force_mts import (
    limit_finder,
    # limit_finder_no_label,
    # plot_stress_hits,
    plot_stress_hits_cluster,
)

# Functions for DBSCAN model
# Functions for plotting
from utils.unsupervised_functions import (
    plot_cluster_feat,
    plot_cluster_tsne,
    # plot_cluster_tsne_3d,
    tsne,
    # tsne_3d,
)

# =============================================================================
# Initial data
# =============================================================================
# Current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Directory to save images
figure_dir = os.path.join(base_dir, "Figures")
os.makedirs(figure_dir, exist_ok=True)

# Directory to save results
results_dir = os.path.join(base_dir, "Results")
os.makedirs(results_dir, exist_ok=True)

models_dir = os.path.join(base_dir, "models")
datasets_dir = os.path.join(base_dir, "datasets")
images_dir = os.path.join(datasets_dir, "cnn_images", "test")

# Import datasets
data = pd.read_csv(os.path.join(datasets_dir, "Dataset_total.csv"))
force = pd.read_csv(os.path.join(datasets_dir, "Datos_MTS.csv"))

# Separate features and labels
# X_original, X, y, hits = train_test_set(data, normalization='std', split=False)
X_original, X, y, hits = train_test_set(
    data, normalization="log-std", columns_to_transform=["energy"], split=False
)
test_ids = hits["test_id"].unique()

# T-SNE
# X_reduced = tsne_3d(X, figure_dir)
X_reduced = tsne(X, figure_dir)

# =============================================================================
# Import images
# =============================================================================
# Set the parameters
target_size = (128, 128)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_data = test_datagen.flow_from_directory(
    images_dir,
    color_mode="grayscale",
    target_size=target_size,
    batch_size=batch_size,
    shuffle=False,
    class_mode=None,
)

# =============================================================================
# Predictions with model
# =============================================================================
# Load the saved model
model_name = "CNN_model_1"
model_path = os.path.join(models_dir, model_name + ".keras")
model = load_model(model_path)

# Get predictions
predictions = model.predict(test_data)
# Round predictions to 0 or 1
labels_cnn = np.round(predictions).astype(int)
# Flatten the predictions array
labels_cnn = labels_cnn.flatten()

# Save predictions to CSV
labels_cnn_df = data.copy()
labels_cnn_df["CNN label"] = labels_cnn

# Save the new data DataFrame with predictions
labels_cnn_df.to_csv(os.path.join(results_dir, f"labels_{model_name}.csv"), index=True)

# =============================================================================
# T-SNE plot
# =============================================================================
# List of labels and subtitle
labels_dbscan = pd.read_csv(os.path.join(results_dir, "labels_dbscan.csv"))
labels_dbscan = labels_dbscan["DBSCAN label"]

labels = [labels_cnn, labels_dbscan]
subtitle = ["CNN", "DBSCAN"]

# plot_cluster_tsne_3d(
#     labels_dbscan,
#     X_reduced,
#     figure_dir,
#     subtitle=subtitle,
#     title=f"t-SNE Clustering 3D- {model_name}",
#     save=False,
# )

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 1)

# Image size in mm
width, height = (90, 130)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
title = f"t-SNE Clustering Comparison - {model_name}"

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    plot_cluster_tsne(
        labels[i - 1],
        X_reduced,
        figure_dir,
        subtitle=subtitle[i - 1],
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        save=False,
        title=title,
    )
plt.show()
# =============================================================================
# Feature vs Feature plot
# =============================================================================
plot_cluster_feat(
    labels_cnn,
    data,
    "p_power_3",
    "w_peak_freq",
    figure_dir,
    width=90,
    height=60,
    title=f"Feature vs. feature scatter plot - {model_name}",
    x_label="Partial power 3 [%]",
    y_label="Weighted peak frequency [kHz]",
    save=True,
)

# =============================================================================
# Feature vs Feature plot separated by cluster (0 and 1)
# =============================================================================
# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
title = f"Feature vs. feature scatter plot - {model_name}"

for i, ax in enumerate(axes.flat, start=1):
    plot_cluster_feat(
        labels_cnn[labels_cnn == i - 1],
        data[labels_cnn == i - 1],
        "p_power_3",
        "w_peak_freq",
        figure_dir,
        width=90,
        height=60,
        title=title,
        x_label="Partial power 3 [%]",
        y_label="Weighted peak frequency [kHz]",
        save=False,
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
    )

plt.show()

# =============================================================================
# Stress vs hits plots - Clustering
# =============================================================================
test_id_prueba = ["P0_2", "P90_3", "P0_90_3", "P0_90W_1", "P45_4", "PQ_3"]
limits = limit_finder(labels_cnn, hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (180, 115)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
title = f"Stress and Cumulative hits vs Time - Clustering {model_name}"
plt.suptitle(title, fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits_cluster(
        labels_cnn,
        hits,
        force,
        test_id_prueba[i - 1],
        figure_dir,
        plot_type="line",
        limits=limits,
        y_label_right="\u03c3 [MPa]",
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        save=False,
        title=title,
    )
plt.show()
```

## File: `CNN_training.py`
```python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf  # type: ignore
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model  # type: ignore

# =============================================================================
# Directories
# =============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
EA_path = os.path.join(base_dir, "Datos_EA")
figure_path = os.path.join(base_dir, "Figures")
images_dir = os.path.join(base_dir, "datasets", "cnn_images", 'train')
dbscan_labels_path = os.path.join(base_dir, "Results", "labels_dbscan_gold_standard.csv")

plot = False

# =============================================================================
# Load images
# =============================================================================
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    images_dir,
    labels=None,
    shuffle=False,
    image_size=(128, 128),
    color_mode="grayscale",
    batch_size=1,
)

test_dir = os.path.join(base_dir, "datasets", "cnn_images", "test")
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    color_mode="grayscale",
    image_size=(128, 128),
    batch_size=1,
    shuffle=False,
    labels=None,
)

# Rescale the images from [0, 255] to [0, 1]
train_dataset = train_dataset.map(lambda x: x / 255.0)
x_images = np.array(list(train_dataset.as_numpy_iterator()))
x_images = x_images.reshape(-1, 128, 128, 1)

test_dataset = test_dataset.map(lambda x: x / 255.0)
x_test = np.array(list(test_dataset.as_numpy_iterator()))
x_test = x_test.reshape(-1, 128, 128, 1)

# Read the DBSCAN labels
dbscan_labels = pd.read_csv(dbscan_labels_path)

# Define test IDs and repeat counts for each label set
# If you used Data augmentation, set the use_data_aug to true to
# adjust the test IDs and repeat counts accordingly

use_data_aug = True
if use_data_aug:
    n_repeat = 3
else:
    n_repeat = 1

label_sets = {
    "T0": (["P0_2", "P0_3", "P0_4", "P0_5", "P0_6"], n_repeat),
    "T90": (["P90_1", "P90_2", "P90_3"], n_repeat),
    "T090": (["P0_90_1", "P0_90_2", "P0_90_3", "P0_90_4"], n_repeat),
}

# Initialize an empty list to store all labels
labels = []

# Loop through each label set and process the labels
for label_name, (test_ids, n_repeat) in label_sets.items():
    dbscan_labels_subset = dbscan_labels[dbscan_labels["test_id"].isin(test_ids)]
    labels_subset = dbscan_labels_subset["DBSCAN label"]
    repeated_labels = np.repeat(labels_subset.values, n_repeat)
    labels.append(repeated_labels)

# Concatenate all labels
y_images = np.concatenate(labels)
classes = np.unique(y_images)
num_classes = len(classes)

test_label_sets = {
    "T090W": ["P0_90W_1", "P0_90W_3"],
    "T45": ["P45_2", "P45_3", "P45_4"],
    "TQ": ["PQ_1", "PQ_2", "PQ_3", "PQ_4"],
}

test_labels = []
for label_name, (test_ids) in test_label_sets.items():
    dbscan_labels_subset = dbscan_labels[dbscan_labels["test_id"].isin(test_ids)]
    labels_subset = dbscan_labels_subset["DBSCAN label"]
    test_labels.append(labels_subset)

y_test = np.concatenate(test_labels)
x_test = x_test[-len(y_test):]  # Keep just the last len(y_test) images
# Because the test set is in order we keep just the last len(y_test) images

# =============================================================================
# Training loop for multiple models
# =============================================================================
# Reset the model number
model_number = 1
# Define variables to store model data
cm_test_prec_list = []
accuracy_arr = []
model_arr = []
history_arr = []
cm_test_arr = []

# Define the iterative variable for model training
random_state = 0
rs_lim = 2

# Once the iteration is complete, check which model has the highest accuracy and store it for later use
while random_state <= rs_lim:
    print("\nStarting iteration: %i\n" % random_state)

    # Split for Train, Validation, and Test
    # x_train_val, x_test, y_train_val, y_test = train_test_split(
    #     x_images, y_images, test_size=0.15, random_state=random_state, stratify=y_images
    # )
    x_train, x_val, y_train, y_val = train_test_split(
        x_images,
        y_images,
        test_size=0.2,
        random_state=random_state,
        stratify=y_images,
    )

    # Ensure y_train is a 1-dimensional array
    y_train = np.array(y_train).flatten()
    y_val = np.array(y_val).flatten()

    # Clear previous model parameters
    tf.keras.backend.clear_session()

    # Get the input dimensions for the CNN model
    input_shape = x_train[0].shape

    # Generate the CNN model architecture
    input_layer = Input(shape=(input_shape[0], input_shape[1], 1))
    x = Conv2D(
        16, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"
    )(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(
        32, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = MaxPooling2D((2, 2))(x)
    conv = Conv2D(
        64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"
    )(x)
    x = Flatten()(conv)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.15)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.15)(x)
    output_layer = Dense(1, activation="sigmoid")(x)

    # Generate the CNN model
    model = Model(inputs=input_layer, outputs=output_layer)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Display model configuration
    # model.summary()

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",  # 'binary_crossentropy', 'mse
        metrics=["accuracy"],
    )

    # Define training parameters
    epochs = 200
    batch_size = 32

    # Assuming y_train contains the class labels
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    # Train the model with class weights
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    # Model prediction
    y_pred = model.predict(x_test)

    # Check with confusion matrix
    y_predn = np.round(y_pred).astype(int)

    # Get the confusion matrix to calculate model accuracy
    cm_test = confusion_matrix(y_test, y_predn).astype(float)
    cm_test_prec = np.zeros((num_classes, num_classes))
    cm_test_sum = np.sum(cm_test, axis=1)
    for row in range(cm_test.shape[0]):
        cm_test_prec[row] = 100 * cm_test[row] / cm_test_sum[row]

    precision = 100 * np.sum(np.diag(cm_test) / np.sum(cm_test, axis=1)) / len(cm_test)
    print("Iteration accuracy = %.2f" % precision, "%")

    # Store the trained model for later selection of the best one
    model_arr.append([model])
    history_arr.append([history])
    accuracy_arr.append([precision])
    cm_test_arr.append([cm_test])
    cm_test_prec_list.append([cm_test_prec])
    print("\nEnd of iteration: %i\n" % random_state)

    # Increase the random_state counter to train the model with a different dataset
    random_state += 1

# =============================================================================
# Extract the highest accuracy model for graphical representation
# =============================================================================
# Convert list to array
np_model = np.array(model_arr)
np_history = np.array(history_arr)
np_acc = np.array(accuracy_arr)
np_cm_test = np.array(cm_test_arr)
np_cm_test_prec = np.array(cm_test_prec_list)

# Get model parameters
max_acc = np_acc.max()
max_acc_pos = np.argmax(np_acc)
best_model = np_model[max_acc_pos][0]
history = np_history[max_acc_pos][0]
cm_test = np_cm_test[max_acc_pos][0]
cm_test_prec = np_cm_test_prec[max_acc_pos][0]

# Get model accuracy and loss
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# =============================================================================
# Plot the temporal evolution of the model
# =============================================================================
model_name = (
    "relu_model_" + str(model_number) + "_" + str(round(precision, 2)).replace(".", "-")
)
if plot:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), dpi=130)
    fig.tight_layout()

    # Accuracy
    ax1.plot(accuracy)
    ax1.plot(val_accuracy)
    ax1.set_title("Evolution of the model's accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.legend(["Training", "Validation"], loc="lower right")
    ax1.grid()

    # Loss
    ax2.plot(loss)
    ax2.plot(val_loss)
    ax2.set_title("Evolution of the model's loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Training", "Validation"], loc="upper right")
    ax2.grid()
    ax2.set_xticks(np.arange(0, int(epochs + 1), 20))
    # Save the generated plot
    # plt.savefig(r'C:\Users\ahercas1\Desktop\EA\Fab aditiva\Article\Images\Evolución-LossVsEpochs-AccVsEpochs_' + model_name + '.pdf', bbox_inches = 'tight')
    plt.show()

    # =============================================================================
    # Graphical representation of confusion matrices 'cm1 = data count' and 'cm2 = percentage' both ranging from 0 to max value
    # =============================================================================
    # Plot configuration
    vertical = 150 / 25.4
    horizontal = 60 / 25.4
    figsize = (vertical, horizontal)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["figure.dpi"] = 300

    # Start plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=300)
    fig.tight_layout()
    # plt.suptitle('Confusion Matrix')

    # Plot predictions by predicted unit
    heatmap = sns.heatmap(
        cm_test,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        vmin=0,
        vmax=cm_test_sum.max(),
        linewidth=0.5,
        ax=ax1,
    )

    # Adjust the colorbar
    colorbar = heatmap.collections[0].colorbar  # Get the colorbar from the heatmap
    colorbar.set_ticks(np.arange(0, cm_test_sum.max(), 50))  # Define the desired ticks

    ax1.set_xlabel("Predicted label")
    ax1.set_ylabel("True label")

    # Plot predictions in percentage
    sns.heatmap(
        cm_test_prec,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=100,
        linewidth=0.5,
        ax=ax2,
    )
    ax2.set_xlabel("Predicted label")

    # Save the generated plot
    figure_path_name = os.path.join(figure_path, "Confussion_matrix.pdf")
    plt.savefig(figure_path_name, bbox_inches="tight")
    plt.show()

# =============================================================================
# Get model accuracy for the test set
# =============================================================================
precision = 100 * np.sum(np.diag(cm_test) / np.sum(cm_test, axis=1)) / len(cm_test)
print("Best model: %i" % max_acc_pos, "\nFinal accuracy = %.2f" % precision, "%")

# =============================================================================
# Save the highest accuracy model
# =============================================================================
model_path = os.path.join(base_dir, "models")
os.makedirs(model_path, exist_ok=True)
model_name = f"CNN_model_{model_number}.keras"
model_path_name = os.path.join(model_path, model_name)
best_model.save(model_path_name)

print('Model "%s"' % model_name, "saved")

```

## File: `data_visualization.py`
```python
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:10:11 2024

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Dataframes
import os

import pandas as pd

# Functions for dataset splitting
# Functions for result visualization
from utils.analysis_AE import plot_feat_vs_feat, train_test_set
from utils.force_mts import plot_force_hits

# =============================================================================
# Initial data
# =============================================================================
# Current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory to save images
figure_path = os.path.join(script_dir, "Figures")
os.makedirs(figure_path, exist_ok=True)

models_path = os.path.join(script_dir, "models")
datasets_path = os.path.join(script_dir, "datasets")

dbscan_model_path = os.path.join(models_path, "dbscan_model.joblib")
kmeans_model_path = os.path.join(models_path, "kmeans_model.joblib")

# Import datasets
data = pd.read_csv(os.path.join(datasets_path, "Dataset_total.csv"))
force = pd.read_csv(os.path.join(datasets_path, "Datos_MTS.csv"))

# =============================================================================
# Dataset splitting
# =============================================================================
X_original, X, y, hits = train_test_set(data, split=False)
test_ids = hits["test_id"].unique()

# =============================================================================
# Data visualization - Segment = 1
# =============================================================================
plot_feat_vs_feat(data, y, "w_peak_freq", "centroid_freq", figure_path)
plot_feat_vs_feat(data, y, "time_norm", "amplitude", figure_path)
plot_feat_vs_feat(data, y, "time_norm", "energy", figure_path, save=True)
plot_feat_vs_feat(data, y, "w_peak_freq", "energy", figure_path, save=True)

# Class 0
data_0 = data[data["Class"] == "[0_A]"]
y_0 = y[y == "[0_A]"]
plot_feat_vs_feat(data_0, y_0, "time_norm", "w_peak_freq", figure_path)

# Class 90
data_90 = data[data["Class"] == "[90_A]"]
y_90 = y[y == "[90_A]"]
plot_feat_vs_feat(data_90, y_90, "time_norm", "w_peak_freq", figure_path)

# Force and hits
for test_id in test_ids:
    plot_force_hits(y, hits, force, test_id, figure_path)

for test_id in test_ids:
    condition = data["test_id"] == test_id
    plot_feat_vs_feat(
        data[condition], y[condition], "time_norm", "w_peak_freq", figure_path
    )

# =============================================================================
# Data visualization - Segment = 3
# =============================================================================
plot_feat_vs_feat(data, y, "w_peak_freq_1", "centroid_freq_1", figure_path)
plot_feat_vs_feat(data, y, "time_norm", "w_peak_freq_1", figure_path)

# Class 0
data_0 = data[data["Class"] == "[0_A]"]
y_0 = y[y == "[0_A]"]
plot_feat_vs_feat(data_0, y_0, "time_norm", "w_peak_freq_1", figure_path)

# Class 90
data_90 = data[data["Class"] == "[90_A]"]
y_90 = y[y == "[90_A]"]
plot_feat_vs_feat(data_90, y_90, "time_norm", "w_peak_freq_1", figure_path)

```

## File: `dataset_generator.py`
```python
"""
Created on Mon Dec 11 12:48:58 2023

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Directories
import os

# Functions for feature extraction
from utils.fft_analysis import Features, unir_df

# Functions to load MTS data
from utils.force_mts import force_data

# =============================================================================
# Initial data
# =============================================================================
# Current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory with AE data
EA_path = os.path.join(script_dir, "Datos_EA")

# Test directories
path_P0_1 = os.path.join(
    EA_path, "P0_1-1"
)  # This is the folder P0_1-1, renamed for convenience
path_P0_2 = os.path.join(EA_path, "P0_2")
path_P0_3 = os.path.join(EA_path, "P0_3")
path_P0_4 = os.path.join(EA_path, "P0_4")
path_P0_5 = os.path.join(EA_path, "P0_5")
path_P0_6 = os.path.join(EA_path, "P0_6")
path_P90_1 = os.path.join(EA_path, "P90_1")
path_P90_2 = os.path.join(EA_path, "P90_2")
path_P90_3 = os.path.join(EA_path, "P90_3")
path_P90_4 = os.path.join(EA_path, "P90_4")

path_P0_90_1 = os.path.join(EA_path, "P0_90_1")
path_P0_90_2 = os.path.join(EA_path, "P0_90_2")
path_P0_90_3 = os.path.join(EA_path, "P0_90_3")
path_P0_90_4 = os.path.join(EA_path, "P0_90_4")
path_P0_90W_1 = os.path.join(EA_path, "P0_90W_1")
path_P0_90W_3 = os.path.join(EA_path, "P0_90W_3")
path_P45_2 = os.path.join(EA_path, "P45_2")
path_P45_3 = os.path.join(EA_path, "P45_3")
path_P45_4 = os.path.join(EA_path, "P45_4")
path_PQ_1 = os.path.join(EA_path, "PQ_1")
path_PQ_2 = os.path.join(EA_path, "PQ_2")
path_PQ_3 = os.path.join(EA_path, "PQ_3")
path_PQ_4 = os.path.join(EA_path, "PQ_4")

path_P0_A1 = os.path.join(EA_path, "P0_A1")
path_P0_A2 = os.path.join(EA_path, "P0_A2")
path_P0_A3 = os.path.join(EA_path, "P0_A3")
path_P0_A4 = os.path.join(EA_path, "P0_A4")
path_P90_A1 = os.path.join(EA_path, "P90_A1")
path_P90_A2 = os.path.join(EA_path, "P90_A2")
path_P90_A3 = os.path.join(EA_path, "P90_A3")
path_P90_A4 = os.path.join(EA_path, "P90_A4")

# Constants
lower_freq = 0  # Lower frequency limit in kHz
upper_freq = 900  # Upper frequency limit in kHz
sampling_rate = 5.0  # Data acquisition frequency per second in MHz
N_samp = 1024  # Number of samples (data) in each segment
N_seg = 1  # Number of segments (if changed, column names must be changed)
desfase = 200  # The offset ranges from [0 to pretrigger*sampling rate] [0-250]

# =============================================================================
# Object generation
# =============================================================================
# Class initialization
P0_1 = Features(
    path_P0_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0,
)
P0_2 = Features(
    path_P0_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=16.875718,
)
P0_3 = Features(
    path_P0_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=14.416501,
)
P0_4 = Features(
    path_P0_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=11.668197,
)
P0_5 = Features(
    path_P0_5,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.58955476,
)
P0_6 = Features(
    path_P0_6,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=1.548062,
)
P90_1 = Features(
    path_P90_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=4.4939521,
)
P90_2 = Features(
    path_P90_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.61250454,
)
P90_3 = Features(
    path_P90_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=2.3312921,
)
P90_4 = Features(
    path_P90_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.03841427,
)

P0_90_1 = Features(
    path_P0_90_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=6.1713131,
)
P0_90_2 = Features(
    path_P0_90_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.85981781,
)
P0_90_3 = Features(
    path_P0_90_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.065987506,
)
P0_90_4 = Features(
    path_P0_90_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-1.3343409,
)
P0_90W_1 = Features(
    path_P0_90W_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.53398163,
)
P0_90W_3 = Features(
    path_P0_90W_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.15772048,
)
P45_2 = Features(
    path_P45_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-13.833989,
)
P45_3 = Features(
    path_P45_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=7.7640255,
)
P45_4 = Features(
    path_P45_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.191836,
)
PQ_1 = Features(
    path_PQ_1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=6.1940507,
)
PQ_2 = Features(
    path_PQ_2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=5.8247346,
)
PQ_3 = Features(
    path_PQ_3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=7.9953848,
)
PQ_4 = Features(
    path_PQ_4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.0063383359,
)

P0_A1 = Features(
    path_P0_A1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.29895222,
)
P0_A2 = Features(
    path_P0_A2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0.21784435,
)
P0_A3 = Features(
    path_P0_A3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.12090737,
)
P0_A4 = Features(
    path_P0_A4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.17492944,
)
P90_A1 = Features(
    path_P90_A1,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=0,
)
P90_A2 = Features(
    path_P90_A2,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.4175458,
)
P90_A3 = Features(
    path_P90_A3,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.25912817,
)
P90_A4 = Features(
    path_P90_A4,
    lower_freq,
    upper_freq,
    sampling_rate,
    N_samp,
    N_seg,
    desfase,
    desfase_carga=-0.61885705,
)

# =============================================================================
# Feature extraction
# =============================================================================
# Filtering values
threshold = 36
counts = 5

# =============================================================================
# Feature extraction - Without breakage
# =============================================================================
P0_1_df = P0_1.feature_extr(threshold, counts, clase="[0]", test_id="P0_1", max_trai=66)
P0_2_df = P0_2.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_2", max_trai=111
)
P0_3_df_previo = P0_3.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_3", max_trai=86
)  # Repair file
P0_3_df = P0_3_df_previo[
    50:
]  # File 3 only takes the second test (100 depends on P0_3_previo)
P0_3_df["time"] = P0_3_df["time"] - 375  # Change time by taking only the second test
P0_4_df = P0_4.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_4", max_trai=401
)
P0_5_df = P0_5.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_5", max_trai=385
)
P0_6_df = P0_6.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_6", max_trai=460
)

P90_1_df = P90_1.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_1", max_trai=28
)
P90_2_df = P90_2.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_2", max_trai=56
)
P90_3_df = P90_3.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_3", max_trai=130
)
P90_4_df = P90_4.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_4", max_trai=616
)

P0_90_1_df = P0_90_1.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_1", max_trai=200
)
P0_90_2_df = P0_90_2.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_2", max_trai=440
)
P0_90_3_df = P0_90_3.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_3", max_trai=474
)
P0_90_4_df = P0_90_4.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_4", max_trai=428
)

P0_90W_1_df = P0_90W_1.feature_extr(
    threshold, counts, clase="[0-90W]", test_id="P0_90W_1", max_trai=220
)
P0_90W_3_df = P0_90W_3.feature_extr(
    threshold, counts, clase="[0-90W]", test_id="P0_90W_3", max_trai=270
)

P45_2_df = P45_2.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_2", max_trai=100
)
P45_3_df = P45_3.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_3", max_trai=180
)  # For tsne originally was at 520
P45_4_df = P45_4.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_4", max_trai=304
)

PQ_1_df = PQ_1.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_1", max_trai=211
)
PQ_2_df = PQ_2.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_2", max_trai=109
)
PQ_3_df = PQ_3.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_3", max_trai=166
)
PQ_4_df = PQ_4.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_4", max_trai=318
)

# =============================================================================
# Feature extraction - With breakage
# =============================================================================
P0_1_df = P0_1.feature_extr(threshold, counts, clase="[0]", test_id="P0_1")
P0_2_df = P0_2.feature_extr(threshold, counts, clase="[0]", test_id="P0_2")
P0_3_df_previo = P0_3.feature_extr(
    threshold, counts, clase="[0]", test_id="P0_3", max_trai=187
)  # Repair file
P0_3_df = P0_3_df_previo[
    111:
]  # File 3 only takes the second test (100 depends on P0_3_previo)
P0_3_df["time"] = P0_3_df["time"] - 375  # Change time by taking only the second test
P0_4_df = P0_4.feature_extr(threshold, counts, clase="[0]", test_id="P0_4")
P0_5_df = P0_5.feature_extr(threshold, counts, clase="[0]", test_id="P0_5")
P0_6_df = P0_6.feature_extr(threshold, counts, clase="[0]", test_id="P0_6")

P90_1_df = P90_1.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_1", max_trai=100
)
P90_2_df = P90_2.feature_extr(
    threshold, counts, clase="[90]", test_id="P90_2", max_trai=425
)
P90_3_df = P90_3.feature_extr(threshold, counts, clase="[90]", test_id="P90_3")
P90_4_df = P90_4.feature_extr(threshold, counts, clase="[90]", test_id="P90_4")

P0_90_1_df = P0_90_1.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_1", max_trai=417
)
P0_90_2_df = P0_90_2.feature_extr(threshold, counts, clase="[0-90]", test_id="P0_90_2")
P0_90_3_df = P0_90_3.feature_extr(threshold, counts, clase="[0-90]", test_id="P0_90_3")
P0_90_4_df = P0_90_4.feature_extr(
    threshold, counts, clase="[0-90]", test_id="P0_90_4", max_trai=557
)

P0_90W_1_df = P0_90W_1.feature_extr(
    threshold, counts, clase="[0-90W]", test_id="P0_90W_1"
)
P0_90W_3_df = P0_90W_3.feature_extr(
    threshold, counts, clase="[0-90W]", test_id="P0_90W_3"
)

P45_2_df = P45_2.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_2", max_trai=143
)
P45_3_df = P45_3.feature_extr(
    threshold, counts, clase="[45]", test_id="P45_3"
)  # For tsne originally was at 520
P45_4_df = P45_4.feature_extr(threshold, counts, clase="[45]", test_id="P45_4")

PQ_1_df = PQ_1.feature_extr(threshold, counts, clase="[Q]", test_id="PQ_1")
PQ_2_df = PQ_2.feature_extr(threshold, counts, clase="[Q]", test_id="PQ_2")
PQ_3_df = PQ_3.feature_extr(threshold, counts, clase="[Q]", test_id="PQ_3")
PQ_4_df = PQ_4.feature_extr(
    threshold, counts, clase="[Q]", test_id="PQ_4", max_trai=431
)

# =============================================================================
# Merging dataframes - Dataset of valid tests
# =============================================================================
# Directory to save the datasets
dataset_path = os.path.join(script_dir, "datasets")

file_name = "Dataset_0_90.csv"
df_0_90 = unir_df(
    P0_2_df, P0_3_df, P0_4_df, P0_5_df, P0_6_df, P90_1_df, P90_2_df, P90_3_df
)
df_0_90.to_csv(os.path.join(dataset_path, file_name), index=False)
print(f"File saved as: {file_name}")

file_name = "Dataset_total.csv"
df_total = unir_df(
    df_0_90,
    P0_90_1_df,
    P0_90_2_df,
    P0_90_3_df,
    P0_90_4_df,
    P0_90W_1_df,
    P0_90W_3_df,
    P45_2_df,
    P45_3_df,
    P45_4_df,
    PQ_1_df,
    PQ_2_df,
    PQ_3_df,
    PQ_4_df,
)
df_total.to_csv(os.path.join(dataset_path, file_name), index=False)
print(f"File saved as: {file_name}")

# =============================================================================
# Dataset of force data
# =============================================================================
force_path = os.path.join(script_dir, "Datos_MTS")
force_df = force_data(force_path)
force_df.to_csv(os.path.join(dataset_path, "Datos_MTS.csv"), index=False)

```

## File: `feature_distribution.py`
```python
# Dataframes
import os

import matplotlib.pyplot as plt
import pandas as pd

from utils.force_mts import plot_feature_time_histogram, plot_feature_histogram


# Directories
base_dir = os.path.dirname(os.path.abspath(__file__))

datasets_path = os.path.join(base_dir, "datasets")
figures_path = os.path.join(base_dir, "Figures")

# Import datasets
data = pd.read_csv(os.path.join(datasets_path, "dataset_total.csv"))

# List of features to plot
features = [
    "peak_freq",
    "w_peak_freq",
    "centroid_freq",
    "p_power_1",
    "p_power_2",
    "p_power_3",
]

y_labels = [
    "Peak Frequency [kHz]",
    "Weighted Peak Frequency [kHz]",
    "Centroid Frequency [kHz]",
    "Partial Power 1 [%]",
    "Partial Power 2 [%]",
    "Partial Power 3 [%]",
]

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (180, 115)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Feature distribution vs Normalized time", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):

    plot_feature_time_histogram(
        data,
        figures_path,
        title="Feature distribution vs Normalized time",
        x="time_norm",
        y=features[i - 1],
        y_label=y_labels[i - 1],
        ax=ax,
        n_col=n_col,
        i=i,
        save=True,
    )
plt.show()

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Feature distribution histogram", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):

    plot_feature_histogram(
        data,
        figures_path,
        title="Feature distribution histogram",
        x=features[i - 1],
        ax=ax,
        x_label=y_labels[i - 1],
        n_col=n_col,
        i=i,
        save=True,
    )
plt.show()

print("Feature vs Time plot generated successfully.")
```

## File: `Generate_cwt_images.py`
```python
import os

import numpy as np
from scipy import signal

from utils import cwt_analysis as eaf

# =============================================================================
# Directories
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
EA_path = os.path.join(base_dir, "Datos_EA")

figure_path = os.path.join(base_dir, "Figures")
os.makedirs(figure_path, exist_ok=True)

# Create directory to save images
images_dir = os.path.join(base_dir, "datasets", "cnn_images")
os.makedirs(images_dir, exist_ok=True)

# Description of wavelet variables
signal_function = signal.ricker
sampling_rate = 2e9  # [MHz]
n_bands = 24  # Number of frequency bands to use, in an article with sam_rate=1MHz

# Transient filtering parameters
t_trans = 204.8
amp_lim = 36
cnts_lim = 5

# Application of white noise
# Randomly applies one of the num_ruido_options white noise options for each standard deviation
# in STD_noise, resulting in 2 additional time signals per original signal
# STD_noise = [1.3e-3, 5e-3]
STD_noise = [0.03, 0.05]

num_ruido_options = 5
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
    num_noisySignals=num_ruido_options,
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
    num_noisySignals=num_ruido_options,
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
    num_noisySignals=num_ruido_options,
    figure_path=figure_path,
)

set_name = "train"
eaf.save_image_pred(cwt_image0, dims, "1_T0", images_dir, set_name)
eaf.save_image_pred(cwt_image90, dims, "2_T90", images_dir, set_name)
eaf.save_image_pred(cwt_image090, dims, "3_T090", images_dir, set_name)

# =============================================================================
# Load data at 0-90W orientation
# =============================================================================
# Test data num_ruido_options = 0
num_ruido_options = 0

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
    num_noisySignals=num_ruido_options,
    figure_path=figure_path,
)

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
    num_noisySignals=num_ruido_options,
    figure_path=figure_path,
)

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
    num_noisySignals=num_ruido_options,
    figure_path=figure_path,
)

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
    num_noisySignals=num_ruido_options,
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
    num_noisySignals=num_ruido_options,
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
    num_noisySignals=num_ruido_options,
    figure_path=figure_path,
)

# =============================================================================
# Save images
# =============================================================================
set_name = "test"
eaf.save_image_pred(cwt_image0, dims, "1_T0", images_dir, set_name)
eaf.save_image_pred(cwt_image90, dims, "2_T90", images_dir, set_name)
eaf.save_image_pred(cwt_image090, dims, "3_T090", images_dir, set_name)
eaf.save_image_pred(cwt_image090W, dims, "4_T090W", images_dir, set_name)
eaf.save_image_pred(cwt_image45, dims, "5_T45", images_dir, set_name)
eaf.save_image_pred(cwt_imageQ, dims, "6_TQ", images_dir, set_name)

```

## File: `layout.py`
```python
from mylib import ProjectLayout

layout = ProjectLayout(["src", "models", "results", "utils"]) # manual list

```

## File: `load_time_difference.py`
```python
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:15:58 2024

@author: bbarmac
"""

# Dataframes
import os

import pandas as pd

# Functions for data splitting
from utils.analysis_AE import train_test_set

# Current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "datasets")

# Import datasets
EA_dataset = "Dataset_total.csv"
data = pd.read_csv(os.path.join(dataset_dir, EA_dataset))
force = pd.read_csv(os.path.join(dataset_dir, "Datos_MTS.csv"))

X_original, X, y, hits = train_test_set(data, split=False)
test_ids = hits["test_id"].unique()

differences = []

for test_id in test_ids:
    # Condition to filter rows based on the string column
    condition = hits["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    # Filtered DataFrame
    filtered_hits = hits[condition]
    filtered_force = force[condition_force]

    # Time difference
    dif = filtered_hits["time"].iloc[-1] - filtered_force["Time [s]"].iloc[-1]
    differences.append(dif)

# Create a DataFrame
df = pd.DataFrame({"test_id": test_ids, "dif_time": differences})
file_name = "Diferencia_tiempo_carga_agujero.csv"
df.to_csv(os.path.join(dataset_dir, file_name), index=False)
print(f"File saved as: {file_name}")

```

## File: `Predictions.py`
```python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:52:46 2024

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Dataframes
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

# Functions for dataset splitting
from utils.analysis_AE import train_test_set
from utils.force_mts import (
    limit_finder,
    limit_finder_no_label,
    plot_stress_hits,
    plot_stress_hits_cluster,
)

# Functions for DBSCAN model
# Functions for plotting
from utils.unsupervised_functions import plot_cluster_feat, plot_cluster_tsne, tsne

# =============================================================================
# Initial data
# =============================================================================
# Current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Directory to save images
figure_path = os.path.join(base_dir, "Figures")
os.makedirs(figure_path, exist_ok=True)

# Directory to save results
results_path = os.path.join(base_dir, "Results")
os.makedirs(results_path, exist_ok=True)

models_path = os.path.join(base_dir, "models")
datasets_path = os.path.join(base_dir, "datasets")

dbscan_model_path = os.path.join(models_path, "dbscan_model.joblib")
kmeans_model_path = os.path.join(models_path, "kmeans_model.joblib")

# Import datasets
data = pd.read_csv(os.path.join(datasets_path, "Dataset_total.csv"))
force = pd.read_csv(os.path.join(datasets_path, "Datos_MTS.csv"))

# Separate features and labels
# X_original, X, y, hits = train_test_set(data, normalization='std', split=False)
X_original, X, y, hits = train_test_set(
    data, normalization="log-std", columns_to_transform=["energy"], split=False
)
test_ids = hits["test_id"].unique()

# T-SNE
X_reduced = tsne(X, figure_path)

# =============================================================================
# Predictions with model
# =============================================================================
dbscan_model = load(dbscan_model_path)
kmeans_model = load(kmeans_model_path)

labels_dbscan = dbscan_model.fit_predict(X_reduced)
labels_kmeans = kmeans_model.predict(X_reduced)

# Save the labels of DBSCAN and KMeans in CSV files
labels_dbscan_df = data.copy()
labels_dbscan_df["DBSCAN label"] = labels_dbscan

labels_kmeans_df = data.copy()
labels_kmeans_df["DBSCAN label"] = labels_dbscan

labels_dbscan_df.to_csv(os.path.join(results_path, "labels_dbscan.csv"), index=False)
labels_kmeans_df.to_csv(os.path.join(results_path, "labels_kmeans.csv"), index=False)

# Get unique values and their counts
unique_labels, counts = np.unique(labels_dbscan, return_counts=True)

# Display the results
for label, count in zip(unique_labels, counts):
    percentage = count / np.sum(counts)
    print(f"Occurrences of {label}: {count}")
    print(f"Percentage: {percentage:.1%}")
    print("-----")

# =============================================================================
# t-SNE Plot
# =============================================================================
# List of labels and subtitle
labels = [labels_kmeans, labels_dbscan]
subtitle = ["k-means", "DBSCAN"]

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 1)

# Image size in mm
width, height = (90, 130)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("t-SNE Clustering Comparison", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    plot_cluster_tsne(
        labels[i - 1],
        X_reduced,
        figure_path,
        subtitle=subtitle[i - 1],
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        save=False,
    )
plt.show()
# =============================================================================
# Feature vs Feature Plot
# =============================================================================
plot_cluster_feat(
    labels_dbscan,
    data,
    "p_power_3",
    "w_peak_freq",
    figure_path,
    width=90,
    height=60,
    title="Feature vs. feature scatter plot",
    x_label="Partial power 3 [%]",
    y_label="Weighted peak frequency [kHz]",
    save=False,
)
plt.show()
# =============================================================================
# Stress vs Hits Plot - Clustering
# =============================================================================
test_id_prueba = ["P0_2", "P90_3", "P0_90_3", "P0_90W_1", "P45_4", "PQ_3"]
limits = limit_finder(labels_dbscan, hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (180, 115)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Stress and Cumulative hits vs Time - Clustering", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits_cluster(
        labels_dbscan,
        hits,
        force,
        test_id_prueba[i - 1],
        figure_path,
        plot_type="line",
        limits=limits,
        y_label_right="\u03c3 [MPa]",
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        save=False,
    )
plt.show()
# =============================================================================
# Stress vs Hits Plot
# =============================================================================
test_id_prueba = ["P0_2", "P90_3", "P0_90_3", "P0_90W_1", "P45_4", "PQ_3"]
limits = limit_finder_no_label(hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (180, 115)
figsize_inches = (width / 25.4, height / 25.4)
# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Stress and Cumulative hits vs Time", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits(
        hits,
        force,
        test_id_prueba[i - 1],
        figure_path,
        plot_type="line",
        limits=limits,
        y_label_right="\u03c3 [MPa]",
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        save=False,
    )
plt.show()
```

## File: `save_md.py`
```python
from pathlib import Path

with open("project_code_AE.md", "w", encoding="utf-8") as salida:
    for root in sorted(Path(".").rglob("*.py")):
        if ".venv" in root.parts or "venv" in root.parts:
            continue
        salida.write(f"\n## File: `{root}`\n```python\n")
        salida.write(root.read_text(encoding="utf-8"))
        salida.write("\n```\n")
```

## File: `setup.py`
```python
from setuptools import find_packages, setup

setup(
    name="ae_damage_id",  
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "vallenae==0.11.0",
        "seaborn==0.13.2",
        "scikit-learn==1.6.1",
        "scipy==1.11.3",
        "scikit-image==0.25.2",
        "tensorflow==2.19.0"
    ],
)

```

## File: `src\fft\analyze_unsupervised.py`
```python
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import load

from utils.unsupervised_functions import (
    plot_cluster_feat,
    plot_cluster_tsne,
    plot_kmeans_per_k,
)

CONFIG_DIR = Path("parameters") / "arcan"
config_name = "analyze_unsupervised.json"


def main(config_name: str) -> None:
    """
    Main function for analyzing unsupervised models.
    """
    config_path = CONFIG_DIR / config_name

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    dataset_name = config["dataset_name"]
    trial_name = config["trial_name"]
    data_trial_name = config.get("data_trial_name", trial_name)
    show_plots = config["plots"]["show"]
    save_plots = config["plots"]["save"]
    format_plots = config["plots"]["format"]

    # Directories
    TRIALS_DIR = Path("results") / dataset_name / "models" / "trials" / trial_name
    FIGURES_DIR = Path("figures") / dataset_name / trial_name
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR = Path("datasets") / dataset_name
    DATA_PATH = (
        DATASETS_DIR / "trial_data" / data_trial_name / "preprocessed_data.joblib"
    )

    # Import datasets
    X_original, X, _, X_reduced, _ = load(DATA_PATH).values()
    print(f'Preprocessed data loaded from: "{DATA_PATH}"')

    # =============================================================================
    # K-means by k
    # =============================================================================

    kmeans_models_path = TRIALS_DIR / "kmeans_models.joblib"
    k_means_results_path = TRIALS_DIR / "kmeans_results.joblib"
    if kmeans_models_path.exists():
        kmeans_models = load(kmeans_models_path)
        k_means_results = load(k_means_results_path)
        k_means_fig_dir = FIGURES_DIR / "kmeans"
        k_means_fig_dir.mkdir(parents=True, exist_ok=True)

        # # plot k-means results
        print("[INFO] Plotting K-means results...")
        print("[INFO] Plotting silhouette scores...")
        plot_kmeans_per_k(
            X=X,
            kmeans_per_k=kmeans_models,
            silhouette_scores=k_means_results["silhouette_scores"],
            figure_path=k_means_fig_dir / f"silhouette_scores.{format_plots}",
            save=save_plots,
            show=show_plots,
        )

        for model in kmeans_models:
            print(
                f"[INFO] Plotting K-means cluster in t-SNE for k={model.n_clusters}..."
            )
            plot_cluster_tsne(
                labels=model.labels_,
                X_reduced=X_reduced,
                figure_path=k_means_fig_dir
                / f"tsne_kmeans_k_{model.n_clusters}.{format_plots}",
                save=save_plots,
                show=show_plots,
            )

            for feature_x, feature_y in combinations(X_original.columns, 2):
                figure_path = (
                    k_means_fig_dir
                    / f"k{model.n_clusters}_{feature_x}_vs_{feature_y}.{format_plots}"
                )
                print(f"[INFO] Plotting K-means cluster {feature_x} and {feature_y}...")
                plot_cluster_feat(
                    labels=model.labels_,
                    X=X_original,
                    feat_x=feature_x,
                    feat_y=feature_y,
                    figure_path=figure_path,
                    save=save_plots,
                    show=show_plots,
                )

    # =============================================================================
    # DBSCAN
    # =============================================================================
    dbscan_models_path = TRIALS_DIR / "dbscan_models.joblib"
    dbscan_models = load(dbscan_models_path)
    dbscan_fig_dir = FIGURES_DIR / "dbscan"
    dbscan_fig_dir.mkdir(parents=True, exist_ok=True)

    for model in dbscan_models:
        eps = model.eps
        min_samples = model.min_samples

        k = len(np.unique(model.labels_)) - (1 if -1 in model.labels_ else 0)

        print(
            f"[INFO] Plotting DBSCAN cluster in t-SNE for eps={eps}, min_samples={min_samples}, k={k}..."
        )
        plot_cluster_tsne(
            labels=model.labels_,
            X_reduced=X_reduced,
            figure_path=dbscan_fig_dir
            / f"tsne_k_{k}_eps_{eps}_min_samples_{min_samples}.{format_plots}",
            save=save_plots,
            show=show_plots,
        )

        for feature_x, feature_y in combinations(X_original.columns, 2):
            print(f"[INFO] Plotting DBSCAN cluster {feature_x} and {feature_y}...")
            figure_path = (
                dbscan_fig_dir
                / f"k{k}_eps_{eps}_min_samples_{min_samples}_{feature_x}_vs_{feature_y}.{format_plots}"
            )
            plot_cluster_feat(
                labels=model.labels_,
                X=X_original,
                feat_x=feature_x,
                feat_y=feature_y,
                figure_path=figure_path,
                save=save_plots,
                show=show_plots,
            )


if __name__ == "__main__":
    main(config_name)

```

## File: `src\fft\feature_extraction.py`
```python
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from utils.fft_analysis import Features, plot_feat_vs_feat, select_significant_indices
from utils.force_mts import force_data

CONFIG_DIR = Path("parameters") / "arcan"
config_name = "feature_extraction_1.json"


def main(config_name: str) -> None:
    """
    Main function for feature extraction.
    """
    config_path = CONFIG_DIR / config_name

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    dataset_name = config["dataset_name"]

    # Directories
    EA_DATA_DIR = Path(config["EA_DATA_DIR"])
    DATASET_DIR = Path("datasets") / dataset_name
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    FIGURES_DIR = Path("figures") / dataset_name
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if config["files"] == "all":
        pri_dirs = [
            path.parent for path in EA_DATA_DIR.glob("**/*.pridb") if path.is_file()
        ]
    else:
        pri_dirs = [EA_DATA_DIR / file for file in config["files"]]

    pri_names = [path.stem for path in pri_dirs]
    print(f"[INFO] Found {len(pri_dirs)} .pridb files in {EA_DATA_DIR}")

    # =============================================================================
    # Object initialization
    # =============================================================================
    parameters = config["parameters"]
    min_trais = config["min_trais"]
    if min_trais is None:
        min_trais = [1 for _ in range(len(pri_dirs))]

    max_trais = config["max_trais"]
    if max_trais is None:
        max_trais = [None for _ in range(len(pri_dirs))]

    dfs = []
    force_dfs = []

    for pri_dir, pri_name in zip(pri_dirs, pri_names):
        print(f"[INFO] Processing {pri_name}...")

        # Force data saving if specified in the configuration
        if config["force"].get("force_data", False):
            data_force_i = force_data(
                path=pri_dir,
                section=config["force"].get("section", None),
                test_id=pri_name,
            )
        force_dfs.append(data_force_i)

        features = Features(
            path=pri_dir,
            lower_freq=parameters["lower_freq"],
            upper_freq=parameters["upper_freq"],
            sampling_rate=parameters["sampling_rate"],
            N_samp=parameters["N_samp"],
            N_seg=parameters["N_seg"],
            desfase=parameters["desfase"],
            desfase_carga=parameters.get("desfase_carga", 0),
            force=data_force_i if config["force"].get("force_data", False) else None,
        )

        df = features.feature_extr(
            threshold=parameters["threshold"],
            counts=parameters["counts"],
            label=pri_name,
            test_id=pri_name,
            max_trai=max_trais[pri_names.index(pri_name)],
            min_trai=min_trais[pri_names.index(pri_name)],
        )
        if df.empty:
            continue  # Skip to the next pri_dir if no transients were found
        dfs.append(df)

    # Merge all the individual DataFrames into a single DataFrame
    data = pd.concat(dfs, ignore_index=True)
    data["label"] = 0
    data.to_csv(DATASET_DIR / config.get("output_name", "features.csv"), index=False)
    print("[INFO] Feature extraction completed. Data saved to CSV.")

    # Force data
    if config["force"].get("force_data", False):
        data_force = pd.concat(force_dfs, ignore_index=True)
        data_force.to_csv(
            DATASET_DIR / config.get("force_output_name", "force_data.csv"), index=False
        )

    # Saving feat vs feat images
    features_to_plot = config["plots"].get("features_to_plot", [])

    if config["plots"].get("save_feat_vs_feat_images", False):
        print("[INFO] Saving Feature vs Feature images...")

        width, height = config["plots"].get("fig_size", [90, 60])
        format_plot = config["plots"].get("format", "png")

        feat_vs_feat_dir = FIGURES_DIR / "feat_vs_feat"
        feat_vs_feat_dir.mkdir(parents=True, exist_ok=True)

        label_column = "test_id" if config["plots"].get("labeled", False) else "label"

        if config["plots"].get("reduced_points", False):
            selected_indices = select_significant_indices(
                data=data,
                label_column=label_column,
                feature_columns=features_to_plot,
                max_points=config["plots"].get("max_points", 20),
            )
            data = data.loc[selected_indices].copy()

        y = data[label_column]

        for feature_x, feature_y in combinations(features_to_plot, 2):
            figure_path = feat_vs_feat_dir / f"{feature_x}_vs_{feature_y}.{format_plot}"

            plot_feat_vs_feat(
                X=data,
                y=y,
                feat_x=feature_x,
                feat_y=feature_y,
                figure_path=figure_path,
                width=width,
                height=height,
                save=True,
            )

        print("[INFO] Feature vs Feature images saved.")


if __name__ == "__main__":
    main(config_name)

```

## File: `src\fft\feature_visualiztion.py`
```python
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from utils.fft_analysis import plot_feat_vs_feat, select_significant_indices

CONFIG_DIR = Path("parameters") / "arcan"
config_name = "feature_extraction_1.json"


def main(config_name: str) -> None:
    """
    Main function for feature visualization.
    """
    config_path = CONFIG_DIR / config_name

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    dataset_name = config["dataset_name"]
    output_name = config.get("output_name", "features.csv")

    # Directories
    DATASET_DIR = Path("datasets") / dataset_name

    FIGURES_DIR = Path("figures") / dataset_name
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    DATA_PATH = DATASET_DIR / output_name

    # Read the datasets
    data = pd.read_csv(DATA_PATH)

    # Saving feat vs feat images
    features_to_plot = [
        "peak_freq",
        "w_peak_freq",
        "centroid_freq",
        "p_power_1",
        "p_power_2",
        "p_power_3",
        "p_power_4",
        "p_power_5",
        "p_power_6",
        "energy",
    ]

    if config["plots"].get("save_feat_vs_feat_images", False):
        print("[INFO] Saving Feature vs Feature images...")
        format_plot = config["plots"].get("format", "png")
        width, height = config["plots"].get("fig_size", [90, 60])

        feat_vs_feat_dir = FIGURES_DIR / "feat_vs_feat"
        feat_vs_feat_dir.mkdir(parents=True, exist_ok=True)

        label_column = "test_id" if config["plots"].get("labeled", False) else "label"

        if config["plots"].get("reduced_points", False):
            selected_indices = select_significant_indices(
                data=data,
                label_column=label_column,
                feature_columns=features_to_plot,
                max_points=config["plots"].get("max_points", 20),
            )
            data = data.loc[selected_indices].copy()

        y = data[label_column]

        for feature_x, feature_y in combinations(features_to_plot, 2):
            figure_path = feat_vs_feat_dir / f"{feature_x}_vs_{feature_y}.{format_plot}"

            plot_feat_vs_feat(
                X=data,
                y=y,
                feat_x=feature_x,
                feat_y=feature_y,
                figure_path=figure_path,
                width=width,
                height=height,
                save=True,
            )

        print("[INFO] Feature vs Feature images saved.")


if __name__ == "__main__":
    main(config_name)

```

## File: `src\fft\results_visualization.py`
```python
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from joblib import load

from utils.force_mts import limit_finder, plot_stress_hits_cluster
from utils.unsupervised_functions import plot_cluster_test_id, plot_cluster_tsne

CONFIG_DIR = Path("parameters") / "arcan"
config_name = "results_visualization.json"


def main(config_name: str) -> None:
    config_path = CONFIG_DIR / config_name

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

        dataset_name = config["dataset_name"]
        model_name = config["model_name"]
        data_tsne_path = config["data_tsne_path"]
        plots_config = config["plots"]
        show_plots = plots_config.get("show", False)
        save_plots = plots_config.get("save", False)
        format_plots = plots_config.get("format", "png")
        fig_size = plots_config.get("fig_size", [90, 60])
        use_log_scale = plots_config.get("use_log_scale", False)
        width, height = fig_size

        # Directories
        MODELS_DIR = Path("results") / dataset_name / "models" / model_name
        FIGURES_DIR = Path("figures") / dataset_name / model_name
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Import data
    data_path = MODELS_DIR / "data_labeled.csv"
    data = pd.read_csv(data_path)
    data["Count"] = 1

    # force data
    force_path = Path("datasets") / dataset_name / "force_data.csv"
    force = pd.read_csv(force_path)

    # plots
    features_to_plot = plots_config.get("features_to_plot", [])

    # feat vs feat test id
    if plots_config.get("feat_vs_feat_test_id", False):
        print("[INFO] Plotting Feature vs Feature Test ID images...")

        feat_vs_feat_test_id_dir = FIGURES_DIR / "feat_vs_feat_test_id"
        feat_vs_feat_test_id_dir.mkdir(parents=True, exist_ok=True)

        for feature_x, feature_y in combinations(features_to_plot, 2):
            for test_id in data["test_id"].unique():
                fig_dir = feat_vs_feat_test_id_dir / test_id
                fig_dir.mkdir(parents=True, exist_ok=True)

                figure_path = (
                    fig_dir
                    / f"{feature_x}_vs_{feature_y}_test_{test_id}.{format_plots}"
                )

                plot_cluster_test_id(
                    data=data,
                    test_id=test_id,
                    feat_x=feature_x,
                    feat_y=feature_y,
                    title=f"Test ID {test_id}: {feature_x} vs {feature_y}",
                    figure_path=figure_path,
                    width=width,
                    height=height,
                    show=show_plots,
                    save=save_plots,
                )

    # Hits vs Stress for each test_id
    if plots_config.get("hits_vs_stress", False):
        print("[INFO] Plotting Hits vs Stress images...")

        hits_vs_stress_dir = FIGURES_DIR / "hits_vs_stress"
        hits_vs_stress_dir.mkdir(parents=True, exist_ok=True)

        if plots_config.get("use_limit_finder", False):
            limits = limit_finder(data, force)
        else:
            limits = None

        for test_id in data["test_id"].unique():
            fig_dir = hits_vs_stress_dir / test_id
            fig_dir.mkdir(parents=True, exist_ok=True)

            figure_path = fig_dir / f"hits_vs_stress_test_{test_id}.{format_plots}"

            plot_stress_hits_cluster(
                data=data,
                force=force,
                test_id=test_id,
                limits=limits,
                use_log_scale=use_log_scale,
                figure_path=figure_path,
                width=width,
                height=height,
                show=show_plots,
                save=save_plots,
            )

    # hits vs stress per label
    if plots_config.get("hits_per_label", False):
        print("[INFO] Plotting Hits vs Stress per label images...")

        for test_id in data["test_id"].unique():
            test_data = data[data["test_id"] == test_id]
            fig_dir = hits_vs_stress_dir / test_id
            fig_dir.mkdir(parents=True, exist_ok=True)

            for label in test_data["label"].unique():
                figure_path = (
                    fig_dir
                    / f"hits_vs_stress_test_{test_id}_label_{label}.{format_plots}"
                )

                plot_stress_hits_cluster(
                    data=test_data[test_data["label"] == label],
                    force=force,
                    test_id=test_id,
                    limits=limits,
                    use_log_scale=use_log_scale,
                    figure_path=figure_path,
                    width=width,
                    height=height,
                    show=show_plots,
                    save=save_plots,
                )

    # tsne clustering
    if plots_config.get("tsne_clustering", False):
        print("[INFO] Plotting t-SNE clustering...")

        tsne_dir = FIGURES_DIR / "tsne_clustering"
        tsne_dir.mkdir(parents=True, exist_ok=True)

        figure_path = tsne_dir / f"t-SNE_Clustering.{format_plots}"

        _, _, _, X_reduced, _ = load(data_tsne_path).values()
        print(f'Preprocessed data loaded from: "{data_tsne_path}"')

        plot_cluster_tsne(
            labels=data["label"],
            X_reduced=X_reduced,
            figure_path=figure_path,
            width=width,
            height=height,
            show=show_plots,
            save=save_plots,
        )


if __name__ == "__main__":
    main(config_name)

```

## File: `src\fft\select_model.py`
```python
import json
from pathlib import Path

import pandas as pd
from joblib import dump, load

CONFIG_DIR = Path("parameters") / "arcan"
config_name = "select_model.json"


def main(config_name: str) -> None:
    config_path = CONFIG_DIR / config_name

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

        dataset_name = config["dataset_name"]
        trial_name = config["trial_name"]
        trial_config_name = config["trial_config_name"]
        model_type = config["model"]["model_type"]
        output_model_name = config["model"]["output_model_name"]
        index = config["model"]["index"]

    # Directories
    TRIALS_DIR = Path("results") / dataset_name / "models" / "trials" / trial_name
    MODELS_DIR = Path("results") / dataset_name / "models" / output_model_name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Import trial
    trial_path = TRIALS_DIR / f"{model_type}_models.joblib"
    trial_results_path = TRIALS_DIR / f"{model_type}_results.joblib"

    models = load(trial_path)
    selected_model = models[index]
    results = load(trial_results_path)
    selected_results = results[index]

    trial_config_path = CONFIG_DIR / trial_config_name
    with trial_config_path.open("r", encoding="utf-8") as file:
        trial_config = json.load(file)

        feat_data_name = trial_config["feat_data_name"]

    DATA_PATH = Path("datasets") / dataset_name / feat_data_name
    data = pd.read_csv(DATA_PATH)
    data["label"] = selected_model.labels_

    data.to_csv(MODELS_DIR / "data_labeled.csv", index=False)
    print("[INFO] Data labeled and saved.")

    # Save the selected model and results
    selected_model_path = MODELS_DIR / f"{output_model_name}.joblib"
    selected_results_path = MODELS_DIR / f"{output_model_name}_results.joblib"
    selected_results_txt_path = MODELS_DIR / f"{output_model_name}_results.txt"

    dump(selected_model, selected_model_path)
    dump(selected_results, selected_results_path)
    selected_results_txt_path.write_text(
        "\n".join(f"{key}: {value}" for key, value in selected_results.items()),
        encoding="utf-8",
    )

    print(f"[INFO] Selected model saved to: {selected_model_path}")
    print(f"[INFO] Selected results saved to: {selected_results_path}")
    print(f"[INFO] Selected results text saved to: {selected_results_txt_path}")


if __name__ == "__main__":
    main(config_name)

```

## File: `src\fft\train_unsupervised.py`
```python
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from mylib.utils.time_report import Profiler  # type: ignore

# Functions for dataset splitting
from utils.fft_analysis import train_test_set

# Functions for DBSCAN model
from utils.unsupervised_functions import (
    grid_search_dbscan,
    kmeans_per_k,
    tsne,
)

CONFIG_DIR = Path("parameters") / "arcan"
config_name = "dbscan_trial_2.json"


class _OutputTee:
    """Send printed output to both the console and a text buffer."""

    def __init__(self, console, buffer):
        self.console = console
        self.buffer = buffer

    def write(self, text):
        self.console.write(text)
        self.buffer.write(text)

    def flush(self):
        self.console.flush()
        self.buffer.flush()


def main(config_name: str) -> None:
    """
    Ex
    """
    timer = Profiler()
    original_stdout = sys.stdout
    output_buffer = io.StringIO()
    sys.stdout = _OutputTee(original_stdout, output_buffer)
    config_path = CONFIG_DIR / config_name

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    dataset_name = config["dataset_name"]
    show_plots = config["plots"]["show"]
    save_plots = config["plots"]["save"]
    format_plot = config["plots"].get("format", "png")
    save_data = config.get("save_data", False)

    models_dir = (
        Path("results") / dataset_name / "models" / "trials" / config["trial_name"]
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    # Directories
    FIGURES_DIR = Path("figures") / dataset_name / config["trial_name"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR = Path("datasets") / dataset_name
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Import datasets
    if config.get("saved_data_path", None) is not None:
        DATA_PATH = Path(config["saved_data_path"])
        X_original, X, y, X_reduced, hits = load(DATA_PATH).values()

    else:
        DATA_PATH = DATASETS_DIR / config["feat_data_name"]
        data = pd.read_csv(DATA_PATH)

        # Separate features and labels
        X_original, X, y, hits = train_test_set(
            data,
            normalization=config.get("normalization", "log-std"),
            columns_to_transform=config.get("columns_to_transform", ["energy"]),
            features=config.get("features", None),
            split=False,
        )  # Normalization: log-std or std

        if config.get("use_tsne", False):
            with timer.section("TSNE"):
                # T-SNE
                X_reduced = tsne(
                    X,
                    FIGURES_DIR / f"t-SNE Visualization.{format_plot}",
                    show=show_plots,
                    save=save_plots,
                )
        else:
            X_reduced = X

    if save_data:
        data_saving_path = (
            DATASETS_DIR
            / "trial_data"
            / config["trial_name"]
            / "preprocessed_data.joblib"
        )
        data_saving_path.parent.mkdir(parents=True, exist_ok=True)

        dump(
            {
                "X_original": X_original,
                "X": X,
                "y": y,
                "X_reduced": X_reduced,
                "hits": hits,
            },
            data_saving_path,
        )
        print(f'Preprocessed data saved at: "{data_saving_path}"')

    # =============================================================================
    # K-means by k
    # =============================================================================
    if config.get("use_kmeans", False):
        with timer.section("K-means"):
            print("[INFO] Performing K-means clustering...")
            k_max = config["dbscan"]["k_max"]
            kmeans_models, silhouette_scores, dbi_kmeans = kmeans_per_k(
                X_reduced, k_max
            )
            kmeans_results = {
                "silhouette_scores": silhouette_scores,
                "dbi_kmeans": dbi_kmeans,
            }

    # =============================================================================
    # DBSCAN
    # =============================================================================
    with timer.section("DBSCAN"):

        # Parameters for DBSCAN
        epsilon_values = config["dbscan"]["epsilon_values"]
        min_samples_limits = config["dbscan"]["min_samples_limits"]
        min_samples_num = config["dbscan"]["min_samples_num"]

        min_samples_values = np.linspace(
            min_samples_limits[0], min_samples_limits[1], min_samples_num, dtype=int
        )

        target_clusters = [k for k in range(2, k_max + 1) ]
        dbscan_models, results = grid_search_dbscan(
            X_reduced, target_clusters, epsilon_values, min_samples_values
        )

    with timer.section("Saving results"):
        # Save the DBSCAN models and results
        dbscan_models_path = models_dir / config["saving_names"]["dbscan_models"]
        dbscan_results_path = models_dir / config["saving_names"]["dbscan_results"]

        dump(dbscan_models, dbscan_models_path)
        dump(results, dbscan_results_path)

        print(f'DBSCAN models saved at: "{dbscan_models_path}"')
        print(f'DBSCAN results saved at: "{dbscan_results_path}"')

        # Save the K-means models and results if used
        if config.get("use_kmeans", False):
            kmeans_models_path = models_dir / config["saving_names"]["kmeans_models"]
            kmeans_results_path = models_dir / config["saving_names"]["kmeans_results"]

            dump(kmeans_models, kmeans_models_path)
            dump(kmeans_results, kmeans_results_path)

            print(f'K-means models saved at: "{kmeans_models_path}"')
            print(f'K-means results saved at: "{kmeans_results_path}"')

        # Save everything printed during the run while keeping console output.
        sys.stdout = original_stdout
        report_path = models_dir / "training_output.txt"
        report_path.write_text(output_buffer.getvalue(), encoding="utf-8")
        print(f'Results printed during training saved at: "{report_path}"')

    timer.report()


if __name__ == "__main__":
    main(config_name)

```

## File: `src\unsupervised_model.py`
```python
"""
Created on Mon Dec 11 12:18:58 2023

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Dataframes
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump

# Functions for dataset splitting
from utils.fft_analysis import train_test_set
from utils.force_mts import (
    limit_finder,
    limit_finder_no_label,
    plot_stress_hits,
    plot_stress_hits_cluster,
)

# Functions for DBSCAN model
# Functions for plotting
from utils.unsupervised_functions import (
    grid_search_dbscan,
    kmeans_per_k,
    plot_cluster_feat,
    plot_cluster_hits,
    plot_cluster_time,
    plot_cluster_tsne,
    plot_dbi,
    plot_kmeans_per_k,
    tsne,
)

# =============================================================================
# Initial data
# =============================================================================
# Current directory
dataset_name = "3d_specimen"

# Directory to save images
FIGURES_DIR = Path("figures") / dataset_name
DATASETS_DIR = Path("datasets") / dataset_name

# Import datasets
DATA_PATH = DATASETS_DIR / "Dataset_total.csv"
FORCE_PATH = DATASETS_DIR / "Datos_MTS.csv"

# Read the datasets
data = pd.read_csv(DATA_PATH)
force = pd.read_csv(FORCE_PATH)

# Separate features and labels
# X_original, X, y, hits = train_test_set(data, normalization='std', split=False)
X_original, X, y, hits = train_test_set(
    data, normalization="log-std", columns_to_transform=["energy"], split=False
)
test_ids = hits["test_id"].unique()

# T-SNE
X_reduced = tsne(X, FIGURES_DIR)

# List with dbscan models for k = 2 3 4 5
dbscan_per_k = []
dbscan_results_per_k = []

# =============================================================================
# DBSCAN
# =============================================================================
# Parameters for DBSCAN
target_clusters = 2  # Specify the desired number of clusters
# epsilon_values = np.linspace(1, 10, 10)
epsilon_values = [1, 5, 10, 15, 20]
min_samples_values = np.linspace(5, 20, 4, dtype=int)

# Model and adjusted parameters
dbscan_models, results = grid_search_dbscan(
    X_reduced, target_clusters, epsilon_values, min_samples_values
)
# Convert results list to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_csv_path = DATASETS_DIR / "dbscan_grid_search_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f'DBSCAN grid search results saved at: "{results_csv_path}"')
# =============================================================================
# DBSCAN by k
# =============================================================================
# Index of the chosen model
index = -2

# Add to the lists
dbscan_per_k.append(dbscan_models[index])
dbscan_results_per_k.append(results[index])

# =============================================================================
# DBSCAN - DBI Scores
# =============================================================================
dbi_dbscan = []

for i in range(len(dbscan_results_per_k)):
    dbi_dbscan.append(dbscan_results_per_k[i]["DBI Score"])

# =============================================================================
# DBSCAN - Chosen model
# =============================================================================
dbscan_model = dbscan_per_k[0]
labels_dbscan = dbscan_model.labels_

# Get unique values and their counts
unique_labels, counts = np.unique(labels_dbscan, return_counts=True)

# Display the results
for label, count in zip(unique_labels, counts):
    percentage = count / np.sum(counts)
    print(f"Occurrences of {label}: {count}")
    print(f"Percentage: {percentage:.1%}")
    print("-----")

# =============================================================================
# K-means by k
# =============================================================================
k = 6  # Maximum number of clusters
kmeans_models, silhouette_scores, dbi_kmeans = kmeans_per_k(X_reduced, k)
plot_kmeans_per_k(
    X=X_reduced,
    kmeans_per_k=kmeans_models,
    silhouette_scores=silhouette_scores,
    figure_path=FIGURES_DIR,
    save=False,
    show=True,
)

# =============================================================================
# K-means - Chosen model
# =============================================================================
kmeans_model = kmeans_models[0]
labels_kmeans = kmeans_model.labels_

# =============================================================================
# DBI Plot
# =============================================================================
plot_dbi(
    dbi_kmeans, dbi_dbscan, figure_path=FIGURES_DIR, save=False
)  # Needs to have been executed for different target_clusters
# The same number of clusters as kmeans

# =============================================================================
# t-SNE Plot
# =============================================================================
# List of labels and subtitle
labels = [labels_kmeans, labels_dbscan]
subtitle = ["k-means", "DBSCAN"]

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 1)

# Image size in mm
width, height = (90, 130)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("t-SNE Clustering Comparison", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    plot_cluster_tsne(
        labels[i - 1],
        X_reduced,
        figure_path=FIGURES_DIR,
        subtitle=subtitle[i - 1],
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        save=False,
    )


# =============================================================================
# Force vs hits plots - Clustering
# =============================================================================
test_id_prueba = ["P0_3", "P90_3", "P0_90_3", "P0_90W_1", "P45_4", "PQ_3"]
limits = limit_finder(labels_dbscan, hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (190, 110)
figsize_inches = (width / 25.4, height / 25.4)

# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Stress and Cumulative hits vs Time - Clustering", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits_cluster(
        labels_dbscan,
        hits,
        force,
        test_id_prueba[i - 1],
        figure_path=FIGURES_DIR,
        plot_type="line",
        limits=limits,
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        save=False,
    )

# =============================================================================
# Force vs hits plots
# =============================================================================
test_id_prueba = ["P0_3", "P90_3", "P0_90_3", "P0_90W_1", "P45_4", "PQ_3"]
limits = limit_finder_no_label(hits, force, test_id_prueba)

# Define the number of rows and columns for the subplot grid
n_row, n_col = (2, 3)

# Image size in mm
width, height = (190, 110)
figsize_inches = (width / 25.4, height / 25.4)
# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Stress and Cumulative hits vs Time", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_stress_hits(
        hits,
        force,
        test_id_prueba[i - 1],
        figure_path=FIGURES_DIR,
        plot_type="line",
        limits=limits,
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        save=False,
    )


# Create a new figure and set the size
fig, axes = plt.subplots(
    n_row, n_col, figsize=figsize_inches, dpi=300, tight_layout=True
)
plt.suptitle("Feature vs Feature Plot", fontsize=10)

# Loop through the function calls and store the plots in an array
for i, ax in enumerate(axes.flat, start=1):
    # Call the plot_dbi function and pass the current subplot axis
    plot_cluster_feat(
        labels_dbscan,
        data,
        "time_norm",
        "w_peak_freq",
        figure_path=FIGURES_DIR,
        x_label="test",
        ax=ax,
        i=i,
        n_col=n_col,
        n_row=n_row,
        save=False,
    )

# =============================================================================
# Other plots
# =============================================================================
# DBSCAN
plot_cluster_feat(labels_dbscan, data, "p_power_1", "w_peak_freq", figure_path=FIGURES_DIR)

for test_id in test_ids:
    plot_stress_hits_cluster(labels_dbscan, hits, force, test_id, figure_path=FIGURES_DIR)
    plot_stress_hits(hits, force, test_id, figure_path=FIGURES_DIR, plot_type="line")

for test_id in test_ids:
    condition = data["test_id"] == test_id
    plot_cluster_feat(
        labels_dbscan[condition],
        data[condition],
        "time_norm",
        "w_peak_freq",
        figure_path=FIGURES_DIR,
    )

# kmeans
plot_cluster_tsne(labels_kmeans, X_reduced, figure_path=FIGURES_DIR)
plot_cluster_feat(labels_kmeans, data, "p_power_3", "w_peak_freq", figure_path=FIGURES_DIR)
for test_id in test_ids:
    plot_cluster_hits(labels_kmeans, hits, test_id, figure_path=FIGURES_DIR)
for test_id in test_ids:
    plot_cluster_time(labels_kmeans, hits, test_id, figure_path=FIGURES_DIR)

# =============================================================================
# Save Model
# =============================================================================
models_path = Path("models") / dataset_name
dbscan_path = models_path / "dbscan_model.joblib"
kmeans_path = models_path / "kmeans_model.joblib"

dump(dbscan_model, dbscan_path)
print('Dbscan model saved at: "./models/dbscan_model.joblib"')

dump(kmeans_model, kmeans_path)
print('k-means model saved at: "./models/kmeans_model.joblib"')

# =============================================================================
# Change class label according to clustering
# =============================================================================
data["Class"] = labels_dbscan
labeled_dataset_name = "Dataset_0_y_90_labeled.csv"
data.to_csv(DATASETS_DIR / labeled_dataset_name, index=False)
print(f'Dataset labeled saved at: "./dataset/{labeled_dataset_name}"')

# =============================================================================
# Filtered hits
# =============================================================================
hits_filtered = hits.drop(
    ["Class", "amplitude", "energy", "time_norm", "counts"], axis=1
).copy()
hits_filtered.to_csv("hits_filtered.csv", index=False)

```

## File: `transient.py`
```python
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
figure_path = os.path.join(script_dir, "Figures")
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
    guardar=False,
)

```

## File: `utils\__init__.py`
```python

```

## File: `utils\cwt_analysis.py`
```python
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
def abreCSV(path, filename, val=0):
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
def abreCSVdot(path, filename, val=0):
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
                    plt.title(f"Trai n {trai}")
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
        # # Use logarithmic spacing for widths
        # widths = np.logspace(np.log10(1), np.log10(n_bands), int(n_bands * 2))

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
                STD_noise, signals, time, figure_path, num_noisySignals, save_noised=False
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

        lx, ly, _ = image.shape
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
        new_vector.extend(image_add)
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
        plt.plot(time, noised_signal_2, color="orange", linewidth=0.5, alpha=0.6)
        plt.plot(time, sig, linewidth=1)
        # plt.scatter(time, noised_signal_2, marker=".", color="orange", linewidths=0.25)
        plt.xlabel("Time [μs]")
        plt.ylabel("Amplitude [mV]")
        plt.legend(["Noised signal", "Original signal"])
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


def save_image_pred(image_list, dims, test_name, images_dir, images_set):
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
    test_name : str
        Name of the test for which the images are being saved.
    images_dir : str
        Directory to store the images.
    images_set : str
        Name of the image set (train, test, etc.).
    Returns
    -------
    None
        Saves the images in the specified directory.

    """
    path = os.path.join(images_dir, images_set, test_name)
    os.makedirs(path, exist_ok=True)
    image_redim = redim(image_list, dims)
    for i, image in enumerate(image_redim):
        image_name = f"CWT_{test_name}_{i:04d}.png"
        image_path = os.path.join(path, image_name)
        plt.imsave(image_path, image, cmap="gray")
    return print(f"Images of test {test_name} for prediction saved")

```

## File: `utils\fft_analysis.py`
```python
"""
Created on Thu Nov 16 10:24:10 2023

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================

# Main libraries
import os
import string

# Import necessary libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import vallenae as vae

# Libraries for signal processing
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hamming

# Model training
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Feature visualization
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
)

# Libraries for hyperparameter optimization
# Libraries for metrics and model evaluation
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_predict,
    train_test_split,
)

# Data processing and normalization
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

# Set font to Times New Roman and size to 8
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["figure.dpi"] = 150

MARKER_LIST = ["o", "s", "^", "v", "D", "p"]  # Customize the markers as desired

# =============================================================================
# Feature extraction
# =============================================================================


class Features:
    def __init__(
        self,
        path: str,
        lower_freq: float,
        upper_freq: float,
        sampling_rate: float,
        N_samp: int,
        N_seg: int,
        desfase: int,
        desfase_carga: int = 0,
        force: pd.DataFrame = None,
    ):
        """
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

        """
        self.path = path
        self.lower_freq = lower_freq  # Lower frequency limit in kHz
        self.upper_freq = upper_freq  # Upper frequency limit in kHz
        self.sampling_rate = (
            sampling_rate  # Data acquisition frequency per second in MHz
        )
        self.N_samp = N_samp  # Number of samples (data points) in each segment
        self.N_seg = N_seg  # Number of segments
        self.desfase = desfase  # Depends on the pretrigger time, allows starting from
        # the beginning of the signal or with a positive offset
        self.desfase_carga = desfase_carga  # Offset of the time taken in AE and in MTS
        self.force = force  # Force data for time alignment

        # Open pri and tra files
        files = os.listdir(self.path)
        for item in files:
            if item.endswith(".tradb"):
                self.tradb_path = os.path.join(self.path, item)

            elif item.endswith(".pridb"):
                self.pridb_path = os.path.join(self.path, item)

        # Calculate values
        self.res_freq = sampling_rate * 1000 / N_samp  # Frequency resolution in kHz
        self.time_samp = 1 / sampling_rate  # Sampling time in µs (microseconds)
        self.time_seg = self.time_samp * N_samp  # Time of each segment in µs

        # Calculate frequency range indices
        self.lower_freq_index = int(
            self.lower_freq // self.res_freq
        )  # Lower frequency index
        self.upper_freq_index = int(
            self.upper_freq // self.res_freq + 1
        )  # Upper frequency index
        self.N_feat_seg = 9  # Number of features per segment

        # Values for FFT that are calculated only once
        self.window = hamming(self.N_samp)  # Create the Hamming window
        self.freq_fft = (fftfreq(self.N_samp, self.time_samp * 1e-6) / 1000)[
            : self.N_samp // 2
        ]  # Positive frequency values of the FFT in kHz
        self.freq_fft_rango = self.freq_fft[
            self.lower_freq_index : self.upper_freq_index + 1
        ]  # Section the frequency values

    def feature_extr(
        self, threshold, counts, label, test_id, max_trai=None, min_trai=1
    ):
        """

        Parameters
        ----------
        threshold : int
            Threshold amplitude value for filtering.
        counts : int
            Minimum counts value for filtering.
        label : int or str
            Class to which the signal belongs.
        test_id : str
            Test code for later identification.
        max_trai : int, optional
            Maximum trai index to consider, if you want to exclude the break.
            The default is None.
        min_trai : int, optional
            Minimum trai index to consider, if you want to exclude some erroneous
            initial data. The default is 1.

        Returns
        -------
        df_total : DataFrame
            DataFrame containing frequency features + rise time, classes, and trai number.

        """

        # Read files
        pridb = vae.io.PriDatabase(self.pridb_path)
        tradb = vae.io.TraDatabase(self.tradb_path)
        df_hits = pridb.read_hits()  # Read the hits that have occurred

        # Signal filtering (trai threshold and counts)
        threshold_V = 10 ** (threshold / 20 - 6)  # Threshold in Volts
        no_saturacion = 10 ** (94 / 20 - 6)

        if max_trai is not None:
            df_hits_filtro = df_hits[
                (df_hits["channel"] >= 1)
                & (df_hits["amplitude"] >= threshold_V)
                & (df_hits["amplitude"] <= no_saturacion)
                & (df_hits["trai"] >= min_trai)
                & (df_hits["trai"] <= max_trai)
                & (df_hits["counts"] >= counts)
            ]  # Select only the values that meet the conditions
        else:
            df_hits_filtro = df_hits[
                (df_hits["channel"] >= 1)
                & (df_hits["amplitude"] >= threshold_V)
                & (df_hits["amplitude"] <= no_saturacion)
                & (df_hits["trai"] >= min_trai)
                & (df_hits["counts"] >= counts)
            ]  # Select only the values that meet the conditions
        if df_hits_filtro.empty:
            print(
                f"[WARNING] No transients found for label '{label}' with the given filtering criteria."
            )
            return (
                pd.DataFrame()
            )  # Return an empty DataFrame if no transients are found

        trai = df_hits_filtro[
            "trai"
        ].to_numpy()  # Extract the column with the TRAI values (transient indices)
        N_trai = trai.size  # Total number of transients

        # Create matrix to store features
        v_features_total = np.zeros(
            (N_trai, self.N_seg * self.N_feat_seg)
        )  # Array that will store the features in each iteration

        x = np.arange(0, self.N_seg)  # Vector with factors for segmentation

        # Main for loop for each transient
        for N_iter, trans in enumerate(trai):
            amp, _ = tradb.read_wave(trans, time_axis=False)  # Read the i-th transient

            amp *= 1e3  # in mV   # The operator *= is updating the amplitude value
            # -- amplitude = amplitude * 1e3
            # The signal was in volts and is represented in mV
            # This is the amplitude
            # time = tiempo * 1e6  # for µs  # Similarly, the signal is stored in
            # seconds and represented in µs

            # Create the segments
            seg_total = np.zeros(
                (self.N_seg, self.N_samp)
            )  # Array that will store the amplitude values in each segment

            # For loop to apply the Fourier transform
            for i in x:
                seg_total[i] = amp[
                    int(i * self.N_samp / 2) + self.desfase : int(
                        (i / 2 + 1) * self.N_samp
                    )
                    + self.desfase
                ]

            # Apply Hamming windows to each segment
            seg_t_window = (
                seg_total * self.window
            )  # Apply the window to the segment matrix

            # Calculate FFT of the segment matrix
            amp_fft = fft(
                seg_t_window
            )  # Calculate the Fourier transform of the segment matrix
            amp_fft = np.abs(
                amp_fft
            )  # Calculate the amplitude modulus to not represent with its phase

            # Select the range of interest frequencies
            amp_fft_rango = amp_fft[
                :, self.lower_freq_index : self.upper_freq_index + 1
            ]  # Section the amplitude values in the established frequency range

            # Initialize frequency feature vectors for each segment
            peak_freq = np.zeros(
                self.N_seg
            )  # Store the peak frequencies of each segment
            centroid_freq = np.zeros(
                self.N_seg
            )  # Store the centroid frequencies of each segment
            part_power_1 = np.zeros(
                self.N_seg
            )  # Store the total partial power 1 of each segment
            part_power_2 = np.zeros(
                self.N_seg
            )  # Store the total partial power 2 of each segment
            part_power_3 = np.zeros(
                self.N_seg
            )  # Store the total partial power 3 of each segment
            part_power_4 = np.zeros(
                self.N_seg
            )  # Store the total partial power 4 of each segment
            part_power_5 = np.zeros(
                self.N_seg
            )  # Store the total partial power 5 of each segment
            part_power_6 = np.zeros(
                self.N_seg
            )  # Store the total partial power 6 of each segment

            # For loop to calculate features
            for i in x:
                # Peak frequency
                peak_index = np.argmax(amp_fft_rango[i])
                peak_freq[i] = self.freq_fft_rango[peak_index]

                # Centroid frequency
                centroid_freq[i] = np.sum(
                    amp_fft_rango[i] * self.freq_fft_rango
                ) / np.sum(amp_fft_rango[i])

                # Partial Power 1 - 6
                denominator = np.sum((amp_fft_rango[i]) ** 2)
                part_power_1[i] = (
                    np.sum(
                        (amp_fft_rango[i, 0 : int(len(self.freq_fft_rango) * 1 / 6)])
                        ** 2
                    )
                    * 100
                    / denominator
                )
                part_power_2[i] = (
                    np.sum(
                        (
                            amp_fft_rango[
                                i,
                                int(len(self.freq_fft_rango) * 1 / 6) : int(
                                    len(self.freq_fft_rango) * 2 / 6
                                ),
                            ]
                        )
                        ** 2
                    )
                    * 100
                    / denominator
                )
                part_power_3[i] = (
                    np.sum(
                        (
                            amp_fft_rango[
                                i,
                                int(len(self.freq_fft_rango) * 2 / 6) : int(
                                    len(self.freq_fft_rango) * 3 / 6
                                ),
                            ]
                        )
                        ** 2
                    )
                    * 100
                    / denominator
                )
                part_power_4[i] = (
                    np.sum(
                        (
                            amp_fft_rango[
                                i,
                                int(len(self.freq_fft_rango) * 3 / 6) : int(
                                    len(self.freq_fft_rango) * 4 / 6
                                ),
                            ]
                        )
                        ** 2
                    )
                    * 100
                    / denominator
                )
                part_power_5[i] = (
                    np.sum(
                        (
                            amp_fft_rango[
                                i,
                                int(len(self.freq_fft_rango) * 4 / 6) : int(
                                    len(self.freq_fft_rango) * 5 / 6
                                ),
                            ]
                        )
                        ** 2
                    )
                    * 100
                    / denominator
                )
                part_power_6[i] = (
                    np.sum(
                        (amp_fft_rango[i, int(len(self.freq_fft_rango) * 5 / 6) :]) ** 2
                    )
                    * 100
                    / denominator
                )

            w_peak_freq = (peak_freq * centroid_freq) ** 0.5

            # Feature vector
            v_features = np.hstack(
                (
                    peak_freq,
                    w_peak_freq,
                    centroid_freq,
                    part_power_1,
                    part_power_2,
                    part_power_3,
                    part_power_4,
                    part_power_5,
                    part_power_6,
                )
            )  # Group the features
            v_features = v_features.reshape(
                1, self.N_feat_seg * self.N_seg
            )  # Reshape the feature matrix into a single row vector
            v_features_total[N_iter] = (
                v_features  # Chain the values to each row of the matrix to create the feature matrix
            )

        # Feature names

        if self.N_seg == 3:
            name_feature = [
                "peak_freq_1",
                "peak_freq_2",
                "peak_freq_3",
                "w_peak_freq_1",
                "w_peak_freq_2",
                "w_peak_freq_3",
                "centroid_freq_1",
                "centroid_freq_2",
                "centroid_freq_3",
                "p_power_1_1",
                "p_power_1_2",
                "p_power_1_3",
                "p_power_2_1",
                "p_power_2_2",
                "p_power_2_3",
                "p_power_3_1",
                "p_power_3_2",
                "p_power_3_3",
                "p_power_4_1",
                "p_power_4_2",
                "p_power_4_3",
                "p_power_5_1",
                "p_power_5_2",
                "p_power_5_3",
                "p_power_6_1",
                "p_power_6_2",
                "p_power_6_3",
            ]  # Array that will store the feature names per segment

        elif self.N_seg == 1:
            name_feature = [
                "peak_freq",
                "w_peak_freq",
                "centroid_freq",
                "p_power_1",
                "p_power_2",
                "p_power_3",
                "p_power_4",
                "p_power_5",
                "p_power_6",
            ]  # Array that will store the feature names per segment

        # Feature DataFrame
        df_features = pd.DataFrame(v_features_total, columns=name_feature)

        # Trai DataFrame
        df_trai = pd.DataFrame(trai, columns=["trai"])

        # Time and rise time DataFrame
        df_time = (
            df_hits_filtro[["rise_time", "time", "amplitude", "energy", "counts"]]
            .reset_index()
            .drop(["set_id"], axis=1)
        )
        df_time["amplitude"] = 20 * (np.log10(df_time["amplitude"]) + 6)
        
        if self.force is not None:
            desfase_carga = df_time["time"].iloc[-1] - self.force["Time [s]"].iloc[-1]
        else:
            desfase_carga = self.desfase_carga
        df_time["time"] = df_time["time"] - desfase_carga
        df_time["time_norm"] = df_time["time"] / max(df_time["time"])

        # Class DataFrame
        df_label = pd.DataFrame({"label": [label] * N_trai})

        # Total DataFrame
        df_total = pd.concat([df_features, df_time, df_label, df_trai], axis=1)
        df_total["test_id"] = test_id

        # The function returns the total DataFrame
        return df_total

    def plot_signal(
        self,
        trai,
        name_figure,
        figure_path,
        time_graph=None,
        title=None,
        x_label="Time [µs]",
        y_label="Amplitude [mV]",
        width=90,
        height=60,
        guardar=False,
    ):
        """
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

        """
        # Open tradb files
        tradb = vae.io.TraDatabase(self.tradb_path)

        # Plot the transient
        amp, tiempo = tradb.read_wave(trai, time_axis=True)
        amp *= 1e3  # in mV   # The operator *= is updating the amplitude value -- amp = amp * 1e3
        # The signal was in volts and is represented in mV
        # This is the amplitude
        time = (
            tiempo * 1e6
        )  # for µs  # Similarly, the signal is stored in seconds and represented in µs
        # This is the time

        figsize_inches = (width / 25.4, height / 25.4)
        plt.figure(figsize=figsize_inches, dpi=300, tight_layout=True)

        if time_graph is None:
            plt.plot(time, amp, linewidth=0.6)
        else:
            points = int(time_graph * self.sampling_rate + 1)
            plt.plot(
                time[self.desfase : points], amp[self.desfase : points], linewidth=0.6
            )
            plt.ylim(
                bottom=min(amp[self.desfase : points]) * 1.6,
                top=max(amp[self.desfase : points]) * 1.6,
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(linewidth=0.3, color=".25")
        if title is None:
            plt.title(f"Transient Wave Plot: {name_figure} - Transient: {trai}")
        else:
            plt.title(title)

        # Save the image in figure_path
        if guardar:
            figure_filename = f"Transient_{name_figure}_{trai}.pdf"
            figure_path_name = os.path.join(figure_path, figure_filename)
            plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

        plt.show()

    def plot_segmentation(self, trai, name_figure, figure_path, guardar=False):
        """
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

        """
        # Open tradb files
        tradb = vae.io.TraDatabase(self.tradb_path)

        # Read amplitude and time
        amp, tiempo = tradb.read_wave(trai, time_axis=True)
        amp *= 1e3  # in mV   # The operator *= is updating the amplitude value -- amp = amp * 1e3
        # The signal was in volts and is represented in mV
        # This is the amplitude
        time = (
            tiempo * 1e6
        )  # for µs  # Similarly, the signal is stored in seconds and represented in µs
        # This is the time

        # Fourier transform calculation
        x = np.arange(0, self.N_seg)  # Vector with factors for segmentation

        # Create segments
        seg_total = np.zeros(
            (self.N_seg, self.N_samp)
        )  # Array that will store the amplitude values in each segment

        for i in x:
            seg_total[i] = amp[
                int(i * self.N_samp / 2) + self.desfase : int((i / 2 + 1) * self.N_samp)
                + self.desfase
            ]

        # Apply Hamming windows to each segment
        seg_t_window = seg_total * self.window  # Apply the window to the segment matrix

        # Calculate FFT of the segment matrix
        amp_fft = fft(
            seg_t_window
        )  # Calculate the Fourier transform of the segment matrix
        amp_fft = np.abs(
            amp_fft
        )  # Calculate the amplitude modulus to not represent with its phase (Real and imaginary part)

        # Select the range of interest frequencies
        amp_fft_rango = amp_fft[
            :, self.lower_freq_index : self.upper_freq_index + 1
        ]  # Section the amplitude values in the established frequency range

        # Multiple plots in one figure
        fig, axs = plt.subplots(2, self.N_seg, figsize=(15, 7.5))
        plt.suptitle(f"Segmentation: {name_figure} - Transient: {trai}", fontsize=14)
        plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9)

        # Maximum and minimum for axes
        seg_t_window_max = np.amax(abs(seg_t_window))
        amp_fft_max = np.amax(amp_fft_rango)

        for j in x:
            # Time domain
            axs[0, j].grid(linewidth=0.3, color=".25")
            axs[0, j].set_ylim(-1.1 * seg_t_window_max, seg_t_window_max * 1.1)
            axs[0, j].plot(
                time[
                    int(j * self.N_samp / 2) + self.desfase : int(
                        (j / 2 + 1) * self.N_samp
                    )
                    + self.desfase
                ],
                seg_t_window[j],
            )
            axs[0, j].set_xlabel("Time [µs]")
            axs[0, 0].set_ylabel("Amplitude [mV]")
            axs[0, j].set_title(f"Segment: {j + 1}")

            # FFT
            axs[1, j].grid(linewidth=0.3, color=".25")
            axs[1, j].set_ylim(0, amp_fft_max * 1.1)
            axs[1, j].plot(self.freq_fft_rango, amp_fft_rango[j][: self.N_samp // 2])

            # Vertical line in the peak frequency
            peak_index = np.argmax(amp_fft_rango[j])
            peak_freq = self.freq_fft_rango[peak_index]
            y_max = np.amax(amp_fft_rango[j]) * 2 / (amp_fft_max * 1.1)
            axs[1, j].axvline(
                x=peak_freq, ymax=y_max, color="red", linestyle="--", label="f_peak"
            )

            # Vertical line in the Centroid frequency
            centroid_freq = np.sum(amp_fft_rango[j] * self.freq_fft_rango) / np.sum(
                amp_fft_rango[j]
            )
            y_max_c = 0.8 * y_max
            axs[1, j].axvline(
                x=centroid_freq,
                ymax=y_max_c,
                color="green",
                linestyle=":",
                label="f_centroid",
            )

            axs[1, j].set_xlabel("Frequency [kHz]")
            axs[1, 0].set_ylabel("FFT-Magnitude [mV]")
            axs[1, j].legend()

        # Add a single border around all subplots
        fig.patch.set_edgecolor("black")
        fig.patch.set_linewidth(1)

        # Save the image in figure_path
        if guardar:
            figure_filename = f"Segmentation_{name_figure}_{trai}.pdf"
            figure_path_name = os.path.join(figure_path, figure_filename)
            plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

        plt.show()

    def plot_segmentation_1(
        self,
        trai,
        name_figure,
        figure_path,
        title=None,
        width=90,
        height=120,
        guardar=False,
    ):
        """
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
            Title of the figure, if left as default it will be Segmentation:
            {name_figure} - Transient: {trai}. The default is None
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

        """
        # Open tradb files
        tradb = vae.io.TraDatabase(self.tradb_path)

        # Read amplitude and time
        amp, tiempo = tradb.read_wave(trai, time_axis=True)
        amp *= 1e3  # in mV   # The operator *= is updating the amplitude value -- amp = amp * 1e3
        # The signal was in volts and is represented in mV
        # This is the amplitude
        time = (
            tiempo * 1e6
        )  # for µs  # Similarly, the signal is stored in seconds and represented in µs
        # This is the time

        # Fourier transform calculation
        x = np.arange(0, self.N_seg)  # Vector with factors for segmentation

        # Create segments
        seg_total = np.zeros(
            (self.N_seg, self.N_samp)
        )  # Array that will store the amplitude values in each segment

        for i in x:
            seg_total[i] = amp[
                int(i * self.N_samp / 2) + self.desfase : int((i / 2 + 1) * self.N_samp)
                + self.desfase
            ]

        # Apply Hamming windows to each segment
        seg_t_window = seg_total * self.window  # Apply the window to the segment matrix

        # Calculate FFT of the segment matrix
        amp_fft = fft(
            seg_t_window
        )  # Calculate the Fourier transform of the segment matrix
        amp_fft = np.abs(
            amp_fft
        )  # Calculate the amplitude modulus to not represent with its phase (Real and imaginary part)

        # Select the range of interest frequencies
        amp_fft_rango = amp_fft[
            :, self.lower_freq_index : self.upper_freq_index + 1
        ]  # Section the amplitude values in the established frequency range

        # Multiple plots in one figure
        figsize_inches = (width / 25.4, height / 25.4)
        _, axs = plt.subplots(
            2, self.N_seg, figsize=figsize_inches, dpi=300, tight_layout=True
        )
        if title is None:
            plt.suptitle(
                f"Segmentation: {name_figure} - Transient: {trai}", fontsize=10
            )
        else:
            plt.suptitle(title, fontsize=10)
        plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9)

        # Maximum and minimum for axes
        seg_t_window_max = np.amax(abs(seg_t_window))
        amp_fft_max = np.amax(amp_fft_rango)

        # Time domain
        axs[0].grid(linewidth=0.3, color=".25")
        axs[0].set_ylim(-1.1 * seg_t_window_max, seg_t_window_max * 1.1)
        axs[0].plot(
            time[
                int(0 * self.N_samp / 2) + self.desfase : int((0 / 2 + 1) * self.N_samp)
                + self.desfase
            ],
            seg_t_window[0],
            linewidth=1,
        )
        axs[0].set_xlabel("Time [µs]")
        axs[0].set_ylabel("Amplitude [mV]")
        axs[0].set_title(f"Segment: {0 + 1}")

        # FFT
        axs[1].grid(linewidth=0.3, color=".25")
        axs[1].set_ylim(0, amp_fft_max * 1.1)
        axs[1].plot(
            self.freq_fft_rango, amp_fft_rango[0][: self.N_samp // 2], linewidth=1
        )

        # Vertical line in the peak frequency
        peak_index = np.argmax(amp_fft_rango[0])
        peak_freq = self.freq_fft_rango[peak_index]
        y_max = np.amax(amp_fft_rango[0]) * 2 / (amp_fft_max * 1.1)
        axs[1].axvline(
            x=peak_freq, ymax=y_max, color="red", linestyle="--", label="f_peak"
        )

        # Vertical line in the Centroid frequency
        centroid_freq = np.sum(amp_fft_rango[0] * self.freq_fft_rango) / np.sum(
            amp_fft_rango[0]
        )
        y_max_c = 0.8 * y_max
        axs[1].axvline(
            x=centroid_freq,
            ymax=y_max_c,
            color="green",
            linestyle=":",
            label="f_centroid",
        )

        axs[1].set_xlabel("Frequency [kHz]")
        axs[1].set_ylabel("FFT-Magnitude [mV]")
        axs[1].legend()

        # Save the image in figure_path
        if guardar:
            figure_filename = f"Segmentation_{name_figure}_{trai}.pdf"
            figure_path_name = os.path.join(figure_path, figure_filename)
            plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

        plt.show()


def unir_df(*dataframes):
    """
    Parameters
    ----------
    *dataframes : DataFrame
        DataFrames to concatenate.

    Returns
    -------
    concatenated_df : DataFrame
        Returns a single concatenated DataFrame.

    """
    # Concatenate DataFrames vertically
    concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)
    return concatenated_df


def train_test_set(
    data,
    normalization=None,
    columns_to_transform=None,
    features=None,
    split=True,
    test_size=0.2,
):
    """
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

    If split=False, only X_original, X_train, y_train, and hits
    (DataFrame for cumulative hits plot) are returned.
    """
    # Separate features from classes
    X = data.drop(
        [
            "label",
            "trai",
            "time",
            "test_id",
        ],
        axis=1,
    )  # Remove class and trais columns (also remove counts and time_norm)

    if features:
        X = X[features]

    y = data["label"].copy()  # Create a copy of the class column

    X_original = X.copy()

    # Feature names
    feature_names = X.columns.tolist()

    # If columns to transform are not defined, select all
    if columns_to_transform is None:
        columns_to_transform = feature_names

    # Normalization options
    if normalization == "log":
        log_transformer = FunctionTransformer(np.log)
        X[columns_to_transform] = log_transformer.transform(
            X[columns_to_transform]
        )  # Apply logarithm to features

    if normalization == "std":
        standardize = StandardScaler()
        X[columns_to_transform] = standardize.fit_transform(
            X[columns_to_transform]
        )  # Apply standardization to features

    if normalization == "min_max":
        min_max = MinMaxScaler()
        X[columns_to_transform] = min_max.fit_transform(
            X[columns_to_transform]
        )  # Apply min-max scaling to features

    if normalization == "log-std":
        log_transformer = FunctionTransformer(np.log)
        X[columns_to_transform] = log_transformer.transform(
            X[columns_to_transform]
        )  # Apply logarithm to features
        standardize = StandardScaler()
        X = standardize.fit_transform(X)  # Apply standardization to features

    # The DataFrame returned by the function depends on whether it is split or not
    if split:
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=data["label"],
            random_state=0,
            shuffle=True,
        )

        # The function returns 4 DataFrames
        return X_train, X_test, y_train, y_test

    else:
        X_train = X
        y_train = y

        # DataFrame for cumulative hits
        hits = data[
            [
                "test_id",
                "trai",
                "label",
                "time",
                "amplitude",
                "energy",
                "time_norm",
                "counts",
            ]
        ].copy()
        hits["Count"] = 1

        return X_original, X_train, y_train, hits


def backward_elimination(X_train, y_train):
    """
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

    """
    # Classifier
    model = RandomForestClassifier(random_state=11)

    # Perform backward elimination with cross-validation
    selector = RFECV(estimator=model, step=1, cv=5, scoring="f1_macro")
    selector.fit(X_train, y_train)

    # Get the selected feature names
    selected_feature_names = selector.get_feature_names_out().tolist()

    # Get the best score
    best_f1_score = selector.cv_results_["mean_test_score"][selector.n_features_ - 1]

    # Print the best feature subset and its performance
    print("", "Backward Elimination", sep="\n")
    print("Best Feature Subset:", selected_feature_names)
    print(f"Best F1 Score: {best_f1_score:.1%}")

    return selected_feature_names


def forest_clf(X_train, X_test, y_train, y_test, selected_features=None):
    """
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

    """
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
    f1_score_test = f1_score(
        y_test, y_test_pred, average="macro"
    )  # Calculate the F1 score
    print("", "Precision and F1 score on the test set", sep="\n")
    print(
        f"Baseline Accuracy: {baseline_accuracy:.1%}"
    )  # Print the model accuracy in % format with 1 decimal
    print(f"F1 score - test = {f1_score_test:.1%}")

    # Cross-validation that returns the model accuracy (scores) and predicted values (predict)
    y_train_pred = cross_val_predict(
        model, X_train, y_train, cv=5
    )  # Predicted values on the training set with cross-validation
    cross_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy
    cross_f1_score = f1_score(
        y_train, y_train_pred, average="macro"
    )  # Calculate the F1 score
    print("", "Cross-validation to check for overfitting", sep="\n")
    print(f"Cross-val Accuracy = {cross_accuracy:.1%}")
    print(f"Cross-val F1 score = {cross_f1_score:.1%}")

    # Calculate the importance of each feature for classification
    feature_importances = (model.feature_importances_ * 100).round(2)
    best_features = pd.DataFrame(
        sorted(
            zip(X_train.columns.tolist(), feature_importances),
            key=lambda x: x[1],
            reverse=True,
        ),
        columns=["Features", "Importances (%)"],
    )

    # Print the sorted DataFrame
    print("", "Feature Importance", sep="\n")
    print(best_features.head(5))

    return y_train_pred, y_test_pred, best_features, model


def forest_clf_hyper(X_train, X_test, y_train, y_test, selected_features=None):
    """
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

    """
    param_grid = {
        "max_features": [
            "sqrt",
            "log2",
        ],  # Maximum number of features to consider at each split
        "n_estimators": np.arange(100, 1000, 100),  # Number of trees in the forest
        "max_depth": [None] + list(np.arange(5, 30, 5)),  # Maximum depth of each tree
        "min_samples_split": np.arange(
            2, 11
        ),  # Minimum number of samples required to split an internal node
        "min_samples_leaf": np.arange(
            1, 5
        ),  # Minimum number of samples required to be at a leaf node
    }

    # Define the scoring metric
    scorer = make_scorer(f1_score, average="macro")

    model = RandomForestClassifier(random_state=42)  # Model name

    # Perform randomized search
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        scoring=scorer,
        cv=5,
        n_iter=10,
        random_state=42,
    )  # Optimizer
    random_search.fit(X_train, y_train)  # Train the optimizer
    best_params = random_search.best_params_  # Get the optimized parameters
    best_score = (
        random_search.best_score_
    )  # Get the F1 score from the optimizer training

    print("", "Hyperparameter tuning", sep="\n")
    print(f"Best parameters: {best_params}")
    print(f"Optimized F1 score: {best_score:.1%}")

    # Train the model with the best estimator
    best_model = random_search.best_estimator_  # Store the best estimator (best model)
    best_model.fit(X_train, y_train)  # Train the machine with the best estimator
    y_test_pred = best_model.predict(
        X_test
    )  # Predictions made with the optimized model on the test set

    # Model accuracy on the test set
    baseline_accuracy = accuracy_score(y_test, y_test_pred)

    # F1 score of the model on the test set
    f1_score_test = f1_score(
        y_test, y_test_pred, average="macro"
    )  # Calculate the F1 score
    print("", "Precision and F1 score on the test set", sep="\n")
    print(
        f"Baseline Accuracy: {baseline_accuracy:.1%}"
    )  # Print the model accuracy in % format with 1 decimal
    print(f"F1 score - test = {f1_score_test:.1%}")

    # Cross-validation that returns the model accuracy (scores) and predicted values (predict)
    y_train_pred = cross_val_predict(
        best_model, X_train, y_train, cv=5
    )  # Predicted values on the training set with cross-validation
    cross_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy
    cross_f1_score = f1_score(
        y_train, y_train_pred, average="macro"
    )  # Calculate the F1 score
    print("", "Cross-validation to check for overfitting", sep="\n")
    print(f"Cross-val Accuracy = {cross_accuracy:.1%}")
    print(f"Cross-val F1 score = {cross_f1_score:.1%}")

    # Calculate the importance of each feature for classification
    feature_importances = (best_model.feature_importances_ * 100).round(2)
    best_features = pd.DataFrame(
        sorted(
            zip(X_train.columns.tolist(), feature_importances),
            key=lambda x: x[1],
            reverse=True,
        ),
        columns=["Features", "Importances (%)"],
    )

    # Print the sorted DataFrame
    print("", "Feature Importance", sep="\n")
    print(best_features.head(5))

    return y_train_pred, y_test_pred, best_features, best_model


def conf_matrix(
    y, y_pred, name_figure, figure_path, width=90, height=60, guardar=False
):
    """
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

    """
    # Plot the confusion matrix
    figsize_inches = (width / 25.4, height / 25.4)
    _, axs = plt.subplots(
        nrows=1, ncols=2, figsize=figsize_inches, dpi=300, tight_layout=True
    )
    plt.subplots_adjust(wspace=0.4)
    plt.rc("font", size=10)
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=axs[0])
    axs[0].set_title("By element")
    plt.rc("font", size=10)
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, ax=axs[1], normalize="true", values_format=".0%"
    )
    axs[1].set_title("Normalized by row")
    plt.suptitle(f"Confusion matrix - {name_figure}", fontsize=14)

    # Save the image in figure_path
    if guardar:
        figure_filename = f"Confusion matrix {name_figure}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    plt.show()


def plot_tsne(
    X,
    y,
    figure_path,
    title="t-SNE Scatter Plot",
    x_label="t-SNE Dimension 1",
    y_label="t-SNE Dimension 2",
    width=90,
    height=60,
    selected_features=None,
    guardar=False,
):
    """
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

    """
    # Feature selection (if backward elimination was used)
    if selected_features is not None:
        X = X[selected_features]

    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X)

    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f"t-SNE_{i + 1}" for i in range(2)])

    y = pd.Series(y, name="Labels")
    y.index = tsne_df.index
    tsne_df["Labels"] = y

    # Define custom markers for each class
    markers = ["o", "s", "^", "v", "D", "p"]  # Customize the markers as desired

    # Create a figure
    figsize_inches = (width / 25.4, height / 25.4)
    plt.figure(figsize=figsize_inches, dpi=300, tight_layout=True)

    # Use Seaborn scatter plot with hue and style parameters
    sns.scatterplot(
        x="t-SNE_1",
        y="t-SNE_2",
        hue="Labels",
        style="Labels",
        data=tsne_df,
        palette=sns.color_palette(),
        markers=markers,
    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if guardar:
        figure_filename = f"{title}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    plt.show()


def plot_histogram(
    X, y, figure_path, width=90, height=60, bins=30, feature_to_save="", guardar=False
):
    """
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

    """
    # Directory and filename to save the image
    figure_filename = f"Histogram_{feature_to_save}.pdf"
    figure_path_name = os.path.join(figure_path, figure_filename)

    # Iterate over each column
    for feature in X.columns:
        # Set up the figure and axis
        figsize_inches = (width / 25.4, height / 25.4)
        _, ax = plt.subplots(figsize=figsize_inches, dpi=300, tight_layout=True)
        # Set the number of bins for the histogram
        bins = 30

        # Plot the histogram for each class
        class_labels = y.unique()
        for class_label in class_labels:
            feat_to_plot = X[y == class_label][feature]
            ax.hist(
                feat_to_plot, bins=bins, alpha=0.6, label=class_label, edgecolor="black"
            )

        # Add labels and a title
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram - {feature}")

        # Add a legend
        ax.legend()

        # Save the image in figure_path
        if guardar and (feature == feature_to_save):
            plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

        # Display the plot
        plt.show()


def plot_feat_vs_feat(
    X,
    y,
    feat_x,
    feat_y,
    figure_path="Feat_vs_Feat.png",
    title=None,
    subtitle=None,
    x_label=None,
    y_label=None,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    n_row=1,
    save=False,
    show=False,
):
    """
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
        Title of the figure, if left as default it will be {feat_1} vs {feat_2}.
        The default is None.
    subtitle : str or list, optional
        Subtitle of the figure, if left as default it will be {feat_1} vs {feat_2}.
        The default is None.
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
    save : boolean, optional
        "True" to save the image. The default is False.
    show : boolean, optional
        "True" to show the image. The default is False.
        e.g. Feat_vs_Feat_{feat_1}_{feat_2}.pdf

    Returns
    -------
    Plots and (saves) a scatter plot of two features with labels and markers in pdf format.

    """
    y = pd.Series(y, name="Labels")
    # Rename 'Labels' column in y if it already exists in X
    if "Labels" in X.columns:
        X = X.drop(["Labels"], axis=1)

    y.index = X.index
    data = pd.concat([X, y], axis=1)
    label_order = list(pd.unique(data["Labels"]))
    n_labels = len(label_order)

    # Define custom markers for each class
    markers = {
        label: MARKER_LIST[i % len(MARKER_LIST)] for i, label in enumerate(label_order)
    }
    # Create a scatter plot
    figsize_inches = (width / 25.4, height / 25.4)

    # If there is just one plot and set the title
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_inches, dpi=300, tight_layout=True)
        if title is None:
            ax.set_title(f"{feat_x} vs {feat_y}")
        else:
            ax.set_title(title)
        if x_label is None:
            ax.set_xlabel(feat_x)
        else:
            ax.set_xlabel(x_label)

    else:
        if x_label is None:
            ax.set_xlabel(f"{feat_x}\n(" + string.ascii_lowercase[i - 1] + ")")
        else:
            ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(f"{feat_x} vs {feat_y}")
    if n_labels > 10:
        palette = sns.color_palette("tab20", n_colors=n_labels)
    else:
        palette = sns.color_palette("tab10", n_colors=n_labels)

    # Create the plot
    sns.scatterplot(
        x=feat_x,
        y=feat_y,
        hue="Labels",
        style="Labels",
        hue_order=label_order,
        style_order=label_order,
        data=data,
        markers=markers,
        palette=palette,
        ax=ax,
        legend=True,
    )

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        if y_label is None:
            ax.set_ylabel(feat_y)
        else:
            ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    # Logarithmic scale if you choose energy
    if feat_x == "energy":
        ax.set_xscale("log")

    if feat_y == "energy":
        ax.set_yscale("log")

    # Activate the grid
    ax.grid()
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    if n_labels > 6:
        ax.legend(
            title="Label",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=7,
        )

    fig = ax.get_figure()
    fig.subplots_adjust(right=0.72)

    # Save the image in figure_path
    if save:
        fig_format = figure_path.name.split(".")[-1]
        plt.savefig(figure_path, format=fig_format, bbox_inches="tight")
    if show:
        plt.show()

    plt.close()

    return ax


def select_significant_indices(
    data,
    label_column,
    feature_columns,
    max_points=20,
):
    selected_indices = []

    for _, group in data.groupby(label_column, sort=False):
        values = group[feature_columns].apply(pd.to_numeric, errors="coerce")
        values = values.dropna(how="all")

        if values.empty:
            continue

        median = values.median()
        mad = (values - median).abs().median()
        mad = mad.mask(mad == 0, 1.0)

        robust_z = 0.6745 * (values - median) / mad
        significance = robust_z.abs().mean(axis=1)

        selected_indices.extend(
            significance.nlargest(min(max_points, len(significance))).index
        )

    return selected_indices

```

## File: `utils\force_mts.py`
```python
"""
Created on Tue Jan  9 11:09:45 2024

@author: bbarmac
"""

# =============================================================================
# Import necessary libraries
# =============================================================================
# Dataframes
import csv
import os
import string
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Figures
import seaborn as sns

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["figure.dpi"] = 150

MARKER_LIST = ["o", "s", "^", "v", "D", "p"]  # Customize the markers as desired


def force_data(path, section=None, test_id=None):
    """
    Parameters
    ----------
    path : str
        Directory where the force mts csv files are stored.
    section : list or array
        Width and thickness measurements of the specimen in mm. The default is [22, 2.4]

    Returns
    -------
    force_df : Dataframe
        Dataframe containing force, displacement, time, and test_id data.

    """
    if section is None:
        section = [22, 2.4]
    files = os.listdir(path)
    forces_list = []

    for item in files:
        # Check if the files are .csv
        if item.endswith(".csv"):
            # Path of the folder joined with the file
            force_filename_path = Path(path) / item

            # Load the csv file
            # force = pd.read_csv(force_filename_path, sep=";")
            # Find the first row whose first column is numeric
            with force_filename_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                reader = csv.reader(file, delimiter=";")

                for row_number, row in enumerate(reader):
                    if not row:
                        continue

                    first_value = row[0].strip().replace(",", ".")

                    try:
                        float(first_value)
                        first_data_row = row_number
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"No numeric data found in {force_filename_path}")

            # Read from the first numeric row
            force = pd.read_csv(
                force_filename_path,
                sep=";",
                skiprows=first_data_row,
                header=None,
                names=["Crosshead [mm]", "Load [kN]", "Time [s]"],
                decimal=",",
            )

            # # Rename the columns and remove the units row
            # new_column_names = ["Crosshead [mm]", "Load [kN]", "Time [s]"]
            # force.columns = new_column_names
            # force = force.iloc[1:]

            # Add a row of 0 where the test starts
            new_row = pd.Series(0, index=force.columns)
            force = pd.concat([pd.DataFrame([new_row]), force], ignore_index=True)

            # Stress
            area = np.prod(section)
            force["Stress [MPa]"] = force["Load [kN]"] * 1000 / area

            # Add the test id column
            if test_id is None:
                test_id = item.split(".")[0]
            force["test_id"] = test_id
            forces_list.append(force)
        if not forces_list:
            print(f"[INFO] No .csv files found in {path}. Please check the directory.")
        else:
            force_df = pd.concat(forces_list, ignore_index=True)
    return force_df


def plot_stress_hits_cluster(
    data,
    force,
    test_id,
    figure_path="Stress and Cumulative hits vs Time - Clustering.pdf",
    plot_type="scatter",
    x="time",
    y="Cumulative_Label",
    x_label="Time [s]",
    y_label="Cumulative hits",
    y_label_right="Stress [MPa]",
    use_log_scale=False,
    subtitle=None,
    limits=None,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    save=False,
    show=True,
):
    """
    Parameters
    ----------
    data : Dataframe
        Set containing the data for the plot labeled and with their id.
    force : Dataframe
        Set containing load displacement, time data, and their id.
    test_id : str
        Id of the test to be plotted.
    figure_path : str
        Directory to save the figure.
    plot_type : str, optional
        Type of plot to create ('scatter' or 'line'). The default is 'scatter'.
    title : str, optional
        Title of the image to display and save. The default is 'Cumulative hits vs Time'.
    x : str, optional
        Time characteristic for the x-axis. The default is 'time'.

        Other options: 'Cumulative_Label', 'Cumulative', 'amplitude', 'energy'
    y : str, optional
        Time characteristic for the y-axis. The default is 'Cumulative_Label'.

        Other options: 'time', 'Cumulative', 'amplitude', 'energy'
    x_label : str, optional
        Label for the x-axis. The default is 'Time [s]'.
    y_label : str, optional
        Label for the y-axis. The default is 'Cumulative hits'.
    y_label_right : str, optional
        Label for the right y-axis. The default is 'Stress [MPa]'.
    subtitle : str or list, optional
        Title for each subplot, if left as default the test_id is plotted. The default is None.
    limits : list, optional
        Upper limits for the axes [time_limit, hits_limit, stress_limit]. The default is None.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    ax : matplotlib axis object, optional
        Axis to plot on. If not provided, a new axis will be created.
    i : int, optional
        Index for subplot. Default is 1.
    n_col : int, optional
        Number of columns. The default is 1.
    n_row : int, optional
        Number of rows. The default is 1.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. Cluster vs Time {test_id}.pdf

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Plots and (saves) a diagram of cumulative hits and stress in pdf format.
    """
    # Extract the limits
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    # Condition to filter rows based on the string column
    condition = data["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    # Filtered DataFrame
    filtered_hits = data.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = filtered_hits.groupby("label")[
        "Count"
    ].cumsum()

    # Force data
    filtered_force = force[condition_force]

    # Count the number of unique labels in the DataFrame
    num_labels = filtered_hits["label"].nunique()
    first_label = np.sort(filtered_hits["label"].unique())[0]

    # Define custom markers for each class
    markers = MARKER_LIST[first_label:first_label + num_labels]
    palette = sns.color_palette()
    palette = palette[first_label:first_label + num_labels]

    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_title(f"Stress and Cumulative hits vs Time - {test_id}")
        ax.set_xlabel(x_label)

    else:
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(test_id)

    # Create the scatter plot
    if plot_type == "scatter":
        sns.scatterplot(
            x=x,
            y=y,
            hue="label",
            style="label",
            data=filtered_hits,
            markers=markers,
            palette=palette,
            linewidth=0.2,
            s=20,
            ax=ax,
        )
        if use_log_scale:
            ax.set_yscale("log")

    elif plot_type == "line":
        sns.lineplot(
            x=x,
            y=y,
            hue="label",
            style="label",
            data=filtered_hits,
            palette=palette,
            linewidth=2,
            ax=ax,
        )

    # Set the x-axis limit
    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    # Set the y-axis limit
    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)

    # Create a twin Axes sharing the x-axis
    ax2 = ax.twinx()

    # Create the second scatter plot on the twin Axes
    sns.lineplot(
        x="Time [s]",
        y="Stress [MPa]",
        data=filtered_force,
        ax=ax2,
        linewidth=1,
        color="black",
        label="Stress",
    )

    # Set the y-axis limit for the second plot
    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force["Stress [MPa]"]) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)

    # Set y-axis label only for the last plot of each row
    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel("")

    # Combine legends
    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    # Legend in the first plot
    if i == 1:
        # Create a new legend with combined handles and legends
        ax2.legend(handles, legends, loc="upper left")

        # Remove the separate legend from ax if it exists
        ax.get_legend().remove()
    else:
        ax.get_legend().remove()
        ax2.get_legend().remove()

    # Plot the grid
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        fig_format = figure_path.name.split(".")[-1]
        plt.savefig(figure_path, format=fig_format, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()  # Close the plot to free up memory

    return ax


def plot_stress_hits(
    hits,
    force,
    test_id,
    figure_path,
    plot_type="scatter",
    title="Stress and Cumulative hits vs Time",
    x="time",
    y="Cumulative_Label",
    x_label="Time [s]",
    y_label="Cumulative hits",
    y_label_right="Stress [MPa]",
    subtitle=None,
    limits=None,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    n_row=1,
    save=False,
):
    """
    Parameters
    ----------
    hits : Dataframe
        Set containing the characteristics over time of the hits and their id.
    force : Dataframe
        Set containing load displacement, time data, and their id.
    test_id : str
        Id of the test to be plotted.
    figure_path : str
        Directory to save the figure.
    plot_type : str, optional
        Type of plot to create ('scatter' or 'line'). The default is 'scatter'.
    title : str, optional
        Title of the image to display and save. The default is 'Stress and Cumulative hits vs Time'.
    x : str, optional
        Time characteristic for the x-axis. The default is 'time'.

        Other options: 'Cumulative_Label', 'Cumulative', 'amplitude', 'energy'
    y : str, optional
        Time characteristic for the y-axis. The default is 'Cumulative_Label'.

        Other options: 'time', 'Cumulative', 'amplitude', 'energy'
    x_label : str, optional
        Label for the x-axis. The default is 'Time [s]'.
    y_label : str, optional
        Label for the y-axis. The default is 'Cumulative hits'.
    y_label_right : str, optional
        Label for the right y-axis. The default is 'Stress [MPa]'.
    subtitle : str or list, optional
        Title for each subplot, if left as default the test_id is plotted. The default is None.
    limits : list, optional
        Upper limits for the axes [time_limit, hits_limit, stress_limit]. The default is None.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    ax : matplotlib axis object, optional
        Axis to plot on. If not provided, a new axis will be created.
    i : int, optional
        Index for subplot. Default is 1.
    n_col : int, optional
        Number of columns. The default is 1.
    n_row : int, optional
        Number of rows. The default is 1.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. Cluster vs Time {test_id}.pdf

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Plots and (saves) a diagram of cumulative hits and stress in pdf format.
    """
    # Extract the limits
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    # Condition to filter rows based on the string column
    condition = hits["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    # Filtered DataFrame
    filtered_hits = hits.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = filtered_hits["Count"].cumsum()

    # Force data
    filtered_force = force[condition_force]

    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_title(f"Stress and Cumulative hits vs Time - {test_id}")
        ax.set_xlabel(x_label)

    else:
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(test_id)

    # Create the scatter plot
    if plot_type == "scatter":
        sns.scatterplot(
            x=x, y=y, data=filtered_hits, label="hits", linewidth=0.2, s=20, ax=ax
        )
    elif plot_type == "line":
        sns.lineplot(x=x, y=y, label="hits", data=filtered_hits, linewidth=2, ax=ax)

    # Set the x-axis limit
    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    # Set the y-axis limit
    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)

    # Create a twin Axes sharing the x-axis
    ax2 = ax.twinx()

    # Create the second scatter plot on the twin Axes
    sns.lineplot(
        x="Time [s]",
        y="Stress [MPa]",
        data=filtered_force,
        ax=ax2,
        linewidth=1,
        color="black",
        label="Stress",
    )

    # Set the y-axis limit for the second plot
    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force["Stress [MPa]"]) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)

    # Set y-axis label only for the last plot of each row
    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel("")

    # Combine legends
    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    # Legend in the first plot
    if i == 1:
        # Create a new legend with combined handles and legends
        ax2.legend(handles, legends, loc="upper left")

        # Remove the separate legend from ax if it exists
        ax.get_legend().remove()
    else:
        ax.get_legend().remove()
        ax2.get_legend().remove()

    # Plot the grid
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        if ax is None:
            figure_filename = f"{title} - {test_id}.pdf"
        else:
            figure_filename = f"{title}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    return ax


def plot_feature_time_histogram(
    data,
    figure_path,
    title="Feature distribution vs Normalized time",
    x="time_norm",
    y="peak_freq",
    x_label="Normalized time",
    y_label="Peak frequency [kHz]",
    subtitle=None,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    save=False,
):
    """
    Parameters
    ----------
    data : DataFrame
        Data containing the features to plot, must include columns for x and y.
    figure_path : str
        Directory to save the figure.
    title : str, optional
        Title of the image to display and save. Default is 'Feature distribution vs Normalized time'.
    x : str, optional
        Column name for the x-axis. Default is 'time_norm'.
    y : str, optional
        Column name for the y-axis. Default is 'peak_freq'.
    x_label : str, optional
        Label for the x-axis. Default is 'Normalized time'.
    y_label : str, optional
        Label for the y-axis. Default is 'Peak frequency [kHz]'.
    subtitle : str or list, optional
        Title for each subplot, if left as default the y variable is plotted. Default is None.
    width : int or float, optional
        Width of the figure in millimeters. Default is 90.
    height : int or float, optional
        Height of the figure in millimeters. Default is 60.
    ax : matplotlib axis object, optional
        Axis to plot on. If not provided, a new axis will be created.
    i : int, optional
        Index for subplot. Default is 1.
    n_col : int, optional
        Number of columns. Default is 1.
    save : boolean, optional
        If True, saves the image. Default is False.

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Plots and (optionally saves) a 2D histogram of the feature distribution vs normalized time.
    """

    # Create the plot
    # Exclude the data in the fracture moment
    data = data[data["time_norm"] < 0.99]

    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_title(f"{y} vs Normalized time")
        ax.set_xlabel(x_label)

    else:
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(y)

    # Create the plot
    sns.histplot(
        data=data,
        x=x,
        y=y,
        bins=60,
        pmax=0.95,
        cmap="rainbow",
        cbar=True,
        ax=ax,
    )

    ax.set_ylabel(y_label)

    # Plot the grid
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        if ax is None:
            figure_filename = f"{title} - {y}.pdf"
        else:
            figure_filename = f"{title}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    return ax


def plot_feature_histogram(
    data,
    figure_path,
    title="Feature distribution vs Normalized time",
    x="peak_freq",
    x_label="Peak frequency [kHz]",
    subtitle=None,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    save=False,
):
    """
    Parameters
    ----------
    data : DataFrame
        Data containing the features to plot, must include columns for x and y.
    figure_path : str
        Directory to save the figure.
    title : str, optional
        Title of the image to display and save. Default is 'Feature distribution vs Normalized time'.
    x : str, optional
        Column name for the x-axis. Default is 'time_norm'.
    y : str, optional
        Column name for the y-axis. Default is 'peak_freq'.
    x_label : str, optional
        Label for the x-axis. Default is 'Normalized time'.
    y_label : str, optional
        Label for the y-axis. Default is 'Peak frequency [kHz]'.
    subtitle : str or list, optional
        Title for each subplot, if left as default the y variable is plotted. Default is None.
    width : int or float, optional
        Width of the figure in millimeters. Default is 90.
    height : int or float, optional
        Height of the figure in millimeters. Default is 60.
    ax : matplotlib axis object, optional
        Axis to plot on. If not provided, a new axis will be created.
    i : int, optional
        Index for subplot. Default is 1.
    n_col : int, optional
        Number of columns. Default is 1.
    save : boolean, optional
        If True, saves the image. Default is False.

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Plots and (optionally saves) a 2D histogram of the feature distribution vs normalized time.
    """

    # Create the plot
    # Exclude the data in the fracture moment

    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_title(f"{x} Histogram")
        ax.set_xlabel(x_label)

    else:
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(x)

    # Create the plot
    sns.histplot(
        data=data,
        x=x,
        bins=40,
        pmax=0.95,
        ax=ax,
    )

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        ax.set_ylabel("Event count")
    else:
        ax.set_ylabel("")

    # Plot the grid
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        if ax is None:
            figure_filename = f"{title} - {x}.pdf"
        else:
            figure_filename = f"{title}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    return ax


def limit_finder(data, force, test_ids=None):
    """
    Parameters
    ----------
    data : Dataframe
        Dataframe containing the labeled data and test ids.
    force : Dataframe
        Dataframe containing load displacement, time data, and their id.
    test_ids : list
        Ids of the tests considered for the calculation.

    Returns
    -------
    max_value_hits : float
        Maximum value for hits.
    max_value_stress : float
        Maximum value for stress.

    """

    # Condition to filter rows based on the string column
    if test_ids is None:
        test_ids = data["test_id"].unique()
    condition = data["test_id"].isin(test_ids)
    condition_force = force["test_id"].isin(test_ids)

    # Filtered DataFrame
    filtered_hits = data.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = None

    for test_id in test_ids:
        single_condition = filtered_hits["test_id"] == test_id
        filtered_hits.loc[single_condition, "Cumulative_Label"] = (
            filtered_hits[single_condition].groupby("label")["Count"].cumsum()
        )

    # Force data
    filtered_force = force[condition_force]

    # Maximum values of the df
    max_value_time = filtered_hits["time"].max()
    max_value_hits = filtered_hits["Cumulative_Label"].max()
    max_value_stress = filtered_force["Stress [MPa]"].max()

    return [max_value_time, max_value_hits, max_value_stress]


def limit_finder_no_label(hits, force, test_ids):
    """
    Parameters
    ----------
    hits : Dataframe
        Dataframe containing the characteristics over time of the hits and their id.
    force : Dataframe
        Dataframe containing load displacement, time data, and their id.
    test_ids : list
        Ids of the tests considered for the calculation.

    Returns
    -------
    max_value_hits : float
        Maximum value for hits.
    max_value_stress : float
        Maximum value for stress.

    """
    # Condition to filter rows based on the string column
    condition = hits["test_id"].isin(test_ids)
    condition_force = force["test_id"].isin(test_ids)

    # Filtered DataFrame
    filtered_hits = hits.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = None

    for test_id in test_ids:
        single_condition = filtered_hits["test_id"] == test_id
        filtered_hits.loc[single_condition, "Cumulative_Label"] = filtered_hits[
            single_condition
        ]["Count"].cumsum()

    # Force data
    filtered_force = force[condition_force]

    # Maximum values of the df
    max_value_time = filtered_hits["time"].max()
    max_value_hits = filtered_hits["Cumulative_Label"].max()
    max_value_stress = filtered_force["Stress [MPa]"].max()

    return [max_value_time, max_value_hits, max_value_stress]


def reorder_hits(hits, y, y_pred):
    """
    Parameters
    ----------
    hits : Dataframe
        Dataframe for plotting cumulative hits.
    y : series or array
        Classes.
    y_pred : series or array
        Predictions.

    Returns
    -------
    hits_ordered : Dataframe
        Hits ordered by ascending time.
    y_pred_ordered : series
        Predictions ordered by ascending time.

    """
    # Create a DataFrame with the predictions
    y_pred_df = pd.Series(y_pred, name="Class")

    # Select only the hits from the training or test set
    hits = hits.loc[y.index]
    hits = hits.reset_index(drop=True)

    # Remove the class column
    hits = hits.drop("Class", axis=1)

    # Concatenate hits and predictions dataframes
    hits_and_pred = pd.concat([hits, y_pred_df], axis=1)

    # Sort dataframe based on the time column
    hits_and_pred = hits_and_pred.sort_values(by="time", ascending=True)

    # Separate hits and predictions
    hits_ordered = hits_and_pred.drop("Class", axis=1)
    y_pred_ordered = hits_and_pred["Class"].copy()

    return hits_ordered, y_pred_ordered

```

## File: `utils\preprocess.py`
```python
from pathlib import Path

import numpy as np
import numpy.typing as npt
import vallenae as vae
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, hilbert


def _find_preceding_valley(
    smooth_env: npt.NDArray[np.float64],
    peak_idx: int,
):
    """
    Walk backward from a peak index to locate the nearest local minimum (valley).

    Parameters
    ----------
    smooth_env : npt.NDArray[np.float64]
        1D smoothed Hilbert envelope array [mV].
    peak_idx : int
        Sample index of the detected burst maximum.

    Returns
    -------
    int
        Sample index of the closest local minimum preceding `peak_idx`.
    """
    idx = peak_idx
    while idx > 0 and smooth_env[idx - 1] <= smooth_env[idx]:
        idx -= 1
    return idx


def _refine_arrival_onset(
    waveform: npt.NDArray[np.floating],
    smooth_env: npt.NDArray[np.float64],
    valley_idx: int,
    peak_idx: int,
    rise_fraction: float,
):
    """
    Pinpoint the sharp arrival onset (t0) of a burst between a valley and its peak.

    Parameters
    ----------
    waveform : npt.NDArray[np.floating]
        1D raw transient voltage signal [mV].
    smooth_env : npt.NDArray[np.float64]
        1D smoothed Hilbert envelope array [mV].
    valley_idx : int
        Sample index of the valley preceding the burst.
    peak_idx : int
        Sample index of the burst maximum.
    rise_fraction : float
        Fraction (0 to 1) of the peak-above-valley jump used to define t0.

    Returns
    -------
    int
        Refined arrival sample index (t0) for the isolated burst.
    """
    search_slice = np.abs(waveform[valley_idx : peak_idx + 1])
    if search_slice.size == 0:
        return valley_idx

    baseline_amp = float(smooth_env[valley_idx])
    peak_raw_amp = float(np.max(search_slice))
    onset_level = baseline_amp + rise_fraction * (peak_raw_amp - baseline_amp)

    above_onset = np.flatnonzero(search_slice >= onset_level)
    onset_offset = int(above_onset[0]) if above_onset.size > 0 else 0
    return valley_idx + onset_offset


def segment_merged_hits(
    waveform: npt.NDArray[np.floating],
    time: npt.NDArray[np.floating],
    fs: float = 5e6,
    threshold_mv: float = 0.178,
    pre_trigger_us: float = 50.0,
    min_separation_us: float = 150.0,
    smoothing_us: float = 100.0,
    valley_decay_ratio: float = 0.35,
    onset_rise_fraction: float = 0.10,
):
    """
    Segment a transient acoustic emission (AE) waveform into individual sub-hits
    when two or more events merge within a single acquisition window.

    Handles arbitrary relative amplitudes (weak-to-strong, strong-to-weak,
    equal-amplitude bursts, and cascades of three or more events) while
    suppressing false splits caused by internal modal reflections.

    Parameters
    ----------
    waveform : npt.NDArray[np.floating]
        1D array containing the digitized transient voltage signal [mV].
    fs : float, default=5e6
        Sampling frequency of the transient recorder [Hz] (e.g., 5 MHz).
    threshold_mv : float, default=0.178
        Minimum envelope peak amplitude [mV] required for a burst to be treated
        as a valid event (0.178 mV corresponds to 45 dBAE ref. 1 uV).
    pre_trigger_us : float, default=50.0
        Configured hardware pre-trigger duration [us] used to identify delayed
        first arrivals caused by sub-threshold precursors.
    min_separation_us : float, default=150.0
        Minimum temporal separation [us] between consecutive burst maxima to
        prevent over-segmenting internal wave reflections.
    smoothing_us : float, default=25.0
        Moving-average window length [us] applied to the Hilbert envelope to
        remove carrier oscillations while preserving burst boundaries.
    valley_decay_ratio : float, default=0.35
        Maximum allowed ratio between the intermediate valley amplitude and the
        smaller adjacent peak: `valley <= valley_decay_ratio * min(P_i, P_{i+1})`.
    onset_rise_fraction : float, default=0.10
        Fraction of the local peak-above-valley amplitude used to locate the
        sharp arrival index (t0) of each separated burst.

    Returns
    -------
    list[tuple[int, int]]
        Ordered list of `(start_idx, end_idx)` sample index pairs defining each
        separated sub-hit segment within `waveform`.
    """
    n_samples = waveform.size
    if n_samples == 0:
        return []

    dt_us = 1e6 / fs
    min_dist_samples = max(1, round(min_separation_us / dt_us))
    smooth_samples = max(1, round(smoothing_us / dt_us))

    
    # 1. Compute analytic envelope via Hilbert transform and smooth carrier cycles
    analytic_env = np.abs(hilbert(waveform))
    kernel = np.ones(smooth_samples, dtype=np.float64) / smooth_samples
    smooth_env = np.convolve(analytic_env, kernel, mode="same")
    # plot_signal(smooth_env, time)
    # 2. Detect all candidate burst maxima exceeding the digital threshold
    candidate_peaks, _ = find_peaks(
        smooth_env,
        height=threshold_mv,
        distance=min_dist_samples,
    )

    if candidate_peaks.size == 0:
        return [(0, n_samples)]

    # 3. Validate true multi-hit boundaries via intermediate valley decay
    valid_peaks = [int(candidate_peaks[0])]
    split_valleys = []

    for raw_peak in candidate_peaks[1:]:
        curr_peak = int(raw_peak)
        prev_peak = valid_peaks[-1]

        # Locate the deepest envelope minimum between consecutive peaks
        valley_rel_idx = int(np.argmin(smooth_env[prev_peak:curr_peak]))
        valley_idx = prev_peak + valley_rel_idx
        valley_amp = float(smooth_env[valley_idx])
        smaller_peak_amp = float(min(smooth_env[prev_peak], smooth_env[curr_peak]))

        # Split only if the preceding burst decayed sufficiently below the smaller peak
        if valley_amp <= valley_decay_ratio * smaller_peak_amp:
            valid_peaks.append(curr_peak)
            split_valleys.append(valley_idx)
        else:
            # Keep the dominant peak when an internal reflection is rejected
            if smooth_env[curr_peak] > smooth_env[prev_peak]:
                valid_peaks[-1] = curr_peak

    # 4. Determine start index of the first valid burst
    # Re-align t0 if a sub-threshold noise spike opened the window early
    first_peak = valid_peaks[0]
    if first_peak * dt_us > (pre_trigger_us + min_separation_us):
        initial_valley = _find_preceding_valley(smooth_env, first_peak)
        current_start = _refine_arrival_onset(
            waveform, smooth_env, initial_valley, first_peak, onset_rise_fraction
        )
    else:
        current_start = 0

    if not split_valleys:
        return [(current_start, n_samples)]

    # 5. Construct (start_idx, end_idx) intervals for all N separated sub-hits
    segments = []
    for idx, valley_idx in enumerate(split_valleys):
        # End current sub-hit at the intermediate valley
        segments.append((current_start, valley_idx))

        # Re-align arrival onset (t0) for the next burst after the valley
        next_peak = valid_peaks[idx + 1]
        current_start = _refine_arrival_onset(
            waveform, smooth_env, valley_idx, next_peak, onset_rise_fraction
        )

    # Append final sub-hit through the end of the digitized record
    segments.append((current_start, n_samples))

    # Plot each separated sub-hit
    # for segment_idx, (start_idx, end_idx) in enumerate(segments, start=1):
    #     plot_signal(
    #         waveform[start_idx:end_idx],
    #         time[start_idx:end_idx],
    #         y_label=f"Segment {segment_idx} [mV]",
    #     )

    # Plot all separated sub-hits together for direct comparison
    plt.figure(figsize=(90 / 25.4, 60 / 25.4), dpi=300, tight_layout=True)

    for segment_idx, (start_idx, end_idx) in enumerate(segments, start=1):
        segment_time = np.copy(time[start_idx:end_idx]) * 1e6
        segment_amp = np.copy(waveform[start_idx:end_idx]) * 1e3
        plt.plot(
            segment_time,
            segment_amp,
            linewidth=0.6,
            label=f"Segment {segment_idx}",
        )

    plt.xlabel("Time [µs]")
    plt.ylabel("Amplitude [mV]")
    plt.grid(linewidth=0.3, color=".25")
    plt.title("Separated Sub-hits")
    plt.legend()
    plt.show()

    return segments

def plot_signal(
        amp: npt.NDArray[np.floating],
        time: npt.NDArray[np.floating],
        time_graph=None,
        x_label="Time [µs]",
        y_label="Amplitude [mV]",
        width=90,
        height=60,
    ):
        """
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

        """

        # Plot the transient
        copy_amp = np.copy(amp)
        copy_time = np.copy(time)
        copy_amp *= 1e3  # in mV   # The operator *= is updating the amplitude value -- amp = amp * 1e3
        # The signal was in volts and is represented in mV
        # This is the amplitude
        copy_time = (
            copy_time * 1e6
        )  # for µs  # Similarly, the signal is stored in seconds and represented in µs
        # This is the time

        figsize_inches = (width / 25.4, height / 25.4)
        plt.figure(figsize=figsize_inches, dpi=300, tight_layout=True)

        if time_graph is None:
            plt.plot(copy_time, copy_amp, linewidth=0.6)
        else:
            # points = int(time_graph * self.sampling_rate + 1)
            # plt.plot(
            #     time[self.desfase : points], amp[self.desfase : points], linewidth=0.6
            # )
            # plt.ylim(
            #     bottom=min(amp[self.desfase : points]) * 1.6,
            #     top=max(amp[self.desfase : points]) * 1.6,
            # )
            pass

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(linewidth=0.3, color=".25")
        plt.title("Transient Wave Plot") 
        plt.show()

# Example usage inside your waveform processing loop:
if __name__ == "__main__":
    # Example waveform
    tradb_path = Path("D:\\ByronGerman\\profile\\Documents\\Datasets\\Arcan\\FV_mod1_1\\FV_mod1_1.tradb")
    tradb = vae.io.TraDatabase(tradb_path)
    # trai = 6740
    trai = 6848
    trai = 7006
    amp, time = tradb.read_wave(trai, time_axis=True)
    # plot_signal(amp, time)
    
    sub_hit_bounds = segment_merged_hits(
        waveform=amp,
        time=time,
        fs=5e6,
        smoothing_us=50.0,
        threshold_mv=0.000178,       # 45 dBAE threshold
        pre_trigger_us=50.0,      # Matches LinWave pre-trigger setting
        min_separation_us=250.0,  # Minimum gap between distinct damage bursts
        valley_decay_ratio=0.3,  # Requires >= 80% envelope decay relative to smaller peak
    )
    
    
    print("Final sub-hit boundaries:")

# for start_idx, end_idx in sub_hit_bounds:
#     sub_waveform: npt.NDArray[np.floating] = raw_waveform[start_idx:end_idx]
#     # Apply your Counts > 5, peak < 94 dBAE, and Hamming window feature extraction here
```

## File: `utils\unsupervised_functions.py`
```python
"""
Created on Mon Dec 11 10:54:49 2023

@author: bbarmac
"""
# =============================================================================
# Import necessary libraries
# =============================================================================

# Main libraries
import os
import string

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Library for silhouette plot
from matplotlib.ticker import FixedFormatter, FixedLocator

# Unsupervised algorithm libraries
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE

# Metric libraries
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score

# Set font to Times New Roman and size to 8
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["figure.dpi"] = 150


MARKER_LIST = ["o", "s", "^", "v", "D", "p"]  # Customize the markers as desired

# =============================================================================
# DBSCAN
# =============================================================================


def grid_search_dbscan(X, target_clusters, epsilon_values, min_samples_values):
    """
    Parameters
    ----------
    X : DataFrame, array
        Feature set for training.
    target_clusters : list
        Target number of clusters.
    epsilon_values : array or list
        Values that epsilon can take for clustering.
    min_samples_values : array or list
        Values that min_samples can take for clustering.

    Returns
    -------
    best_models : list
        List containing the DBSCAN models that meet the target clusters condition.
    results : list
        List containing the values of epsilon, min_samples, number of clusters,
        number and percentage of anomalies for each DBSCAN model.

    """
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
                num_clusters = (
                    len(unique_labels) - 1
                )  # Subtract 1 for the noise cluster
            else:
                num_clusters = len(unique_labels)

            # Check if the current model meets the target_clusters
            if num_clusters in target_clusters:
                # if num_clusters > 1:
                results.append(
                    {
                        "Epsilon": epsilon,
                        "min_samples": min_samples,
                        "Number of Clusters": num_clusters,
                        "Number of Anomalies": np.sum(labels == -1),
                        "Percentage of Anomalies": (
                            np.sum(labels == -1) / total_samples
                        )
                        * 100,
                        "Labels Distribution": {
                            int(label): int(np.sum(labels == label))
                            for label in np.unique(labels)
                            if label != -1
                        },
                        "Silhouette Score": silhouette_score(X, labels),
                        "DBI Score": davies_bouldin_score(X, labels),
                    }
                )
                best_models.append(dbscan_model)
            else:
                print(
                    f"Model with eps={epsilon}, min_samples={min_samples} does not meet the target clusters condition."
                    f"Found {num_clusters} clusters instead of {target_clusters}."
                )

    # Print the results
    print(
        f"\nDBSCAN models that meet the target clusters condition ({target_clusters} clusters):\n"
    )
    for i, result in enumerate(results):
        for key, value in result.items():
            if key == "Labels Distribution":
                print(f"{key}:")
                for label, count in value.items():
                    print(f"    {label}: {count}")
            else:
                print(f"{key}: {value}")
        print(f"Index: {i}")
        print("-----")

    return best_models, results


def kmeans_per_k(X, k, selected_features=None):
    """
    Parameters
    ----------
    X : DataFrame, array
        Feature set for training.
    k : int
        Maximum number of clusters to evaluate.
    selected_features : list, optional
        Names of the features to train the model. The default is None.

    Returns
    -------
    kmeans_per_k : List
        List containing the k-means models trained from 2 to k.
    silhouette_scores : List
        Silhouette score values for each model.
    dbi_scores : List
        Davies-Bouldin score values for each model.

    """
    # Feature selection
    if selected_features is not None:
        X = X[selected_features]

    kmeans_per_k = [
        KMeans(n_clusters=i, random_state=42).fit(X) for i in range(2, k + 1)
    ]

    # Calculate scores for each k
    silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k]

    dbi_scores = [davies_bouldin_score(X, model.labels_) for model in kmeans_per_k]

    # Print silhouette scores for each model
    for i, score in enumerate(silhouette_scores, start=2):
        print(f"Model index: {i - 2}, k={i}, Silhouette Score: {score:.4f}")

    for i, score in enumerate(dbi_scores, start=2):
        print(f"Model index: {i - 2}, k={i}, Davies-Bouldin Score: {score:.2f}")

    return kmeans_per_k, silhouette_scores, dbi_scores


def tsne(
    X: pd.DataFrame | np.ndarray,
    figure_path: str = "t-SNE Visualization.pdf",
    width: float = 90,
    height: float = 60,
    selected_features: list | None = None,
    save: bool = False,
    show: bool = False,
):
    """
    Parameters
    ----------
    X : DataFrame, array
        Feature set for training.
    figure_path : str
        Directory to save the figure.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    selected_features : list, optional
        Names of the features to train the model. The default is None.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. t-SNE Visualization.pdf
    Returns
    -------
    X_reduced : Array
        Array with 2 columns containing the dimensionality reduction for clustering training.

    Plots and (saves) a scatter plot of the data with dimensionality reduction in PDF format.

    """
    print("Running t-SNE...")
    # Feature selection (if backward elimination was used)
    if selected_features is not None:
        X = X[selected_features]

    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X)

    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f"t-SNE_{i + 1}" for i in range(2)])

    # Create a scatter plot with Seaborn
    figsize_inches = (width / 25.4, height / 25.4)
    plt.figure(figsize=figsize_inches, tight_layout=True)
    sns.scatterplot(x="t-SNE_1", y="t-SNE_2", data=tsne_df, palette="viridis")
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid()
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        fig_format = figure_path.name.split(".")[-1]
        plt.savefig(figure_path, format=fig_format, bbox_inches="tight")
    if show:
        plt.show()

    return X_reduced


def plot_kmeans_per_k(
    X,
    kmeans_per_k,
    silhouette_scores,
    figure_path="Silhouette Analysis for K-Means Clustering.pdf",
    save=False,
    show=False,
):
    """
    Parameters
    ----------
    X : DataFrame, array
        Feature set for training.
    kmeans_per_k : List
        List of k-means models.
    silhouette_scores : List
        Silhouette score values for each model.
    figure_path : str
        Directory to save the figure.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. Silhouette Analysis for K-Means Clustering.pdf
    Returns
    -------
    Plots and (saves) the silhouette score graphs in PDF format.

    """

    n_plots = len(kmeans_per_k)
    if n_plots == 0:
        raise ValueError("kmeans_per_k must contain at least one model.")

    ncols = min(3, int(np.ceil(np.sqrt(n_plots))))
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    # Add a super title
    fig.suptitle("Silhouette Analysis for K-Means Clustering", fontsize=16)

    for plot_index, (model, score) in enumerate(zip(kmeans_per_k, silhouette_scores)):
        ax = axes[plot_index]
        labels = np.asarray(model.labels_)
        cluster_labels = np.unique(labels)
        n_clusters = len(cluster_labels)

        silhouette_coefficients = silhouette_samples(X, labels)

        padding = max(1, len(X) // 30)
        pos = padding
        ticks = []
        for cluster_label in cluster_labels:
            coeffs = silhouette_coefficients[labels == cluster_label]
            coeffs.sort()

            color = plt.cm.nipy_spectral(float(len(ticks)) / max(1, n_clusters))
            ax.fill_betweenx(
                np.arange(pos, pos + len(coeffs)),
                0,
                coeffs,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(cluster_labels))
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        row, col = divmod(plot_index, ncols)
        if col == 0:
            ax.set_ylabel("Cluster")

        if row == nrows - 1:
            ax.set_xlabel("Silhouette Coefficient")

        ax.axvline(x=score, color="red", linestyle="--")
        ax.set_title(f"$k={n_clusters}$")
        # Activate grid
        ax.grid(axis="x", linestyle="--", which="major", color="grey", alpha=0.5)
        ax.set_axisbelow(True)  # Set grid lines behind the data points

    for ax in axes[n_plots:]:
        ax.remove()

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    # Save the image in figure_path
    if save:
        fig_format = figure_path.name.split(".")[-1]
        fig.savefig(figure_path, format=fig_format, bbox_inches="tight")
    if show:
        plt.show()


def plot_silhouette(
    labels, X, silhouette_score, figure_path, width=90, height=60, save=False
):
    """
    Parameters
    ----------
    labels : array
        Labels of the clusters obtained by k-means or dbscan.
    X : DataFrame, array
        Feature set for training.
    silhouette_score : List
        Silhouette score values for each model.
    figure_path : str
        Directory to save the figure.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90

    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. Silhouette Analysis for K-Means Clustering.pdf
    Returns
    -------
    Plots and (saves) the silhouette score graphs in PDF format.

    """

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

        color = plt.cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(
            np.arange(pos, pos + len(coeffs)),
            0,
            coeffs,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.ylabel("Cluster")
    plt.xlabel("Silhouette Coefficient")

    plt.axvline(x=silhouette_score, color="red", linestyle="--")
    plt.title("Silhouette Analysis")
    # Activate grid
    plt.grid(axis="x", linestyle="--", which="major", color="grey", alpha=0.5)
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        figure_filename = "Silhouette Analysis.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    plt.show()


def tsne_3d(X, figure_path, width=180, height=120, selected_features=None, save=False):
    """
    Parameters
    ----------
    X : DataFrame, array
        Feature set for training.
    figure_path : str
        Directory to save the figure.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    selected_features : list, optional
        Names of the features to train the model. The default is None.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. t-SNE 3D Visualization.pdf
    Returns
    -------
    X_reduced : Array
        Array with 3 columns containing the dimensionality reduction for clustering training.

    Plots and (saves) a scatter plot of the data with dimensionality reduction in PDF format.

    """
    print("Running t-SNE in 3D...")
    # Feature selection (if backward elimination was used)
    if selected_features is not None:
        X = X[selected_features]

    tsne = TSNE(n_components=3, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X)

    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f"t-SNE_{i + 1}" for i in range(3)])

    # Create a 3D scatter plot
    figsize_inches = (width / 25.4, height / 25.4)
    fig = plt.figure(figsize=figsize_inches, tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        tsne_df["t-SNE_1"],
        tsne_df["t-SNE_2"],
        tsne_df["t-SNE_3"],
        c=tsne_df.index,
        cmap="viridis",
    )
    ax.set_title("t-SNE 3D Visualization")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

    # Save the image in figure_path
    if save:
        figure_filename = "t-SNE 3D Visualization.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    plt.show()

    return X_reduced


def plot_cluster_feat(
    labels,
    X,
    feat_x,
    feat_y,
    figure_path="Feat_vs_Feat_clustering.pdf",
    title=None,
    subtitle=None,
    x_label=None,
    y_label=None,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    n_row=1,
    save=False,
    show=False,
):
    """
    Parameters
    ----------
    labels : array
        Labels of the clusters obtained by k-means or dbscan.
    X : DataFrame, array
        Feature set for training.
    feat_x : str
        Name of feature 1 to plot.
    feat_y : str
        Name of feature 2 to plot.
    figure_path : str
        Directory to save the figure.
    title : str or list, optional
        Title of the figure. If left as default, it will be {feat_1} vs {feat_2} -
        Clustering. The default is None
    subtitle : str or list, optional
        Subtitle of the figure. If left as default, it will be {feat_1} vs {feat_2} -
        Clustering. The default is None
    x_label : str, optional
        Label for the x-axis. The default is None
    y_label : str, optional
        Label for the y-axis. The default is None
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    ax : axes, optional
        Axes subplot. The default is None.
    i : int, optional
        Counter for plotting each subplot. The default is 1.
    n_col : int, optional
        Number of columns. The default is 1.
    n_row : int, optional
        Number of rows. The default is 1.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. Feat_vs_Feat_{feat_1}_{feat_2} - Clustering.pdf
    Returns
    -------
    Plots and (saves) a scatter plot of two features with labels and markers in PDF format.

    """
    labels = pd.Series(labels, name="Labels")
    # Rename 'Labels' column in y if it already exists in X
    if isinstance(X, pd.DataFrame):
        labels.index = X.index
        if "Labels" in X.columns:
            X = X.drop(["Labels"], axis=1)

    data = pd.concat([X, labels], axis=1)

    # Masks to change markers
    valid_points = data[data["Labels"] != -1]
    anomalies = data[data["Labels"] == -1]

    # Count the number of unique labels in the DataFrame
    num_labels = valid_points["Labels"].nunique()

    # Define custom markers for each class
    palette = sns.color_palette()
    # Customize the markers list to match the number of unique labels
    markers = MARKER_LIST[:num_labels]
    palette = palette[:num_labels]

    # Create a figure
    figsize_inches = (width / 25.4, height / 25.4)

    # If there is just one plot and set the title
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        if title is None:
            ax.set_title(f"{feat_x} vs {feat_y} - Clustering")
        else:
            ax.set_title(title)
        if x_label is None:
            ax.set_xlabel(feat_x)
        else:
            ax.set_xlabel(x_label)

    else:
        if x_label is None:
            ax.set_xlabel(f"{feat_x}\n(" + string.ascii_lowercase[i - 1] + ")")
        else:
            ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(f"{feat_x} vs {feat_y}")

    # Use Seaborn scatter plot with hue and style parameters
    sns.scatterplot(
        x=valid_points[feat_x],
        y=valid_points[feat_y],
        hue="Labels",
        style="Labels",
        data=valid_points,
        palette=palette,
        markers=markers,
        ax=ax,
    )

    # Scatter points with label '-1' separately as red 'x'
    if not anomalies.empty:
        sns.scatterplot(
            x=anomalies[feat_x],
            y=anomalies[feat_y],
            hue="Labels",
            style="Labels",
            data=anomalies,
            palette=["r"],
            markers="x",
            linewidth=1.5,
            ax=ax,
        )

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        if y_label is None:
            ax.set_ylabel(feat_y)
        else:
            ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    # Logarithmic scale if you choose energy
    if feat_x == "energy":
        ax.set_xscale("log")

    if feat_y == "energy":
        ax.set_yscale("log")

    # Activate the grid
    ax.grid()
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        fig_format = figure_path.name.split(".")[-1]
        plt.savefig(figure_path, format=fig_format, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return ax


def plot_cluster_time(
    labels, hits, test_id, figure_path, width=90, height=60, save=False
):
    """
    Parameters
    ----------
    labels : array
        Labels of the clusters obtaied by k-means or dbscan.
    hits : DataFrame
        DataFrame containing the features over time of the hits and their id.
    test_id : str
        Id of the test to be plotted.
    figure_path : str
        Directory to save the figure.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. Cluster vs Time - {test_id}.pdf

    Returns
    -------
    Plots and (saves) a scatter plot of cluster labels vs time in PDF format.

    """
    # Create a DataFrame with the labels
    labels_df = pd.Series(labels, name="Labels")
    labels_df.index = hits.index

    # Concatenate the new DataFrame with the existing one
    hits_labels = pd.concat([hits, labels_df], axis=1)

    # Condition to filter rows based on the string column
    condition = hits_labels["test_id"] == test_id

    # Filtered DataFrame
    filtered_hits = hits_labels[condition]

    # Define custom markers for each class
    markers = ["o", "s", "^", "v", "D", "p"]  # Customize the markers as desired

    # Plotting with Seaborn scatter plot
    figsize_inches = (width / 25.4, height / 25.4)
    plt.figure(figsize=figsize_inches, tight_layout=True)
    sns.scatterplot(
        x="time",
        y="Labels",
        hue="Labels",
        style="Labels",
        data=filtered_hits,
        markers=markers,
        palette=sns.color_palette(),
        s=100,
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Labels")
    plt.title(f"Cluster vs Time - {test_id}")
    plt.grid()
    plt.grid(axis="y", linestyle="--", which="major", color="grey", alpha=0.5)
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points

    # Set y-axis ticks to display only integer values
    plt.yticks(
        range(
            int(filtered_hits["Labels"].min()), int(filtered_hits["Labels"].max()) + 1
        )
    )

    # Save the image in figure_path
    if save:
        figure_filename = f"Cluster vs Time - {test_id}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    plt.show()


def plot_cluster_hits(
    labels,
    hits,
    test_id,
    figure_path,
    width=90,
    height=60,
    plot_type="scatter",
    x="time",
    y="Cumulative_Label",
    x_label="Time [s]",
    y_label="Cumulative hits",
    title="Cumulative hits vs Time",
    save=False,
):
    """
    Parameters
    ----------
    labels : array
        Labels of the clusters obtained by k-means or dbscan.
    hits : DataFrame
        DataFrame containing the features over time of the hits and their id.
    test_id : str
        Id of the test to be plotted.
    figure_path : str
        Directory to save the figure.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    plot_type : str, optional
        Type of plot to create ('scatter' or 'line'). The default is 'scatter'.
    x : str, optional
        Feature over time for the x-axis. The default is 'time'.

        Other options: 'Cumulative_Label', 'Cumulative', 'amplitude', 'energy'
    y : str, optional
        Feature over time for the y-axis. The default is 'Cumulative_Label'.

        Other options: 'time', 'Cumulative', 'amplitude', 'energy'
    x_label : str, optional
        Label for the x-axis. The default is 'Time [s]'.
    y_label : str, optional
        Label for the y-axis. The default is 'Cumulative hits'.
    title : str, optional
        Title of the image to display and save. The default is 'Cumulative hits vs Time'.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. Cluster vs Time {test_id}.pdf

    Returns
    -------
    Plots and (saves) a scatter plot of temporal features with labels and markers in PDF format.

    """

    # Create a DataFrame with the labels
    labels_df = pd.Series(labels, name="Labels")
    labels_df.index = hits.index

    # Concatenate the new DataFrame with the existing one
    hits_labels = pd.concat([hits, labels_df], axis=1)

    # Condition to filter rows based on the string column
    condition = hits_labels["test_id"] == test_id

    # Filtered DataFrame
    filtered_hits = hits_labels[condition]
    filtered_hits["Cumulative_Label"] = filtered_hits.groupby("Labels")[
        "Count"
    ].cumsum()

    # Define custom markers for each class
    markers = ["o", "s", "^", "v", "D", "p"]  # Customize the markers as desired

    # Plotting with Seaborn scatter plot or line plot based on plot_type
    figsize_inches = (width / 25.4, height / 25.4)
    plt.figure(figsize=figsize_inches, tight_layout=True)
    if plot_type == "scatter":
        sns.scatterplot(
            x=x,
            y=y,
            hue="Labels",
            style="Labels",
            data=filtered_hits,
            markers=markers,
            palette=sns.color_palette(),
        )
    elif plot_type == "line":
        sns.lineplot(
            x=x,
            y=y,
            hue="Labels",
            style="Labels",
            data=filtered_hits,
            palette=sns.color_palette(),
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title} - {test_id}")
    plt.grid()
    plt.gca().set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        figure_filename = f"{title} - {test_id}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    plt.show()


def plot_cluster_tsne(
    labels,
    X_reduced,
    figure_path="t-SNE Clustering.pdf",
    title="t-SNE Clustering",
    subtitle="",
    x_label="t-SNE Dimension 1",
    y_label="t-SNE Dimension 2",
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    n_row=1,
    save=False,
    show=False,
):
    """
    Parameters
    ----------
    labels : array
        Labels of the clusters obtained by k-means or dbscan.
    X_reduced : array
        Dimensionality reduction set obtained with t-SNE and used for clustering training.
    figure_path : str
        Directory to save the figure.
    title : str, optional
        Title of the image to display and save. The default is 't-SNE Clustering'.
    x_label : str, optional
        Label for the x-axis. The default is 't-SNE Dimension 1'.
    y_label : str, optional
        Label for the y-axis. The default is 't-SNE Dimension 2'.
    subtitle : str or list, optional
        Subtitle for each of the subplots. The default is ''.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    ax : matplotlib axis object, optional
        Axis to plot on. If not provided, a new axis will be created.
    i : int, optional
        Index for subplot. Default is 1.
    n_col : int, optional
        Number of columns. The default is 1.
    n_row : int, optional
        Number of rows. The default is 1.
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. t-SNE - Clustering.pdf

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Plots and (saves) a scatter plot of the data with dimensionality reduction with
        cluster markers in PDF format.
    """
    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f"t-SNE_{i + 1}" for i in range(2)])
    tsne_df["Labels"] = labels

    # Masks to change markers
    valid_points = tsne_df[tsne_df["Labels"] != -1]
    anomalies = tsne_df[tsne_df["Labels"] == -1]

    # Count the number of unique labels in the DataFrame
    num_labels = valid_points["Labels"].nunique()

    # Define custom markers for each class
    palette = sns.color_palette()
    # Customize the markers list to match the number of unique labels
    markers = MARKER_LIST[:num_labels]
    palette = palette[:num_labels]

    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_title(title)
        ax.set_xlabel(x_label)
    else:
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        ax.set_title(subtitle)

    # Use Seaborn scatter plot with hue and style parameters
    sns.scatterplot(
        x=valid_points["t-SNE_1"],
        y=valid_points["t-SNE_2"],
        hue="Labels",
        style="Labels",
        data=valid_points,
        palette=palette,
        markers=markers,
        ax=ax,
    )

    # Scatter points with label '-1' separately as red 'x'
    if not anomalies.empty:
        sns.scatterplot(
            x=anomalies["t-SNE_1"],
            y=anomalies["t-SNE_2"],
            hue="Labels",
            style="Labels",
            data=anomalies,
            palette=["r"],
            markers="x",
            linewidth=1.5,
            ax=ax,
        )

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    ax.grid()
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        fig_format = figure_path.name.split(".")[-1]
        plt.savefig(figure_path, format=fig_format, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()  # Close the plot to free up memory

    return ax


def plot_cluster_tsne_3d(
    labels,
    X_reduced,
    figure_path,
    title="t-SNE 3D Clustering",
    subtitle="",
    x_label="t-SNE Dimension 1",
    y_label="t-SNE Dimension 2",
    z_label="t-SNE Dimension 3",
    width=180,
    height=120,
    save=False,
):
    """
    Parameters
    ----------
    labels : array
        Labels of the clusters obtained by k-means or dbscan.
    X_reduced : array
        Dimensionality reduction set obtained with t-SNE and used for clustering training.
    figure_path : str
        Directory to save the figure.
    title : str, optional
        Title of the image to display and save. The default is 't-SNE 3D Clustering'.
    x_label : str, optional
        Label for the x-axis. The default is 't-SNE Dimension 1'.
    y_label : str, optional
        Label for the y-axis. The default is 't-SNE Dimension 2'.
    z_label : str, optional
        Label for the z-axis. The default is 't-SNE Dimension 3'.
    subtitle : str, optional
        Subtitle for each of the subplots. The default is ''.
    width : int or float, optional
        Width of the figure in millimeters. The default is 180
    height : int or float, optional
        Height of the figure in millimeters. The default is 120
    save : boolean, optional
        "True" to save the image. The default is False.

        e.g. t-SNE 3D - Clustering.pdf

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the plot.
        Plots and (saves) a 3D scatter plot of the data with dimensionality reduction
        and cluster markers in PDF format.
    """
    # Create a DataFrame for easy use with Seaborn
    tsne_df = pd.DataFrame(X_reduced, columns=[f"t-SNE_{i + 1}" for i in range(3)])
    tsne_df["Labels"] = labels

    # Masks to change markers
    valid_points = tsne_df[tsne_df["Labels"] != -1]
    anomalies = tsne_df[tsne_df["Labels"] == -1]

    # Count the number of unique labels in the DataFrame
    num_labels = valid_points["Labels"].nunique()

    # Define custom markers for each class
    markers = ["o", "s", "^", "v", "D", "p"]  # Customize the markers as desired
    palette = sns.color_palette()

    # Customize the markers list to match the number of unique labels
    markers = markers[:num_labels]
    palette = palette[:num_labels]

    # Create the plot
    figsize_inches = (width / 25.4, height / 25.4)
    fig = plt.figure(figsize=figsize_inches, tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # Plot valid points with different markers and colors
    for label, marker, color in zip(valid_points["Labels"].unique(), markers, palette):
        subset = valid_points[valid_points["Labels"] == label]
        ax.scatter(
            subset["t-SNE_1"],
            subset["t-SNE_2"],
            subset["t-SNE_3"],
            label=f"Class {label}",
            marker=marker,
            color=color,
        )

    # Scatter points with label '-1' separately as red 'x'
    ax.scatter(
        anomalies["t-SNE_1"],
        anomalies["t-SNE_2"],
        anomalies["t-SNE_3"],
        c="r",
        marker="x",
        label="Anomalies",
    )

    # Save the image in figure_path
    if save:
        figure_filename = f"{title}.pdf"
        figure_path_name = os.path.join(figure_path, figure_filename)
        plt.savefig(figure_path_name, format="pdf", bbox_inches="tight")

    plt.show()

    return ax


def plot_dbi(
    dbi_kmeans,
    dbi_dbscan,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    n_row=1,
    k_min=2,
    figure_path="Davies-Bouldin Index.pdf",
    save=False,
    show=False,
):
    """
    Parameters
    ----------
    dbi_kmeans : array, list
        DBI scores obtained by k-means.
    dbi_dbscan : array, list
        DBI scores obtained by DBSCAN.
    figure_path : str
        Directory to save the figure.
    k_min : int, optional
        Minimum number of clusters. The default is 2.
    width : int or float, optional
        Width of the figure in millimeters. The default is 90
    height : int or float, optional
        Height of the figure in millimeters. The default is 60
    ax : axes, optional
        Axes subplot. The default is None.
    i : int, optional
        Counter for plotting each subplot. The default is 1.
    n_col : int, optional
        Number of columns. The default is 1.
    n_row : int, optional
        Number of rows. The default is 1.
    save : boolean, optional
        "True" to save the image. The default is False.

    Returns
    -------
    ax : axes
        Subplot of the figure.
        Plots and (saves) a diagram of the DBI scores for k-means and DBSCAN in PDF format.
    """
    # Create a DataFrame with the labels
    n_k = len(dbi_kmeans)
    k_max = k_min + n_k
    kmeans_k = range(k_min, k_min + len(dbi_kmeans))
    dbscan_k = range(k_min, k_min + len(dbi_dbscan))

    dbi_kmeans_df = pd.DataFrame(
        {
            "k": kmeans_k,
            "DBI": dbi_kmeans,
            "Algorithm": "k-means",
        }
    )

    dbi_dbscan_df = pd.DataFrame(
        {
            "k": dbscan_k,
            "DBI": dbi_dbscan,
            "Algorithm": "DBSCAN",
        }
    )

    dbi_scores = pd.concat(
        [dbi_kmeans_df, dbi_dbscan_df],
        ignore_index=True,
    )

    # Define custom markers for each class
    markers = MARKER_LIST[:n_k]

    # Plotting with Seaborn scatter plot
    figsize_inches = (width / 25.4, height / 25.4)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        ax.set_xlabel("Number of Clusters")
        ax.set_title("Davies-Bouldin Index for Different Cluster Numbers")

    elif i > n_col * (n_row - 1):
        ax.set_xlabel("Number of Clusters\n(" + string.ascii_lowercase[i - 1] + ")")

    else:
        ax.set_xlabel("(" + string.ascii_lowercase[i - 1] + ")")

    # Create the plot
    sns.lineplot(
        x="k",
        y="DBI",
        hue="Algorithm",
        style="Algorithm",
        data=dbi_scores,
        markers=markers,
        palette=sns.color_palette(n_colors=n_k),
        ax=ax,
    )

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        ax.set_ylabel("Davies-Bouldin Index")
    else:
        ax.set_ylabel("")

    # Set x-axis ticks to display only integer values
    ax.set_xticks(range(k_min, k_max + 1))
    ax.grid()
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        fig_format = figure_path.name.split(".")[-1]
        plt.savefig(figure_path, format=fig_format, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return ax


def plot_cluster_test_id(
    data,
    test_id,
    feat_x,
    feat_y,
    figure_path="Test ID Clustering.pdf",
    title=None,
    subtitle=None,
    x_label=None,
    y_label=None,
    width=90,
    height=60,
    ax=None,
    i=1,
    n_col=1,
    save=False,
    show=False,
):
    data = data[data["test_id"] == test_id]

    # Masks to change markers
    valid_points = data[data["label"] != -1]
    anomalies = data[data["label"] == -1]

    # Count the number of unique labels in the DataFrame
    num_labels = valid_points["label"].nunique()

    # Define custom markers for each class
    palette = sns.color_palette()
    # Customize the markers list to match the number of unique labels
    markers = MARKER_LIST[:num_labels]
    palette = palette[:num_labels]

    # Create a figure
    figsize_inches = (width / 25.4, height / 25.4)

    # If there is just one plot and set the title
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_inches, tight_layout=True)
        if title is None:
            ax.set_title(f"{feat_x} vs {feat_y} - Clustering")
        else:
            ax.set_title(title)
        if x_label is None:
            ax.set_xlabel(feat_x)
        else:
            ax.set_xlabel(x_label)

    else:
        if x_label is None:
            ax.set_xlabel(f"{feat_x}\n(" + string.ascii_lowercase[i - 1] + ")")
        else:
            ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        # Title for each test id
        if subtitle is not None:
            ax.set_title(subtitle)
        else:
            ax.set_title(f"{feat_x} vs {feat_y}")

    # Use Seaborn scatter plot with hue and style parameters
    sns.scatterplot(
        x=valid_points[feat_x],
        y=valid_points[feat_y],
        hue="label",
        style="label",
        data=valid_points,
        palette=palette,
        markers=markers,
        ax=ax,
    )

    # Scatter points with label '-1' separately as red 'x'
    if not anomalies.empty:
        sns.scatterplot(
            x=anomalies[feat_x],
            y=anomalies[feat_y],
            hue="label",
            style="label",
            data=anomalies,
            palette=["r"],
            markers="x",
            linewidth=1.5,
            ax=ax,
        )

    # Set y-axis label only for the first plot of each row
    if (i - 1) % n_col == 0:
        if y_label is None:
            ax.set_ylabel(feat_y)
        else:
            ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    # Logarithmic scale if you choose energy
    if feat_x == "energy":
        ax.set_xscale("log")

    if feat_y == "energy":
        ax.set_yscale("log")

    # Activate the grid
    ax.grid()
    ax.set_axisbelow(True)  # Set grid lines behind the data points

    # Save the image in figure_path
    if save:
        fig_format = figure_path.name.split(".")[-1]
        plt.savefig(figure_path, format=fig_format, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return ax

```
