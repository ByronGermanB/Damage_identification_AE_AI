
## File: `backup_results\scripts\CNN_accuracy.py`
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

## File: `backup_results\scripts\CNN_predictions.py`
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

## File: `backup_results\scripts\CNN_training.py`
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

## File: `backup_results\scripts\data_visualization.py`
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

## File: `backup_results\scripts\dataset_generator.py`
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

## File: `backup_results\scripts\feature_distribution.py`
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

## File: `backup_results\scripts\Generate_cwt_images.py`
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

## File: `backup_results\scripts\load_time_difference.py`
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

## File: `backup_results\scripts\Predictions.py`
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

## File: `backup_results\scripts\transient.py`
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

## File: `backup_results\scripts\unsupervised_model.py`
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

## File: `layout.py`
```python
from mylib import ProjectLayout  #type: ignore

layout = ProjectLayout(["src", "parameters", "results", "utils", "datasets", "figures"]) # manual list

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

import numpy as np
from joblib import load

from utils.paths import DATASETS_DIR, FIGURES_DIR, PARAMETERS_DIR, RESULTS_DIR
from utils.unsupervised_functions import (
    plot_cluster_feat,
    plot_cluster_tsne,
    plot_kmeans_per_k,
)

CONFIG_DIR = PARAMETERS_DIR / "arcan"
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
    TRIALS_DIR = RESULTS_DIR / dataset_name / "models" / "trials" / trial_name
    FIG_DIR = FIGURES_DIR / dataset_name / trial_name
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = DATASETS_DIR / dataset_name
    DATA_PATH = (
        DATA_DIR / "trial_data" / data_trial_name / "preprocessed_data.joblib"
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
        k_means_fig_dir = FIG_DIR / "kmeans"
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
    dbscan_fig_dir = FIG_DIR / "dbscan"
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

## File: `src\fft\feature_visualiztion.py`
```python
import json
from itertools import combinations

import pandas as pd

from utils.fft_analysis import plot_feat_vs_feat, select_significant_indices
from utils.paths import DATASETS_DIR, FIGURES_DIR, PARAMETERS_DIR

CONFIG_DIR = PARAMETERS_DIR / "arcan"
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
    DATA_DIR = DATASETS_DIR / dataset_name

    FIG_DIR = FIGURES_DIR / dataset_name
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    DATA_PATH = DATA_DIR / output_name

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

        feat_vs_feat_dir = FIG_DIR / "feat_vs_feat"
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

## File: `src\fft\n01_feature_extraction.py`
```python
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from mylib import configure_cache_dir, configure_plots, disk_cache, load_config  # type: ignore

from src.fft.pipeline_utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_DIR,
    get_fft_paths,
    parse_config_args,
)
from utils.fft_analysis import (
    Features,
    plot_feat_vs_feat,
    resolve_feature_columns,
    select_significant_indices,
)
from utils.force_mts import force_data
from utils.paths import PROJECT_ROOT
from utils.segmentation import load_segmentation_params

configure_plots(font="IBM Plex Sans")

@disk_cache()
def extract_campaign_features(
    data_dir: Path,
    files: str | list[str],
    parameters: dict[str, Any],
    trai_limits: dict[str, Any],
    force_cfg: dict[str, Any],
    segmentation_params: dict[str, Any] | None = None,
    sub_signal_offset_us: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run heavy .pridb/.tradb FFT feature extraction and MTS force loading."""
    if files == "all":
        pri_dirs = [
            path.parent for path in data_dir.glob("**/*.pridb") if path.is_file()
        ]
    else:
        pri_dirs = [data_dir / file for file in files]

    pri_names = [path.stem for path in pri_dirs]
    print(f"[INFO] Found {len(pri_dirs)} .pridb files in {data_dir}")

    dfs: list[pd.DataFrame] = []
    force_dfs: list[pd.DataFrame] = []
    force_enabled = force_cfg.get("enabled", False)

    for pri_dir, pri_name in zip(pri_dirs, pri_names):
        print(f"[INFO] Processing {pri_name}...")

        data_force_i = None
        if force_enabled:
            data_force_i = force_data(
                path=pri_dir,
                section=force_cfg.get("section", None),
                test_id=pri_name,
            )
            force_dfs.append(data_force_i)

        features = Features(
            path=pri_dir,
            lower_freq=parameters.get("lower_freq", 100),
            upper_freq=parameters.get("upper_freq", 900),
            sampling_rate=parameters.get("sampling_rate", 5e6),
            N_samp=parameters.get("N_samp", 1024),
            N_seg=parameters.get("N_seg", 1),
            signal_offset=parameters.get("signal_offset", 200),
            load_offset=parameters.get("load_offset", 0),
            force=data_force_i if force_cfg.get("force_data", False) else None,
            segmentation_params=segmentation_params,
            sub_signal_offset_us=sub_signal_offset_us,
        )

        df = features.feature_extr(
            threshold=parameters["threshold"],
            counts=parameters["counts"],
            label=pri_name,
            test_id=pri_name,
            min_trai=trai_limits.get(pri_name, {}).get("min_trai", 1),
            max_trai=trai_limits.get(pri_name, {}).get("max_trai", None),
        )
        if df.empty:
            continue
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data["label"] = 0

    data_force = (
        pd.concat(force_dfs, ignore_index=True)
        if force_enabled and force_dfs
        else None
    )
    return data, data_force


def main(config_arg: str | Path | None = DEFAULT_CONFIG_PATH) -> None:
    """Main function for feature extraction."""
    configure_cache_dir(PROJECT_ROOT / ".cache")

    config = load_config(config_arg, return_path=False)
    paths = get_fft_paths(config, create=True)

    extraction_cfg = config["extraction"]
    data_dir = Path(extraction_cfg.get("data_dir") or DEFAULT_DATA_DIR)

    seg_cfg = extraction_cfg.get("segmentation", {})
    segmentation_params = load_segmentation_params(
        seg_cfg=seg_cfg,
        dataset_name=config["dataset_name"],
    )
    if segmentation_params is not None:
        print(f"[INFO] Sub-hit segmentation enabled: {segmentation_params}")
    
    # 1st run: computes and caches to .cache/ ([CACHE MISS])
    # Subsequent runs: loads instantly from disk ([CACHE HIT]) while tweaking plots
    data, data_force = extract_campaign_features(
        data_dir=data_dir,
        files=extraction_cfg["files"],
        parameters=extraction_cfg["parameters"],
        trai_limits=extraction_cfg.get("trai_limits", {}),
        force_cfg=extraction_cfg.get("force", {}),
        segmentation_params=segmentation_params,
        sub_signal_offset_us=float(seg_cfg.get("sub_signal_offset_us", 0.0)), 
    )

    data.to_csv(paths["features_path"], index=False)
    print("[INFO] Feature extraction completed. Data saved to CSV.")

    if data_force is not None:
        data_force.to_csv(paths["force_data_path"], index=False)

    # Visualization block (runs fresh every time without re-extracting features)
    plot_cfg = config["feature_visualization"]
    if plot_cfg.get("enabled", False):
        features_to_plot = plot_cfg.get("features_to_plot", [])
        print("[INFO] Saving Feature vs Feature images...")

        feat_vs_feat_dir = paths["trial_figures_dir"] / "feat_vs_feat"
        feat_vs_feat_dir.mkdir(parents=True, exist_ok=True)

        label_column = "test_id" if plot_cfg.get("labeled", False) else "label"
        plot_data = data
        resolved_features = resolve_feature_columns(plot_data, features_to_plot)

        if plot_cfg.get("reduced_points", False):
            selected_indices = select_significant_indices(
                data=plot_data,
                label_column=label_column,
                feature_columns=resolved_features,
                max_points=plot_cfg.get("max_points", 20),
            )
            plot_data = plot_data.loc[selected_indices].copy()

        y = plot_data[label_column]
        img_format = plot_cfg.get("img_format", "png")

        for feature_x, feature_y in combinations(resolved_features, 2):
            figure_path = feat_vs_feat_dir / f"{feature_x}_vs_{feature_y}.{img_format}"

            plot_feat_vs_feat(
                X=plot_data,
                y=y,
                feat_x=feature_x,
                feat_y=feature_y,
                fig_path=figure_path,
                figsize=plot_cfg.get("figsize", (90, 60)),
                save=True,
            )

        print("[INFO] Feature vs Feature images saved.")


if __name__ == "__main__":
    args = parse_config_args(
        "Stage 1: Extract fft features from all the AE tests in an experimental campaign."
    )
    main(config_arg=args.config)
```

## File: `src\fft\pipeline_utils.py`
```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mylib import (  # type: ignore
    parse_config_args as _mylib_parse_config_args,
)

from utils.paths import (
    DATASETS_DIR,
    FIGURES_DIR,
    PARAMETERS_DIR,
    RESULTS_DIR,
)

DEFAULT_CONFIG_SUBDIR = PARAMETERS_DIR / "example_data"
DEFAULT_CONFIG_NAME = "trial_1.json"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_SUBDIR / DEFAULT_CONFIG_NAME
DEFAULT_DATA_DIR = DATASETS_DIR / "example_data" / "data"


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATA_DIR",
    "get_fft_paths",
    "parse_config_args",
]


def get_fft_paths(
    config: dict[str, Any],
    create: bool = True,
) -> dict[str, Path]:
    """Build standard dataset, trial, model, and figure paths for a config."""
    dataset_name = config["dataset_name"]
    trial_name = config["trial_name"]

    dataset_dir = DATASETS_DIR / dataset_name / trial_name
    trial_results_dir = RESULTS_DIR / dataset_name / "models" / "trials" / trial_name
    trial_figures_dir = FIGURES_DIR / dataset_name / trial_name
    features_path = dataset_dir / "features.csv"
    force_data_path = dataset_dir / "force_data.csv"
    preprocessed_data_path = dataset_dir / "preprocessed_data.joblib"

    if create:
        for directory in (dataset_dir, trial_results_dir, trial_figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

    return {
        "dataset_dir": dataset_dir,
        "trial_dir": trial_results_dir,
        "trial_figures_dir": trial_figures_dir,
        "features_path": features_path,
        "force_data_path": force_data_path,
        "preprocessed_data_path": preprocessed_data_path,
    }


def parse_config_args(
    description: str,
) -> argparse.Namespace:
    """Parse `-c` / `--config` (defaulting to `DEFAULT_CONFIG_PATH`) ."""

    return _mylib_parse_config_args(
        description=description,
        default_config=DEFAULT_CONFIG_PATH,
    )

```

## File: `src\fft\results_visualization.py`
```python
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from joblib import load

from utils.force_mts import limit_finder, plot_stress_hits_cluster
from utils.paths import FIGURES_DIR, PARAMETERS_DIR, RESULTS_DIR
from utils.unsupervised_functions import plot_cluster_test_id, plot_cluster_tsne

CONFIG_DIR = PARAMETERS_DIR / "arcan"
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
        MODELS_DIR = RESULTS_DIR / dataset_name / "models" / model_name
        FIG_DIR = FIGURES_DIR / dataset_name / model_name
        FIG_DIR.mkdir(parents=True, exist_ok=True)

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

        feat_vs_feat_test_id_dir = FIG_DIR / "feat_vs_feat_test_id"
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

        hits_vs_stress_dir = FIG_DIR / "hits_vs_stress"
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

        tsne_dir = FIG_DIR / "tsne_clustering"
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

import pandas as pd
from joblib import dump, load

from utils.paths import DATASETS_DIR, PARAMETERS_DIR, RESULTS_DIR

CONFIG_DIR = PARAMETERS_DIR / "arcan"
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
    TRIALS_DIR = RESULTS_DIR / dataset_name / "models" / "trials" / trial_name
    MODELS_DIR = RESULTS_DIR / dataset_name / "models" / output_model_name
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

    DATA_PATH = DATASETS_DIR / dataset_name / feat_data_name
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
from utils.paths import DATASETS_DIR, FIGURES_DIR, PARAMETERS_DIR, RESULTS_DIR

# Functions for DBSCAN model
from utils.unsupervised_functions import (
    grid_search_dbscan,
    kmeans_per_k,
    tsne,
)

CONFIG_DIR = PARAMETERS_DIR / "arcan"
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
        RESULTS_DIR / dataset_name / "models" / "trials" / config["trial_name"]
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    # Directories
    FIG_DIR = FIGURES_DIR / dataset_name / config["trial_name"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = DATASETS_DIR / dataset_name
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Import datasets
    if config.get("saved_data_path", None) is not None:
        DATA_PATH = Path(config["saved_data_path"])
        X_original, X, y, X_reduced, hits = load(DATA_PATH).values()

    else:
        DATA_PATH = DATA_DIR / config["feat_data_name"]
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
                    FIG_DIR / f"t-SNE Visualization.{format_plot}",
                    show=show_plots,
                    save=save_plots,
                )
        else:
            X_reduced = X

    if save_data:
        data_saving_path = (
            DATA_DIR
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

## File: `src\preprocessing\n01_select_waveforms.py`
```python
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import vallenae as vae
from matplotlib.widgets import Button, Slider, TextBox
from mylib import load_config, save_config  # type: ignore

from src.preprocessing.pipeline_utils import (
    DEFAULT_CONFIG_PATH,
    get_segmentation_paths,
    parse_config_args,
)
from utils.segmentation import dbae_to_mV


def _discover_trais(
    tradb_path: Path,
    min_amp_dbae: float = 36.0,
    min_counts: int = 5,
    max_trai: int | None = None,
) -> list[int]:
    """Return sorted valid TRAIs using `.pridb` metadata when available, else `.tradb`."""
    pridb_path = tradb_path.with_suffix(".pridb")
    if pridb_path.exists():
        with vae.io.PriDatabase(pridb_path) as pridb:
            hits = pridb.read_hits().reset_index(drop=True)

        if not hits.empty and "trai" in hits.columns:
            amp_dbae = 20.0 * np.log10(
                np.maximum(hits["amplitude"].to_numpy(), 1e-12) / 1e-6
            )
            mask = (
                (hits["trai"] > 0)
                & (amp_dbae >= min_amp_dbae)
                & (amp_dbae <= 94.0)
                & (hits["counts"] >= min_counts)
            )
            if max_trai is not None:
                mask &= hits["trai"] <= max_trai

            trais = sorted(int(t) for t in hits.loc[mask, "trai"].dropna().unique())
            if trais:
                return trais

    with sqlite3.connect(tradb_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT trai FROM tr_data WHERE trai > 0 ORDER BY trai"
        ).fetchall()
    trais = [int(r[0]) for r in rows]
    if max_trai is not None:
        trais = [t for t in trais if t <= max_trai]
    return trais


class InteractiveWaveformSelector:
    """Interactive Matplotlib browser to inspect AE transients and annotate sub-hits."""

    def __init__(
        self,
        tradb_path: Path,
        config_path: Path,
        config: dict[str, Any],
        test_id: str,
        trais: list[int],
        min_amp_dbae: float = 36.0,
    ) -> None:
        if not trais:
            raise ValueError(f"No valid TRAIs found in {tradb_path}.")

        self.tradb_path = tradb_path
        self.config_path = config_path
        self.config = config
        self.test_id = test_id
        self.trais = trais
        self.current_idx = 0
        self.min_amp_dbae = min_amp_dbae

        self.tradb = vae.io.TraDatabase(self.tradb_path)
        self.selections: dict[int, int] = {}
        self.save_to_json()

        self._build_ui()
        self._update_plot()

    def save_to_json(self) -> None:
        """Persist annotated TRAIs into `self.config_path` via `mylib.save_config`."""
        ordered_items = [
            {"trai": trai, "expected_segments": self.selections[trai]}
            for trai in sorted(self.selections)
        ]
        payload: dict[str, Any] = {
            **self.config,
            "test_id": self.test_id,
            "tradb_path": self.config.get("tradb_path", None),
            "selections": ordered_items,
            "trai_list": [item["trai"] for item in ordered_items],
            "expected_segments": [item["expected_segments"] for item in ordered_items],
            "param_grid": self.config.get("param_grid"),
            "grid_search": self.config.get(
                "grid_search",
                {
                    "output_dir": None,
                    "save_valid_only": True,
                    "img_format": "png",
                    "figsize_mm": [140, 90],
                    "save_figures": True,
                    "show_figures": False,
                    "save_txt": True,
                },
            ),
        }
        self.config = payload
        save_config(payload, self.config_path)

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(13.0, 7.2))
        self.fig.canvas.manager.set_window_title(
            f"AE Waveform Selector - {self.test_id}"
        )

        # Main waveform panel and right-hand selection summary panel
        self.ax_wave = self.fig.add_axes([0.07, 0.25, 0.66, 0.65])
        self.ax_summary = self.fig.add_axes([0.75, 0.25, 0.23, 0.65])
        self.ax_summary.axis("off")

        # Index scrubber slider
        ax_slider = self.fig.add_axes([0.07, 0.15, 0.66, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label="Index",
            valmin=0,
            valmax=max(0, len(self.trais) - 1),
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider_changed)

        # Navigation & annotation controls
        self.btn_prev = Button(self.fig.add_axes([0.07, 0.05, 0.08, 0.06]), "Prev (<-)")
        self.btn_next = Button(self.fig.add_axes([0.16, 0.05, 0.08, 0.06]), "Next (->)")

        self.txt_jump = TextBox(
            self.fig.add_axes([0.31, 0.05, 0.08, 0.06]), "TRAI: ", initial=""
        )

        self.btn_seg2 = Button(
            self.fig.add_axes([0.42, 0.05, 0.085, 0.06]), "Add: 2 segs", color="#e6f2ff"
        )
        self.btn_seg3 = Button(
            self.fig.add_axes([0.515, 0.05, 0.085, 0.06]),
            "Add: 3 segs",
            color="#d9f2e6",
        )
        self.btn_seg4 = Button(
            self.fig.add_axes([0.61, 0.05, 0.085, 0.06]), "Add: 4 segs", color="#fff2cc"
        )
        self.btn_seg5 = Button(
            self.fig.add_axes([0.705, 0.05, 0.085, 0.06]),
            "Add: 5 segs",
            color="#ffe6cc",
        )
        self.btn_remove = Button(
            self.fig.add_axes([0.80, 0.05, 0.085, 0.06]), "Remove", color="#ffd6d6"
        )
        self.btn_save = Button(
            self.fig.add_axes([0.895, 0.05, 0.085, 0.06]), "Save JSON", color="#d4edda"
        )

        self.btn_prev.on_clicked(lambda _: self.step(-1))
        self.btn_next.on_clicked(lambda _: self.step(1))
        self.txt_jump.on_submit(self._on_jump_submit)
        self.btn_seg2.on_clicked(lambda _: self.assign_segments(2))
        self.btn_seg3.on_clicked(lambda _: self.assign_segments(3))
        self.btn_seg4.on_clicked(lambda _: self.assign_segments(4))
        self.btn_seg5.on_clicked(lambda _: self.assign_segments(5))
        self.btn_remove.on_clicked(lambda _: self.remove_current())
        self.btn_save.on_clicked(lambda _: self._manual_save())

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.status_msg = (
            "Ready. Use buttons or keys [2-5] to add TRAI, Left/Right to browse."
        )

    def _on_slider_changed(self, val: float) -> None:
        new_idx = round(val)
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self._update_plot()

    def _on_jump_submit(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned.isdigit():
            self.status_msg = f"Invalid TRAI input: '{text}'"
            self._update_summary_panel()
            return
        target = int(cleaned)
        if target in self.trais:
            self.current_idx = self.trais.index(target)
            self.slider.set_val(self.current_idx)
            self.status_msg = f"Jumped to TRAI {target}."
        else:
            nearest_idx = int(np.argmin(np.abs(np.array(self.trais) - target)))
            self.current_idx = nearest_idx
            self.slider.set_val(self.current_idx)
            self.status_msg = f"TRAI {target} not in filtered list; showing nearest TRAI {self.trais[nearest_idx]}."
        self._update_plot()

    def _on_key_press(self, event: Any) -> None:
        if event.inaxes == self.txt_jump.ax:
            return
        if event.key == "left":
            self.step(-1)
        elif event.key == "right":
            self.step(1)
        elif event.key in {"2", "3", "4", "5"}:
            self.assign_segments(int(event.key))
        elif event.key in {"delete", "backspace"}:
            self.remove_current()
        elif event.key == "s":
            self._manual_save()

    def step(self, delta: int) -> None:
        self.current_idx = (self.current_idx + delta) % len(self.trais)
        self.slider.eventson = False
        self.slider.set_val(self.current_idx)
        self.slider.eventson = True
        self._update_plot()

    def assign_segments(self, n_segments: int) -> None:
        trai = self.trais[self.current_idx]
        self.selections[trai] = n_segments
        self.save_to_json()
        self.status_msg = (
            f"Added TRAI {trai} -> {n_segments} segment(s) and saved to JSON."
        )
        self._update_plot()

    def remove_current(self) -> None:
        trai = self.trais[self.current_idx]
        if trai in self.selections:
            del self.selections[trai]
            self.save_to_json()
            self.status_msg = f"Removed TRAI {trai} from selection and updated JSON."
        else:
            self.status_msg = f"TRAI {trai} is not currently selected."
        self._update_plot()

    def _manual_save(self) -> None:
        self.save_to_json()
        self.status_msg = (
            f"Saved {len(self.selections)} TRAIs to {self.config_path.name}."
        )
        self._update_summary_panel()

    def _update_plot(self) -> None:
        trai = self.trais[self.current_idx]
        amp_v, time = self.tradb.read_wave(trai, time_axis=True)
        waveform = amp_v * 1e3
        time_us = time * 1e6

        peak_mv = float(np.max(np.abs(waveform)))
        peak_dbae = 20.0 * np.log10(max(peak_mv, 1e-9) * 1e3)
        dur_us = float(time_us[-1] - time_us[0]) if time_us.size > 1 else 0.0

        self.ax_wave.clear()
        self.ax_wave.plot(
            time_us, waveform, color="#2b5c8f", linewidth=0.65, label="Waveform [mV]"
        )
        self.ax_wave.axhline(
            dbae_to_mV(self.min_amp_dbae),
            color="crimson",
            linestyle=":",
            linewidth=0.8,
            label=f"Ref. threshold ({self.min_amp_dbae:.0f} dBAE)",
        )
        
        self.ax_wave.axhline(
                - dbae_to_mV(self.min_amp_dbae),
                color="crimson",
                linestyle=":",
                linewidth=0.8,
            )

        selected_tag = ""
        if trai in self.selections:
            selected_tag = f"  |  SELECTED: {self.selections[trai]} segment(s)"
            self.ax_wave.set_facecolor("#f4fbf6")
        else:
            self.ax_wave.set_facecolor("white")

        self.ax_wave.set_title(
            f"{self.test_id}  |  TRAI {trai} ({self.current_idx + 1}/{len(self.trais)})  |  "
            f"Peak: {peak_mv:.2f} mV ({peak_dbae:.1f} dBAE)  |  Dur: {dur_us:.1f} us{selected_tag}",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_wave.set_xlabel("Time [us]")
        self.ax_wave.set_ylabel("Amplitude [mV]")
        self.ax_wave.grid(True, linestyle="-", linewidth=0.35, alpha=0.7)
        self.ax_wave.legend(loc="upper right", fontsize=8)

        self._update_summary_panel()
        self.fig.canvas.draw_idle()

    def _update_summary_panel(self) -> None:
        self.ax_summary.clear()
        self.ax_summary.axis("off")

        lines = [
            "SELECTED TRAIs",
            "=" * 26,
            f"Config JSON:\n{self.config_path.name}",
            "-" * 26,
        ]
        if self.selections:
            for trai in sorted(self.selections):
                marker = ">> " if trai == self.trais[self.current_idx] else "   "
                lines.append(
                    f"{marker}TRAI {trai:<6} -> {self.selections[trai]} seg(s)"
                )
        else:
            lines.append("  (No TRAIs selected yet)")

        lines.extend(
            [
                "-" * 26,
                "Shortcuts:",
                "  Left/Right : Prev / Next",
                "  2, 3, 4, 5 : Add N segments",
                "  Backspace  : Remove TRAI",
                "  S          : Save JSON",
                "-" * 26,
                f"Status:\n{self.status_msg}",
            ]
        )
        self.ax_summary.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
        )

    def show(self) -> None:
        plt.show()


def main(config_arg: str | Path | None = DEFAULT_CONFIG_PATH) -> None:
    """Launch the interactive waveform selector driven by a JSON configuration file."""
    config, cfg_path = load_config(config_arg, return_path=True)
    paths = get_segmentation_paths(config, create=False)

    test_id = str(config["test_id"])
    selector_cfg = config.get("selector", {})

    min_amp_dbae = float(selector_cfg.get("min_amp_dbae", 36.0))
    min_counts = int(selector_cfg.get("min_counts", 5))
    max_trai_raw = selector_cfg.get("max_trai", None)
    max_trai = int(max_trai_raw) if max_trai_raw is not None else None

    tradb_path = paths["tradb_path"]
    if not tradb_path.exists():
        raise FileNotFoundError(f"Transient database not found: {tradb_path}")

    trais = _discover_trais(
        tradb_path=tradb_path,
        min_amp_dbae=min_amp_dbae,
        min_counts=min_counts,
        max_trai=max_trai,
    )
    print(f"[INFO] Loaded config from   : {cfg_path}")
    print(f"[INFO] Loaded {len(trais)} candidate TRAIs from {tradb_path}")

    selector = InteractiveWaveformSelector(
        tradb_path=tradb_path,
        config_path=cfg_path,
        config=config,
        test_id=test_id,
        trais=trais,
        min_amp_dbae=min_amp_dbae
    )
    selector.show()


if __name__ == "__main__":
    args = parse_config_args(
        "Stage 1: Inspect AE transient waveforms and select example TRAIs."
    )
    main(config_arg=args.config)

```

## File: `src\preprocessing\n02_run_grid_search.py`
```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from mylib import configure_plots, load_config, save_config  # type: ignore

from src.preprocessing.pipeline_utils import (
    DEFAULT_CONFIG_PATH,
    get_segmentation_paths,
    parse_config_args,
)
from utils.segmentation import run_segmentation_grid_search

configure_plots(font="IBM Plex Sans")


def _extract_trais_and_expectations(
    config: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Read `(trai_list, expected_segments)` from `selections` or flat arrays."""
    selections = config.get("selections")
    if selections:
        trai_list = [int(item["trai"]) for item in selections]
        expected_segments = [int(item["expected_segments"]) for item in selections]
        return trai_list, expected_segments

    trai_list = [int(t) for t in config.get("trai_list", [])]
    expected_segments = [int(e) for e in config.get("expected_segments", [])]
    return trai_list, expected_segments


def main(
    config_arg: str | Path | None = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the sub-hit segmentation grid search from a stage-01 selection JSON."""
    config, cfg_path = load_config(config_arg, return_path=True)
    paths = get_segmentation_paths(config, create=True)

    dataset_name = str(config["dataset_name"])
    test_id = str(config["test_id"])
    sampling_rate = float(config.get("sampling_rate", 5.0))

    trai_list, expected_segments = _extract_trais_and_expectations(config)
    if not trai_list:
        raise ValueError(
            f"No TRAIs found in '{cfg_path}'. Run 01_select_waveforms.py first."
        )

    tradb_path = paths["tradb_path"]
    if not tradb_path.exists():
        raise FileNotFoundError(f"Transient database not found: {tradb_path}")

    param_grid = config.get("param_grid")
    if not param_grid:
        raise ValueError(f"Missing 'param_grid' dictionary in '{cfg_path}'.")

    gs_cfg = config.get("grid_search", {})
    save_valid_only = bool(gs_cfg.get("save_valid_only", True))
    img_format = str(gs_cfg.get("img_format", "png"))
    figsize_mm = tuple(gs_cfg.get("figsize_mm", [140, 90]))
    save_figures = bool(gs_cfg.get("save_figures", True))
    show_figures = bool(gs_cfg.get("show_figures", False))
    save_txt = bool(gs_cfg.get("save_txt", True))

    print(f"[INFO] Loaded config from   : {cfg_path}")
    print(f"[INFO] Dataset / Test ID    : {dataset_name} / {test_id}")
    print(f"[INFO] TRAIs                : {trai_list}")
    print(f"[INFO] Expected segments    : {expected_segments}")
    print(f"[INFO] Figures directory    : {paths['figures_dir']}")
    print(f"[INFO] Results directory    : {paths['results_dir']}")

    summary_df = run_segmentation_grid_search(
        tradb_path=tradb_path,
        trai_list=trai_list,
        param_grid=param_grid,
        fig_dir=paths["figures_dir"],
        results_dir=paths["results_dir"],
        expected_segments=expected_segments,
        save_valid_only=save_valid_only,
        test_id=test_id,
        sampling_rate=sampling_rate,
        img_format=img_format,
        figsize=figsize_mm,
        save=save_figures,
        show=show_figures,
        save_txt=save_txt,
    )

    # Save an exact snapshot of the configuration used for reproducibility
    save_config(config, paths["results_dir"] / "grid_search_config_used.json")
    print(f"[INFO] Completed {len(summary_df)} total evaluations.")


if __name__ == "__main__":
    args = parse_config_args(
        "Stage 2: Run sub-hit segmentation grid search from selected TRAIs JSON.",
    )
    main(
        config_arg=args.config,
    )

```

## File: `src\preprocessing\pipeline_utils.py`
```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mylib import (  # type: ignore
    parse_config_args as _mylib_parse_config_args,
)
from mylib import resolve_path  # type: ignore

from utils.paths import (
    DATASETS_DIR,
    EXAMPLE_TRADB,
    FIGURES_DIR,
    PARAMETERS_DIR,
    RESULTS_DIR,
)

DEFAULT_CONFIG_SUBDIR = PARAMETERS_DIR / "example_data"
DEFAULT_CONFIG_NAME = "segmentation.json"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_SUBDIR / DEFAULT_CONFIG_NAME


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "get_segmentation_paths",
    "parse_config_args",
]


def get_segmentation_paths(
    config: dict[str, Any],
    create: bool = True,
) -> dict[str, Path]:
    """Build (and optionally create) standard dataset, result, and figure paths."""
    dataset_name = str(config["dataset_name"])
    test_id = str(config["test_id"])

    raw_tradb = config.get("tradb_path")
    tradb_path = resolve_path(raw_tradb) if raw_tradb else EXAMPLE_TRADB

    dataset_dir = DATASETS_DIR / dataset_name
    results_dir = RESULTS_DIR / dataset_name / "segmentation" / test_id
    figures_dir = FIGURES_DIR / dataset_name / "segmentation" / test_id

    if create:
        for directory in (dataset_dir, results_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

    return {
        "tradb_path": tradb_path,
        "dataset_dir": dataset_dir,
        "results_dir": results_dir,
        "figures_dir": figures_dir,
    }


def parse_config_args(
    description: str,
) -> argparse.Namespace:
    """Parse `-c` / `--config` (defaulting to `DEFAULT_CONFIG_PATH`) ."""

    return _mylib_parse_config_args(
        description=description,
        default_config=DEFAULT_CONFIG_PATH,
    )

```

## File: `thesis_scripts\plot_single_segmentation.py`
```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import vallenae as vae
from mylib import configure_plots, load_config, parse_config_args  # type: ignore

from src.preprocessing.pipeline_utils import get_segmentation_paths
from thesis_scripts.thesis_utils import DEFAULT_CONFIG_DIR
from utils.segmentation import (
    PARAM_COLUMNS,
    compute_smoothed_envelope,
    plot_subhit_segmentation,
    segment_merged_hits,
)

configure_plots(font="IBM Plex Sans")

DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "single_segmentation.json"


def load_combo_params(csv_path: str, test_id: str, combo_id: str) -> dict[str, Any]:
    """Read a specific `combo_id` parameter row from `grid_search_summary.csv`."""
    df = pd.read_csv(csv_path)
    match = df[df["combo_id"] == combo_id]
    if match.empty:
        raise ValueError(f"Combo '{combo_id}' not found in {csv_path}.")

    row = match.iloc[0]
    return {
        col: (row[col].item() if hasattr(row[col], "item") else row[col])
        for col in PARAM_COLUMNS
        if col in row and pd.notna(row[col])
    }


def main(
    config_arg: str | Path | None = DEFAULT_CONFIG_PATH,
) -> None:
    config = load_config(config_arg, return_path=False)
    paths = get_segmentation_paths(config, create=False)

    # 1. Choose specimen, single TRAI, and output styling for the thesis
    test_id = config["test_id"]
    trai = config["trai"]
    combo_id: str | None = config.get("combo_id", None)

    manual_params: dict[str, Any] = {
        "sampling_rate": 5e6,
        "threshold_dbae": 30.0,
        "pre_trigger_us": 40.0,
        "smoothing_us": 100.0,
        "min_separation_us": 300.0,
        "min_prominence_ratio": 0.25,
        "valley_decay_ratio": 0.35,
        "min_secondary_ratio": 0.1,
    }

    # 2. Establish the parameters
    csv_path = paths["results_dir"] / "grid_search_summary.csv"
    params = (
        load_combo_params(csv_path=csv_path, test_id=test_id, combo_id=combo_id)
        if combo_id is not None
        else manual_params
    )

    # 3. Load the single transient and segment it
    tradb_path = paths["tradb_path"]
    tradb = vae.io.TraDatabase(tradb_path)
    amp_v, time = tradb.read_wave(trai, time_axis=True)
    waveform = amp_v * 1e3

    sampling_rate = float(params.get("sampling_rate", 5.0))
    smooth_env = compute_smoothed_envelope(
        waveform, sampling_rate=sampling_rate, smoothing_us=float(params.get("smoothing_us", 100.0))
    )
    segments = segment_merged_hits(waveform=waveform, **params)

    # 4. Plot & save without main title (caption goes in LaTeX/Word)
    plot_cfg = config["plot"]
    img_format = plot_cfg.get("img_format", "pdf")
    figsize = tuple(plot_cfg.get("figsize_mm", [120, 90]))
    fig_path = paths["figures_dir"] / f"trai_{trai}_segmentation.{img_format}"
    plot_subhit_segmentation(
        waveform=waveform,
        time=time,
        segments=segments,
        params=params,
        figure_path=fig_path,
        smooth_env=smooth_env,
        trai=trai,
        test_id=test_id,
        show_title=plot_cfg.get("show_tile", False),
        show_subplot_titles=plot_cfg.get("show_subplot_tiles", True),
        legend_loc=plot_cfg.get("legend_loc", "lower right"),
        figsize=figsize,
        save=plot_cfg.get("save_figures", False),
        show=plot_cfg.get("show_figures", True),
        save_txt=False,
    )


if __name__ == "__main__":
    args = parse_config_args(
        description="Single Segmention: Run sub-hit segmentation from selected TRAI.",
        default_config=DEFAULT_CONFIG_PATH,
    )
    main(
        config_arg=args.config,
    )

```

## File: `thesis_scripts\thesis_utils.py`
```python
from mylib import configure_plots  # type: ignore

from utils.paths import (
    PARAMETERS_DIR,
)

configure_plots(font="IBM Plex Sans")

DEFAULT_CONFIG_DIR = PARAMETERS_DIR / "thesis_figures"

__all__ = [
    "DEFAULT_CONFIG_DIR"
]
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
import re
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import vallenae as vae
from mylib import (  # type: ignore
    IMG_SIZE,
    MARKER_LIST,
    configure_plots,
    get_active_palette,
    img_size_inches,
    resolve_save_target,
    save_fig,
)

# Libraries for signal processing
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hamming

# Model training
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Feature visualization
from sklearn.manifold import TSNE
from sklearn.metrics import (
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

from utils.segmentation import SubHitSegment, segment_merged_hits

configure_plots(font="IBM Plex Sans")

# =============================================================================
# Feature extraction
# =============================================================================


class Features:
    def __init__(
        self,
        path: str | Path,
        lower_freq: float,
        upper_freq: float,
        sampling_rate: float,
        N_samp: int,
        N_seg: int,
        signal_offset: float,
        load_offset: float = 0,
        force: pd.DataFrame | None = None,
        segmentation_params: dict[str, Any] | None = None,
        sub_signal_offset_us: float = 0.0,
    ):
        """
        Parameters
        ----------
        path : str or Path
            Directory where the `.pridb` and `.tradb` files are located.
        lower_freq : float
            Lower frequency limit in kHz.
        upper_freq : float
            Upper frequency limit in kHz.
        sampling_rate : float
            Data acquisition frequency per second in MHz.
        N_samp : int
            Number of samples in each segment for Fourier analysis (power of 2).
        N_seg : int
            Total number of overlapping segments to analyze.
        signal_offset : float
            Start offset in microseconds [µs] depending on pretrigger.
        load_offset : float, default=0
            Offset between AE and MTS time bases [s].
        force : DataFrame, optional
            MTS force data for automatic time alignment.
        """
        self.path = Path(path)
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        self.sampling_rate = sampling_rate
        self.N_samp = N_samp
        self.N_seg = N_seg
        self.signal_offset_us = signal_offset
        self.signal_offset = int(signal_offset * sampling_rate)
        self.load_offset = load_offset
        self.force = force

        for item in os.listdir(self.path):
            if item.endswith(".tradb"):
                self.tradb_path = self.path / item
            elif item.endswith(".pridb"):
                self.pridb_path = self.path / item

        self.res_freq = sampling_rate * 1000.0 / N_samp
        self.time_samp = 1.0 / sampling_rate
        self.time_seg = self.time_samp * N_samp

        self.lower_freq_index = int(self.lower_freq // self.res_freq)
        self.upper_freq_index = int(self.upper_freq // self.res_freq + 1)
        self.N_feat_seg = 9

        self.window = hamming(self.N_samp)
        self.freq_fft = (fftfreq(self.N_samp, self.time_samp * 1e-6) / 1000.0)[
            : self.N_samp // 2
        ]
        self.freq_fft_rango = self.freq_fft[
            self.lower_freq_index : self.upper_freq_index + 1
        ]
        self.segmentation_params = segmentation_params
        self.sub_signal_offset = int(sub_signal_offset_us * sampling_rate)

    def _extract_fft_vector(
        self,
        waveform_mv: np.ndarray,
        offset_pts: int,
        bin_edges: list[int],
    ) -> np.ndarray:
        """Compute the 1D spectral feature vector (`9 * N_seg`) for one waveform or sub-hit."""
        min_required = int(((self.N_seg - 1) / 2 + 1) * self.N_samp) + offset_pts
        if waveform_mv.size < min_required:
            waveform_mv = np.pad(waveform_mv, (0, min_required - waveform_mv.size))

        seg_total = np.zeros((self.N_seg, self.N_samp))
        for i in range(self.N_seg):
            start = int(i * self.N_samp / 2 + offset_pts)
            end = int((i / 2 + 1) * self.N_samp + offset_pts)
            seg_total[i] = waveform_mv[start:end]

        amp_fft = np.abs(fft(seg_total * self.window))
        amp_fft_rango = amp_fft[:, self.lower_freq_index : self.upper_freq_index + 1]

        peak_freq = np.zeros(self.N_seg)
        centroid_freq = np.zeros(self.N_seg)
        part_powers = np.zeros((6, self.N_seg))

        for i in range(self.N_seg):
            spectrum = amp_fft_rango[i]
            spec_sum = np.sum(spectrum)
            if spec_sum > 0:
                peak_freq[i] = self.freq_fft_rango[np.argmax(spectrum)]
                centroid_freq[i] = np.sum(spectrum * self.freq_fft_rango) / spec_sum

            power_spec = spectrum**2
            denominator = np.sum(power_spec)
            if denominator > 0:
                for band in range(6):
                    part_powers[band, i] = (
                        np.sum(power_spec[bin_edges[band] : bin_edges[band + 1]])
                        * 100.0
                        / denominator
                    )

        w_peak_freq = (peak_freq * centroid_freq) ** 0.5
        return np.hstack(
            (
                peak_freq,
                w_peak_freq,
                centroid_freq,
                part_powers[0],
                part_powers[1],
                part_powers[2],
                part_powers[3],
                part_powers[4],
                part_powers[5],
            )
        )

    def read_transient(self, trai: int) -> tuple[np.ndarray, np.ndarray]:
        """Load `(time_us, amp_mv)` for a given `trai` from `self.tradb_path`."""
        return load_transient_wave(self.tradb_path, trai)

    def feature_extr(
        self,
        threshold: float,
        counts: int,
        label: int | str,
        test_id: str,
        max_trai: int | None = None,
        min_trai: int = 1,
    ) -> pd.DataFrame:
        """Extract frequency and time-domain features for all filtered transients."""
        pridb = vae.io.PriDatabase(str(self.pridb_path))
        tradb = vae.io.TraDatabase(str(self.tradb_path))
        df_hits = pridb.read_hits() 

        threshold_V = 10 ** (threshold / 20 - 6)
        threshold_mv = threshold_V * 1e3
        no_saturation = 10 ** (94 / 20 - 6)

        mask = (
            (df_hits["channel"] >= 1)
            & (df_hits["amplitude"] >= threshold_V)
            & (df_hits["amplitude"] <= no_saturation)
            & (df_hits["trai"] >= min_trai)
            & (df_hits["counts"] >= counts)
        )
        if max_trai is not None:
            mask &= df_hits["trai"] <= max_trai

        df_hits_filtro = df_hits[mask]
        if df_hits_filtro.empty:
            print(
                f"[WARNING] No transients found for label '{label}' with the given filtering criteria."
            )
            return pd.DataFrame()

        # trai = df_hits_filtro["trai"].to_numpy()
        # N_trai = trai.size
        # v_features_total = np.zeros((N_trai, self.N_seg * self.N_feat_seg))

        n_bins = len(self.freq_fft_rango)
        bin_edges = [int(n_bins * k / 6) for k in range(7)]

        feature_rows: list[np.ndarray] = []
        meta_rows: list[dict[str, Any]] = []

        for hit in df_hits_filtro.itertuples():
            trans = int(hit.trai)
            amp_v, _ = tradb.read_wave(trans, time_axis=False)
            amp_mv = amp_v * 1e3
            total_sq_energy = float(np.sum(amp_mv**2))

            if self.segmentation_params is not None:
                segments = segment_merged_hits(
                    waveform=amp_mv,
                    **self.segmentation_params,
                )

                # Fallback to single full segment if no peak crossed the segmentation threshold
                if not segments:
                    segments = [
                        SubHitSegment(
                            sub_trai=1,
                            start_idx=0,
                            end_idx=len(amp_mv),
                            onset_idx=0,
                            peak_idx=int(np.argmax(np.abs(amp_mv))),
                        )
                    ]

                for seg in segments:
                    sub_amp = amp_mv[seg.start_idx : seg.end_idx]
                    if sub_amp.size == 0:
                        continue

                    # Count positive threshold crossings in this sub-hit
                    above = sub_amp >= threshold_mv
                    sub_counts = int(np.sum(above[1:] & ~above[:-1]) + int(above[0]))
                    if sub_counts < counts:
                        continue

                    peak_mv = float(np.max(np.abs(sub_amp)))
                    if peak_mv <= 0:
                        continue

                    sub_amp_dbae = 20.0 * (np.log10(peak_mv * 1e-3) + 6.0)
                    energy_ratio = (
                        float(np.sum(sub_amp**2)) / total_sq_energy
                        if total_sq_energy > 0
                        else 1.0
                    )
                    sub_time_s = float(hit.time) + (
                        seg.onset_idx * self.time_samp * 1e-6
                    )
                    sub_rise_us = max(
                        0.0, (seg.peak_idx - seg.onset_idx) * self.time_samp
                    )

                    feat_vec = self._extract_fft_vector(
                        waveform_mv=sub_amp,
                        offset_pts=self.sub_signal_offset,
                        bin_edges=bin_edges,
                    )
                    feature_rows.append(feat_vec)
                    meta_rows.append(
                        {
                            "rise_time": sub_rise_us,
                            "time": sub_time_s,
                            "amplitude": sub_amp_dbae,
                            "energy": float(hit.energy) * energy_ratio,
                            "counts": sub_counts,
                            "label": label,
                            "trai": trans,
                            "sub_trai": int(seg.sub_trai),
                            "hit_id": f"{trans}_{seg.sub_trai}",
                            "test_id": test_id,
                        }
                    )

            else:
                feat_vec = self._extract_fft_vector(
                    waveform_mv=amp_mv,
                    offset_pts=self.signal_offset,
                    bin_edges=bin_edges,
                )
                feature_rows.append(feat_vec)
                meta_rows.append(
                    {
                        "rise_time": float(hit.rise_time),
                        "time": float(hit.time),
                        "amplitude": 20.0 * (np.log10(float(hit.amplitude)) + 6.0),
                        "energy": float(hit.energy),
                        "counts": int(hit.counts),
                        "label": label,
                        "trai": trans,
                        "sub_trai": 1,
                        "hit_id": f"{trans}_1",
                        "test_id": test_id,
                    }
                )

        if not feature_rows:
            return pd.DataFrame()
        
        # for N_iter, trans in enumerate(trai):
        #     amp, _ = tradb.read_wave(trans, time_axis=False)
        #     amp *= 1e3  # mV

        #     seg_total = np.zeros((self.N_seg, self.N_samp))
        #     for i in range(self.N_seg):
        #         seg_total[i] = amp[
        #             int(i * self.N_samp / 2 + self.signal_offset) : int(
        #                 (i / 2 + 1) * self.N_samp + self.signal_offset
        #             )
        #         ]

        #     amp_fft = np.abs(fft(seg_total * self.window))
        #     amp_fft_rango = amp_fft[
        #         :, self.lower_freq_index : self.upper_freq_index + 1
        #     ]

        #     peak_freq = np.zeros(self.N_seg)
        #     centroid_freq = np.zeros(self.N_seg)
        #     part_powers = np.zeros((6, self.N_seg))

        #     for i in range(self.N_seg):
        #         spectrum = amp_fft_rango[i]
        #         peak_freq[i] = self.freq_fft_rango[np.argmax(spectrum)]
        #         centroid_freq[i] = np.sum(spectrum * self.freq_fft_rango) / np.sum(
        #             spectrum
        #         )

        #         power_spec = spectrum**2
        #         denominator = np.sum(power_spec)
        #         for band in range(6):
        #             part_powers[band, i] = (
        #                 np.sum(power_spec[bin_edges[band] : bin_edges[band + 1]])
        #                 * 100.0
        #                 / denominator
        #             )

        #     w_peak_freq = (peak_freq * centroid_freq) ** 0.5

        #     v_features = np.hstack(
        #         (
        #             peak_freq,
        #             w_peak_freq,
        #             centroid_freq,
        #             part_powers[0],
        #             part_powers[1],
        #             part_powers[2],
        #             part_powers[3],
        #             part_powers[4],
        #             part_powers[5],
        #         )
        #     )
        #     v_features_total[N_iter] = v_features.reshape(
        #         1, self.N_feat_seg * self.N_seg
        #     )

        base_features = [
            "peak_freq",
            "w_peak_freq",
            "centroid_freq",
            *(f"p_power_{k}" for k in range(1, 7)),
        ]

        name_feature = (
            base_features
            if self.N_seg == 1
            else [
                f"{feat}_{seg}"
                for feat in base_features
                for seg in range(1, self.N_seg + 1)
            ]
        )

        df_features = pd.DataFrame(np.vstack(feature_rows), columns=name_feature)
        df_meta = pd.DataFrame(meta_rows)

        # df_time = (
        #     df_hits_filtro[["rise_time", "time", "amplitude", "energy", "counts"]]
        #     .reset_index()
        #     .drop(columns=["set_id"], errors="ignore")
        # )
        # df_time["amplitude"] = 20 * (np.log10(df_time["amplitude"]) + 6)

        if self.force is not None:
            load_offset = df_meta["time"].iloc[-1] - self.force["Time [s]"].iloc[-1]
        else:
            load_offset = self.load_offset

        df_meta["time"] = df_meta["time"] - load_offset
        df_meta["time_norm"] = df_meta["time"] / df_meta["time"].max()

        # df_label = pd.DataFrame({"label": [label] * N_trai})
        # df_total = pd.concat([df_features, df_time, df_label, df_trai], axis=1)
        # df_total["test_id"] = test_id

        return pd.concat([df_features, df_meta], axis=1)

    def plot_signal(
        self,
        trai: int,
        name_figure: str,
        figure_path: str | Path | None = None,
        time_graph: float | None = None,
        title: str | None = None,
        x_label: str = "Time [µs]",
        y_label: str = "Amplitude [mV]",
        figsize: tuple[float, float] = IMG_SIZE,
        ax: plt.Axes | None = None,
        save: bool = False,
        show: bool = True,
    ) -> plt.Axes:
        """Convenience wrapper to plot a transient directly from `self.tradb_path`."""
        time_us, amp_mv = self.read_transient(trai)
        plot_title = title or f"Transient Wave Plot: {name_figure} - Transient: {trai}"
        target = resolve_save_target(figure_path, f"Transient_{name_figure}_{trai}.pdf")

        return plot_signal(
            time_us=time_us,
            amp_mv=amp_mv,
            figure_path=target,
            time_graph=time_graph,
            signal_offset_us=self.signal_offset_us,
            sampling_rate=self.sampling_rate,
            title=plot_title,
            x_label=x_label,
            y_label=y_label,
            figsize=figsize,
            ax=ax,
            save=save,
            show=show,
        )

    def plot_segmentation(
        self,
        trai: int,
        name_figure: str,
        figure_path: str | Path | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        save: bool = False,
        show: bool = True,
    ) -> np.ndarray:
        """Convenience wrapper to plot segmentation + FFT directly from `self.tradb_path`."""
        time_us, amp_mv = self.read_transient(trai)
        plot_title = title or f"Segmentation: {name_figure} - Transient: {trai}"
        target = resolve_save_target(
            figure_path, f"Segmentation_{name_figure}_{trai}.pdf"
        )

        return plot_segmentation(
            time_us=time_us,
            amp_mv=amp_mv,
            sampling_rate=self.sampling_rate,
            lower_freq=self.lower_freq,
            upper_freq=self.upper_freq,
            n_samp=self.N_samp,
            n_seg=self.N_seg,
            signal_offset_us=self.signal_offset_us,
            figure_path=target,
            title=plot_title,
            figsize=figsize,
            save=save,
            show=show,
        )


def load_transient_wave(
    tradb_path: str | Path,
    trai: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a single transient from a Vallen `.tradb` file.

    Returns
    -------
    time_us : np.ndarray
        Time vector in microseconds [µs].
    amp_mv : np.ndarray
        Signal amplitude in millivolts [mV].
    """
    tradb = vae.io.TraDatabase(str(tradb_path))
    amp_v, time_s = tradb.read_wave(trai, time_axis=True)
    return time_s * 1e6, amp_v * 1e3


def load_transient_csv(
    csv_path: str | Path,
    time_col: str | int = 0,
    amp_col: str | int = 1,
    time_scale: float = 1.0,
    amp_scale: float = 1.0,
    sep: str = ",",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a transient waveform from a `.csv` file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    time_col, amp_col : str or int, default=(0, 1)
        Column names or positional indices for time and amplitude.
    time_scale, amp_scale : float, default=1.0
        Multiplicative unit conversion factors (e.g., `1e6` for s -> µs, `1e3` for V -> mV).
    sep : str, default=","
        CSV delimiter.

    Returns
    -------
    time_us : np.ndarray
    amp_mv : np.ndarray
    """
    df = pd.read_csv(csv_path, sep=sep)
    t = (
        df[time_col].to_numpy()
        if isinstance(time_col, str)
        else df.iloc[:, time_col].to_numpy()
    )
    a = (
        df[amp_col].to_numpy()
        if isinstance(amp_col, str)
        else df.iloc[:, amp_col].to_numpy()
    )
    return t * time_scale, a * amp_scale


def compute_segmented_fft(
    amp_mv: np.ndarray,
    sampling_rate: float,
    n_samp: int = 1024,
    n_seg: int = 1,
    lower_freq: float = 100.0,
    upper_freq: float = 900.0,
    signal_offset_us: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Segment a waveform with 50% overlap, apply a Hamming window, and compute FFT spectra.

    Parameters
    ----------
    amp_mv : np.ndarray
        Waveform amplitude in mV.
    sampling_rate : float
        Sampling frequency in MHz (samples/µs).
    n_samp : int
        Samples per segment (power of 2).
    n_seg : int
        Number of 50%-overlapping segments.
    lower_freq, upper_freq : float
        Frequency band of interest in kHz.
    signal_offset_us : float
        Start offset in µs.

    Returns
    -------
    dict with keys:
        `seg_indices`, `seg_windowed`, `freq_range`, `amp_fft_range`,
        `peak_freq`, `centroid_freq`.
    """
    offset_pts = int(signal_offset_us * sampling_rate)
    res_freq = sampling_rate * 1000.0 / n_samp  # kHz
    time_samp = 1.0 / sampling_rate  # µs

    lower_idx = int(lower_freq // res_freq)
    upper_idx = int(upper_freq // res_freq + 1)

    window = hamming(n_samp)
    freq_fft = (fftfreq(n_samp, time_samp * 1e-6) / 1000.0)[: n_samp // 2]
    freq_range = freq_fft[lower_idx : upper_idx + 1]

    seg_total = np.zeros((n_seg, n_samp))
    seg_indices: list[tuple[int, int]] = []

    for i in range(n_seg):
        start = int(i * n_samp / 2) + offset_pts
        end = int((i / 2 + 1) * n_samp) + offset_pts
        seg_total[i] = amp_mv[start:end]
        seg_indices.append((start, end))

    seg_windowed = seg_total * window
    amp_fft = np.abs(fft(seg_windowed))
    amp_fft_range = amp_fft[:, lower_idx : upper_idx + 1]

    peak_freq = np.zeros(n_seg)
    centroid_freq = np.zeros(n_seg)

    for i in range(n_seg):
        peak_idx = int(np.argmax(amp_fft_range[i]))
        peak_freq[i] = freq_range[peak_idx]
        centroid_freq[i] = np.sum(amp_fft_range[i] * freq_range) / np.sum(
            amp_fft_range[i]
        )

    return {
        "seg_indices": np.array(seg_indices),
        "seg_windowed": seg_windowed,
        "freq_range": freq_range,
        "amp_fft_range": amp_fft_range,
        "peak_freq": peak_freq,
        "centroid_freq": centroid_freq,
    }


# =============================================================================
# Standalone Plotting Functions (mylib-compatible)
# =============================================================================
def plot_signal(
    time_us: np.ndarray,
    amp_mv: np.ndarray,
    figure_path: str | Path | None = None,
    time_graph: float | None = None,
    signal_offset_us: float = 0.0,
    sampling_rate: float | None = None,
    title: str | None = "Transient Wave Plot",
    subtitle: str | None = None,
    x_label: str = "Time [µs]",
    y_label: str = "Amplitude [mV]",
    figsize: tuple[float, float] = IMG_SIZE,
    ax: plt.Axes | None = None,
    i: int = 1,
    n_col: int = 1,
    save: bool = False,
    show: bool = True,
) -> plt.Axes:
    """
    Plot and optionally save a time-domain transient waveform.
    `figsize` is specified in millimeters `(width, height)`.
    """
    primary_color = get_active_palette(1)[0]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(f"{x_label}\n({string.ascii_lowercase[i - 1]})")
        if subtitle is not None:
            ax.set_title(subtitle)
        elif title is not None:
            ax.set_title(title)

    if time_graph is None:
        ax.plot(time_us, amp_mv, color=primary_color, linewidth=0.6)
    else:
        if sampling_rate is not None:
            start_idx = int(signal_offset_us * sampling_rate)
            end_idx = int(time_graph * sampling_rate + 1)
            t_slice = time_us[start_idx:end_idx]
            a_slice = amp_mv[start_idx:end_idx]
        else:
            mask = (time_us >= signal_offset_us) & (time_us <= time_graph)
            t_slice = time_us[mask]
            a_slice = amp_mv[mask]

        ax.plot(t_slice, a_slice, color=primary_color, linewidth=0.6)
        if a_slice.size > 0:
            ax.set_ylim(bottom=a_slice.min() * 1.6, top=a_slice.max() * 1.6)

    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        default_name = f"{title or 'Transient'}.pdf"
        save_fig(fig, resolve_save_target(figure_path, default_name))

    if show:
        plt.show()
    if own_fig:
        plt.close(fig)

    return ax


def plot_segmentation(
    time_us: np.ndarray,
    amp_mv: np.ndarray,
    sampling_rate: float,
    lower_freq: float = 100.0,
    upper_freq: float = 900.0,
    n_samp: int = 1024,
    n_seg: int = 1,
    signal_offset_us: float = 0.0,
    figure_path: str | Path | None = None,
    title: str | None = "Signal Segmentation & FFT",
    figsize: tuple[float, float] | None = None,
    save: bool = False,
    show: bool = True,
) -> np.ndarray:
    """
    Plot windowed time-domain segments (top row) and their FFT magnitude spectra
    with `f_peak` and `f_centroid` markers (bottom row).

    Works identically for `n_seg == 1` and `n_seg > 1`.
    `figsize` is specified in millimeters `(width, height)`.
    """
    if figsize is None:
        figsize = (90 * n_seg, 110)

    dsp = compute_segmented_fft(
        amp_mv=amp_mv,
        sampling_rate=sampling_rate,
        n_samp=n_samp,
        n_seg=n_seg,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        signal_offset_us=signal_offset_us,
    )

    seg_indices = dsp["seg_indices"]
    seg_windowed = dsp["seg_windowed"]
    freq_range = dsp["freq_range"]
    amp_fft_range = dsp["amp_fft_range"]
    peak_freq = dsp["peak_freq"]
    centroid_freq = dsp["centroid_freq"]

    palette = get_active_palette(3)
    sig_color, peak_color, centroid_color = palette[0], palette[1], palette[2]

    fig, axs = plt.subplots(
        2,
        n_seg,
        figsize=img_size_inches(figsize),
        squeeze=False,
    )
    if title is not None:
        fig.suptitle(title)

    seg_t_window_max = np.amax(np.abs(seg_windowed))
    amp_fft_max = np.amax(amp_fft_range)

    for j in range(n_seg):
        start, end = seg_indices[j]

        # Row 0: Windowed time-domain segment
        ax_time = axs[0, j]
        ax_time.plot(
            time_us[start:end], seg_windowed[j], color=sig_color, linewidth=0.8
        )
        ax_time.set_ylim(-1.1 * seg_t_window_max, 1.1 * seg_t_window_max)
        ax_time.set_xlabel("Time [µs]")
        if j == 0:
            ax_time.set_ylabel("Amplitude [mV]")
        ax_time.set_title(f"Segment {j + 1}")
        ax_time.grid(True, linestyle="-", alpha=0.7)
        ax_time.set_axisbelow(True)

        # Row 1: FFT spectrum + Peak & Centroid frequencies
        ax_fft = axs[1, j]
        ax_fft.plot(freq_range, amp_fft_range[j], color=sig_color, linewidth=0.8)
        ax_fft.set_ylim(0, amp_fft_max * 1.1)

        y_max = np.amax(amp_fft_range[j]) / (amp_fft_max * 1.1)
        ax_fft.axvline(
            x=peak_freq[j],
            ymax=min(y_max * 1.05, 1.0),
            color=peak_color,
            linestyle="--",
            linewidth=1.0,
            label=f"$f_{{peak}}$ ({peak_freq[j]:.0f} kHz)",
        )
        ax_fft.axvline(
            x=centroid_freq[j],
            ymax=min(y_max * 0.85, 1.0),
            color=centroid_color,
            linestyle=":",
            linewidth=1.0,
            label=f"$f_{{centroid}}$ ({centroid_freq[j]:.0f} kHz)",
        )

        ax_fft.set_xlabel("Frequency [kHz]")
        if j == 0:
            ax_fft.set_ylabel("FFT Magnitude [mV]")
        ax_fft.legend(loc="upper right")
        ax_fft.grid(True, linestyle="-", alpha=0.7)
        ax_fft.set_axisbelow(True)

    if save:
        default_name = f"{title or 'Segmentation'}.pdf"
        save_fig(fig, resolve_save_target(figure_path, default_name))

    if show:
        plt.show()
    plt.close(fig)

    return axs


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


def plot_tsne(
    X,
    y,
    figure_path=None,
    title="t-SNE Scatter Plot",
    x_label="t-SNE Dimension 1",
    y_label="t-SNE Dimension 2",
    figsize: tuple[float, float] = IMG_SIZE,
    selected_features=None,
    save: bool = False,
    show: bool = True,
):
    """
    Plot and optionally save a 2D t-SNE scatter plot with class labels and markers.
    `figsize` is specified in millimeters `(width, height)`.
    """
    if selected_features is not None:
        X = X[selected_features]

    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42)
    X_reduced = tsne.fit_transform(X)

    tsne_df = pd.DataFrame(X_reduced, columns=[f"t-SNE_{i + 1}" for i in range(2)])
    y_series = pd.Series(y, name="Labels").reset_index(drop=True)
    tsne_df["Labels"] = y_series

    label_order = list(pd.unique(tsne_df["Labels"]))
    n_labels = len(label_order)
    markers = {
        label: MARKER_LIST[idx % len(MARKER_LIST)]
        for idx, label in enumerate(label_order)
    }
    palette = get_active_palette(n_labels)

    fig, ax = plt.subplots(figsize=img_size_inches(figsize))

    sns.scatterplot(
        x="t-SNE_1",
        y="t-SNE_2",
        hue="Labels",
        style="Labels",
        hue_order=label_order,
        style_order=label_order,
        data=tsne_df,
        palette=palette,
        markers=markers,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.set_axisbelow(True)

    if save:
        save_fig(fig, resolve_save_target(figure_path, f"{title}.pdf"))

    if show:
        plt.show()
    plt.close(fig)

    return ax


def plot_histogram(
    X,
    y,
    figure_path=None,
    figsize: tuple[float, float] = IMG_SIZE,
    bins: int = 30,
    feature_to_save: str = "",
    save: bool = False,
    show: bool = True,
):
    """
    Plot feature histograms grouped by class and optionally save `feature_to_save`.
    `figsize` is specified in millimeters `(width, height)`.
    """
    class_labels = list(pd.unique(y))
    palette = get_active_palette(len(class_labels))

    for feature in X.columns:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))

        for class_label, color in zip(class_labels, palette):
            feat_to_plot = X[y == class_label][feature]
            ax.hist(
                feat_to_plot,
                bins=bins,
                alpha=0.6,
                label=class_label,
                color=color,
                edgecolor="black",
            )

        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram - {feature}")
        ax.legend()

        if save and (feature == feature_to_save):
            save_fig(
                fig,
                resolve_save_target(figure_path, f"Histogram_{feature_to_save}.pdf"),
            )

        if show:
            plt.show()
        plt.close(fig)


def plot_feat_vs_feat(
    X,
    y,
    feat_x,
    feat_y,
    fig_path="Feat_vs_Feat.png",
    title=None,
    subtitle=None,
    x_label=None,
    y_label=None,
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    save=False,
    show=False,
):
    """
    Plot and optionally save a scatter plot of two features with class labels and markers.
    `figsize` is specified in millimeters `(width, height)`.
    """
    y = pd.Series(y, name="Labels")
    if "Labels" in X.columns:
        X = X.drop(["Labels"], axis=1)

    y.index = X.index
    data = pd.concat([X, y], axis=1)
    label_order = list(pd.unique(data["Labels"]))
    n_labels = len(label_order)

    markers = {
        label: MARKER_LIST[idx % len(MARKER_LIST)]
        for idx, label in enumerate(label_order)
    }
    palette = get_active_palette(n_labels)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"{feat_x} vs {feat_y}" if title is None else title)
        ax.set_xlabel(feat_x if x_label is None else x_label)
    else:
        fig = ax.get_figure()
        base_xlabel = feat_x if x_label is None else x_label
        ax.set_xlabel(f"{base_xlabel}\n({string.ascii_lowercase[i - 1]})")
        ax.set_title(subtitle if subtitle is not None else f"{feat_x} vs {feat_y}")

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

    if (i - 1) % n_col == 0:
        ax.set_ylabel(feat_y if y_label is None else y_label)
    else:
        ax.set_ylabel("")

    if feat_x == "energy":
        ax.set_xscale("log")
    if feat_y == "energy":
        ax.set_yscale("log")

    ax.grid(True)
    ax.set_axisbelow(True)

    if n_labels > 6:
        ax.legend(
            title="Label",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )
        if own_fig:
            fig.subplots_adjust(right=0.72)

    if save:
        save_fig(fig, fig_path)
    if show:
        plt.show()
    if own_fig:
        plt.close(fig)

    return ax


def resolve_feature_columns(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    n_seg: int | None = None,
) -> list[str]:
    """
    Resolve base feature names against `data.columns` for both `N_seg == 1`
    and segmented (`N_seg > 1`) DataFrames.

    - If a name already exists in `data.columns` (e.g., `"energy"` or `"peak_freq"`
      when `N_seg == 1`, or an explicit `"peak_freq_2"`), it is kept as-is.
    - Otherwise, it expands to matching segmented columns (`f"{col}_{seg}"`).
    """
    resolved: list[str] = []
    missing: list[str] = []

    for col in feature_columns:
        if col in data.columns:
            resolved.append(col)
            continue

        if n_seg is not None and n_seg > 1:
            seg_cols = [
                f"{col}_{seg}"
                for seg in range(1, n_seg + 1)
                if f"{col}_{seg}" in data.columns
            ]
        else:
            pattern = re.compile(rf"^{re.escape(col)}_(\d+)$")
            matches = [
                (int(m.group(1)), c) for c in data.columns if (m := pattern.match(c))
            ]
            seg_cols = [c for _, c in sorted(matches, key=lambda item: item[0])]

        if seg_cols:
            resolved.extend(seg_cols)
        else:
            missing.append(col)

    if missing:
        raise KeyError(f"Feature column(s) not found in DataFrame: {missing}")

    # Preserve order while removing any accidental duplicates
    return list(dict.fromkeys(resolved))


def select_significant_indices(
    data: pd.DataFrame,
    label_column: str,
    feature_columns: Sequence[str],
    max_points: int = 20,
    n_seg: int | None = None,
) -> list:
    """
    Select the top `max_points` most significant (highest mean |robust z-score|)
    row indices per class in `label_column`, supporting both `N_seg == 1` and
    segmented (`N_seg > 1`) feature columns.
    """
    resolved_cols = resolve_feature_columns(data, feature_columns, n_seg=n_seg)
    selected_indices = []

    for _, group in data.groupby(label_column, sort=False):
        values = group[resolved_cols].apply(pd.to_numeric, errors="coerce")
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

from __future__ import annotations

import csv
import os
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mylib import (  # type: ignore
    IMG_SIZE,
    MARKER_LIST,
    get_active_palette,
    img_size_inches,
    resolve_save_target,
    save_fig,
)


def force_data(
    path: str | Path,
    section: list[float] | None = None,
    test_id: str | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    path : str or Path
        Directory where the force MTS CSV files are stored.
    section : list or array, optional
        Width and thickness measurements of the specimen in mm. Defaults to [22, 2.4].
    test_id : str, optional
        Identifier assigned to the loaded test. Defaults to the CSV filename stem.

    Returns
    -------
    force_df : DataFrame
        DataFrame containing force, displacement, time, stress, and test_id data.
    """
    if section is None:
        section = [22, 2.4]

    files = os.listdir(path)
    forces_list: list[pd.DataFrame] = []

    for item in files:
        if not item.endswith(".csv"):
            continue

        force_filename_path = Path(path) / item

        # Find the first row whose first column is numeric
        with force_filename_path.open("r", encoding="utf-8-sig", newline="") as file:
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

        # Add a row of zeros where the test starts
        new_row = pd.Series(0, index=force.columns)
        force = pd.concat([pd.DataFrame([new_row]), force], ignore_index=True)

        # Stress
        area = np.prod(section)
        force["Stress [MPa]"] = force["Load [kN]"] * 1000 / area

        # Add the test id column
        current_test_id = item.split(".")[0] if test_id is None else test_id
        force["test_id"] = current_test_id
        forces_list.append(force)

    if not forces_list:
        print(f"[INFO] No .csv files found in {path}. Please check the directory.")
        return pd.DataFrame()

    return pd.concat(forces_list, ignore_index=True)


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
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    save=False,
    show=True,
):
    """
    Plot and optionally save cumulative hits per cluster and stress vs time.
    `figsize` is specified in millimeters `(width, height)`.
    """
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    condition = data["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    filtered_hits = data.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = filtered_hits.groupby("label")[
        "Count"
    ].cumsum()
    filtered_force = force[condition_force]

    num_labels = filtered_hits["label"].nunique()
    markers = [MARKER_LIST[idx % len(MARKER_LIST)] for idx in range(num_labels)]
    palette = get_active_palette(num_labels)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"Stress and Cumulative hits vs Time - {test_id}")
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        ax.set_title(subtitle if subtitle is not None else test_id)

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

    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)

    ax2 = ax.twinx()
    sns.lineplot(
        x="Time [s]",
        y="Stress [MPa]",
        data=filtered_force,
        ax=ax2,
        linewidth=1,
        color="black",
        label="Stress",
    )

    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force["Stress [MPa]"]) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)

    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel("")

    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    if i == 1:
        ax2.legend(handles, legends, loc="upper left")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()

    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        save_fig(fig, figure_path)

    if show:
        plt.show()
    if own_fig:
        plt.close(fig)

    return ax


def plot_stress_hits(
    hits,
    force,
    test_id,
    figure_path=None,
    plot_type="scatter",
    title="Stress and Cumulative hits vs Time",
    x="time",
    y="Cumulative_Label",
    x_label="Time [s]",
    y_label="Cumulative hits",
    y_label_right="Stress [MPa]",
    subtitle=None,
    limits=None,
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    n_row=1,
    save=False,
):
    """
    Plot and optionally save cumulative hits and stress vs time.
    `figsize` is specified in millimeters `(width, height)`.
    """
    if limits is None:
        time_limit, hits_limit, stress_limit = [None, None, None]
    else:
        time_limit, hits_limit, stress_limit = limits

    condition = hits["test_id"] == test_id
    condition_force = force["test_id"] == test_id

    filtered_hits = hits.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = filtered_hits["Count"].cumsum()
    filtered_force = force[condition_force]

    primary_color = get_active_palette(1)[0]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"Stress and Cumulative hits vs Time - {test_id}")
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        ax.set_title(subtitle if subtitle is not None else test_id)

    if plot_type == "scatter":
        sns.scatterplot(
            x=x,
            y=y,
            data=filtered_hits,
            color=primary_color,
            label="hits",
            linewidth=0.2,
            s=20,
            ax=ax,
        )
    elif plot_type == "line":
        sns.lineplot(
            x=x,
            y=y,
            label="hits",
            data=filtered_hits,
            color=primary_color,
            linewidth=2,
            ax=ax,
        )

    if time_limit is not None:
        ax.set_xlim(left=0, right=time_limit * 1.05)

    if (i - 1) % n_col == 0:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    if hits_limit is None:
        ax.set_ylim(bottom=0, top=max(filtered_hits[y]) * 1.1)
    else:
        ax.set_ylim(bottom=0, top=hits_limit * 1.1)

    ax2 = ax.twinx()
    sns.lineplot(
        x="Time [s]",
        y="Stress [MPa]",
        data=filtered_force,
        ax=ax2,
        linewidth=1,
        color="black",
        label="Stress",
    )

    if stress_limit is None:
        ax2.set_ylim(bottom=0, top=max(filtered_force["Stress [MPa]"]) * 1.1)
    else:
        ax2.set_ylim(bottom=0, top=stress_limit * 1.1)

    if i % n_col == 0:
        ax2.set_ylabel(y_label_right)
    else:
        ax2.set_ylabel("")

    handles, legends = [], []
    for ax_temp in [ax, ax2]:
        for handle, label in zip(*ax_temp.get_legend_handles_labels()):
            handles.append(handle)
            legends.append(label)

    if i == 1:
        ax2.legend(handles, legends, loc="upper left")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()

    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        default_name = f"{title} - {test_id}.pdf" if own_fig else f"{title}.pdf"
        save_fig(fig, resolve_save_target(figure_path, default_name))

    return ax


def plot_feature_time_histogram(
    data,
    figure_path=None,
    title="Feature distribution vs Normalized time",
    x="time_norm",
    y="peak_freq",
    x_label="Normalized time",
    y_label="Peak frequency [kHz]",
    subtitle=None,
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    save=False,
):
    """
    Plot and optionally save a 2D histogram of feature distribution vs normalized time.
    `figsize` is specified in millimeters `(width, height)`.
    """
    data = data[data["time_norm"] < 0.99]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"{y} vs Normalized time")
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        ax.set_title(subtitle if subtitle is not None else y)

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
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        default_name = f"{title} - {y}.pdf" if own_fig else f"{title}.pdf"
        save_fig(fig, resolve_save_target(figure_path, default_name))

    return ax


def plot_feature_histogram(
    data,
    figure_path=None,
    title="Feature distribution vs Normalized time",
    x="peak_freq",
    x_label="Peak frequency [kHz]",
    subtitle=None,
    figsize: tuple[float, float] = IMG_SIZE,
    ax=None,
    i=1,
    n_col=1,
    save=False,
):
    """
    Plot and optionally save a 1D histogram of a feature distribution.
    `figsize` is specified in millimeters `(width, height)`.
    """
    primary_color = get_active_palette(1)[0]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=img_size_inches(figsize))
        ax.set_title(f"{x} Histogram")
        ax.set_xlabel(x_label)
    else:
        fig = ax.get_figure()
        ax.set_xlabel(x_label + "\n(" + string.ascii_lowercase[i - 1] + ")")
        ax.set_title(subtitle if subtitle is not None else x)

    sns.histplot(
        data=data,
        x=x,
        bins=40,
        pmax=0.95,
        color=primary_color,
        ax=ax,
    )

    if (i - 1) % n_col == 0:
        ax.set_ylabel("Event count")
    else:
        ax.set_ylabel("")

    ax.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    if save:
        default_name = f"{title} - {x}.pdf" if own_fig else f"{title}.pdf"
        save_fig(fig, resolve_save_target(figure_path, default_name))

    return ax


def limit_finder(data, force, test_ids=None):
    """
    Compute upper axis limits [time_limit, hits_limit, stress_limit] across labeled tests.
    """
    if test_ids is None:
        test_ids = data["test_id"].unique()
    condition = data["test_id"].isin(test_ids)
    condition_force = force["test_id"].isin(test_ids)

    filtered_hits = data.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = None

    for test_id in test_ids:
        single_condition = filtered_hits["test_id"] == test_id
        filtered_hits.loc[single_condition, "Cumulative_Label"] = (
            filtered_hits[single_condition].groupby("label")["Count"].cumsum()
        )

    filtered_force = force[condition_force]

    max_value_time = filtered_hits["time"].max()
    max_value_hits = filtered_hits["Cumulative_Label"].max()
    max_value_stress = filtered_force["Stress [MPa]"].max()

    return [max_value_time, max_value_hits, max_value_stress]


def limit_finder_no_label(hits, force, test_ids):
    """
    Compute upper axis limits [time_limit, hits_limit, stress_limit] across unlabeled tests.
    """
    condition = hits["test_id"].isin(test_ids)
    condition_force = force["test_id"].isin(test_ids)

    filtered_hits = hits.loc[condition].copy()
    filtered_hits["Cumulative_Label"] = None

    for test_id in test_ids:
        single_condition = filtered_hits["test_id"] == test_id
        filtered_hits.loc[single_condition, "Cumulative_Label"] = filtered_hits[
            single_condition
        ]["Count"].cumsum()

    filtered_force = force[condition_force]

    max_value_time = filtered_hits["time"].max()
    max_value_hits = filtered_hits["Cumulative_Label"].max()
    max_value_stress = filtered_force["Stress [MPa]"].max()

    return [max_value_time, max_value_hits, max_value_stress]


def reorder_hits(hits, y, y_pred):
    """
    Order hits and predicted class labels by ascending time.
    """
    y_pred_df = pd.Series(y_pred, name="Class")

    hits = hits.loc[y.index].reset_index(drop=True)
    hits = hits.drop("Class", axis=1)

    hits_and_pred = pd.concat([hits, y_pred_df], axis=1)
    hits_and_pred = hits_and_pred.sort_values(by="time", ascending=True)

    hits_ordered = hits_and_pred.drop("Class", axis=1)
    y_pred_ordered = hits_and_pred["Class"].copy()

    return hits_ordered, y_pred_ordered
```

## File: `utils\paths.py`
```python
from mylib.utils.io import ProjectLayout  #type: ignore

PROJECT_DIRS = ['src', 'parameters', 'results', 'utils', 'datasets', 'figures']

# Resolve root and directory paths without recreating folders on import
layout = ProjectLayout(PROJECT_DIRS, create=False)

PROJECT_ROOT = layout.root
SRC_DIR = layout["src"]
PARAMETERS_DIR = layout["parameters"]
RESULTS_DIR = layout["results"]
UTILS_DIR = layout["utils"]
DATASETS_DIR = layout["datasets"]
FIGURES_DIR = layout["figures"]
EXAMPLE_DATA_DIR = DATASETS_DIR / "example_data" / "data" / "FV_mod1_1"
EXAMPLE_TRADB = EXAMPLE_DATA_DIR / "FV_mod1_1.tradb"

```

## File: `utils\segmentation.py`
```python
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import vallenae as vae
from mylib import Profiler, img_size_inches, resolve_path, save_fig  # type: ignore
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert

from utils.paths import FIGURES_DIR, RESULTS_DIR

PARAM_COLUMNS = [
    "sampling_rate",
    "threshold_dbae",
    "pre_trigger_us",
    "smoothing_us",
    "min_separation_us",
    "min_prominence_ratio",
    "valley_decay_ratio",
    "min_secondary_ratio",
]


@dataclass(frozen=True)
class SubHitSegment:
    """Boundaries and landmark indices for an isolated sub-hit within a transient."""

    sub_trai: int
    start_idx: int
    end_idx: int
    onset_idx: int
    peak_idx: int


def dbae_to_mV(dbae: float) -> float:
    """Convert acoustic emission amplitude from dBAE (ref. 1 uV) to mV."""
    return float(10.0 ** ((dbae / 20.0) - 3.0))


def compute_smoothed_envelope(
    waveform: npt.NDArray[np.floating],
    sampling_rate: float = 5.0,
    smoothing_us: float = 80.0,
) -> npt.NDArray[np.float64]:
    """
    Compute the zero-phase smoothed Hilbert envelope [mV] of a transient signal.
    Parameters:
    - waveform: 1D array of transient signal amplitudes [mV].
    - sampling_rate: Sampling frequency [MHz].
    - smoothing_us: Smoothing window width [µs] for the uniform filter.

    Returns:
    - smooth_env: 1D array of the smoothed Hilbert envelope [mV].
    """

    dt_us = 1 / sampling_rate
    smooth_samples = max(1, round(smoothing_us / dt_us))
    analytic_env = np.abs(hilbert(waveform))
    return uniform_filter1d(
        analytic_env.astype(np.float64), size=smooth_samples, mode="nearest"
    )


def _find_preceding_valley(
    smooth_env: npt.NDArray[np.float64],
    peak_idx: int,
) -> int:
    """Walk backward from `peak_idx` to locate the nearest local minimum."""
    idx = peak_idx
    while idx > 0 and smooth_env[idx - 1] <= smooth_env[idx]:
        idx -= 1
    return idx


def _refine_arrival_onset(
    waveform: npt.NDArray[np.floating],
    smooth_env: npt.NDArray[np.float64],
    valley_idx: int,
    peak_idx: int,
    rise_fraction: float = 0.10,
) -> int:
    """Pinpoint the arrival onset (t0) between a preceding valley and burst peak."""
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
    sampling_rate: float = 5.0,
    threshold_dbae: float = 30.0,
    pre_trigger_us: float = 40.0,
    smoothing_us: float = 100.0,
    min_separation_us: float = 300.0,
    min_prominence_ratio: float = 0.25,
    valley_decay_ratio: float = 0.35,
    min_secondary_ratio: float = 0.10,
    onset_rise_fraction: float = 0.10,
) -> list[SubHitSegment]:
    """
    Split a transient `waveform` [mV] into individual `SubHitSegment` objects when
    multiple AE bursts overlap within the same acquisition window.
    """
    n_samples = waveform.size
    if n_samples == 0:
        return []

    # 1. Compute zero-phase smoothed Hilbert envelope [mV]
    dt_us = 1 / sampling_rate
    threshold_mV = dbae_to_mV(threshold_dbae)
    smooth_env = compute_smoothed_envelope(
        waveform, sampling_rate=sampling_rate, smoothing_us=smoothing_us
    )

    # 2. Detect candidate peaks using both height and prominence to reject coda ripples
    min_dist_samples = max(1, round(min_separation_us / dt_us))
    pre_trigger_samples = max(0, round(pre_trigger_us / dt_us))
    min_prominence = max(threshold_mV * 0.5, min_prominence_ratio * threshold_mV)

    candidate_peaks, _ = find_peaks(
        smooth_env,
        height=threshold_mV,
        distance=min_dist_samples,
        prominence=min_prominence,
    )

    if candidate_peaks.size == 0:
        peak_idx = int(np.argmax(smooth_env))
        return [
            SubHitSegment(
                sub_trai=1,
                start_idx=0,
                end_idx=n_samples,
                onset_idx=min(pre_trigger_samples, peak_idx),
                peak_idx=peak_idx,
            )
        ]

    # 3. Validate true multi-hit boundaries via valley decay and relative amplitude
    valid_peaks = [int(candidate_peaks[0])]
    split_valleys: list[int] = []

    for raw_peak in candidate_peaks[1:]:
        curr_peak = int(raw_peak)
        prev_peak = valid_peaks[-1]

        valley_rel_idx = int(np.argmin(smooth_env[prev_peak:curr_peak]))
        valley_idx = prev_peak + valley_rel_idx
        valley_amp = float(smooth_env[valley_idx])
        prev_amp = float(smooth_env[prev_peak])
        curr_amp = float(smooth_env[curr_peak])
        smaller_peak_amp = min(prev_amp, curr_amp)

        has_deep_valley = valley_amp <= valley_decay_ratio * smaller_peak_amp
        is_significant_burst = curr_amp >= min_secondary_ratio * prev_amp

        if has_deep_valley and is_significant_burst:
            valid_peaks.append(curr_peak)
            split_valleys.append(valley_idx)
        elif curr_amp > prev_amp:
            # Replace dominant peak if an early precursor preceded the main burst
            valid_peaks[-1] = curr_peak

    # 4. Locate physical arrival onsets (t0) and apply pre-trigger buffer
    onsets: list[int] = []
    starts: list[int] = []

    # First burst: realign t0 if a sub-threshold precursor triggered early
    first_peak = valid_peaks[0]
    if first_peak * dt_us > (pre_trigger_us + min_separation_us):
        initial_valley = _find_preceding_valley(smooth_env, first_peak)
        first_onset = _refine_arrival_onset(
            waveform, smooth_env, initial_valley, first_peak, onset_rise_fraction
        )
    else:
        first_onset = min(pre_trigger_samples, first_peak)

    onsets.append(first_onset)
    starts.append(max(0, first_onset - pre_trigger_samples))

    # Subsequent bursts: clamp pre-trigger at the preceding valley to avoid tail leakage
    for idx, valley_idx in enumerate(split_valleys):
        next_peak = valid_peaks[idx + 1]
        next_onset = _refine_arrival_onset(
            waveform, smooth_env, valley_idx, next_peak, onset_rise_fraction
        )
        onsets.append(next_onset)
        starts.append(max(valley_idx, next_onset - pre_trigger_samples))

    # 5. Build SubHitSegment objects
    segments: list[SubHitSegment] = []
    n_bursts = len(valid_peaks)

    for i in range(n_bursts):
        if i == n_bursts - 1:
            end_idx = n_samples
        else:
            end_idx = starts[i + 1]

        segments.append(
            SubHitSegment(
                sub_trai=i + 1,
                start_idx=starts[i],
                end_idx=end_idx,
                onset_idx=onsets[i],
                peak_idx=valid_peaks[i],
            )
        )

    return segments


def _format_segmentation_report(
    waveform: npt.NDArray[np.floating],
    time: npt.NDArray[np.floating],
    segments: list[SubHitSegment],
    params: dict[str, Any],
    trai: int | None = None,
    test_id: str | None = None,
) -> str:
    """Format a plain-text report summarizing parameters and extracted sub-hits."""
    sampling_rate = float(params.get("sampling_rate", 5.0))
    dt_us = 1 / sampling_rate
    lines = [
        "=" * 72,
        "SUB-HIT SEGMENTATION REPORT",
        "=" * 72,
        f"Test ID          : {test_id or 'N/A'}",
        f"TRAI             : {trai if trai is not None else 'N/A'}",
        f"Total Samples    : {len(waveform)} ({len(waveform) * dt_us:.2f} us)",
        f"Segments Found   : {len(segments)}",
        "-" * 72,
        "PARAMETERS:",
    ]
    for key, val in params.items():
        lines.append(f"  {key:<22}: {val}")

    lines.extend(["-" * 72, "SEGMENTS:"])
    for seg in segments:
        t_start = time[seg.start_idx] * 1e6
        t_onset = time[seg.onset_idx] * 1e6
        t_peak = time[seg.peak_idx] * 1e6
        t_end = time[seg.end_idx - 1] * 1e6
        dur_us = (seg.end_idx - seg.start_idx) * dt_us
        pre_trig_us = (seg.onset_idx - seg.start_idx) * dt_us
        peak_mv = float(np.max(np.abs(waveform[seg.start_idx : seg.end_idx])))
        lines.append(
            f"  Sub-hit {seg.sub_trai}: [{seg.start_idx}:{seg.end_idx}] | "
            f"Start={t_start:.2f} us | Onset={t_onset:.2f} us (PreTrig={pre_trig_us:.2f} us) | "
            f"Peak={t_peak:.2f} us ({peak_mv:.2f} mV) | End={t_end:.2f} us | Dur={dur_us:.2f} us"
        )
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


def plot_subhit_segmentation(
    waveform: npt.NDArray[np.floating],
    time: npt.NDArray[np.floating],
    segments: list[SubHitSegment],
    params: dict[str, Any],
    figure_path: Path | str = "subhit_segmentation.png",
    smooth_env: npt.NDArray[np.float64] | None = None,
    trai: int | None = None,
    test_id: str | None = None,
    title: str | None = None,
    show_title: bool = True,
    show_subplot_titles: bool = True,
    legend_loc: str = "lower right",
    figsize: tuple[float, float] = (140, 90),
    save: bool = True,
    show: bool = True,
    save_txt: bool = True,
) -> np.ndarray:
    """Plot transient envelope boundaries and extracted sub-hits using mylib.
    Parameters:
    - waveform: 1D array of transient signal amplitudes [mV].
    - time: 1D array of time values [s] corresponding to `waveform`.
    - segments: List of `SubHitSegment` objects defining extracted sub-hits.
    - params: Dictionary of segmentation parameters used for this analysis.
    - figure_path: Path to save the generated figure.
    - smooth_env: Optional pre-computed smoothed Hilbert envelope [mV].
    - trai: Optional transient ID for labeling.
    - test_id: Optional test identifier for labeling.
    - title: Optional custom title for the figure.
    - figsize: Tuple specifying figure size in inches (width, height).
    - save: Whether to save the figure to `figure_path`.
    - show: Whether to display the figure interactively.
    - save_txt: Whether to save a plain-text segmentation report alongside the figure.
    Returns:
    - axs: The matplotlib Axes array containing the generated subplots.
    """

    fig_path = Path(figure_path)
    time_us = time * 1e6
    sampling_rate = float(params.get("sampling_rate", 5.0))
    smoothing_us = float(params.get("smoothing_us", 80.0))
    threshold_dbae = float(params.get("threshold_dbae", 45.0))
    threshold_mV = dbae_to_mV(threshold_dbae)

    if smooth_env is None:
        smooth_env = compute_smoothed_envelope(
            waveform, sampling_rate=sampling_rate, smoothing_us=smoothing_us
        )

    fig, axs = plt.subplots(
        2, 1, figsize=img_size_inches(figsize), sharex=True, tight_layout=True
    )

    if show_title:
        if title is None:
            id_prefix = f"{test_id} - " if test_id else ""
            trai_label = f"TRAI {trai}" if trai is not None else "Transient"
            fig.suptitle(
                f"Sub-hit Segmentation: {id_prefix}{trai_label} ({len(segments)} segments)",
            )
        else:
            fig.suptitle(title)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    axs[0].plot(time_us, waveform, color="0.65", linewidth=0.5, label="Raw waveform")
    axs[0].plot(
        time_us, smooth_env, color="darkorange", linewidth=0.9, label="Hilbert envelope"
    )
    axs[0].axhline(
        threshold_mV,
        color="crimson",
        linestyle=":",
        linewidth=0.7,
        label=f"Threshold ({threshold_dbae:.1f} dBAE)",
    )
    axs[0].axhline(
        -threshold_mV,
        color="crimson",
        linestyle=":",
        linewidth=0.7,
    )

    for idx, seg in enumerate(segments):
        color = colors[idx % len(colors)]
        axs[0].axvspan(
            time_us[seg.start_idx], time_us[seg.end_idx - 1], color=color, alpha=0.12
        )
        axs[0].axvline(
            time_us[seg.onset_idx],
            color=color,
            linestyle="--",
            linewidth=0.7,
            label="Onset ($t_0$)" if idx == 0 else None,
        )
        axs[0].plot(
            time_us[seg.peak_idx],
            smooth_env[seg.peak_idx],
            marker="v",
            markersize=4,
            color=color,
            linestyle="None",
            label="Detected peak" if idx == 0 else None,
        )

    axs[0].set_ylabel("Amplitude [mV]")
    if show_subplot_titles:
        axs[0].set_title("Envelope Detection & Event Landmarks", fontsize=8)
    axs[0].grid(True, linestyle="-", linewidth=0.3, alpha=0.7)
    axs[0].set_axisbelow(True)
    axs[0].legend(loc=legend_loc, fontsize=6.5, ncol=2)

    for idx, seg in enumerate(segments):
        color = colors[idx % len(colors)]
        axs[1].plot(
            time_us[seg.start_idx : seg.end_idx],
            waveform[seg.start_idx : seg.end_idx],
            color=color,
            linewidth=0.65,
            label=f"Sub-hit {seg.sub_trai}",
        )
        axs[1].axvline(
            time_us[seg.onset_idx], color=color, linestyle="--", linewidth=0.6
        )

    axs[1].set_xlabel("Time [us]")
    axs[1].set_ylabel("Amplitude [mV]")
    if show_subplot_titles:
        axs[1].set_title("Extracted Sub-hit Waveforms", fontsize=8)
    axs[1].grid(True, linestyle="-", linewidth=0.3, alpha=0.7)
    axs[1].set_axisbelow(True)
    axs[1].legend(loc=legend_loc, fontsize=6.5)

    if save:
        save_fig(fig, fig_path)

    if save_txt:
        txt_path = fig_path.with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(
            _format_segmentation_report(
                waveform=waveform,
                time=time,
                segments=segments,
                params=params,
                trai=trai,
                test_id=test_id,
            ),
            encoding="utf-8",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return axs


def _write_valid_combos_report(
    summary_df: pd.DataFrame,
    expected_map: dict[int, int],
    report_path: Path,
    param_keys: list[str],
    test_id: str = "",
) -> list[str]:
    """Identify combinations matching `expected_map` across all signals and write a report."""
    trai_order = list(expected_map.keys())
    valid_combos: list[str] = []
    combo_scores: list[tuple[int, int, str]] = []

    for combo_id, group in summary_df.groupby("combo_id", sort=False):
        by_trai = dict(zip(group["trai"], group["n_segments"]))
        matches = sum(1 for t, exp in expected_map.items() if by_trai.get(t) == exp)
        abs_error = sum(
            abs(int(by_trai.get(t, 0)) - exp) for t, exp in expected_map.items()
        )
        combo_scores.append((matches, -abs_error, str(combo_id)))

        if matches == len(expected_map) and len(by_trai) == len(expected_map):
            valid_combos.append(combo_id)

    total_combos = summary_df["combo_id"].nunique()
    lines = [
        "=" * 76,
        "GRID SEARCH VALIDATION REPORT",
        "=" * 76,
        f"Test ID            : {test_id or 'N/A'}",
        f"TRAI List          : {trai_order}",
        f"Expected Segments  : {[expected_map[t] for t in trai_order]}",
        f"Combos Evaluated   : {total_combos}",
        f"Valid Combos Found : {len(valid_combos)}",
        "=" * 76,
        "",
    ]

    if valid_combos:
        for combo_id in valid_combos:
            sub = summary_df[summary_df["combo_id"] == combo_id].set_index("trai")
            first_row = sub.iloc[0]
            param_dict = {
                k: (
                    first_row[k].item()
                    if hasattr(first_row[k], "item")
                    else first_row[k]
                )
                for k in param_keys
                if k in first_row
            }
            lines.append(f"[{combo_id}]  ->  {param_dict}")
            for t in trai_order:
                row = sub.loc[t]
                lines.append(
                    f"    TRAI {t:<6} | detected={int(row['n_segments'])} "
                    f"(expected={expected_map[t]}) | "
                    f"durations_us={row['durations_us']} | "
                    f"file={row['figure_file'] or 'not_saved'}"
                )
            lines.append("-" * 76)
    else:
        lines.append("No parameter combination matched all expected segment counts.")
        lines.append(
            "Top 5 closest combinations (ranked by matched signals & lowest error):"
        )
        lines.append("-" * 76)
        combo_scores.sort(reverse=True)
        for matches, neg_err, combo_id in combo_scores[:5]:
            sub = summary_df[summary_df["combo_id"] == combo_id].set_index("trai")
            first_row = sub.iloc[0]
            param_dict = {
                k: (
                    first_row[k].item()
                    if hasattr(first_row[k], "item")
                    else first_row[k]
                )
                for k in param_keys
                if k in first_row
            }
            detected_str = ", ".join(
                f"{t}:{int(sub.loc[t, 'n_segments'])}/{expected_map[t]}"
                for t in trai_order
                if t in sub.index
            )
            lines.append(
                f"[{combo_id}] matched {matches}/{len(expected_map)} "
                f"(total_err={-neg_err}) | {detected_str}"
            )
            lines.append(f"    params = {param_dict}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return valid_combos


def run_segmentation_grid_search(
    tradb_path: Path | str,
    trai_list: list[int],
    param_grid: dict[str, list[Any]],
    fig_dir: Path | str = FIGURES_DIR,
    results_dir: Path | str = RESULTS_DIR,
    expected_segments: list[int] | None = None,
    save_valid_only: bool = False,
    test_id: str = "",
    sampling_rate: float = 5.0,
    img_format: str = "png",
    show_title: bool = True,
    show_subplot_titles: bool = True,
    legend_loc: str = "lower right",
    figsize: tuple[float, float] = (140, 90),
    save: bool = True,
    show: bool = False,
    save_txt: bool = True,
) -> pd.DataFrame:
    """Run a parametric grid search for `segment_merged_hits` across selected transients."""
    expected_map: dict[int, int] | None = None
    if expected_segments is not None:
        if len(expected_segments) != len(trai_list):
            raise ValueError(
                f"expected_segments length ({len(expected_segments)}) must match "
                f"trai_list length ({len(trai_list)})."
            )
        expected_map = dict(zip(trai_list, expected_segments))
    elif save_valid_only:
        raise ValueError(
            "expected_segments must be provided when save_valid_only=True."
        )

    profiler = Profiler()
    with profiler.section("Load .tradb waveforms"):
        tradb = vae.io.TraDatabase(Path(tradb_path))
        waveforms: dict[
            int, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = {}
        for trai in trai_list:
            amp_v, time = tradb.read_wave(trai, time_axis=True)
            waveforms[trai] = (amp_v * 1e3, time)

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    summary_rows: list[dict[str, Any]] = []
    dt_us = 1 / sampling_rate

    t_seg_total = 0.0
    t_plot_total = 0.0

    for combo_idx, values in enumerate(combinations, start=1):
        params = {"sampling_rate": sampling_rate, **dict(zip(keys, values))}

        combo_evals: list[tuple[int, list[SubHitSegment], list[float]]] = []
        with profiler.section("Envelope & segmentation"):
            for trai in trai_list:
                waveform, _ = waveforms[trai]
                segments = segment_merged_hits(waveform=waveform, **params)
                durations_us = [
                    round((s.end_idx - s.start_idx) * dt_us, 2) for s in segments
                ]
                combo_evals.append((trai, segments, durations_us))
        t_seg_total += profiler._times["Envelope & segmentation"]

        if expected_map is not None:
            combo_is_valid = all(
                len(segs) == expected_map[t] for t, segs, _ in combo_evals
            )
        else:
            combo_is_valid = True

        should_export = (not save_valid_only) or combo_is_valid

        for trai, segments, durations_us in combo_evals:
            waveform, time = waveforms[trai]
            prefix = f"{test_id}_" if test_id else ""
            stem = f"{prefix}trai_{trai}_{combo_idx}"
            fig_name = f"{stem}.{img_format}"
            fig_path = fig_dir / fig_name

            if should_export and (save or save_txt or show):
                with profiler.section("Plot & export reports"):
                    smooth_env = compute_smoothed_envelope(
                        waveform,
                        sampling_rate=sampling_rate,
                        smoothing_us=float(params.get("smoothing_us", 80.0)),
                    )
                    plot_subhit_segmentation(
                        waveform=waveform,
                        time=time,
                        segments=segments,
                        params=params,
                        figure_path=fig_path,
                        smooth_env=smooth_env,
                        trai=trai,
                        test_id=test_id,
                        show_title=show_title,
                        show_subplot_titles=show_subplot_titles,
                        legend_loc=legend_loc,
                        figsize=figsize,
                        save=save,
                        show=show,
                        save_txt=save_txt,
                    )
                t_plot_total += profiler._times["Plot & export reports"]

            row: dict[str, Any] = {
                "combo_id": combo_idx,
                "test_id": test_id,
                "trai": trai,
                "n_segments": len(segments),
            }
            if expected_map is not None:
                row["expected_segments"] = expected_map[trai]
                row["valid_signal"] = len(segments) == expected_map[trai]
                row["valid_combo"] = combo_is_valid

            row.update(
                {
                    "durations_us": str(durations_us),
                    "figure_file": fig_name if (should_export and save) else "",
                    **params,
                }
            )
            summary_rows.append(row)

    profiler._times["Envelope & segmentation"] = t_seg_total
    profiler._times["Plot & export reports"] = t_plot_total
    profiler.report()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "grid_search_summary.csv", index=False)

    if expected_map is not None:
        valid_report_path = results_dir / "valid_combinations.txt"
        valid_ids = _write_valid_combos_report(
            summary_df=summary_df,
            expected_map=expected_map,
            report_path=valid_report_path,
            param_keys=["sampling_rate", *keys],
            test_id=test_id,
        )
        print(
            f"[INFO] {len(valid_ids)}/{len(combinations)} combinations matched all "
            f"expected segments. Report saved at: {valid_report_path}"
        )

    return summary_df


def load_segmentation_params(
    seg_cfg: dict[str, Any],
    dataset_name: str,
    default_test_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Resolve `segment_merged_hits()` keyword arguments from the `"segmentation"`
    config block, reading `grid_search_summary.csv` when needed.
    """
    if not seg_cfg.get("enabled", False):
        return None

    # 1. Direct parameter dictionary override
    if seg_cfg.get("parameters"):
        return dict(seg_cfg["parameters"])

    # 2. Locate grid_search_summary.csv
    if seg_cfg.get("summary_csv"):
        summary_path = resolve_path(seg_cfg["summary_csv"])
    else:
        source_test_id = seg_cfg.get("source_test_id") or default_test_id
        if not source_test_id:
            raise ValueError(
                "Specify 'source_test_id' or 'summary_csv' under extraction.segmentation."
            )
        summary_path = (
            RESULTS_DIR
            / dataset_name
            / "segmentation"
            / source_test_id
            / "grid_search_summary.csv"
        )

    if not summary_path.exists():
        raise FileNotFoundError(f"Grid search summary not found: {summary_path}")

    df = pd.read_csv(summary_path)
    combo_id = seg_cfg.get("combo_id")

    if combo_id and str(combo_id).lower() != "best":
        matches = df[df["combo_id"] == combo_id]
        if matches.empty:
            raise ValueError(f"Combo '{combo_id}' not found in {summary_path}.")
    else:
        if "valid_combo" in df.columns and df["valid_combo"].any():
            matches = df[df["valid_combo"]]
        else:
            raise ValueError(
                f"No valid combos found in {summary_path}. Specify an explicit 'combo_id'."
            )

    row = matches.iloc[0]
    params: dict[str, Any] = {}
    for col in PARAM_COLUMNS:
        if col in row and pd.notna(row[col]):
            val = row[col]
            params[col] = val.item() if hasattr(val, "item") else val

    return params

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
