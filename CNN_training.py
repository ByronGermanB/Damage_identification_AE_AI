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
figure_path = os.path.join(base_dir, "Figuras")
images_set = "train_3"
images_dir = os.path.join(base_dir, "datasets", "images", images_set)
dbscan_labels_path = os.path.join(base_dir, "Results", "labels_dbscan.csv")

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

# Rescale the images from [0, 255] to [0, 1]
train_dataset = train_dataset.map(lambda x: x / 255.0)
x_images = np.array(list(train_dataset.as_numpy_iterator()))
x_images = x_images.reshape(-1, 128, 128, 1)
# Read the DBSCAN labels
dbscan_labels = pd.read_csv(dbscan_labels_path)

# Define test IDs and repeat counts for each label set
label_sets = {
    "T0": (["P0_2", "P0_3", "P0_4", "P0_5", "P0_6"], 3),
    "T90": (["P90_1", "P90_2", "P90_3"], 3),
    "T090": (["P0_90_1", "P0_90_2", "P0_90_3", "P0_90_4"], 3),
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

# =============================================================================
# Training loop for multiple models
# =============================================================================
# Reset the model number
model_number = 3
# Define variables to store model data
cm_test_prec_list = []
accuracy_arr = []
model_arr = []
history_arr = []
cm_test_arr = []

# Define the iterative variable for model training
random_state = 0
rs_lim = 5

# Once the iteration is complete, check which model has the highest accuracy and store it for later use
while random_state <= rs_lim:
    print("\nStarting iteration: %i\n" % random_state)

    # Split for Train, Validation, and Test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_images, y_images, test_size=0.15, random_state=random_state, stratify=y_images
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train_val,
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
