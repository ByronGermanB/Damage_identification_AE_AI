import os

import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Directory to save results
results_dir = os.path.join(base_dir, "Results")

predicted_csv = "labels_CNN_model_2.csv"
ground_truth_csv = "labels_dbscan.csv"

# Load CSV files
predicted = pd.read_csv(os.path.join(results_dir, predicted_csv))
ground_truth = pd.read_csv(os.path.join(results_dir, ground_truth_csv))


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

plt.rcParams.update({"font.size": 8, "font.family": "Segoe UI"})
figsize_mm = (90, 60)
fig_size = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)  # Convert mm to inches
plt.figure(figsize=(fig_size))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "CNN_model_2_confusion_matrix.pdf"), dpi=300)
plt.show()
plt.close()

# Save results to a text file
results_file = os.path.join(results_dir, "CNN_model_2_accuracy_results.txt")
with open(results_file, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
