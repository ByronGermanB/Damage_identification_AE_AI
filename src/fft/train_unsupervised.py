import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from mylib.utils.time_report import Profiler  # type: ignore

# Functions for dataset splitting
from utils.fft_analysis import train_test_set

# Functions for DBSCAN model
from utils.unsupervised_functions import (
    grid_search_dbscan,
    kmeans_per_k,
    tsne,
)

CONFIG_DIR = Path("parameters") / "3d_specimen"
config_name = "dbscan.json"


def main(config_name: str) -> None:
    """
    Ex
    """
    timer = Profiler()
    config_path = CONFIG_DIR / config_name

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    dataset_name = config["dataset_name"]
    show_plots = config["plots"]["show"]
    save_plots = config["plots"]["save"]

    models_dir = Path("results") / dataset_name / "models" / "trials"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Directories
    FIGURES_DIR = Path("figures") / dataset_name
    DATASETS_DIR = Path("datasets") / dataset_name

    # Import datasets
    DATA_PATH = DATASETS_DIR / "Dataset_total.csv"

    # Read the datasets
    data = pd.read_csv(DATA_PATH)

    # Separate features and labels
    X_original, X, y, hits = train_test_set(
        data,
        normalization=config.get("normalization", "log-std"),
        columns_to_transform=config.get("columns_to_transform", ["energy"]),
        split=False,
    )  # Normalization: log-std or std


    if config.get("use_tsne", False):
        with timer.section("TSNE"):
            # T-SNE
            X_reduced = tsne(
                X,
                FIGURES_DIR,
                show=show_plots,
                save=save_plots,
            )
    else:
        X_reduced = X

    # =============================================================================
    # K-means by k
    # =============================================================================
    if config.get("use_kmeans", False):
        with timer.section("K-means"):
            k_num = config["dbscan"]["k_num"]
            kmeans_models, silhouette_scores, dbi_kmeans = kmeans_per_k(
                X_reduced, k_num
            )
            kmeans_results = {
                "silhouette_scores": silhouette_scores,
                "dbi_kmeans": dbi_kmeans,
            }

    # =============================================================================
    # DBSCAN
    # =============================================================================
    with timer.section("DBSCAN"):
        # List with dbscan models for k
        dbscan_per_k = []
        dbscan_results_per_k = []

        # Parameters for DBSCAN
        epsilon_values = config["dbscan"]["epsilon_values"]
        min_samples_limits = config["dbscan"]["min_samples_limits"]
        min_samples_num = config["dbscan"]["min_samples_num"]

        min_samples_values = np.linspace(
            min_samples_limits[0], min_samples_limits[1], min_samples_num, dtype=int
        )

        for k in range(2, k_num + 1):
            print(f"\nDBSCAN for k={k} clusters")
            # Adjust the target_clusters for the current k
            # Model and adjusted parameters
            dbscan_models, results = grid_search_dbscan(
                X_reduced, k, epsilon_values, min_samples_values
            )

            dbscan_per_k.append(dbscan_models)
            dbscan_results_per_k.append(results)

    with timer.section("Saving results"):
        # Save the DBSCAN models and results
        dbscan_models_path = models_dir / config["saving_names"]["dbscan_models"]
        dbscan_results_path = models_dir / config["saving_names"]["dbscan_results"]

        dump(dbscan_per_k, dbscan_models_path)
        dump(dbscan_results_per_k, dbscan_results_path)

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
    timer.report()


if __name__ == "__main__":
    main(config_name)
