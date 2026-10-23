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
config_name = "select_model.json"


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
            figure_path=k_means_fig_dir / "silhouette_scores.png",
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
                figure_path=k_means_fig_dir / f"tsne_kmeans_k_{model.n_clusters}.png",
                save=save_plots,
                show=show_plots,
            )

            for feature_x, feature_y in combinations(X_original.columns, 2):
                figure_path = (
                    k_means_fig_dir
                    / f"k{model.n_clusters}_{feature_x}_vs_{feature_y}.png"
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
            / f"tsne_k_{k}_eps_{eps}_min_samples_{min_samples}.png",
            save=save_plots,
            show=show_plots,
        )

        for feature_x, feature_y in combinations(X_original.columns, 2):
            print(f"[INFO] Plotting DBSCAN cluster {feature_x} and {feature_y}...")
            figure_path = (
                dbscan_fig_dir
                / f"k{k}_eps_{eps}_min_samples_{min_samples}_{feature_x}_vs_{feature_y}.png"
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
