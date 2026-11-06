import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from utils.force_mts import limit_finder, plot_stress_hits_cluster
from utils.unsupervised_functions import plot_cluster_test_id

CONFIG_DIR = Path("parameters") / "arcan"
config_name = "results_visualization.json"


def main(config_name: str) -> None:
    config_path = CONFIG_DIR / config_name

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

        dataset_name = config["dataset_name"]
        model_name = config["model_name"]
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


if __name__ == "__main__":
    main(config_name)
