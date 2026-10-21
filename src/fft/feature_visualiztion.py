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
            figure_path = feat_vs_feat_dir / f"{feature_x}_vs_{feature_y}.png"

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
