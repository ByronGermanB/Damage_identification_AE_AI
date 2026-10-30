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

        features = Features(
            path=pri_dir,
            lower_freq=parameters["lower_freq"],
            upper_freq=parameters["upper_freq"],
            sampling_rate=parameters["sampling_rate"],
            N_samp=parameters["N_samp"],
            N_seg=parameters["N_seg"],
            desfase=parameters["desfase"],
            desfase_carga=parameters.get("desfase_carga", 0),
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

        # Force data saving if specified in the configuration
        if config["force"].get("force_data", False):
            data_force_i = force_data(
                path=pri_dir,
                section=config["force"].get("section", None),
                test_id=pri_name,
            )
        force_dfs.append(data_force_i)


    # Merge all the individual DataFrames into a single DataFrame
    data = pd.concat(dfs, ignore_index=True)
    data["label"] = 0
    data.to_csv(
        DATASET_DIR / config.get("output_name", "features.csv"), index=False
    )
    print("[INFO] Feature extraction completed. Data saved to CSV.")

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
