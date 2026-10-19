import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from utils.fft_analysis import Features, plot_feat_vs_feat

CONFIG_DIR = Path("parameters") / "trial"
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

    # Merge all the individual DataFrames into a single DataFrame
    df_total = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df_total["label"] = 0
    df_total.to_csv(
        DATASET_DIR / config.get("output_name", "features.csv"), index=False
    )
    print("[INFO] Feature extraction completed. Data saved to CSV.")

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

        feat_vs_feat_dir = FIGURES_DIR / "feat_vs_feat"
        feat_vs_feat_dir.mkdir(parents=True, exist_ok=True)

        if config["plots"].get("labeled", False):
            y = df_total["test_id"]
        else:
            y = df_total["label"]

        for feature_x, feature_y in combinations(features_to_plot, 2):
            figure_path = feat_vs_feat_dir / f"{feature_x}_vs_{feature_y}.png"
            plot_feat_vs_feat(
                X=df_total,
                y=y,
                feat_1=feature_x,
                feat_2=feature_y,
                figure_path=figure_path,
                save=True,
            )
        print("[INFO] Feature vs Feature images saved.")


if __name__ == "__main__":
    main(config_name)
