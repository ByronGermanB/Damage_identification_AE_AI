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
