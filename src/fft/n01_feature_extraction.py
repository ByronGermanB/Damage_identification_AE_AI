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