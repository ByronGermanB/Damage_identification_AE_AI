from mylib.utils.io import ProjectLayout  #type: ignore

PROJECT_DIRS = ['src', 'parameters', 'results', 'utils', 'datasets', 'figures']

# Resolve root and directory paths without recreating folders on import
layout = ProjectLayout(PROJECT_DIRS, create=False)

PROJECT_ROOT = layout.root
SRC_DIR = layout["src"]
PARAMETERS_DIR = layout["parameters"]
RESULTS_DIR = layout["results"]
UTILS_DIR = layout["utils"]
DATASETS_DIR = layout["datasets"]
FIGURES_DIR = layout["figures"]
EXAMPLE_DATA_DIR = DATASETS_DIR / "example_data" / "data" / "FV_mod1_1"
EXAMPLE_TRADB = EXAMPLE_DATA_DIR / "FV_mod1_1.tradb"
