from mylib.utils.io import ProjectLayout

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
