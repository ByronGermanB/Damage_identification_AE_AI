from pathlib import Path


def get_project_root(markers: tuple[str, ...] = ("setup.py", ".git")) -> Path:
    """Walk up from this file until a project root marker is found."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    # Fallback: assumes utils/paths.py is one level below the project root
    return current.parents[1]


PROJECT_ROOT = get_project_root()

# Standard project directories
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_EA_DIR = PROJECT_ROOT / "Datos_EA"
RAW_MTS_DIR = PROJECT_ROOT / "Datos_MTS"
PARAMETERS_DIR = PROJECT_ROOT / "parameters"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"