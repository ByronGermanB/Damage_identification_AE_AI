from mylib import configure_plots  # type: ignore

from utils.paths import (
    PARAMETERS_DIR,
)

configure_plots(font="IBM Plex Sans")

DEFAULT_CONFIG_DIR = PARAMETERS_DIR / "thesis_figures"

__all__ = [
    "DEFAULT_CONFIG_DIR"
]