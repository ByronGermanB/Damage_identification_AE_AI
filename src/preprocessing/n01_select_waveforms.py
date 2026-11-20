from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import vallenae as vae
from matplotlib.widgets import Button, Slider, TextBox
from mylib import load_config, save_config  # type: ignore

from src.preprocessing.pipeline_utils import (
    DEFAULT_CONFIG_PATH,
    get_segmentation_paths,
    parse_config_args,
)
from utils.segmentation import dbae_to_mV


def _discover_trais(
    tradb_path: Path,
    min_amp_dbae: float = 36.0,
    min_counts: int = 5,
    max_trai: int | None = None,
) -> list[int]:
    """Return sorted valid TRAIs using `.pridb` metadata when available, else `.tradb`."""
    pridb_path = tradb_path.with_suffix(".pridb")
    if pridb_path.exists():
        with vae.io.PriDatabase(pridb_path) as pridb:
            hits = pridb.read_hits().reset_index(drop=True)

        if not hits.empty and "trai" in hits.columns:
            amp_dbae = 20.0 * np.log10(
                np.maximum(hits["amplitude"].to_numpy(), 1e-12) / 1e-6
            )
            mask = (
                (hits["trai"] > 0)
                & (amp_dbae >= min_amp_dbae)
                & (amp_dbae <= 94.0)
                & (hits["counts"] >= min_counts)
            )
            if max_trai is not None:
                mask &= hits["trai"] <= max_trai

            trais = sorted(int(t) for t in hits.loc[mask, "trai"].dropna().unique())
            if trais:
                return trais

    with sqlite3.connect(tradb_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT trai FROM tr_data WHERE trai > 0 ORDER BY trai"
        ).fetchall()
    trais = [int(r[0]) for r in rows]
    if max_trai is not None:
        trais = [t for t in trais if t <= max_trai]
    return trais


class InteractiveWaveformSelector:
    """Interactive Matplotlib browser to inspect AE transients and annotate sub-hits."""

    def __init__(
        self,
        tradb_path: Path,
        config_path: Path,
        config: dict[str, Any],
        test_id: str,
        trais: list[int],
        min_amp_dbae: float = 36.0,
    ) -> None:
        if not trais:
            raise ValueError(f"No valid TRAIs found in {tradb_path}.")

        self.tradb_path = tradb_path
        self.config_path = config_path
        self.config = config
        self.test_id = test_id
        self.trais = trais
        self.current_idx = 0
        self.min_amp_dbae = min_amp_dbae

        self.tradb = vae.io.TraDatabase(self.tradb_path)
        self.selections: dict[int, int] = {}
        self.save_to_json()

        self._build_ui()
        self._update_plot()

    def save_to_json(self) -> None:
        """Persist annotated TRAIs into `self.config_path` via `mylib.save_config`."""
        ordered_items = [
            {"trai": trai, "expected_segments": self.selections[trai]}
            for trai in sorted(self.selections)
        ]
        payload: dict[str, Any] = {
            **self.config,
            "test_id": self.test_id,
            "tradb_path": self.config.get("tradb_path", None),
            "selections": ordered_items,
            "trai_list": [item["trai"] for item in ordered_items],
            "expected_segments": [item["expected_segments"] for item in ordered_items],
            "param_grid": self.config.get("param_grid"),
            "grid_search": self.config.get(
                "grid_search",
                {
                    "output_dir": None,
                    "save_valid_only": True,
                    "img_format": "png",
                    "figsize_mm": [140, 90],
                    "save_figures": True,
                    "show_figures": False,
                    "save_txt": True,
                },
            ),
        }
        self.config = payload
        save_config(payload, self.config_path)

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(13.0, 7.2))
        self.fig.canvas.manager.set_window_title(
            f"AE Waveform Selector - {self.test_id}"
        )

        # Main waveform panel and right-hand selection summary panel
        self.ax_wave = self.fig.add_axes([0.07, 0.25, 0.66, 0.65])
        self.ax_summary = self.fig.add_axes([0.75, 0.25, 0.23, 0.65])
        self.ax_summary.axis("off")

        # Index scrubber slider
        ax_slider = self.fig.add_axes([0.07, 0.15, 0.66, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label="Index",
            valmin=0,
            valmax=max(0, len(self.trais) - 1),
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider_changed)

        # Navigation & annotation controls
        self.btn_prev = Button(self.fig.add_axes([0.07, 0.05, 0.08, 0.06]), "Prev (<-)")
        self.btn_next = Button(self.fig.add_axes([0.16, 0.05, 0.08, 0.06]), "Next (->)")

        self.txt_jump = TextBox(
            self.fig.add_axes([0.31, 0.05, 0.08, 0.06]), "TRAI: ", initial=""
        )

        self.btn_seg2 = Button(
            self.fig.add_axes([0.42, 0.05, 0.085, 0.06]), "Add: 2 segs", color="#e6f2ff"
        )
        self.btn_seg3 = Button(
            self.fig.add_axes([0.515, 0.05, 0.085, 0.06]),
            "Add: 3 segs",
            color="#d9f2e6",
        )
        self.btn_seg4 = Button(
            self.fig.add_axes([0.61, 0.05, 0.085, 0.06]), "Add: 4 segs", color="#fff2cc"
        )
        self.btn_seg5 = Button(
            self.fig.add_axes([0.705, 0.05, 0.085, 0.06]),
            "Add: 5 segs",
            color="#ffe6cc",
        )
        self.btn_remove = Button(
            self.fig.add_axes([0.80, 0.05, 0.085, 0.06]), "Remove", color="#ffd6d6"
        )
        self.btn_save = Button(
            self.fig.add_axes([0.895, 0.05, 0.085, 0.06]), "Save JSON", color="#d4edda"
        )

        self.btn_prev.on_clicked(lambda _: self.step(-1))
        self.btn_next.on_clicked(lambda _: self.step(1))
        self.txt_jump.on_submit(self._on_jump_submit)
        self.btn_seg2.on_clicked(lambda _: self.assign_segments(2))
        self.btn_seg3.on_clicked(lambda _: self.assign_segments(3))
        self.btn_seg4.on_clicked(lambda _: self.assign_segments(4))
        self.btn_seg5.on_clicked(lambda _: self.assign_segments(5))
        self.btn_remove.on_clicked(lambda _: self.remove_current())
        self.btn_save.on_clicked(lambda _: self._manual_save())

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.status_msg = (
            "Ready. Use buttons or keys [2-5] to add TRAI, Left/Right to browse."
        )

    def _on_slider_changed(self, val: float) -> None:
        new_idx = round(val)
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self._update_plot()

    def _on_jump_submit(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned.isdigit():
            self.status_msg = f"Invalid TRAI input: '{text}'"
            self._update_summary_panel()
            return
        target = int(cleaned)
        if target in self.trais:
            self.current_idx = self.trais.index(target)
            self.slider.set_val(self.current_idx)
            self.status_msg = f"Jumped to TRAI {target}."
        else:
            nearest_idx = int(np.argmin(np.abs(np.array(self.trais) - target)))
            self.current_idx = nearest_idx
            self.slider.set_val(self.current_idx)
            self.status_msg = f"TRAI {target} not in filtered list; showing nearest TRAI {self.trais[nearest_idx]}."
        self._update_plot()

    def _on_key_press(self, event: Any) -> None:
        if event.inaxes == self.txt_jump.ax:
            return
        if event.key == "left":
            self.step(-1)
        elif event.key == "right":
            self.step(1)
        elif event.key in {"2", "3", "4", "5"}:
            self.assign_segments(int(event.key))
        elif event.key in {"delete", "backspace"}:
            self.remove_current()
        elif event.key == "s":
            self._manual_save()

    def step(self, delta: int) -> None:
        self.current_idx = (self.current_idx + delta) % len(self.trais)
        self.slider.eventson = False
        self.slider.set_val(self.current_idx)
        self.slider.eventson = True
        self._update_plot()

    def assign_segments(self, n_segments: int) -> None:
        trai = self.trais[self.current_idx]
        self.selections[trai] = n_segments
        self.save_to_json()
        self.status_msg = (
            f"Added TRAI {trai} -> {n_segments} segment(s) and saved to JSON."
        )
        self._update_plot()

    def remove_current(self) -> None:
        trai = self.trais[self.current_idx]
        if trai in self.selections:
            del self.selections[trai]
            self.save_to_json()
            self.status_msg = f"Removed TRAI {trai} from selection and updated JSON."
        else:
            self.status_msg = f"TRAI {trai} is not currently selected."
        self._update_plot()

    def _manual_save(self) -> None:
        self.save_to_json()
        self.status_msg = (
            f"Saved {len(self.selections)} TRAIs to {self.config_path.name}."
        )
        self._update_summary_panel()

    def _update_plot(self) -> None:
        trai = self.trais[self.current_idx]
        amp_v, time = self.tradb.read_wave(trai, time_axis=True)
        waveform = amp_v * 1e3
        time_us = time * 1e6

        peak_mv = float(np.max(np.abs(waveform)))
        peak_dbae = 20.0 * np.log10(max(peak_mv, 1e-9) * 1e3)
        dur_us = float(time_us[-1] - time_us[0]) if time_us.size > 1 else 0.0

        self.ax_wave.clear()
        self.ax_wave.plot(
            time_us, waveform, color="#2b5c8f", linewidth=0.65, label="Waveform [mV]"
        )
        self.ax_wave.axhline(
            dbae_to_mV(self.min_amp_dbae),
            color="crimson",
            linestyle=":",
            linewidth=0.8,
            label=f"Ref. threshold ({self.min_amp_dbae:.0f} dBAE)",
        )
        
        self.ax_wave.axhline(
                - dbae_to_mV(self.min_amp_dbae),
                color="crimson",
                linestyle=":",
                linewidth=0.8,
            )

        selected_tag = ""
        if trai in self.selections:
            selected_tag = f"  |  SELECTED: {self.selections[trai]} segment(s)"
            self.ax_wave.set_facecolor("#f4fbf6")
        else:
            self.ax_wave.set_facecolor("white")

        self.ax_wave.set_title(
            f"{self.test_id}  |  TRAI {trai} ({self.current_idx + 1}/{len(self.trais)})  |  "
            f"Peak: {peak_mv:.2f} mV ({peak_dbae:.1f} dBAE)  |  Dur: {dur_us:.1f} us{selected_tag}",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_wave.set_xlabel("Time [us]")
        self.ax_wave.set_ylabel("Amplitude [mV]")
        self.ax_wave.grid(True, linestyle="-", linewidth=0.35, alpha=0.7)
        self.ax_wave.legend(loc="upper right", fontsize=8)

        self._update_summary_panel()
        self.fig.canvas.draw_idle()

    def _update_summary_panel(self) -> None:
        self.ax_summary.clear()
        self.ax_summary.axis("off")

        lines = [
            "SELECTED TRAIs",
            "=" * 26,
            f"Config JSON:\n{self.config_path.name}",
            "-" * 26,
        ]
        if self.selections:
            for trai in sorted(self.selections):
                marker = ">> " if trai == self.trais[self.current_idx] else "   "
                lines.append(
                    f"{marker}TRAI {trai:<6} -> {self.selections[trai]} seg(s)"
                )
        else:
            lines.append("  (No TRAIs selected yet)")

        lines.extend(
            [
                "-" * 26,
                "Shortcuts:",
                "  Left/Right : Prev / Next",
                "  2, 3, 4, 5 : Add N segments",
                "  Backspace  : Remove TRAI",
                "  S          : Save JSON",
                "-" * 26,
                f"Status:\n{self.status_msg}",
            ]
        )
        self.ax_summary.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
        )

    def show(self) -> None:
        plt.show()


def main(config_arg: str | Path | None = DEFAULT_CONFIG_PATH) -> None:
    """Launch the interactive waveform selector driven by a JSON configuration file."""
    config, cfg_path = load_config(config_arg, return_path=True)
    paths = get_segmentation_paths(config, create=False)

    test_id = str(config["test_id"])
    selector_cfg = config.get("selector", {})

    min_amp_dbae = float(selector_cfg.get("min_amp_dbae", 36.0))
    min_counts = int(selector_cfg.get("min_counts", 5))
    max_trai_raw = selector_cfg.get("max_trai", None)
    max_trai = int(max_trai_raw) if max_trai_raw is not None else None

    tradb_path = paths["tradb_path"]
    if not tradb_path.exists():
        raise FileNotFoundError(f"Transient database not found: {tradb_path}")

    trais = _discover_trais(
        tradb_path=tradb_path,
        min_amp_dbae=min_amp_dbae,
        min_counts=min_counts,
        max_trai=max_trai,
    )
    print(f"[INFO] Loaded config from   : {cfg_path}")
    print(f"[INFO] Loaded {len(trais)} candidate TRAIs from {tradb_path}")

    selector = InteractiveWaveformSelector(
        tradb_path=tradb_path,
        config_path=cfg_path,
        config=config,
        test_id=test_id,
        trais=trais,
        min_amp_dbae=min_amp_dbae
    )
    selector.show()


if __name__ == "__main__":
    args = parse_config_args(
        "Stage 1: Inspect AE transient waveforms and select example TRAIs."
    )
    main(config_arg=args.config)
