"""
Visualizer for the Adaptive Video Streaming simulation.

Generates publication-quality matplotlib plots:
  1. Bandwidth vs Selected Bitrate  (time-series, stall regions shaded)
  2. Time at Each Quality Level     (horizontal bar chart)
  3. Buffer Level Over Time         (time-series)

All plots are saved as PNG files to config.OUTPUT_DIR.

Usage:
    from metrics import MetricsCollector
    from visualizer import Visualizer
    mc = MetricsCollector()
    mc.ingest_player_summary(summary)
    vis = Visualizer(mc)
    vis.plot_all()                   # generates all charts
    vis.plot_bandwidth_vs_bitrate()  # individual chart
"""

import os
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for headless servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import config

logger = logging.getLogger(__name__)

# ── Style constants ──────────────────────────────────────────────────
COLOR_BW = "#1f77b4"         # bandwidth line
COLOR_BR = "#ff7f0e"         # selected bitrate line
COLOR_STALL = "#d62728"      # stall shading
COLOR_BUFFER = "#2ca02c"     # buffer level
RENDITION_COLORS = {
    "240p":  "#66c2a5",
    "480p":  "#fc8d62",
    "720p":  "#8da0cb",
    "1080p": "#e78ac3",
}


class Visualizer:
    """
    Creates and saves analysis charts from a populated MetricsCollector.

    Parameters
    ----------
    metrics : MetricsCollector
        A metrics collector that has already ingested a simulation run.
    output_dir : str or None
        Directory to save PNG files.  Defaults to config.OUTPUT_DIR.
    """

    def __init__(self, metrics, output_dir: Optional[str] = None):
        self.metrics = metrics
        self.output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Bandwidth vs Selected Bitrate (with stall events)
    # ------------------------------------------------------------------

    def plot_bandwidth_vs_bitrate(self, save: bool = True) -> plt.Figure:
        """
        Time-series plot: available bandwidth and ABR-selected bitrate.

        Buffering stall periods are highlighted as translucent red regions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        samples = self.metrics.bandwidth_samples
        stalls = self.metrics.stall_events

        if not samples:
            logger.warning("No bandwidth samples to plot.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig

        times = [s["time"] for s in samples]
        bw_mbps = [s["bandwidth_bps"] / 1e6 for s in samples]
        br_mbps = [s["selected_bitrate_kbps"] / 1e3 for s in samples]

        fig, ax = plt.subplots(figsize=(12, 5))

        # Bandwidth line
        ax.plot(times, bw_mbps, color=COLOR_BW, linewidth=1.4,
                alpha=0.85, label="Available Bandwidth")

        # Selected bitrate as step plot
        ax.step(times, br_mbps, color=COLOR_BR, linewidth=2.0,
                where="post", label="Selected Bitrate")

        # Shade stall regions
        for s in stalls:
            t_start = s["time"]
            t_end = t_start + s["duration"]
            ax.axvspan(t_start, t_end, color=COLOR_STALL, alpha=0.25)

        # Add rendition reference lines (dotted)
        for label, _, _, br_kbps in config.RENDITIONS:
            ax.axhline(br_kbps / 1e3, color="gray", linestyle=":",
                       linewidth=0.7, alpha=0.5)
            ax.text(times[-1] * 1.01, br_kbps / 1e3, label,
                    fontsize=8, color="gray", va="center")

        # Legend with stall patch
        handles, labels = ax.get_legend_handles_labels()
        if stalls:
            stall_patch = mpatches.Patch(color=COLOR_STALL, alpha=0.25,
                                          label="Buffering Stall")
            handles.append(stall_patch)
            labels.append("Buffering Stall")
        ax.legend(handles, labels, loc="upper right", fontsize=9)

        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Mbps", fontsize=11)
        ax.set_title("Bandwidth vs Selected Bitrate Over Time", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "bandwidth_vs_bitrate.png")
            fig.savefig(path, dpi=150)
            logger.info("Saved: %s", path)
            print(f"  Plot saved → {path}")

        return fig

    # ------------------------------------------------------------------
    # 2. Time at Each Quality Level
    # ------------------------------------------------------------------

    def plot_quality_distribution(self, save: bool = True) -> plt.Figure:
        """
        Horizontal bar chart showing time spent at each rendition,
        plus a stacked bar variant.

        Returns
        -------
        matplotlib.figure.Figure
        """
        chunks = self.metrics.chunk_downloads
        if not chunks:
            logger.warning("No chunk data to plot quality distribution.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig

        # Collect time per rendition
        time_map: dict[str, float] = {}
        for c in chunks:
            label = c.get("rendition", f"{c['bitrate_kbps']}k")
            time_map[label] = time_map.get(label, 0.0) + config.CHUNK_DURATION

        # Sort renditions by bitrate (lowest first)
        rendition_order = [r[0] for r in config.RENDITIONS]
        labels_sorted = [l for l in rendition_order if l in time_map]
        # Add any labels not in RENDITIONS
        for l in time_map:
            if l not in labels_sorted:
                labels_sorted.append(l)

        durations = [time_map[l] for l in labels_sorted]
        total_vid = sum(durations) or 1.0
        percentages = [d / total_vid * 100 for d in durations]

        colors = [RENDITION_COLORS.get(l, "#999999") for l in labels_sorted]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                                  gridspec_kw={"width_ratios": [2, 1]})

        # ── Left: horizontal bar chart ──
        ax1 = axes[0]
        y_pos = np.arange(len(labels_sorted))
        bars = ax1.barh(y_pos, durations, color=colors, edgecolor="white", height=0.6)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels_sorted, fontsize=11)
        ax1.set_xlabel("Time (s)", fontsize=11)
        ax1.set_title("Time Spent at Each Quality Level", fontsize=13, fontweight="bold")
        ax1.invert_yaxis()
        ax1.grid(True, axis="x", alpha=0.3)

        # Annotate bars with percentage
        for bar, pct in zip(bars, percentages):
            width = bar.get_width()
            ax1.text(width + total_vid * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{pct:.1f}%", va="center", fontsize=10)

        # ── Right: stacked bar (single column) ──
        ax2 = axes[1]
        bottom = 0.0
        for label, dur, color in zip(labels_sorted, durations, colors):
            ax2.bar(0, dur, bottom=bottom, color=color, edgecolor="white",
                    width=0.5, label=label)
            # Place rendition label in the middle of its segment
            if dur / total_vid > 0.05:  # only label if segment is big enough
                ax2.text(0, bottom + dur / 2, f"{label}\n{dur:.0f}s",
                         ha="center", va="center", fontsize=9, fontweight="bold")
            bottom += dur

        ax2.set_xlim(-0.5, 0.5)
        ax2.set_ylabel("Cumulative Time (s)", fontsize=11)
        ax2.set_title("Stacked View", fontsize=12, fontweight="bold")
        ax2.set_xticks([])
        ax2.legend(loc="upper right", fontsize=8)

        fig.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "quality_distribution.png")
            fig.savefig(path, dpi=150)
            logger.info("Saved: %s", path)
            print(f"  Plot saved → {path}")

        return fig

    # ------------------------------------------------------------------
    # 3. Buffer Level Over Time
    # ------------------------------------------------------------------

    def plot_buffer_level(self, save: bool = True) -> plt.Figure:
        """
        Time-series of buffer occupancy with threshold lines.

        Returns
        -------
        matplotlib.figure.Figure
        """
        levels = self.metrics.buffer_levels
        stalls = self.metrics.stall_events

        if not levels:
            logger.warning("No buffer-level data to plot.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig

        times = [b["time"] for b in levels]
        bufs = [b["buffer_seconds"] for b in levels]

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.fill_between(times, bufs, alpha=0.25, color=COLOR_BUFFER)
        ax.plot(times, bufs, color=COLOR_BUFFER, linewidth=1.4, label="Buffer Level")

        # Threshold lines
        ax.axhline(config.BUFFER_MIN, color=COLOR_STALL, linestyle="--",
                   linewidth=1, alpha=0.7, label=f"Rebuffer threshold ({config.BUFFER_MIN}s)")
        ax.axhline(config.BUFFER_LOW, color="#ff9800", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"Low threshold ({config.BUFFER_LOW}s)")
        ax.axhline(config.BUFFER_TARGET, color="#4caf50", linestyle="--",
                   linewidth=1, alpha=0.5, label=f"Target ({config.BUFFER_TARGET}s)")

        # Shade stall regions
        for s in stalls:
            t_start = s["time"]
            t_end = t_start + s["duration"]
            ax.axvspan(t_start, t_end, color=COLOR_STALL, alpha=0.15)

        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Buffer (s)", fontsize=11)
        ax.set_title("Playback Buffer Level Over Time", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "buffer_level.png")
            fig.savefig(path, dpi=150)
            logger.info("Saved: %s", path)
            print(f"  Plot saved → {path}")

        return fig

    # ------------------------------------------------------------------
    # Convenience: generate all plots
    # ------------------------------------------------------------------

    def plot_all(self) -> list[plt.Figure]:
        """
        Generate and save all available plots.

        Returns
        -------
        list[matplotlib.figure.Figure]
        """
        print("\n" + "=" * 60)
        print("  Generating visualizations")
        print("=" * 60)

        figures = [
            self.plot_bandwidth_vs_bitrate(),
            self.plot_quality_distribution(),
            self.plot_buffer_level(),
        ]

        print(f"\n  All plots saved to {self.output_dir}/")

        # Close figures to free memory
        for fig in figures:
            plt.close(fig)

        return figures


# ------------------------------------------------------------------
# Stand-alone demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from network_sim import NetworkSimulator
    from abr_algorithm import ABRController
    from player import Player
    from metrics import MetricsCollector

    print("Running a short 10-chunk simulation for visualizer demo...\n")
    net = NetworkSimulator(use_wall_clock=True)
    abr = ABRController()
    player = Player()
    summary = player.play(total_chunks=10, network_sim=net, abr=abr)

    mc = MetricsCollector()
    mc.ingest_player_summary(summary)

    vis = Visualizer(mc)
    vis.plot_all()
