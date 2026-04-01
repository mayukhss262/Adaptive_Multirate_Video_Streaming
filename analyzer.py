"""
Post-run analysis for the Adaptive Video Streaming simulation.

Reads from a MetricsCollector (metrics.py) and computes:
  • Total buffering (stall) time
  • Number and frequency of quality switches
  • Average / min / max quality (bitrate)
  • Time spent at each rendition level
  • A composite QoE score combining all factors
  • Per-chunk download efficiency

The report is printed to the console and optionally saved as JSON.

Usage:
    from metrics import MetricsCollector
    from analyzer import Analyzer
    mc = MetricsCollector()
    mc.ingest_player_summary(summary)
    report = Analyzer(mc).full_report()
    Analyzer(mc).print_report()
    Analyzer(mc).save_report("output/analysis_report.json")
"""

import json
import os
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class Analyzer:
    """
    Computes QoE statistics from a populated MetricsCollector.

    Parameters
    ----------
    metrics : MetricsCollector
        A metrics collector that has already ingested a simulation run.
    """

    def __init__(self, metrics):
        self.metrics = metrics

    # ------------------------------------------------------------------
    # Core statistics
    # ------------------------------------------------------------------

    def total_stall_time(self) -> float:
        """Total seconds spent buffering (sum of stall durations)."""
        return sum(s["duration"] for s in self.metrics.stall_events)

    def stall_count(self) -> int:
        """Number of distinct buffering stall events."""
        return len(self.metrics.stall_events)

    def quality_switch_count(self) -> int:
        """Number of quality-level switches during the session."""
        return len(self.metrics.switch_events)

    def average_bitrate(self) -> float:
        """
        Average selected bitrate in kbps across all chunks.
        Returns 0.0 if no chunk data is available.
        """
        chunks = self.metrics.chunk_downloads
        if not chunks:
            return 0.0
        return sum(c["bitrate_kbps"] for c in chunks) / len(chunks)

    def min_bitrate(self) -> int:
        """Lowest bitrate selected during the session (kbps)."""
        chunks = self.metrics.chunk_downloads
        if not chunks:
            return 0
        return min(c["bitrate_kbps"] for c in chunks)

    def max_bitrate(self) -> int:
        """Highest bitrate selected during the session (kbps)."""
        chunks = self.metrics.chunk_downloads
        if not chunks:
            return 0
        return max(c["bitrate_kbps"] for c in chunks)

    def time_at_each_quality(self) -> dict[str, float]:
        """
        Estimate how many seconds of video were played at each rendition.

        Each chunk contributes config.CHUNK_DURATION seconds at its
        selected rendition label.

        Returns
        -------
        dict  mapping rendition label -> seconds
        """
        time_map: dict[str, float] = {}
        for c in self.metrics.chunk_downloads:
            label = c.get("rendition", f"{c['bitrate_kbps']}k")
            time_map[label] = time_map.get(label, 0.0) + config.CHUNK_DURATION
        return time_map

    def time_at_each_bitrate(self) -> dict[int, float]:
        """
        Estimate seconds of video at each bitrate (kbps).

        Returns
        -------
        dict  mapping bitrate_kbps -> seconds
        """
        time_map: dict[int, float] = {}
        for c in self.metrics.chunk_downloads:
            br = c["bitrate_kbps"]
            time_map[br] = time_map.get(br, 0.0) + config.CHUNK_DURATION
        return time_map

    def average_buffer_level(self) -> float:
        """Average buffer occupancy (seconds) across all recorded samples."""
        levels = self.metrics.buffer_levels
        if not levels:
            return 0.0
        return sum(b["buffer_seconds"] for b in levels) / len(levels)

    def switch_frequency(self) -> float:
        """
        Quality switches per minute of video.

        Returns 0.0 if no chunks were downloaded.
        """
        n_chunks = len(self.metrics.chunk_downloads)
        if n_chunks == 0:
            return 0.0
        video_minutes = (n_chunks * config.CHUNK_DURATION) / 60.0
        if video_minutes == 0:
            return 0.0
        return self.quality_switch_count() / video_minutes

    def stall_ratio(self) -> float:
        """
        Fraction of total session time spent stalling (0.0 – 1.0).

        session_time = video_duration + total_stall_time
        """
        n_chunks = len(self.metrics.chunk_downloads)
        video_dur = n_chunks * config.CHUNK_DURATION
        stall_t = self.total_stall_time()
        total = video_dur + stall_t
        if total == 0:
            return 0.0
        return stall_t / total

    # ------------------------------------------------------------------
    # QoE score
    # ------------------------------------------------------------------

    def qoe_score(
        self,
        w_quality: float = 1.0,
        w_switch: float = 1.0,
        w_stall: float = 3.0,
    ) -> float:
        """
        Compute a composite Quality-of-Experience score.

        The formula (inspired by common ABR literature) is:

            QoE = w_quality * avg_quality_norm
                  - w_switch  * switch_penalty
                  - w_stall   * stall_penalty

        Where:
          • avg_quality_norm = avg_bitrate / max_available_bitrate   (0-1)
          • switch_penalty   = switch_frequency / 60                 (switches/s → 0-~1)
          • stall_penalty    = stall_ratio                           (0-1)

        Higher is better.  The default weights heavily penalise stalls.

        Parameters
        ----------
        w_quality : float  — weight for average quality (default 1.0)
        w_switch  : float  — weight for switching penalty (default 1.0)
        w_stall   : float  — weight for stalling penalty (default 3.0)

        Returns
        -------
        float  — QoE score (theoretical range roughly –3 to +1)
        """
        max_available = max(r[3] for r in config.RENDITIONS)
        avg_br = self.average_bitrate()

        avg_quality_norm = avg_br / max_available if max_available > 0 else 0.0
        switch_penalty = self.switch_frequency() / 60.0   # per-second
        stall_penalty = self.stall_ratio()

        score = (
            w_quality * avg_quality_norm
            - w_switch * switch_penalty
            - w_stall * stall_penalty
        )

        logger.debug(
            "QoE components: quality=%.3f, switch_pen=%.3f, stall_pen=%.3f => %.3f",
            avg_quality_norm, switch_penalty, stall_penalty, score,
        )
        return score

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self) -> dict:
        """
        Build a comprehensive report dict with all computed statistics.

        Returns
        -------
        dict  — JSON-serializable report.
        """
        n_chunks = len(self.metrics.chunk_downloads)
        video_duration = n_chunks * config.CHUNK_DURATION

        report = {
            "video_duration_s": video_duration,
            "total_chunks": n_chunks,
            "chunk_duration_s": config.CHUNK_DURATION,

            # Stalls
            "total_stall_time_s": round(self.total_stall_time(), 4),
            "stall_count": self.stall_count(),
            "stall_ratio": round(self.stall_ratio(), 4),
            "stall_events": self.metrics.stall_events,

            # Quality
            "avg_bitrate_kbps": round(self.average_bitrate(), 2),
            "min_bitrate_kbps": self.min_bitrate(),
            "max_bitrate_kbps": self.max_bitrate(),
            "quality_switch_count": self.quality_switch_count(),
            "switch_frequency_per_min": round(self.switch_frequency(), 2),
            "time_at_each_quality_s": self.time_at_each_quality(),
            "time_at_each_bitrate_s": {
                str(k): v for k, v in self.time_at_each_bitrate().items()
            },

            # Buffer
            "avg_buffer_level_s": round(self.average_buffer_level(), 2),

            # Switch events detail
            "switch_events": self.metrics.switch_events,

            # QoE
            "qoe_score": round(self.qoe_score(), 4),
        }
        return report

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def print_report(self) -> dict:
        """
        Print a human-readable analysis report to the console.

        Returns the underlying report dict for programmatic use.
        """
        report = self.full_report()

        print()
        print("=" * 60)
        print("  Streaming Session — Analysis Report")
        print("=" * 60)

        print(f"\n  Video duration      : {report['video_duration_s']:.0f}s "
              f"({report['total_chunks']} chunks × {report['chunk_duration_s']}s)")

        # Stalls
        print(f"\n  ── Buffering ──")
        print(f"  Total stall time    : {report['total_stall_time_s']:.2f}s")
        print(f"  Stall events        : {report['stall_count']}")
        print(f"  Stall ratio         : {report['stall_ratio'] * 100:.1f}%")

        if report["stall_events"]:
            for s in report["stall_events"]:
                ci = s.get("chunk_index", "?")
                print(f"    • chunk {ci}: at {s['time']:.1f}s for {s['duration']:.2f}s")

        # Quality
        print(f"\n  ── Quality ──")
        print(f"  Avg bitrate         : {report['avg_bitrate_kbps']:.0f} kbps")
        print(f"  Min / Max           : {report['min_bitrate_kbps']} / "
              f"{report['max_bitrate_kbps']} kbps")
        print(f"  Quality switches    : {report['quality_switch_count']}")
        print(f"  Switch frequency    : {report['switch_frequency_per_min']:.1f} / min")

        # Time per rendition
        taq = report["time_at_each_quality_s"]
        if taq:
            print(f"\n  ── Time at each rendition ──")
            total_vid = report["video_duration_s"] or 1.0
            for label in sorted(taq, key=lambda l: taq[l], reverse=True):
                pct = taq[label] / total_vid * 100
                bar = "█" * int(pct // 2)
                print(f"    {label:>6}  {taq[label]:6.1f}s  ({pct:5.1f}%)  {bar}")

        # Buffer
        print(f"\n  ── Buffer ──")
        print(f"  Avg buffer level    : {report['avg_buffer_level_s']:.1f}s")

        # QoE
        print(f"\n  ── QoE Score ──")
        print(f"  Composite QoE       : {report['qoe_score']:.4f}")
        print(f"    (higher is better; range ≈ -3 to +1)")

        print(f"\n{'=' * 60}\n")
        return report

    def save_report(self, path: Optional[str] = None) -> str:
        """
        Save the full analysis report as a JSON file.

        Parameters
        ----------
        path : str or None
            Output file path.  Defaults to output/analysis_report.json.

        Returns
        -------
        str  — the path that was written.
        """
        if path is None:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            path = os.path.join(config.OUTPUT_DIR, "analysis_report.json")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        report = self.full_report()
        with open(path, "w") as fp:
            json.dump(report, fp, indent=2)
        logger.info("Analysis report saved to %s", path)
        print(f"  Report saved → {path}")
        return path


# ------------------------------------------------------------------
# Stand-alone demo / quick test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Run a short simulation to get data, then analyse
    from network_sim import NetworkSimulator
    from abr_algorithm import ABRController
    from player import Player
    from metrics import MetricsCollector

    print("Running a short 10-chunk simulation for analysis demo...\n")
    net = NetworkSimulator(use_wall_clock=True)
    abr = ABRController()
    player = Player()
    summary = player.play(total_chunks=10, network_sim=net, abr=abr)

    mc = MetricsCollector()
    mc.ingest_player_summary(summary)

    analyzer = Analyzer(mc)
    analyzer.print_report()
    analyzer.save_report()
