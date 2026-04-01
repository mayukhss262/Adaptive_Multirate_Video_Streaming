"""
Event logger for the Adaptive Video Streaming simulation.

Collects timestamped events from player.py and streamer.py:
  - Buffering stalls  (start time + duration)
  - Quality switches  (timestamp + old/new bitrate)
  - Buffer level over time
  - Bandwidth vs selected-bitrate samples

All data is kept in-memory (lists/dicts) during the run and can be
serialized to JSON or CSV via export helpers.

Usage:
    from metrics import MetricsCollector
    mc = MetricsCollector()
    mc.log_buffer_level(0.5, 4.0)
    mc.log_quality(1.0, 1200, "480p")
    mc.log_stall(2.0, 0.35, chunk_index=1)
    mc.export_json("output/metrics.json")
"""

import csv
import json
import os
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Central data-collection sink for the streaming simulation.

    Attributes
    ----------
    buffer_levels : list[dict]
        {"time": float, "buffer_seconds": float}
    quality_events : list[dict]
        {"time": float, "bitrate_kbps": int, "rendition": str}
    switch_events : list[dict]
        {"time": float, "old_bitrate_kbps": int, "new_bitrate_kbps": int,
         "old_rendition": str, "new_rendition": str}
    stall_events : list[dict]
        {"time": float, "duration": float, "chunk_index": int | None}
    bandwidth_samples : list[dict]
        {"time": float, "bandwidth_bps": float, "selected_bitrate_kbps": int}
    chunk_downloads : list[dict]
        Full per-chunk download records forwarded from streamer/player.
    """

    def __init__(self):
        self.buffer_levels: list[dict] = []
        self.quality_events: list[dict] = []
        self.switch_events: list[dict] = []
        self.stall_events: list[dict] = []
        self.bandwidth_samples: list[dict] = []
        self.chunk_downloads: list[dict] = []

        # Internal tracking for detecting quality switches
        self._last_bitrate: Optional[int] = None
        self._last_rendition: Optional[str] = None

    # ------------------------------------------------------------------
    # Logging methods (called during simulation)
    # ------------------------------------------------------------------

    def log_buffer_level(self, time_s: float, buffer_seconds: float) -> None:
        """Record the buffer occupancy at a given simulation time."""
        self.buffer_levels.append({
            "time": time_s,
            "buffer_seconds": buffer_seconds,
        })

    def log_quality(self, time_s: float, bitrate_kbps: int, rendition: str) -> None:
        """
        Record the rendition playing at *time_s*.

        Also auto-detects quality switches and appends to switch_events.
        """
        self.quality_events.append({
            "time": time_s,
            "bitrate_kbps": bitrate_kbps,
            "rendition": rendition,
        })

        # Detect switches
        if self._last_bitrate is not None and bitrate_kbps != self._last_bitrate:
            self.switch_events.append({
                "time": time_s,
                "old_bitrate_kbps": self._last_bitrate,
                "new_bitrate_kbps": bitrate_kbps,
                "old_rendition": self._last_rendition,
                "new_rendition": rendition,
            })
            logger.debug(
                "Quality switch at %.2fs: %s (%d kbps) -> %s (%d kbps)",
                time_s, self._last_rendition, self._last_bitrate,
                rendition, bitrate_kbps,
            )

        self._last_bitrate = bitrate_kbps
        self._last_rendition = rendition

    def log_stall(self, time_s: float, duration: float,
                  chunk_index: Optional[int] = None) -> None:
        """Record a buffering stall event."""
        self.stall_events.append({
            "time": time_s,
            "duration": duration,
            "chunk_index": chunk_index,
        })
        logger.debug("Stall at %.2fs for %.2fs (chunk %s)", time_s, duration, chunk_index)

    def log_bandwidth_sample(self, time_s: float, bandwidth_bps: float,
                             selected_bitrate_kbps: int) -> None:
        """Record a bandwidth measurement alongside the ABR-selected bitrate."""
        self.bandwidth_samples.append({
            "time": time_s,
            "bandwidth_bps": bandwidth_bps,
            "selected_bitrate_kbps": selected_bitrate_kbps,
        })

    def log_chunk_download(self, record: dict) -> None:
        """
        Forward a full per-chunk result dict (from streamer / player).

        Expected keys: chunk_index, rendition, bitrate_kbps, bandwidth_bps,
                        download_time, buffer_seconds
        """
        self.chunk_downloads.append(record)

    # ------------------------------------------------------------------
    # Bulk import from Player.play() summary
    # ------------------------------------------------------------------

    def ingest_player_summary(self, summary: dict) -> None:
        """
        Populate this collector from the summary dict returned by
        Player.play().  This is the easiest way to wire metrics into
        the existing codebase without modifying player.py or streamer.py.

        Parameters
        ----------
        summary : dict
            The dict returned by Player.play(), containing keys like
            'chunk_log', 'quality_log', 'stall_events', etc.
        """
        # Chunk-level data
        for entry in summary.get("chunk_log", []):
            self.log_chunk_download(entry)
            # Derive a timestamp from cumulative download time
            # (chunk_index * chunk_duration is approximate playback time)
            approx_time = entry["chunk_index"] * config.CHUNK_DURATION
            self.log_bandwidth_sample(
                approx_time,
                entry["bandwidth_bps"],
                entry["bitrate_kbps"],
            )
            self.log_buffer_level(approx_time, entry["buffer_seconds"])

        # Quality log (has real timestamps from player)
        for q in summary.get("quality_log", []):
            self.log_quality(q["time"], q["bitrate_kbps"], q["rendition"])

        # Stall events
        for s in summary.get("stall_events", []):
            self.log_stall(
                time_s=s["start"],
                duration=s["duration"],
                chunk_index=s.get("chunk_index"),
            )

        logger.info(
            "Ingested player summary: %d chunks, %d quality events, "
            "%d stalls, %d bandwidth samples",
            len(self.chunk_downloads),
            len(self.quality_events),
            len(self.stall_events),
            len(self.bandwidth_samples),
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return all collected data as a plain dict (JSON-serializable)."""
        return {
            "buffer_levels": self.buffer_levels,
            "quality_events": self.quality_events,
            "switch_events": self.switch_events,
            "stall_events": self.stall_events,
            "bandwidth_samples": self.bandwidth_samples,
            "chunk_downloads": self.chunk_downloads,
        }

    def export_json(self, path: Optional[str] = None) -> str:
        """
        Write all metrics to a JSON file.

        Parameters
        ----------
        path : str or None
            Output file path.  Defaults to  output/metrics.json.

        Returns
        -------
        str  — the path that was written.
        """
        if path is None:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            path = os.path.join(config.OUTPUT_DIR, "metrics.json")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp, indent=2)
        logger.info("Metrics exported to %s", path)
        return path

    def export_csv(self, directory: Optional[str] = None) -> list[str]:
        """
        Write separate CSV files for each event category.

        Parameters
        ----------
        directory : str or None
            Output directory.  Defaults to config.OUTPUT_DIR.

        Returns
        -------
        list[str]  — paths of the CSV files that were written.
        """
        if directory is None:
            directory = config.OUTPUT_DIR
        os.makedirs(directory, exist_ok=True)

        written: list[str] = []

        # Helper to write a list-of-dicts as CSV
        def _write_csv(filename: str, rows: list[dict]) -> None:
            if not rows:
                return
            path = os.path.join(directory, filename)
            with open(path, "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            written.append(path)

        _write_csv("buffer_levels.csv", self.buffer_levels)
        _write_csv("quality_events.csv", self.quality_events)
        _write_csv("switch_events.csv", self.switch_events)
        _write_csv("stall_events.csv", self.stall_events)
        _write_csv("bandwidth_samples.csv", self.bandwidth_samples)
        _write_csv("chunk_downloads.csv", self.chunk_downloads)

        logger.info("Metrics CSV files written to %s: %s", directory,
                     [os.path.basename(p) for p in written])
        return written

    # ------------------------------------------------------------------
    # Quick summary (for console / debugging)
    # ------------------------------------------------------------------

    def summary_string(self) -> str:
        """Return a brief multi-line summary of what was collected."""
        lines = [
            "MetricsCollector summary",
            f"  Buffer-level samples : {len(self.buffer_levels)}",
            f"  Quality events       : {len(self.quality_events)}",
            f"  Quality switches     : {len(self.switch_events)}",
            f"  Stall events         : {len(self.stall_events)}",
            f"  Bandwidth samples    : {len(self.bandwidth_samples)}",
            f"  Chunk downloads      : {len(self.chunk_downloads)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MetricsCollector(chunks={len(self.chunk_downloads)}, "
            f"stalls={len(self.stall_events)}, "
            f"switches={len(self.switch_events)})"
        )
