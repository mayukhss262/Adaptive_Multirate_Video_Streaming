"""
Top-level orchestrator for the Adaptive Multirate Video Streaming Simulator.
  Phase 1: Encode input video into renditions and segment into chunks.
  Phase 2: Launch network simulator, streamer, and player to simulate streaming.
  Phase 3: Stitch ABR-selected chunks into a playable video and launch playback.
"""

import os
import glob
import math
import logging
import shutil
import subprocess

import config
from encoder import encode_and_segment
from network_sim import NetworkSimulator
from abr_algorithm import ABRController
from player import Player
from metrics import MetricsCollector
from analyzer import Analyzer
from visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def count_chunks() -> int:
    """Count available chunks by looking at the first rendition's chunk directory."""
    label = config.RENDITIONS[0][0]
    chunk_dir = os.path.join(config.CHUNKS_DIR, label)
    if not os.path.isdir(chunk_dir):
        return 0
    return len(glob.glob(os.path.join(chunk_dir, "chunk_*.mp4")))


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


# ------------------------------------------------------------------
# Phase 3 helpers
# ------------------------------------------------------------------

def _generate_stall_clip(duration: float, width: int, height: int, out_path: str) -> None:
    """Generate a black clip with 'Buffering...' text for stall events."""
    duration = max(duration, 0.5)  # minimum visible length
    cmd = [
        config.FFMPEG_BIN,
        "-loglevel", config.FFMPEG_LOG_LEVEL,
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:d={duration:.2f}:r=30",
        "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", f"{duration:.2f}",
        "-vf", (
            f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf'"
            f":text='Buffering...'"
            f":fontsize={height//8}:fontcolor=white"
            f":x=(w-text_w)/2:y=(h-text_h)/2"
        ),
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        "-y", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _add_overlay_to_chunk(
    src_path: str,
    out_path: str,
    bitrate_kbps: int,
    rendition: str,
    buffer_sec: float,
) -> None:
    """Re-encode a chunk with a bitrate/buffer overlay in the top-left corner."""
    overlay_text = (
        f"drawtext=text='{rendition}  {bitrate_kbps} kbps  buf\\: {buffer_sec:.1f}s'"
        f":fontsize=24:fontcolor=white"
        f":box=1:boxcolor=black@0.6:boxborderw=8"
        f":x=16:y=16"
    )
    cmd = [
        config.FFMPEG_BIN,
        "-loglevel", config.FFMPEG_LOG_LEVEL,
        "-i", src_path,
        "-vf", overlay_text,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "copy",
        "-y", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def build_playback_video(summary: dict) -> str:
    """
    Stitch the ABR-selected chunks (with overlays + stall clips) into
    a single playable video.

    Returns the path to the output video.
    """
    chunk_log = summary["chunk_log"]
    stall_events = summary["stall_events"]

    # build a lookup: chunk_index -> stall_duration
    stall_map = {}
    for s in stall_events:
        stall_map[s["chunk_index"]] = s["duration"]

    # target resolution = highest rendition
    _, max_w, max_h, _ = config.RENDITIONS[-1]

    # temp dir for overlaid chunks & stall clips
    tmp_dir = os.path.join(config.OUTPUT_DIR, "playback_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    segment_files = []

    for entry in chunk_log:
        idx = entry["chunk_index"]
        rendition = entry["rendition"]
        bitrate = entry["bitrate_kbps"]
        buf = entry["buffer_seconds"]

        # source chunk
        src = os.path.join(config.CHUNKS_DIR, rendition, f"chunk_{idx:03d}.mp4")
        if not os.path.exists(src):
            logger.warning("Chunk not found: %s — skipping", src)
            continue

        # insert stall clip before this chunk if it caused a stall
        if idx in stall_map and stall_map[idx] > 0:
            stall_path = os.path.join(tmp_dir, f"stall_{idx:03d}.mp4")
            logger.info("  Generating stall clip: %.1fs at chunk %d", stall_map[idx], idx)
            _generate_stall_clip(stall_map[idx], max_w, max_h, stall_path)
            segment_files.append(stall_path)

        # overlay bitrate/buffer info and scale to target resolution
        overlaid = os.path.join(tmp_dir, f"overlay_{idx:03d}.mp4")
        logger.info("  Overlaying chunk %d (%s @ %d kbps)", idx, rendition, bitrate)
        overlay_and_scale = (
            f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease,"
            f"pad={max_w}:{max_h}:(ow-iw)/2:(oh-ih)/2,"
            f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf'"
            f":text='{rendition}  {bitrate} kbps  buf\\: {buf:.1f}s'"
            f":fontsize=28:fontcolor=white"
            f":box=1:boxcolor=black@0.6:boxborderw=8"
            f":x=16:y=16"
        )
        cmd = [
            config.FFMPEG_BIN,
            "-loglevel", config.FFMPEG_LOG_LEVEL,
            "-i", src,
            "-vf", overlay_and_scale,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-y", overlaid,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        segment_files.append(overlaid)

    # write FFmpeg concat file
    concat_list = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for path in segment_files:
            escaped = path.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    # concatenate into final video
    output_video = os.path.join(config.OUTPUT_DIR, "streamed_playback.mp4")
    cmd = [
        config.FFMPEG_BIN,
        "-loglevel", config.FFMPEG_LOG_LEVEL,
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        "-y", output_video,
    ]
    logger.info("Concatenating %d segments into %s", len(segment_files), output_video)
    subprocess.run(cmd, check=True)

    # clean up temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Removed temporary directory: %s", tmp_dir)

    return output_video


def play_video(video_path: str) -> None:
    """Launch ffplay to play the stitched video."""
    logger.info("Playing %s", video_path)
    cmd = [
        "ffplay",
        "-autoexit",
        "-window_title", "Adaptive Streaming Playback",
        video_path,
    ]
    subprocess.Popen(cmd)


# ------------------------------------------------------------------
# Analysis reference file
# ------------------------------------------------------------------

def _write_contents_file(analysis_dir: str) -> None:
    """Write a CONTENTS.md file explaining every output in the analysis directory."""
    contents = """\
# Analysis Output Reference

This directory contains the metrics, analysis reports, and visualizations
generated by the Adaptive Multirate Video Streaming Simulator.

## Data Files

| File | Description |
|---|---|
| `metrics.json` | Raw timestamped simulation events — buffer levels, bandwidth samples, quality events, quality switches, stall events, and per-chunk download records. This is the primary data export from the `MetricsCollector`. |
| `analysis_report.json` | Computed QoE statistics — average/min/max bitrate, total stall time, stall ratio, quality switch count & frequency, time spent at each rendition, average buffer level, and a composite QoE score. |
| `buffer_levels.csv` | Time-series of buffer occupancy (columns: `time`, `buffer_seconds`). |
| `quality_events.csv` | Per-chunk quality selection log (columns: `time`, `bitrate_kbps`, `rendition`). |
| `switch_events.csv` | Quality switch events (columns: `time`, `old_bitrate_kbps`, `new_bitrate_kbps`, `old_rendition`, `new_rendition`). |
| `stall_events.csv` | Buffering stall events (columns: `time`, `duration`, `chunk_index`). |
| `bandwidth_samples.csv` | Bandwidth measurements paired with ABR decisions (columns: `time`, `bandwidth_bps`, `selected_bitrate_kbps`). |
| `chunk_downloads.csv` | Full per-chunk download records forwarded from the streamer/player. |

## Plots

| File | Description |
|---|---|
| `bandwidth_vs_bitrate.png` | Time-series plot showing available network bandwidth and the ABR-selected bitrate over time. Buffering stall periods are highlighted as translucent red regions. Dotted reference lines mark each rendition's bitrate. |
| `quality_distribution.png` | Dual chart — a horizontal bar chart of time spent at each quality level (with percentage annotations) alongside a stacked bar view of the same data. |
| `buffer_level.png` | Time-series of playback buffer occupancy with horizontal threshold lines (rebuffer, low, target) and stall regions shaded in red. |
"""
    path = os.path.join(analysis_dir, "CONTENTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)
    logger.info("Contents reference saved to %s", path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Adaptive Multirate Video Streaming Simulator")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Preprocessing — encode and segment
    # ------------------------------------------------------------------
    print("\n>>> PHASE 1: Encoding & Segmenting\n")

    if not os.path.exists(config.INPUT_VIDEO):
        logger.error("Input video not found: %s", config.INPUT_VIDEO)
        return

    num_chunks = count_chunks()
    if num_chunks > 0:
        logger.info("Found %d existing chunks — skipping encoding", num_chunks)
    else:
        encode_and_segment(config.INPUT_VIDEO)
        num_chunks = count_chunks()
        if num_chunks == 0:
            logger.error("Encoding produced no chunks — aborting")
            return

    try:
        duration = get_video_duration(config.INPUT_VIDEO)
        logger.info("Input video: %s (%.1fs, %d chunks)", config.INPUT_VIDEO, duration, num_chunks)
    except Exception:
        duration = num_chunks * config.CHUNK_DURATION
        logger.info("Input video: %s (%d chunks)", config.INPUT_VIDEO, num_chunks)

    # ------------------------------------------------------------------
    # Phase 2: Simulation — stream over simulated network
    # ------------------------------------------------------------------
    print(f"\n>>> PHASE 2: Streaming Simulation ({num_chunks} chunks)\n")

    network_sim = NetworkSimulator(use_wall_clock=True)
    abr = ABRController()
    player = Player()

    summary = player.play(
        total_chunks=num_chunks,
        network_sim=network_sim,
        abr=abr,
    )

    # results
    print(f"\n{'='*60}")
    print("  Simulation Results")
    print(f"{'='*60}")
    print(f"  Chunks streamed  : {summary['total_chunks']}")
    print(f"  Session time     : {summary['total_time']:.1f}s")
    print(f"  Video duration   : {num_chunks * config.CHUNK_DURATION:.0f}s")
    print(f"  Total stall time : {summary['total_stall_time']:.2f}s")
    print(f"  Stall events     : {summary['stall_count']}")
    print(f"  Quality switches : {summary['total_switches']}")
    print(f"  Avg bitrate      : {summary['avg_bitrate_kbps']:.0f} kbps")
    print(f"  Min / Max        : {summary['min_bitrate_kbps']} / {summary['max_bitrate_kbps']} kbps")
    print(f"  Final buffer     : {summary['final_buffer']:.1f}s")

    if summary["stall_events"]:
        print(f"\n  Stall events:")
        for s in summary["stall_events"]:
            print(f"    chunk {s['chunk_index']}: at {s['start']:.1f}s for {s['duration']:.2f}s")

    # ------------------------------------------------------------------
    # Metrics collection, analysis & visualization
    # ------------------------------------------------------------------
    print(f"\n>>> Collecting Metrics & Running Analysis\n")

    analysis_dir = os.path.join(config.OUTPUT_DIR, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    mc = MetricsCollector()
    mc.ingest_player_summary(summary)

    # Export raw metrics as JSON + CSV into analysis/
    metrics_json_path = mc.export_json(os.path.join(analysis_dir, "metrics.json"))
    csv_paths = mc.export_csv(analysis_dir)
    print(f"  Raw metrics saved  → {metrics_json_path}")

    # Detailed analysis report (console + JSON)
    analyzer = Analyzer(mc)
    analyzer.print_report()
    report_path = analyzer.save_report(os.path.join(analysis_dir, "analysis_report.json"))
    print(f"  Analysis report    → {report_path}")

    # Publication-quality visualizations
    vis = Visualizer(mc, output_dir=analysis_dir)
    vis.plot_all()
    print(f"  Plots saved        → {analysis_dir}/")

    # Generate reference file explaining all outputs
    _write_contents_file(analysis_dir)
    print(f"  Contents reference → {os.path.join(analysis_dir, 'CONTENTS.md')}")

    # ------------------------------------------------------------------
    # Phase 3: Build and play the streamed video
    # ------------------------------------------------------------------
    print(f"\n>>> PHASE 3: Building Playback Video\n")

    output_video = build_playback_video(summary)
    logger.info("Playback video saved: %s", output_video)

    print(f"\n>>> Launching playback...")
    play_video(output_video)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
