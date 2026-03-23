"""
Encodes video renditions using ffmpeg and segments them into chunks
"""

import os
import subprocess
import config


def _run(cmd: list[str]) -> None:
    """Run an FFmpeg command and raise on failure."""
    print(f"  >> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def encode_renditions(input_video: str = config.INPUT_VIDEO) -> None:
    """
    Encode the input video into every rendition defined in config.RENDITIONS.
    Output files go to config.ENCODED_DIR as  <label>.mp4
    """
    os.makedirs(config.ENCODED_DIR, exist_ok=True)

    for label, width, height, bitrate_kbps in config.RENDITIONS:
        out_path = os.path.join(config.ENCODED_DIR, f"{label}.mp4")
        if os.path.exists(out_path):
            print(f"  [skip] {out_path} already exists")
            continue

        cmd = [
            config.FFMPEG_BIN,
            "-loglevel", config.FFMPEG_LOG_LEVEL,
            "-i", input_video,
            "-vf", f"scale={width}:{height}",
            "-b:v", f"{bitrate_kbps}k",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y",
            out_path,
        ]
        print(f"  Encoding {label} ({width}x{height} @ {bitrate_kbps} kbps) ...")
        _run(cmd)
        print(f"  -> {out_path}")


def segment_renditions() -> None:
    """
    Segment each encoded rendition into chunks of config.CHUNK_DURATION seconds.
    Chunks go to config.CHUNKS_DIR/<label>/chunk_%03d.mp4
    """
    for label, *_ in config.RENDITIONS:
        src = os.path.join(config.ENCODED_DIR, f"{label}.mp4")
        if not os.path.exists(src):
            print(f"  [skip] {src} not found — encode first")
            continue

        chunk_dir = os.path.join(config.CHUNKS_DIR, label)
        os.makedirs(chunk_dir, exist_ok=True)

        chunk_pattern = os.path.join(chunk_dir, "chunk_%03d.mp4")
        cmd = [
            config.FFMPEG_BIN,
            "-loglevel", config.FFMPEG_LOG_LEVEL,
            "-i", src,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(config.CHUNK_DURATION),
            "-reset_timestamps", "1",
            "-y",
            chunk_pattern,
        ]
        print(f"  Segmenting {label} into {config.CHUNK_DURATION}s chunks ...")
        _run(cmd)
        print(f"  -> {chunk_dir}/")


def encode_and_segment(input_video: str = config.INPUT_VIDEO) -> None:
    """Convenience: encode all renditions then segment them."""
    print("=" * 50)
    print("Step 1: Encoding renditions")
    print("=" * 50)
    encode_renditions(input_video)

    print()
    print("=" * 50)
    print("Step 2: Segmenting into chunks")
    print("=" * 50)
    segment_renditions()

    print("\nDone.")


if __name__ == "__main__":
    encode_and_segment()
