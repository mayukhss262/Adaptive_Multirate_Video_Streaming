"""
Central config module
"""

import os

INPUT_VIDEO = "big_buck_bunny_1080p_60fps.mp4"

#Bitrate renditions ( label, width, height, bitrate(kbps) )
RENDITIONS = [
    ("240p",   426,  240,   400),
    ("480p",   854,  480,  1200),
    ("720p",  1280,  720,  3000),
    ("1080p", 1920, 1080,  6000),
]

CHUNK_DURATION = 2   #seconds per chunk

BUFFER_MIN = 2.0  # rebuffer threshold
BUFFER_LOW = 5.0  # consider switching down
BUFFER_TARGET = 10.0 # comfortable mid-point
BUFFER_MAX = 20.0 # stop downloading until buffer drains

NET_BASE_BW = 5e6           # 5 Mbps centre bandwidth
NET_AMPLITUDE = 2e6         # ±2 Mbps sine swing
NET_PERIOD = 10.0           # oscillation period (seconds)
NET_NOISE_STD = 3e5         # Gaussian noise std-dev
NET_SEED = 42               # RNG seed (None for random)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
ENCODED_DIR = os.path.join(OUTPUT_DIR, "encoded")    # full rendition files
CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")      # segmented chunks

FFMPEG_BIN = "ffmpeg"       # path to ffmpeg binary (or just "ffmpeg" if on PATH)
FFMPEG_LOG_LEVEL = "error"  # ffmpeg log verbosity: quiet, error, warning, info
