# Adaptive Multirate Video Streaming Simulator

Simulates adaptive bitrate (ABR) video streaming over a fluctuating network — encodes multiple renditions, streams chunk-by-chunk with bandwidth-aware quality switching, analyses QoE metrics, and plays back the result showing quality changes and buffering events.

## Files

| File | Role |
|---|---|
| `big_buck_bunny_1080p_60fps.mp4` | Input video file (1080p, 60fps, MP4 format) |
| `config.py` | Central settings — rendition profiles, chunk duration, buffer thresholds, network params, FFmpeg paths |
| `network_sim.py` | Bandwidth simulator — generates fluctuating bandwidth using a sine wave + Gaussian noise model |
| `encoder.py` | FFmpeg wrapper — encodes the input video into multiple renditions and segments each into chunks |
| `abr_algorithm.py` | ABR controller — hybrid throughput-buffer algorithm with EWMA smoothing and hysteresis to select renditions |
| `streamer.py` | Chunk downloader — fetches BW, asks ABR for rendition, locates chunk, sleeps for download time |
| `player.py` | Leaky-bucket buffer — fills on chunk download, drains in real time, detects stalls, logs quality |
| `metrics.py` | Event logger — collects buffer levels, quality switches, stall events & bandwidth samples; exports to JSON/CSV |
| `analyzer.py` | QoE analyser — computes stall ratio, avg/min/max bitrate, switch frequency, time-per-rendition & composite QoE score |
| `visualizer.py` | Chart generator — produces bandwidth-vs-bitrate, quality distribution & buffer level plots (PNG) |
| `multirate_video_streaming_simulator.py` | Orchestrator — runs encoding, simulation, metrics analysis, visualization, then stitches and plays the streamed video |

## Usage

```bash
# Run the full pipeline (encode → simulate → analyse → play)
python multirate_video_streaming_simulator.py
```

This runs the following stages:
1. **Encode & Segment** — encodes 4 renditions (240p–1080p) and segments into 2s chunks (skipped if chunks exist)
2. **Simulate** — streams chunks over the simulated network with ABR quality switching
3. **Metrics & Analysis** — collects all simulation events and saves everything to `output/analysis/`:
   - `metrics.json` — raw timestamped event data (buffer levels, bandwidth samples, stalls, quality switches)
   - `analysis_report.json` — computed statistics (avg bitrate, stall ratio, QoE score, time per rendition, etc.)
   - `*.csv` — per-category CSV files for further processing
   - `bandwidth_vs_bitrate.png` — bandwidth and selected bitrate over time with stall regions
   - `quality_distribution.png` — time spent at each rendition level
   - `buffer_level.png` — buffer occupancy over time with threshold lines
   - `CONTENTS.md` — reference file explaining every output in the directory
4. **Playback** — stitches ABR-selected chunks with bitrate/buffer overlays and stall clips into `output/streamed_playback.mp4`, then plays it

Individual scripts can also be run standalone:
```bash
python encoder.py          # encode and segment only
python network_sim.py      # bandwidth simulator demo
python streamer.py         # chunk download demo
python player.py           # buffer playback demo
python analyzer.py         # run a short simulation and print analysis report
python visualizer.py       # run a short simulation and generate plots
```

## Dependencies

| Dependency | Install |
|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| NumPy | `pip install numpy` |
| FFmpeg | `winget install Gyan.FFmpeg` (ALTERNATE METHOD TO SOLVE THIS DEPENDENCE IS GIVEN BELOW IN INSTALLATION INSTRUCTIONS) |
| Matplotlib | `pip install matplotlib` |

Installation Instructions:

Clone this repository to your local machine.

Go to the Releases section on the right side of the GitHub page.

Download the ffmpeg.exe and ffprobe.exe files attached to the latest release.

Place these .exe files directly into the root directory of the cloned project.

Run the simulator.

python multirate_video_streaming_simulator.py --model 0   # Sine
python multirate_video_streaming_simulator.py --model 1   # Constant
python multirate_video_streaming_simulator.py --model 2   # Ramp
python multirate_video_streaming_simulator.py --model 3   # Step
python multirate_video_streaming_simulator.py --model 4   # Random Walk
python multirate_video_streaming_simulator.py --model 5   # Congested
python multirate_video_streaming_simulator.py --model 6   # Degrading
