# Adaptive Multirate Video Streaming Simulator

Simulates adaptive bitrate (ABR) video streaming over a fluctuating network — encodes multiple renditions, streams chunk-by-chunk with bandwidth-aware quality switching, and plays back the result showing quality changes and buffering events.

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
| `multirate_video_streaming_simulator.py` | Orchestrator — runs encoding, simulation, then stitches and plays the streamed video |

## Usage

```bash
# Run the full pipeline (encode → simulate → play)
python multirate_video_streaming_simulator.py
```

This runs three phases:
1. **Encode & Segment** — encodes 4 renditions (240p–1080p) and segments into 2s chunks (skipped if chunks exist)
2. **Simulate** — streams chunks over the simulated network with ABR quality switching
3. **Playback** — stitches ABR-selected chunks with bitrate/buffer overlays and stall clips into `output/streamed_playback.mp4`, then plays it

Individual scripts can also be run standalone:
```bash
python encoder.py          # encode and segment only
python network_sim.py      # bandwidth simulator demo
python streamer.py         # chunk download demo
python player.py           # buffer playback demo
```

## Dependencies

| Dependency | Install |
|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| NumPy | `pip install numpy` |
| FFmpeg | `winget install Gyan.FFmpeg` |
| Matplotlib *(optional, for plots)* | `pip install matplotlib` |