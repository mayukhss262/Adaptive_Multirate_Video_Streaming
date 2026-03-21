# Adaptive Multirate Video Streaming Simulator

Simulates adaptive bitrate (ABR) video streaming over a fluctuating network, including multi-rendition encoding, chunk-based delivery, and bandwidth estimation.

## Files

| File | Role |
|---|---|
| `big_buck_bunny_1080p_60fps.mp4` | Input video file (1080p, 60fps, MP4 format) |
| `config.py` | Central settings — rendition profiles, chunk duration, buffer thresholds, network params, FFmpeg paths |
| `network_sim.py` | Bandwidth simulator — generates fluctuating bandwidth using a sine wave + Gaussian noise model |
| `encoder.py` | FFmpeg wrapper — encodes the input video into multiple renditions and segments each into chunks |

## Usage

```bash
# 1. Encode all renditions and segment into chunks
python encoder.py

# 2. Run the bandwidth simulator demo
python network_sim.py
```

All settings (bitrates, resolutions, chunk duration, etc.) are configured in `config.py`.

## Dependencies

| Dependency | Install |
|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| NumPy | `pip install numpy` |
| FFmpeg | `winget install Gyan.FFmpeg` |
| Matplotlib | `pip install matplotlib` |