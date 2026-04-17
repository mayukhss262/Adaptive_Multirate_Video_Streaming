"""
Streams video chunk-by-chunk over a simulated network.
Each iteration: fetch BW → ask ABR for rendition → pick chunk → sleep for download time → return result.
"""

import os
import time
import logging

import config
from network_sim import NetworkSimulator
from abr_algorithm import ABRController

logger = logging.getLogger(__name__)


def download_next_chunk(
    chunk_index: int,
    network_sim: NetworkSimulator,
    abr: ABRController,
    buffer_seconds: float,
) -> dict:
    """
    Download one chunk:
      1. Fetch current bandwidth from network simulator
      2. Ask ABR controller to pick a rendition
      3. Locate the corresponding chunk file
      4. Compute download time = chunk_bits / bandwidth
      5. Sleep for that duration (simulates the download)

    Returns dict with: chunk_index, rendition, bitrate_kbps, bandwidth_bps,
                        chunk_path, download_time, buffer_seconds
    """
    # 1 — current bandwidth
    bw_bps = network_sim.get_current_bandwidth()

    # 2 — ABR selects bitrate
    bitrate_kbps = abr.select_bitrate(
        current_buffer_seconds=buffer_seconds,
        measured_throughput_bps=bw_bps,
    )

    # find rendition label that matches this bitrate
    rendition = None
    for label, w, h, br in config.RENDITIONS:
        if br == bitrate_kbps:
            rendition = label
            break
    if rendition is None:
        rendition = f"{bitrate_kbps}k"

    # 3 — chunk file path
    chunk_path = os.path.join(
        config.CHUNKS_DIR, rendition, f"chunk_{chunk_index:03d}.mp4"
    )

    # 4 — download time
    chunk_bits = bitrate_kbps * 1000.0 * config.CHUNK_DURATION
    download_time = chunk_bits / bw_bps if bw_bps > 0 else float("inf")

    # 5 — sleep to simulate download
    if download_time != float("inf"):
        time.sleep(download_time)

    logger.info(
        "chunk %d | %s @ %d kbps | bw %.2f Mbps | dl %.2fs",
        chunk_index, rendition, bitrate_kbps,
        bw_bps / 1e6, download_time,
    )

    return {
        "chunk_index": chunk_index,
        "rendition": rendition,
        "bitrate_kbps": bitrate_kbps,
        "bandwidth_bps": bw_bps,
        "chunk_path": chunk_path,
        "download_time": download_time,
        "buffer_seconds": buffer_seconds,
    }


def stream(total_chunks: int = 50, model: int = config.NET_MODEL) -> list[dict]:
    """
    Main loop — downloads chunks back-to-back.
    Returns the list of per-chunk result dicts.
    """
    network_sim = NetworkSimulator(model=model, use_wall_clock=True)
    abr = ABRController()

    buffer = 0.0
    results = []

    for idx in range(total_chunks):
        result = download_next_chunk(idx, network_sim, abr, buffer)
        results.append(result)

        # update buffer: drains during download, fills by chunk duration
        dl = result["download_time"]
        if dl != float("inf"):
            buffer = max(buffer - dl, 0.0) + config.CHUNK_DURATION
            buffer = min(buffer, config.BUFFER_MAX)
        else:
            buffer = 0.0

    return results


if __name__ == "__main__":
    import argparse
    from network_sim import MODEL_NAMES, VALID_MODELS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Adaptive Streamer demo")
    parser.add_argument("--model", "-m", type=int, default=config.NET_MODEL,
                        choices=sorted(VALID_MODELS), metavar="MODEL",
                        help="Network model (0-6). Default: %(default)s")
    parser.add_argument("--chunks", "-c", type=int, default=20,
                        help="Number of chunks to stream. Default: %(default)s")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Adaptive Streamer — chunk-by-chunk demo (model {args.model}: {MODEL_NAMES[args.model]})")
    print("=" * 60)

    results = stream(total_chunks=args.chunks, model=args.model)

    print(f"\n{'Chunk':>5}  {'Rendition':>9}  {'Bitrate':>10}  {'BW (Mbps)':>10}  {'DL Time':>8}")
    print(f"{'-'*5}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*8}")
    for r in results:
        print(
            f"{r['chunk_index']:5d}  {r['rendition']:>9}  "
            f"{r['bitrate_kbps']:>7d} kbps  "
            f"{r['bandwidth_bps']/1e6:>7.2f} Mbps  "
            f"{r['download_time']:>7.3f}s"
        )
