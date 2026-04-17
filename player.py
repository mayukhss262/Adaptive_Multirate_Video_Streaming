"""
Leaky-bucket video player buffer.
Calls streamer to fetch chunks, drains buffer in real time,
detects stalls, and logs playback quality.
"""

import time
import logging

import config
from network_sim import NetworkSimulator
from abr_algorithm import ABRController
from streamer import download_next_chunk

logger = logging.getLogger(__name__)


class Player:
    """
    Buffer manager for adaptive video playback.

    Two forces act on the buffer simultaneously:
      • streamer fills it  — +chunk_duration per downloaded chunk
      • playback drains it — -elapsed_time every tick (real-time rate)

    Operations:
      add_chunk(duration)        — called after a successful fetch
      drain(elapsed)             — subtracts wall-clock time
      check_stall()              — fires if buffer <= 0
      record_quality(t, bitrate) — logs rendition at time t
    """

    def __init__(self, chunk_duration: float = config.CHUNK_DURATION):
        self.chunk_duration = chunk_duration
        self.buffer: float = 0.0
        self.stalled: bool = False

        # logs
        self.stall_events: list[dict] = []      # {start, duration}
        self.quality_log: list[dict] = []        # {time, bitrate_kbps, rendition}
        self.chunk_log: list[dict] = []          # raw results from streamer

    # ---- buffer ops -------------------------------------------------------
    def add_chunk(self, duration: float | None = None) -> None:
        """Add a downloaded chunk's worth of playback time to the buffer."""
        if duration is None:
            duration = self.chunk_duration
        self.buffer += duration
        self.buffer = min(self.buffer, config.BUFFER_MAX)

    def drain(self, elapsed: float) -> None:
        """Drain buffer by elapsed wall-clock seconds."""
        self.buffer -= elapsed
        if self.buffer < 0:
            self.buffer = 0.0

    def check_stall(self) -> bool:
        """Return True if the buffer is empty (stall / rebuffer)."""
        return self.buffer <= 0.0

    def record_quality(self, t: float, bitrate_kbps: int, rendition: str) -> None:
        """Log what rendition is playing at time t."""
        self.quality_log.append({
            "time": t,
            "bitrate_kbps": bitrate_kbps,
            "rendition": rendition,
        })

    def get_buffer_state(self) -> float:
        """Return current buffer level in seconds (needed by ABR)."""
        return self.buffer

    # ---- main playback loop -----------------------------------------------
    def play(
        self,
        total_chunks: int,
        network_sim: NetworkSimulator,
        abr: ABRController,
    ) -> dict:
        """
        Run the playback session.

        For each chunk:
          1. Record wall-clock time before download
          2. Call streamer.download_next_chunk() (sleeps for download time)
          3. After return: drain buffer by elapsed wall-clock time
          4. Add chunk_duration to buffer
          5. Check for stall — if buffer hit 0 during download, log stall
          6. Record quality

        Returns a summary dict with all logs and aggregate stats.
        """
        sim_start = time.time()
        total_stall_time = 0.0
        total_switches = 0
        prev_bitrate = None

        for idx in range(total_chunks):
            # current buffer state for ABR
            buf_before = self.get_buffer_state()

            # wall-clock before download
            t_before = time.time()

            # fetch chunk (blocks for download_time via sleep)
            result = download_next_chunk(
                chunk_index=idx,
                network_sim=network_sim,
                abr=abr,
                buffer_seconds=buf_before,
            )
            self.chunk_log.append(result)

            # wall-clock after download
            t_after = time.time()
            elapsed = t_after - t_before

            # Advance network sim time by chunk_duration (video playback time),
            # NOT by real elapsed wall-clock time. This ensures network models
            # like Ramp, Degrading, Congested evolve correctly with video time.
            network_sim.advance(self.chunk_duration)

            # drain buffer by real elapsed time
            self.drain(elapsed)

            # check stall before refilling
            stall_duration = 0.0
            if self.check_stall():
                # stall: the buffer went empty during this download
                # stall duration = how much the download exceeded the buffer
                stall_duration = elapsed - buf_before if buf_before < elapsed else 0.0
                self.stalled = True
                self.stall_events.append({
                    "chunk_index": idx,
                    "start": t_after - sim_start - stall_duration,
                    "duration": stall_duration,
                })
                total_stall_time += stall_duration
                logger.warning(
                    "STALL at chunk %d — %.2fs (buffer was %.2fs, download took %.2fs)",
                    idx, stall_duration, buf_before, elapsed,
                )
            else:
                self.stalled = False

            # add chunk to buffer
            self.add_chunk()

            # record quality
            playback_time = t_after - sim_start
            self.record_quality(playback_time, result["bitrate_kbps"], result["rendition"])

            # track switches
            if prev_bitrate is not None and result["bitrate_kbps"] != prev_bitrate:
                total_switches += 1
            prev_bitrate = result["bitrate_kbps"]

            logger.info(
                "chunk %d | buf %.1fs → %.1fs | %s @ %d kbps | dl %.2fs%s",
                idx, buf_before, self.get_buffer_state(),
                result["rendition"], result["bitrate_kbps"], elapsed,
                " | STALL" if stall_duration > 0 else "",
            )

        # aggregate stats
        bitrates = [r["bitrate_kbps"] for r in self.chunk_log]
        return {
            "total_chunks": total_chunks,
            "total_time": time.time() - sim_start,
            "total_stall_time": total_stall_time,
            "stall_count": len(self.stall_events),
            "total_switches": total_switches,
            "avg_bitrate_kbps": sum(bitrates) / len(bitrates),
            "min_bitrate_kbps": min(bitrates),
            "max_bitrate_kbps": max(bitrates),
            "final_buffer": self.get_buffer_state(),
            "stall_events": self.stall_events,
            "quality_log": self.quality_log,
            "chunk_log": self.chunk_log,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    network_sim = NetworkSimulator(use_wall_clock=True)
    abr = ABRController()
    player = Player()

    print("=" * 60)
    print("Player — leaky bucket demo (20 chunks)")
    print("=" * 60)

    summary = player.play(total_chunks=20, network_sim=network_sim, abr=abr)

    print(f"\n{'='*60}")
    print("Session Summary")
    print(f"{'='*60}")
    print(f"  Chunks played    : {summary['total_chunks']}")
    print(f"  Session time     : {summary['total_time']:.1f}s")
    print(f"  Total stall time : {summary['total_stall_time']:.2f}s")
    print(f"  Stall events     : {summary['stall_count']}")
    print(f"  Quality switches : {summary['total_switches']}")
    print(f"  Avg bitrate      : {summary['avg_bitrate_kbps']:.0f} kbps")
    print(f"  Min / Max        : {summary['min_bitrate_kbps']} / {summary['max_bitrate_kbps']} kbps")
    print(f"  Final buffer     : {summary['final_buffer']:.1f}s")

    if summary["stall_events"]:
        print(f"\n  Stall events:")
        for s in summary["stall_events"]:
            print(f"    chunk {s['chunk_index']}: stall at {s['start']:.1f}s for {s['duration']:.2f}s")
