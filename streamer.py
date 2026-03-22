"""
Adaptive Video Streaming Simulator

Simulates chunk-by-chunk delivery of video segments over a fluctuating
network, using ABR logic to select renditions and tracking buffer state,
rebuffering events, and quality switches.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import config
from abr_algorithm import ABRController
from network_sim import NetworkSimulator

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Record of a single segment's streaming event."""
    segment_index: int
    timestamp: float  # simulation time in seconds
    selected_bitrate_kbps: int
    bandwidth_bps: float  # measured bandwidth at decision time
    download_time_seconds: float
    buffer_before_seconds: float
    buffer_after_seconds: float
    rebuffer_duration_seconds: float
    is_bitrate_switch: bool
    reason: str = ""


@dataclass
class StreamingResult:
    """Aggregate result of a complete streaming session."""
    events: list[StreamEvent] = field(default_factory=list)
    total_simulated_time: float = 0.0
    total_rebuffer_time: float = 0.0
    total_segments: int = 0
    total_switches: int = 0
    average_bitrate_kbps: float = 0.0
    min_bitrate_kbps: int = 0
    max_bitrate_kbps: int = 0
    final_buffer_seconds: float = 0.0


class Streamer:
    """
    Simulates adaptive video streaming over a virtual network.

    For each video segment, the streamer:
    1. Queries current bandwidth from the network simulator
    2. Asks the ABR controller which bitrate to use
    3. Computes download time based on segment size and bandwidth
    4. Updates playback buffer (drains during download, fills after)
    5. Detects rebuffering when buffer becomes empty
    6. Records all metrics for analysis

    Parameters
    ----------
    network_sim : NetworkSimulator
        Bandwidth simulator instance.
    abr_controller : ABRController
        ABR decision controller.
    chunk_duration : float
        Duration of each video segment in seconds.
    buffer_max : float
        Maximum buffer occupancy in seconds.
    total_segments : int
        Total number of segments to simulate.
    """

    def __init__(
        self,
        network_sim: NetworkSimulator,
        abr_controller: ABRController,
        chunk_duration: float = config.CHUNK_DURATION,
        buffer_max: float = config.BUFFER_MAX,
        total_segments: int = 100,
    ):
        if chunk_duration <= 0:
            raise ValueError(f"Chunk duration must be positive, got {chunk_duration}")
        if buffer_max < chunk_duration:
            raise ValueError(
                f"Buffer max ({buffer_max}) must be >= chunk duration ({chunk_duration})"
            )
        if total_segments <= 0:
            raise ValueError(f"Total segments must be positive, got {total_segments}")

        self.network_sim = network_sim
        self.abr_controller = abr_controller
        self.chunk_duration = chunk_duration
        self.buffer_max = buffer_max
        self.total_segments = total_segments

        # Simulation state
        self._current_time: float = 0.0
        self._buffer_seconds: float = 0.0
        self._last_bitrate: Optional[int] = None

        logger.info(
            "Streamer initialized: chunk_duration=%.1fs, buffer_max=%.1fs, "
            "total_segments=%d",
            chunk_duration, buffer_max, total_segments
        )

    def run(self) -> StreamingResult:
        """
        Run the complete streaming simulation.

        Returns
        -------
        StreamingResult
            Aggregate metrics and per-segment event records.
        """
        logger.info("Starting streaming simulation for %d segments", self.total_segments)

        # Reset state
        self._current_time = 0.0
        self._buffer_seconds = 0.0
        self._last_bitrate = None
        self.abr_controller.reset()
        self.network_sim.reset()

        events: list[StreamEvent] = []
        total_rebuffer = 0.0
        total_switches = 0

        for seg_idx in range(self.total_segments):
            event = self._stream_segment(seg_idx)
            events.append(event)

            total_rebuffer += event.rebuffer_duration_seconds
            if event.is_bitrate_switch:
                total_switches += 1

            # Log progress periodically
            if (seg_idx + 1) % 10 == 0 or seg_idx == 0:
                logger.info(
                    "Segment %d/%d: bitrate=%d kbps, buffer=%.1fs, rebuffer=%.1fs",
                    seg_idx + 1, self.total_segments,
                    event.selected_bitrate_kbps,
                    event.buffer_after_seconds,
                    event.rebuffer_duration_seconds
                )

        # Compute aggregate metrics
        bitrates = [e.selected_bitrate_kbps for e in events]
        result = StreamingResult(
            events=events,
            total_simulated_time=self._current_time,
            total_rebuffer_time=total_rebuffer,
            total_segments=self.total_segments,
            total_switches=total_switches,
            average_bitrate_kbps=sum(bitrates) / len(bitrates) if bitrates else 0.0,
            min_bitrate_kbps=min(bitrates) if bitrates else 0,
            max_bitrate_kbps=max(bitrates) if bitrates else 0,
            final_buffer_seconds=self._buffer_seconds,
        )

        logger.info(
            "Streaming simulation complete: total_time=%.1fs, rebuffer=%.1fs, "
            "switches=%d, avg_bitrate=%.0f kbps",
            result.total_simulated_time, result.total_rebuffer_time,
            result.total_switches, result.average_bitrate_kbps
        )

        return result

    def _stream_segment(self, segment_index: int) -> StreamEvent:
        """
        Simulate streaming of a single video segment.

        Parameters
        ----------
        segment_index : int
            Zero-based segment index.

        Returns
        -------
        StreamEvent
            Record of the streaming event.
        """
        # Record buffer before download
        buffer_before = self._buffer_seconds

        # Get current bandwidth from network simulator
        bandwidth_bps = self.network_sim.get_current_bandwidth()

        # Ask ABR controller to select bitrate
        selected_bitrate_kbps = self.abr_controller.select_bitrate(
            current_buffer_seconds=self._buffer_seconds,
            measured_throughput_bps=bandwidth_bps,
        )

        # Compute segment size in bits
        # bitrate is in kbps, so convert to bps: bitrate_kbps * 1000
        segment_bits = selected_bitrate_kbps * 1000.0 * self.chunk_duration

        # Compute download time
        if bandwidth_bps <= 0:
            # Zero bandwidth: cannot download
            download_time = float('inf')
            logger.warning(
                "Segment %d: zero bandwidth, cannot download", segment_index
            )
        else:
            download_time = segment_bits / bandwidth_bps

        # Simulate buffer drain during download
        rebuffer_duration = 0.0
        remaining_download = download_time

        if download_time == float('inf'):
            # Infinite download time: buffer will drain completely
            rebuffer_duration = self._buffer_seconds
            self._buffer_seconds = 0.0
            # Advance network simulator time by buffer drain time
            self.network_sim.advance(rebuffer_duration)
            self._current_time += rebuffer_duration
        else:
            # Normal download: buffer drains during download
            if self._buffer_seconds >= download_time:
                # Buffer sufficient: no rebuffering
                self._buffer_seconds -= download_time
            else:
                # Buffer insufficient: rebuffering occurs
                rebuffer_duration = download_time - self._buffer_seconds
                self._buffer_seconds = 0.0

            # Advance network simulator time by download time
            self.network_sim.advance(download_time)
            self._current_time += download_time

        # After download completes, add segment duration to buffer
        # (segment is now available for playback)
        self._buffer_seconds += self.chunk_duration

        # Cap buffer at maximum
        if self._buffer_seconds > self.buffer_max:
            self._buffer_seconds = self.buffer_max

        # Check if bitrate switch occurred
        is_switch = False
        if self._last_bitrate is not None:
            is_switch = selected_bitrate_kbps != self._last_bitrate

        # Determine reason for bitrate selection
        reason = self._get_decision_reason(
            selected_bitrate_kbps, buffer_before, bandwidth_bps
        )

        # Update state
        self._last_bitrate = selected_bitrate_kbps

        # Create event record
        event = StreamEvent(
            segment_index=segment_index,
            timestamp=self._current_time,
            selected_bitrate_kbps=selected_bitrate_kbps,
            bandwidth_bps=bandwidth_bps,
            download_time_seconds=download_time if download_time != float('inf') else -1.0,
            buffer_before_seconds=buffer_before,
            buffer_after_seconds=self._buffer_seconds,
            rebuffer_duration_seconds=rebuffer_duration,
            is_bitrate_switch=is_switch,
            reason=reason,
        )

        if rebuffer_duration > 0:
            logger.warning(
                "Segment %d: REBUFFER %.1fs (buffer=%.1fs, download=%.1fs)",
                segment_index, rebuffer_duration, buffer_before, download_time
            )

        return event

    def _get_decision_reason(
        self,
        selected_bitrate: int,
        buffer_before: float,
        bandwidth_bps: float,
    ) -> str:
        """Generate a human-readable reason for the bitrate decision."""
        if self._last_bitrate is None:
            return "initial"

        if selected_bitrate < self._last_bitrate:
            if buffer_before < config.BUFFER_MIN:
                return "buffer_critical"
            elif buffer_before < config.BUFFER_LOW:
                return "buffer_low"
            else:
                return "throughput_drop"
        elif selected_bitrate > self._last_bitrate:
            if buffer_before >= config.BUFFER_MAX:
                return "buffer_healthy"
            else:
                return "throughput_increase"
        else:
            return "stable"


def run_streaming_session(
    total_segments: int = 100,
    chunk_duration: float = config.CHUNK_DURATION,
    available_bitrates: Optional[list[int]] = None,
    safety_margin: float = 0.8,
    ewma_alpha: float = 0.3,
    seed: Optional[int] = config.NET_SEED,
) -> StreamingResult:
    """
    Convenience function to run a complete streaming session.

    Parameters
    ----------
    total_segments : int
        Number of segments to simulate.
    chunk_duration : float
        Duration of each segment in seconds.
    available_bitrates : list[int] or None
        Available bitrates in kbps. If None, uses config.RENDITIONS.
    safety_margin : float
        ABR safety margin (0 < safety_margin < 1.0).
    ewma_alpha : float
        EWMA smoothing factor for throughput estimation.
    seed : int or None
        Random seed for network simulator.

    Returns
    -------
    StreamingResult
        Aggregate metrics and per-segment event records.
    """
    # Initialize network simulator
    network_sim = NetworkSimulator(seed=seed, use_wall_clock=False)

    # Initialize ABR controller
    abr_controller = ABRController(
        available_bitrates=available_bitrates,
        safety_margin=safety_margin,
        ewma_alpha=ewma_alpha,
    )

    # Initialize streamer
    streamer = Streamer(
        network_sim=network_sim,
        abr_controller=abr_controller,
        chunk_duration=chunk_duration,
        total_segments=total_segments,
    )

    # Run simulation
    return streamer.run()


if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("Adaptive Video Streaming Simulator — Demo")
    print("=" * 60)

    # Run a demo session
    result = run_streaming_session(total_segments=50)

    print(f"\n{'='*60}")
    print("Simulation Results")
    print(f"{'='*60}")
    print(f"  Total segments     : {result.total_segments}")
    print(f"  Total time         : {result.total_simulated_time:.1f} s")
    print(f"  Total rebuffer     : {result.total_rebuffer_time:.1f} s")
    print(f"  Bitrate switches   : {result.total_switches}")
    print(f"  Average bitrate    : {result.average_bitrate_kbps:.0f} kbps")
    print(f"  Min bitrate        : {result.min_bitrate_kbps} kbps")
    print(f"  Max bitrate        : {result.max_bitrate_kbps} kbps")
    print(f"  Final buffer       : {result.final_buffer_seconds:.1f} s")

    # Show first few events
    print(f"\n{'='*60}")
    print("First 10 Segment Events")
    print(f"{'='*60}")
    print(f"  {'Seg':>3}  {'Time':>6}  {'Bitrate':>8}  {'BW':>8}  {'Buffer':>6}  {'Rebuf':>6}  {'Switch':>6}  {'Reason':>15}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*15}")
    for event in result.events[:10]:
        bw_mbps = event.bandwidth_bps / 1e6
        switch_str = "YES" if event.is_bitrate_switch else "no"
        print(
            f"  {event.segment_index:3d}  {event.timestamp:6.1f}  "
            f"{event.selected_bitrate_kbps:5d} kbps  {bw_mbps:5.2f} Mbps  "
            f"{event.buffer_after_seconds:5.1f}s  {event.rebuffer_duration_seconds:5.1f}s  "
            f"{switch_str:>6}  {event.reason:>15}"
        )
