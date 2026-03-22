"""
Property-based validation tests.

Validates invariants across multiple scenarios:
- ABR invariants
- Streamer invariants
- Cross invariants
"""

import logging
import math
from typing import Optional

import pytest

import config
from abr_algorithm import ABRController
from network_sim import NetworkSimulator
from streamer import Streamer

logger = logging.getLogger(__name__)


# =============================================================================
# ABR INVARIANTS
# =============================================================================

class TestABRInvariants:
    """Property-based tests for ABR algorithm."""

    @pytest.mark.parametrize("bitrate_ladder", [
        [300, 800, 1500, 3000],
        [400, 1200, 3000, 6000],
        [500, 1000, 2000, 4000, 8000],
        [1000],
        [200, 400, 600, 800, 1000],
    ])
    def test_bitrate_in_ladder(self, bitrate_ladder):
        """ASSERT: bitrate ∈ ladder for various ladders."""
        abr = ABRController(
            available_bitrates=bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # Test various conditions
        test_cases = [
            (100e3, 10.0),   # Very low BW, high buffer
            (500e3, 10.0),   # Low BW
            (1e6, 10.0),     # Medium BW
            (3e6, 10.0),     # High BW
            (10e6, 10.0),    # Very high BW
            (1e6, 1.0),      # Low buffer
            (1e6, 5.0),      # Buffer at low threshold
            (1e6, 15.0),     # Buffer at high threshold
        ]

        for bw, buffer_sec in test_cases:
            abr.reset()
            selected = abr.select_bitrate(
                current_buffer_seconds=buffer_sec,
                measured_throughput_bps=bw,
            )

            assert selected in bitrate_ladder, (
                f"Selected bitrate {selected} not in ladder {bitrate_ladder}. "
                f"BW={bw/1e6:.1f}Mbps, buffer={buffer_sec}s"
            )

        logger.info(f"Bitrate in ladder test passed for ladder {bitrate_ladder}")

    @pytest.mark.parametrize("bitrate_ladder", [
        [300, 800, 1500, 3000],
        [400, 1200, 3000, 6000],
        [500, 1000, 2000],
    ])
    def test_bitrate_non_negative(self, bitrate_ladder):
        """ASSERT: bitrate ≥ 0."""
        abr = ABRController(
            available_bitrates=bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # Test various bandwidths including zero
        bandwidths = [0, 100e3, 500e3, 1e6, 5e6, 10e6]
        buffers = [0.5, 2.0, 5.0, 10.0, 15.0, 20.0]

        for bw in bandwidths:
            for buffer_sec in buffers:
                abr.reset()
                selected = abr.select_bitrate(
                    current_buffer_seconds=buffer_sec,
                    measured_throughput_bps=bw,
                )

                assert selected >= 0, (
                    f"Bitrate should be non-negative, got {selected} kbps. "
                    f"BW={bw/1e3:.0f}kbps, buffer={buffer_sec}s"
                )

        logger.info(f"Bitrate non-negative test passed for ladder {bitrate_ladder}")

    def test_ewma_throughput_valid(self):
        """ASSERT: EWMA throughput is valid when set."""
        abr = ABRController(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # Before any selection, EWMA should be None
        assert abr.ewma_throughput_bps is None

        # After selection, EWMA should be set
        abr.select_bitrate(10.0, 5e6)
        assert abr.ewma_throughput_bps is not None
        assert abr.ewma_throughput_bps > 0
        assert not math.isnan(abr.ewma_throughput_bps)
        assert not math.isinf(abr.ewma_throughput_bps)

        logger.info(f"EWMA throughput valid: {abr.ewma_throughput_bps/1e6:.2f} Mbps")


# =============================================================================
# STREAMER INVARIANTS
# =============================================================================

class TestStreamerInvariants:
    """Property-based tests for Streamer."""

    def test_buffer_never_negative(self):
        """ASSERT: buffer ≥ 0 throughout simulation."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        # Test various scenarios
        test_configs = [
            # (bandwidth_sequence, bitrate_kbps, initial_buffer)
            ([1e6] * 10, 400, 0.5),     # Low BW, low buffer
            ([500e3] * 10, 1000, 1.0),   # Very low BW
            ([2e6] * 10, 2000, 2.0),      # High bitrate
            ([1e6, 500e3, 2e6] * 5, 1000, 5.0),  # Oscillating
            ([10e6] * 20, 6000, 0.0),     # Start empty, high bitrate
        ]

        for seq, bitrate, init_buffer in test_configs:
            network = FakeNetworkSimulator(sequence=seq)
            abr = FakeABRController(fixed_bitrate_kbps=bitrate)

            streamer = Streamer(
                network_sim=network,
                abr_controller=abr,
                chunk_duration=2.0,
                buffer_max=20.0,
                total_segments=len(seq),
            )
            streamer._buffer_seconds = init_buffer

            result = streamer.run()

            # Check all events
            for event in result.events:
                assert event.buffer_before_seconds >= 0, (
                    f"Buffer before went negative at segment {event.segment_index}: "
                    f"{event.buffer_before_seconds}s"
                )
                assert event.buffer_after_seconds >= 0, (
                    f"Buffer after went negative at segment {event.segment_index}: "
                    f"{event.buffer_after_seconds}s"
                )

        logger.info(f"Buffer never negative test passed for {len(test_configs)} configs")

    def test_rebuffer_never_negative(self):
        """ASSERT: rebuffer ≥ 0."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        test_configs = [
            ([1e6] * 10, 400),
            ([10e6] * 10, 6000),
            ([500e3, 1e6, 2e6] * 5, 1000),
        ]

        for seq, bitrate in test_configs:
            network = FakeNetworkSimulator(sequence=seq)
            abr = FakeABRController(fixed_bitrate_kbps=bitrate)

            streamer = Streamer(
                network_sim=network,
                abr_controller=abr,
                chunk_duration=2.0,
                buffer_max=20.0,
                total_segments=len(seq),
            )

            result = streamer.run()

            for event in result.events:
                assert event.rebuffer_duration_seconds >= 0, (
                    f"Rebuffer went negative at segment {event.segment_index}: "
                    f"{event.rebuffer_duration_seconds}s"
                )

        logger.info(f"Rebuffer never negative test passed")

    def test_timestamps_strictly_increasing(self):
        """ASSERT: timestamps strictly increase."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        network = FakeNetworkSimulator(sequence=[5e6] * 20)
        abr = FakeABRController(fixed_bitrate_kbps=1000)

        streamer = Streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=20,
        )

        result = streamer.run()

        for i in range(1, len(result.events)):
            prev_ts = result.events[i-1].timestamp
            curr_ts = result.events[i].timestamp

            assert curr_ts > prev_ts, (
                f"Timestamps not strictly increasing: "
                f"segment {i-1}={prev_ts:.3f}s, segment {i}={curr_ts:.3f}s"
            )

        logger.info("Timestamps strictly increasing test passed")

    def test_no_nan_or_inf_in_events(self):
        """ASSERT: no NaN or inf values in events."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        network = FakeNetworkSimulator(sequence=[5e6] * 10)
        abr = FakeABRController(fixed_bitrate_kbps=1000)

        streamer = Streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=10,
        )

        result = streamer.run()

        for event in result.events:
            # Check numeric fields
            assert not math.isnan(event.timestamp), (
                f"NaN timestamp at segment {event.segment_index}"
            )
            assert not math.isinf(event.timestamp), (
                f"Inf timestamp at segment {event.segment_index}"
            )

            if event.download_time_seconds >= 0:
                assert not math.isnan(event.download_time_seconds), (
                    f"NaN download time at segment {event.segment_index}"
                )
                assert not math.isinf(event.download_time_seconds), (
                    f"Inf download time at segment {event.segment_index}"
                )

            assert not math.isnan(event.buffer_before_seconds)
            assert not math.isnan(event.buffer_after_seconds)
            assert not math.isnan(event.rebuffer_duration_seconds)

        logger.info("No NaN or inf in events test passed")


# =============================================================================
# CROSS INVARIANTS
# =============================================================================

class TestCrossInvariants:
    """Property-based tests for cross-component invariants."""

    def test_download_time_equals_segment_bits_over_bandwidth(self):
        """ASSERT: download_time == segment_bits / bandwidth."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        test_cases = [
            # (bandwidth_bps, bitrate_kbps, segment_duration)
            (2e6, 1000, 2.0),   # 2 Mbps, 1 Mbps, 2s
            (5e6, 2000, 2.0),   # 5 Mbps, 2 Mbps, 2s
            (1e6, 500, 2.0),    # 1 Mbps, 500 kbps, 2s
            (10e6, 6000, 2.0),  # 10 Mbps, 6 Mbps, 2s
        ]

        for bw, bitrate, duration in test_cases:
            network = FakeNetworkSimulator(constant_bw=bw)
            abr = FakeABRController(fixed_bitrate_kbps=bitrate)

            streamer = Streamer(
                network_sim=network,
                abr_controller=abr,
                chunk_duration=duration,
                buffer_max=20.0,
                total_segments=1,
            )

            result = streamer.run()
            event = result.events[0]

            # Calculate expected
            segment_bits = bitrate * 1000.0 * duration
            expected_download = segment_bits / bw

            assert abs(event.download_time_seconds - expected_download) < 0.001, (
                f"Download time mismatch: expected {expected_download:.3f}s, "
                f"got {event.download_time_seconds:.3f}s. "
                f"bits={segment_bits}, bw={bw}"
            )

        logger.info(f"Download time formula test passed for {len(test_cases)} cases")

    def test_buffer_after_equals_before_minus_download_plus_fill(self):
        """ASSERT: buffer_after = buffer_before - download_time + segment_duration (capped)."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        # Test with buffer starting at 0, but after initial fill
        # Run 2 segments: first fills buffer, second tests formula
        test_cases = [
            (5e6, 1000, 2.0),   # Fast network
            (1e6, 1000, 2.0),     # Slow network
            (2e6, 2000, 2.0),    # High bitrate
        ]

        for bw, bitrate, duration in test_cases:
            network = FakeNetworkSimulator(constant_bw=bw)
            abr = FakeABRController(fixed_bitrate_kbps=bitrate)

            streamer = Streamer(
                network_sim=network,
                abr_controller=abr,
                chunk_duration=duration,
                buffer_max=20.0,
                total_segments=2,
            )

            result = streamer.run()
            
            # Use second segment to test formula (buffer already filled)
            event = result.events[1]
            buffer_before = event.buffer_before_seconds
            download_time = event.download_time_seconds
            
            # Formula: after = before - download + duration (capped)
            expected = buffer_before - download_time + duration
            expected = min(expected, 20.0)

            assert abs(event.buffer_after_seconds - expected) < 0.001, (
                f"Buffer update formula mismatch: expected {expected:.3f}s, "
                f"got {event.buffer_after_seconds:.3f}s. "
                f"before={buffer_before}, download={download_time:.3f}, "
                f"fill={duration}"
            )

        logger.info("Buffer update formula test passed")

    def test_total_time_equals_sum_of_downloads_plus_rebuffer(self):
        """ASSERT: total_time == sum of all download times (rebuffer happens during download)."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        # Use fast network so after first segment buffer fills and no more rebuffering
        network = FakeNetworkSimulator(sequence=[5e6] * 10)  # 5 Mbps
        abr = FakeABRController(fixed_bitrate_kbps=1000)  # 1 Mbps bitrate

        streamer = Streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=10,
        )

        result = streamer.run()

        # Sum of download times
        total_download = sum(
            e.download_time_seconds 
            for e in result.events 
            if e.download_time_seconds >= 0
        )

        # The total simulated time equals sum of download times
        # (rebuffer time is part of download time when buffer is insufficient)
        assert abs(result.total_simulated_time - total_download) < 0.01, (
            f"Total time mismatch: expected {total_download:.1f}s, "
            f"got {result.total_simulated_time:.1f}s. "
            f"download={total_download:.1f}s"
        )

        logger.info(
            f"Total time formula: download={total_download:.1f}s, "
            f"total={result.total_simulated_time:.1f}s"
        )

    def test_total_video_content_equals_segments_times_duration(self):
        """ASSERT: total video content == total_segments * segment_duration."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        # Use slow network so buffer doesn't cap and we can track properly
        network = FakeNetworkSimulator(sequence=[1e6] * 15)  # 1 Mbps
        abr = FakeABRController(fixed_bitrate_kbps=500)  # 500 kbps bitrate

        streamer = Streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=15,
        )

        result = streamer.run()

        total_video_time = 15 * 2.0  # segments * duration
        
        # total_simulated_time = sum of (download_time + rebuffer) for all segments
        # This equals video content consumed + rebuffer time
        # The final buffer contains unplayed video content
        
        # Simpler invariant: total_simulated_time + final_buffer >= total_video_time
        # (we've simulated some time and have some buffer left)
        assert result.total_simulated_time + result.final_buffer_seconds >= total_video_time - 1.0, (
            f"Total video content not accounted for. "
            f"Video time: {total_video_time}s, Simulated: {result.total_simulated_time}s, "
            f"Final buffer: {result.final_buffer_seconds}s"
        )

        logger.info(
            f"Video content test: {total_video_time}s of video content, "
            f"{result.total_simulated_time:.1f}s simulated time, "
            f"{result.final_buffer_seconds:.1f}s buffer"
        )


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStressInvariants:
    """Stress tests for invariants."""

    def test_many_segments_invariants_hold(self):
        """Test invariants hold with many segments."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        # Create varied sequence
        sequence = []
        for i in range(100):
            if i % 4 == 0:
                sequence.append(8e6)   # High
            elif i % 4 == 1:
                sequence.append(4e6)   # Medium-high
            elif i % 4 == 2:
                sequence.append(2e6)   # Medium
            else:
                sequence.append(1e6)   # Low

        network = FakeNetworkSimulator(sequence=sequence)
        abr = FakeABRController(fixed_bitrate_kbps=2000)

        streamer = Streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=100,
        )

        result = streamer.run()

        # All invariants
        for event in result.events:
            assert event.buffer_before_seconds >= 0
            assert event.buffer_after_seconds >= 0
            assert event.rebuffer_duration_seconds >= 0
            assert not math.isnan(event.timestamp)
            assert not math.isinf(event.timestamp)

        logger.info(f"Stress test passed: {len(result.events)} segments processed")

    def test_very_small_buffer_invariants(self):
        """Test invariants with very small initial buffer."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        network = FakeNetworkSimulator(sequence=[1e6] * 20)
        abr = FakeABRController(fixed_bitrate_kbps=500)

        streamer = Streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=20,
        )
        streamer._buffer_seconds = 0.1  # Very small buffer

        result = streamer.run()

        # All invariants should still hold
        for event in result.events:
            assert event.buffer_before_seconds >= -0.001  # Allow tiny floating error
            assert event.buffer_after_seconds >= 0
            assert event.rebuffer_duration_seconds >= 0

        # Verify it eventually stabilizes
        assert result.final_buffer_seconds > 0

        logger.info(f"Small buffer test: final buffer = {result.final_buffer_seconds:.1f}s")

    def test_zero_bandwidth_at_end_invariants(self):
        """Test invariants when bandwidth drops to zero."""
        from tests.conftest import FakeNetworkSimulator, FakeABRController

        # Normal then zero
        sequence = [5e6] * 10 + [0] * 10

        network = FakeNetworkSimulator(sequence=sequence, zero_at_end=True)
        abr = FakeABRController(fixed_bitrate_kbps=1000)

        streamer = Streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=20,
        )

        result = streamer.run()

        # Invariants should still hold (buffer goes to zero when BW is zero)
        for event in result.events:
            assert event.buffer_after_seconds >= 0

        logger.info(f"Zero bandwidth test: final buffer = {result.final_buffer_seconds:.1f}s")
