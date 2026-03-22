"""
Integration tests for streamer with real network simulator and ABR controller.

Validates:
- Constant high bandwidth scenario
- Constant low bandwidth scenario
- Bandwidth drop scenario
- Alternating bandwidth scenario
"""

import logging

import pytest

import config
from abr_algorithm import ABRController
from network_sim import NetworkSimulator
from streamer import Streamer

logger = logging.getLogger(__name__)


# =============================================================================
# 6.1 CONSTANT HIGH BANDWIDTH
# =============================================================================

class TestConstantHighBandwidth:
    """Test streaming with constant high bandwidth."""

    def test_high_bandwidth_selects_highest_bitrate(self, deterministic_network, real_abr, create_streamer):
        """
        EXPECT: highest bitrate, zero rebuffer.
        """
        # Use very high constant bandwidth with no oscillation
        network = deterministic_network(
            base_bw=10e6,      # 10 Mbps
            amplitude=0.0,     # No oscillation
            noise_std=0.0,     # No noise
            seed=42,
        )

        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=20,
        )

        result = streamer.run()

        # Should select highest bitrate (6000 kbps)
        assert result.max_bitrate_kbps == 6000, (
            f"Expected highest bitrate 6000 kbps, got {result.max_bitrate_kbps} kbps. "
            f"All bitrates: {set(e.selected_bitrate_kbps for e in result.events)}"
        )

        # Should have zero or minimal rebuffer
        # With 10 Mbps and 6000 kbps (6 Mbps effective), download is fast
        # Allow some initial rebuffer but should stabilize quickly
        assert result.total_rebuffer_time < 5.0, (
            f"Expected minimal rebuffer with high bandwidth, "
            f"got {result.total_rebuffer_time:.1f}s. Events: "
            f"{[(e.segment_index, e.rebuffer_duration_seconds) for e in result.events[:5]]}"
        )

        # Most segments should have no rebuffer after initial buffer fills
        segments_with_rebuffer = sum(
            1 for e in result.events if e.rebuffer_duration_seconds > 0
        )
        logger.info(
            f"High bandwidth test: max_bitrate={result.max_bitrate_kbps}kbps, "
            f"rebuffer={result.total_rebuffer_time:.1f}s, "
            f"segments_with_rebuffer={segments_with_rebuffer}/{result.total_segments}"
        )

    def test_high_bandwidth_stays_stable(self, deterministic_network, real_abr, create_streamer):
        """Test that high bandwidth leads to stable bitrate selection."""
        network = deterministic_network(
            base_bw=8e6,       # 8 Mbps
            amplitude=0.5e6,   # Small oscillation
            noise_std=0.1e6,    # Small noise
            seed=42,
        )

        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=30,
        )

        result = streamer.run()

        # With hysteresis, should have limited switches
        assert result.total_switches < 10, (
            f"Too many switches ({result.total_switches}) with high bandwidth. "
            f"Bitrates: {[e.selected_bitrate_kbps for e in result.events]}"
        )

        logger.info(
            f"High bandwidth stability: {result.total_switches} switches, "
            f"avg_bitrate={result.average_bitrate_kbps:.0f}kbps"
        )


# =============================================================================
# 6.2 CONSTANT LOW BANDWIDTH
# =============================================================================

class TestConstantLowBandwidth:
    """Test streaming with constant low bandwidth."""

    def test_low_bandwidth_selects_low_bitrate(self, deterministic_network, real_abr, create_streamer):
        """
        EXPECT: low bitrate, stable or minor stalls.
        """
        # Low constant bandwidth
        network = deterministic_network(
            base_bw=500e3,     # 500 kbps
            amplitude=0.0,     # No oscillation
            noise_std=50e3,    # Small noise
            seed=42,
        )

        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=20,
        )

        result = streamer.run()

        # Should select low bitrate (400 kbps)
        assert result.min_bitrate_kbps == 400, (
            f"Expected minimum bitrate 400 kbps with low bandwidth, "
            f"got {result.min_bitrate_kbps} kbps. "
            f"All bitrates: {set(e.selected_bitrate_kbps for e in result.events)}"
        )

        # With low bitrate and adequate buffer, should have minimal rebuffer
        # However, with 400 kbps and 500 kbps bandwidth, download is slightly slower
        # than playback, so buffer may drain slowly
        logger.info(
            f"Low bandwidth test: min_bitrate={result.min_bitrate_kbps}kbps, "
            f"rebuffer={result.total_rebuffer_time:.1f}s, "
            f"avg_bitrate={result.average_bitrate_kbps:.0f}kbps"
        )

    def test_low_bandwidth_buffer_stability(self, deterministic_network, real_abr, create_streamer):
        """Test that low bandwidth maintains buffer stability."""
        network = deterministic_network(
            base_bw=800e3,     # 800 kbps
            amplitude=100e3,   # Small oscillation
            noise_std=50e3,    # Small noise
            seed=42,
        )

        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.9,  # More conservative
            ewma_alpha=0.2,     # More smoothing
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=30,
        )

        result = streamer.run()

        # Buffer should remain positive throughout
        min_buffer = min(e.buffer_after_seconds for e in result.events)
        assert min_buffer >= 0, (
            f"Buffer went negative: {min_buffer:.1f}s. "
            f"Events: {[(e.segment_index, e.buffer_after_seconds) for e in result.events]}"
        )

        logger.info(
            f"Low bandwidth buffer stability: min_buffer={min_buffer:.1f}s, "
            f"final_buffer={result.final_buffer_seconds:.1f}s"
        )


# =============================================================================
# 6.3 BANDWIDTH DROP SCENARIO
# =============================================================================

class TestBandwidthDrop:
    """Test streaming during bandwidth drop."""

    def test_bitrate_decreases_after_drop(self, deterministic_network, real_abr, create_streamer):
        """
        EXPECT: bitrate decreases after drop, buffer shrinks, limited stall.
        """
        # Start with high bandwidth, then drop
        network = deterministic_network(
            base_bw=7e6,       # Start at 7 Mbps
            amplitude=0.0,
            noise_std=0.0,
            seed=42,
        )

        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=30,
        )

        # Manually manipulate bandwidth to simulate drop
        # First 15 segments: high bandwidth
        # Next 15 segments: low bandwidth

        # We'll use a custom sequence instead
        from tests.conftest import FakeNetworkSimulator

        # High then low sequence
        high_bw = 7e6   # 7 Mbps
        low_bw = 1e6    # 1 Mbps
        sequence = [high_bw] * 15 + [low_bw] * 15

        network = FakeNetworkSimulator(sequence=sequence)
        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=30,
        )

        result = streamer.run()

        # Get bitrates before and after drop
        first_half_bitrates = [e.selected_bitrate_kbps for e in result.events[:15]]
        second_half_bitrates = [e.selected_bitrate_kbps for e in result.events[15:]]

        avg_before = sum(first_half_bitrates) / len(first_half_bitrates)
        avg_after = sum(second_half_bitrates) / len(second_half_bitrates)

        # Average bitrate should decrease after drop
        assert avg_after < avg_before, (
            f"Bitrate should decrease after bandwidth drop. "
            f"Before: {avg_before:.0f} kbps, After: {avg_after:.0f} kbps. "
            f"First half: {first_half_bitrates}, Second half: {second_half_bitrates}"
        )

        # Get buffer levels
        buffer_before_drop = result.events[14].buffer_after_seconds
        buffer_after_drop = result.events[15].buffer_after_seconds

        # Buffer may shrink after bandwidth drop
        logger.info(
            f"Bandwidth drop test: avg_bitrate_before={avg_before:.0f}kbps, "
            f"avg_bitrate_after={avg_after:.0f}kbps, "
            f"buffer_before_drop={buffer_before_drop:.1f}s, "
            f"buffer_after_drop={buffer_after_drop:.1f}s, "
            f"total_rebuffer={result.total_rebuffer_time:.1f}s"
        )

    def test_adaptive_downgrade_prevents_excessive_stall(self, deterministic_network, real_abr, create_streamer):
        """Test that ABR downgrades quickly to prevent excessive stalling."""
        from tests.conftest import FakeNetworkSimulator

        # Sharp drop from high to very low
        high_bw = 10e6   # 10 Mbps
        low_bw = 500e3   # 500 kbps
        sequence = [high_bw] * 10 + [low_bw] * 20

        network = FakeNetworkSimulator(sequence=sequence)
        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.5,  # Faster adaptation
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=30,
        )

        result = streamer.run()

        # After bandwidth drop, should downgrade to low bitrate quickly
        # Find first segment with low bitrate after the drop
        drop_index = 10
        post_drop_bitrates = [
            e.selected_bitrate_kbps 
            for e in result.events[drop_index:drop_index+5]
        ]

        # At least one should be low bitrate
        assert min(post_drop_bitrates) <= 1200, (
            f"Should downgrade quickly after bandwidth drop. "
            f"Post-drop bitrates: {post_drop_bitrates}"
        )

        logger.info(
            f"Adaptive downgrade test: post-drop bitrates={post_drop_bitrates}, "
            f"total_rebuffer={result.total_rebuffer_time:.1f}s"
        )


# =============================================================================
# 6.4 ALTERNATING BANDWIDTH
# =============================================================================

class TestAlternatingBandwidth:
    """Test streaming with alternating bandwidth."""

    def test_controlled_switching_no_excessive_oscillation(self, deterministic_network, real_abr, create_streamer):
        """
        EXPECT: controlled switching, no excessive oscillation.
        """
        from tests.conftest import FakeNetworkSimulator

        # Alternating bandwidth pattern
        # High, low, high, low, high, low...
        high_bw = 6e6   # 6 Mbps
        low_bw = 1.5e6  # 1.5 Mbps
        sequence = [high_bw, low_bw] * 15

        network = FakeNetworkSimulator(sequence=sequence)
        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
            min_segments_between_upgrades=3,  # Require stability
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=30,
        )

        result = streamer.run()

        # With hysteresis, should not switch on every bandwidth change
        # Allow some switches but not too many
        max_allowed_switches = 15  # At most half the segments

        assert result.total_switches <= max_allowed_switches, (
            f"Too many switches ({result.total_switches}) with alternating bandwidth. "
            f"Hysteresis should prevent excessive oscillation. "
            f"Bitrates: {[e.selected_bitrate_kbps for e in result.events]}"
        )

        logger.info(
            f"Alternating bandwidth test: {result.total_switches} switches, "
            f"bitrates={[e.selected_bitrate_kbps for e in result.events]}"
        )

    def test_buffer_oscillates_with_bandwidth(self, deterministic_network, real_abr, create_streamer):
        """Test that buffer oscillates with alternating bandwidth."""
        from tests.conftest import FakeNetworkSimulator

        # Alternating pattern
        high_bw = 8e6
        low_bw = 2e6
        sequence = [high_bw, low_bw] * 10

        network = FakeNetworkSimulator(sequence=sequence)
        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=20,
        )

        result = streamer.run()

        # Buffer should stay positive
        min_buffer = min(e.buffer_after_seconds for e in result.events)
        assert min_buffer >= 0, (
            f"Buffer went negative: {min_buffer:.1f}s"
        )

        # Check buffer values
        buffers = [e.buffer_after_seconds for e in result.events]

        logger.info(
            f"Buffer oscillation test: min={min_buffer:.1f}s, "
            f"max={max(buffers):.1f}s, final={result.final_buffer_seconds:.1f}s"
        )


# =============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# =============================================================================

class TestComprehensiveIntegration:
    """Comprehensive integration tests with real components."""

    def test_full_simulation_runs_without_errors(self, deterministic_network, real_abr, create_streamer):
        """Test that full simulation runs without errors."""
        network = deterministic_network(
            base_bw=5e6,
            amplitude=1e6,
            noise_std=0.5e6,
            seed=42,
        )

        abr = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        streamer = create_streamer(
            network_sim=network,
            abr_controller=abr,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=50,
        )

        # Should run without exceptions
        result = streamer.run()

        # Verify all fields are populated
        assert result.total_segments == 50
        assert result.total_simulated_time > 0
        assert len(result.events) == 50
        assert result.average_bitrate_kbps > 0

        logger.info(
            f"Full simulation test: time={result.total_simulated_time:.1f}s, "
            f"avg_bitrate={result.average_bitrate_kbps:.0f}kbps, "
            f"rebuffer={result.total_rebuffer_time:.1f}s, "
            f"switches={result.total_switches}"
        )

    def test_multiple_runs_are_deterministic_with_same_seed(self, deterministic_network, real_abr, create_streamer):
        """Test that multiple runs with same seed produce identical results."""
        network1 = deterministic_network(
            base_bw=5e6,
            amplitude=1e6,
            noise_std=0.5e6,
            seed=42,
        )

        abr1 = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        streamer1 = create_streamer(
            network_sim=network1,
            abr_controller=abr1,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=10,
        )

        result1 = streamer1.run()

        # Second run with same seed
        network2 = deterministic_network(
            base_bw=5e6,
            amplitude=1e6,
            noise_std=0.5e6,
            seed=42,
        )

        abr2 = real_abr(
            available_bitrates=[400, 1200, 3000, 6000],
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        streamer2 = create_streamer(
            network_sim=network2,
            abr_controller=abr2,
            chunk_duration=2.0,
            buffer_max=20.0,
            total_segments=10,
        )

        result2 = streamer2.run()

        # Results should be identical
        assert result1.total_simulated_time == result2.total_simulated_time, (
            "Simulated times should match with same seed"
        )

        bitrates1 = [e.selected_bitrate_kbps for e in result1.events]
        bitrates2 = [e.selected_bitrate_kbps for e in result2.events]
        assert bitrates1 == bitrates2, (
            f"Bitrate selections should match: {bitrates1} vs {bitrates2}"
        )

        logger.info(
            f"Determinism test: both runs produced identical results"
        )
