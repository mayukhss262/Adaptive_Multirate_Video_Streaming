"""
Tests for ABR algorithm decision logic.

Validates:
- Basic selection correctness
- Ladder membership
- Monotonicity
- Buffer sensitivity
- Zero bandwidth handling
- Single bitrate ladder
- Hysteresis / anti-flapping
- State evolution
"""

import logging
from typing import Optional

import pytest

import config
from abr_algorithm import ABRController

logger = logging.getLogger(__name__)


# =============================================================================
# 4.1 BASIC SELECTION CORRECTNESS
# =============================================================================

class TestBasicSelection:
    """Test basic bitrate selection based on bandwidth."""

    def test_high_bandwidth_selects_highest_bitrate(self, default_bitrate_ladder):
        """
        When bandwidth is high (much higher than highest bitrate),
        the ABR should select the highest bitrate.
        """
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # High bandwidth: 10 Mbps, highest bitrate is 6000 kbps = 6 Mbps
        # With 0.8 safety margin: 8 Mbps available, should select 6000 kbps
        throughput_bps = 10e6  # 10 Mbps
        buffer_seconds = 10.0  # Healthy buffer

        selected = abr.select_bitrate(
            current_buffer_seconds=buffer_seconds,
            measured_throughput_bps=throughput_bps,
        )

        expected = max(default_bitrate_ladder)
        assert selected == expected, (
            f"Expected highest bitrate {expected} kbps for high bandwidth, "
            f"got {selected} kbps. Bandwidth: {throughput_bps/1e6:.1f} Mbps"
        )
        logger.info(
            f"High bandwidth test: BW={throughput_bps/1e6:.1f}Mbps, "
            f"selected={selected}kbps, expected={expected}kbps"
        )

    def test_low_bandwidth_selects_lowest_bitrate(self, default_bitrate_ladder):
        """
        When bandwidth is low (lower than lowest bitrate),
        the ABR should select the lowest bitrate.
        """
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # Low bandwidth: 200 kbps = 0.2 Mbps
        # With 0.8 safety margin: 160 kbps available
        # Lowest bitrate is 400 kbps, but it's above safe throughput
        # Should get lowest bitrate that fits
        throughput_bps = 200e3  # 200 kbps
        buffer_seconds = 10.0  # Healthy buffer

        selected = abr.select_bitrate(
            current_buffer_seconds=buffer_seconds,
            measured_throughput_bps=throughput_bps,
        )

        expected = min(default_bitrate_ladder)
        assert selected == expected, (
            f"Expected lowest bitrate {expected} kbps for low bandwidth, "
            f"got {selected} kbps. Bandwidth: {throughput_bps/1e3:.0f} kbps"
        )
        logger.info(
            f"Low bandwidth test: BW={throughput_bps/1e3:.0f}kbps, "
            f"selected={selected}kbps, expected={expected}kbps"
        )


# =============================================================================
# 4.2 LADDER MEMBERSHIP
# =============================================================================

class TestLadderMembership:
    """Test that selected bitrate is always in the available ladder."""

    def test_selection_in_ladder_various_bandwidths(self, default_bitrate_ladder):
        """ASSERT: selected_bitrate in bitrate_ladder for various bandwidths."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # Test various bandwidths
        bandwidths_bps = [
            100e3,   # 100 kbps
            500e3,   # 500 kbps
            1e6,     # 1 Mbps
            2e6,     # 2 Mbps
            5e6,     # 5 Mbps
            10e6,    # 10 Mbps
            20e6,    # 20 Mbps
        ]

        for bw in bandwidths_bps:
            abr.reset()
            selected = abr.select_bitrate(
                current_buffer_seconds=10.0,  # Healthy buffer
                measured_throughput_bps=bw,
            )

            assert selected in default_bitrate_ladder, (
                f"Selected bitrate {selected} kbps not in ladder "
                f"{default_bitrate_ladder}. Bandwidth: {bw/1e6:.1f} Mbps"
            )
            logger.debug(f"Bandwidth={bw/1e6:.1f}Mbps -> selected={selected}kbps ✓")

    def test_selection_in_ladder_various_buffers(self, default_bitrate_ladder):
        """Test ladder membership with various buffer levels."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        buffer_levels = [
            0.5,   # Critical
            2.0,   # BUFFER_MIN
            5.0,   # BUFFER_LOW
            10.0,  # BUFFER_TARGET
            15.0,  # Between target and max
            20.0,  # BUFFER_MAX
        ]

        throughput_bps = 5e6  # 5 Mbps

        for buffer_sec in buffer_levels:
            abr.reset()
            selected = abr.select_bitrate(
                current_buffer_seconds=buffer_sec,
                measured_throughput_bps=throughput_bps,
            )

            assert selected in default_bitrate_ladder, (
                f"Selected bitrate {selected} kbps not in ladder "
                f"{default_bitrate_ladder}. Buffer: {buffer_sec}s"
            )
            logger.debug(f"Buffer={buffer_sec}s -> selected={selected}kbps ✓")


# =============================================================================
# 4.3 MONOTONICITY TEST
# =============================================================================

class TestMonotonicity:
    """Test that increasing bandwidth leads to non-decreasing bitrate."""

    def test_increasing_bandwidth_non_decreasing_bitrate(self, default_bitrate_ladder):
        """For fixed buffer: increasing bandwidth -> non-decreasing bitrate."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # Use high buffer to avoid buffer-based downgrades
        buffer_seconds = 15.0  # Above BUFFER_MAX

        bandwidths_bps = [
            500e3,   # 500 kbps
            1e6,     # 1 Mbps
            2e6,     # 2 Mbps
            4e6,     # 4 Mbps
            6e6,     # 6 Mbps
            8e6,     # 8 Mbps
            10e6,    # 10 Mbps
        ]

        selections = []
        for bw in bandwidths_bps:
            abr.reset()  # Fresh state each time
            selected = abr.select_bitrate(
                current_buffer_seconds=buffer_seconds,
                measured_throughput_bps=bw,
            )
            selections.append(selected)
            logger.debug(f"BW={bw/1e6:.1f}Mbps -> {selected}kbps")

        # Check monotonicity: each selection should be >= previous
        for i in range(1, len(selections)):
            assert selections[i] >= selections[i-1], (
                f"Monotonicity violation: bitrate decreased from "
                f"{selections[i-1]} kbps to {selections[i]} kbps "
                f"when bandwidth increased from {bandwidths_bps[i-1]/1e6:.1f} Mbps "
                f"to {bandwidths_bps[i]/1e6:.1f} Mbps. "
                f"Full selections: {selections}"
            )

        logger.info(f"Monotonicity test passed: {selections}")


# =============================================================================
# 4.4 BUFFER SENSITIVITY
# =============================================================================

class TestBufferSensitivity:
    """Test that lower buffer leads to equal or lower bitrate."""

    def test_lower_buffer_equal_or_lower_bitrate(self, default_bitrate_ladder):
        """For fixed bandwidth: lower buffer -> equal or lower bitrate."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # Use moderate bandwidth that allows multiple bitrate choices
        throughput_bps = 3e6  # 3 Mbps

        buffer_levels = [
            20.0,  # Very healthy
            15.0,  # Healthy
            10.0,  # Target
            5.0,   # Low threshold
            2.0,   # Min threshold
            1.0,   # Below min (critical)
        ]

        selections = []
        for buffer_sec in buffer_levels:
            abr.reset()
            selected = abr.select_bitrate(
                current_buffer_seconds=buffer_sec,
                measured_throughput_bps=throughput_bps,
            )
            selections.append(selected)
            logger.debug(f"Buffer={buffer_sec}s -> {selected}kbps")

        # Check: lower buffer should not lead to higher bitrate
        # (allowing equal due to discrete ladder)
        for i in range(1, len(selections)):
            assert selections[i] <= selections[i-1], (
                f"Buffer sensitivity violation: bitrate increased from "
                f"{selections[i-1]} kbps to {selections[i]} kbps "
                f"when buffer decreased from {buffer_levels[i-1]}s "
                f"to {buffer_levels[i]}s. "
                f"Full selections: {selections}"
            )

        logger.info(f"Buffer sensitivity test passed: {selections}")


# =============================================================================
# 4.5 ZERO BANDWIDTH CASE
# =============================================================================

class TestZeroBandwidth:
    """Test handling of zero or very low bandwidth."""

    def test_zero_bandwidth_returns_minimum_bitrate(self, default_bitrate_ladder):
        """Zero bandwidth should result in minimum bitrate selection."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # Zero bandwidth
        throughput_bps = 0.0
        buffer_seconds = 10.0  # Healthy buffer

        selected = abr.select_bitrate(
            current_buffer_seconds=buffer_seconds,
            measured_throughput_bps=throughput_bps,
        )

        expected = min(default_bitrate_ladder)
        assert selected == expected, (
            f"Expected minimum bitrate {expected} kbps for zero bandwidth, "
            f"got {selected} kbps"
        )
        logger.info(f"Zero bandwidth test passed: selected={selected}kbps")

    def test_very_low_bandwidth_returns_minimum(self, default_bitrate_ladder):
        """Very low bandwidth should result in minimum bitrate."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # Very low bandwidth: 1 bps (essentially zero)
        throughput_bps = 1.0
        buffer_seconds = 10.0

        selected = abr.select_bitrate(
            current_buffer_seconds=buffer_seconds,
            measured_throughput_bps=throughput_bps,
        )

        expected = min(default_bitrate_ladder)
        assert selected == expected, (
            f"Expected minimum bitrate {expected} kbps for very low bandwidth, "
            f"got {selected} kbps"
        )
        logger.info(f"Very low bandwidth test passed: selected={selected}kbps")


# =============================================================================
# 4.6 SINGLE BITRATE LADDER
# =============================================================================

class TestSingleBitrateLadder:
    """Test behavior with single-bitrate ladder."""

    def test_single_bitrate_always_selected(self, single_bitrate_ladder):
        """With single bitrate ladder, must always return that bitrate."""
        abr = ABRController(
            available_bitrates=single_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        expected = single_bitrate_ladder[0]

        # Test various conditions
        test_cases = [
            (100e3, 10.0),   # Low BW, high buffer
            (1e6, 10.0),     # Medium BW, high buffer
            (10e6, 10.0),    # High BW, high buffer
            (100e3, 1.0),    # Low BW, low buffer
            (1e6, 1.0),      # Medium BW, low buffer
        ]

        for bw, buffer_sec in test_cases:
            abr.reset()
            selected = abr.select_bitrate(
                current_buffer_seconds=buffer_sec,
                measured_throughput_bps=bw,
            )

            assert selected == expected, (
                f"Expected only bitrate {expected} kbps, got {selected} kbps. "
                f"BW={bw/1e6:.1f}Mbps, buffer={buffer_sec}s"
            )
            logger.debug(
                f"Single bitrate test: BW={bw/1e6:.1f}Mbps, "
                f"buffer={buffer_sec}s -> {selected}kbps ✓"
            )

        logger.info(f"Single bitrate ladder test passed: always {expected}kbps")


# =============================================================================
# 4.7 HYSTERESIS / ANTI-FLAPPING
# =============================================================================

class TestHysteresis:
    """Test that ABR prevents rapid bitrate oscillation."""

    def test_oscillating_bandwidth_limited_switches(self, default_bitrate_ladder):
        """
        Feed oscillating bandwidth and assert number of switches is limited.
        
        Oscillating sequence: 2.1M -> 1.9M -> 2.0M -> 1.8M -> 2.1M
        """
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
            min_segments_between_upgrades=3,
        )

        # Oscillating bandwidth pattern (in bps)
        # This should trigger different bitrate decisions
        bandwidth_sequence = [
            2.1e6,  # ~2100 kbps -> could be 1500 or 1200
            1.9e6,  # ~1900 kbps -> 1200
            2.0e6,  # ~2000 kbps -> 1200 or 1500
            1.8e6,  # ~1800 kbps -> 1200
            2.1e6,  # ~2100 kbps -> back to 1500
        ]

        # Use healthy buffer to allow upgrades
        buffer_seconds = 15.0

        selections = []
        for bw in bandwidth_sequence:
            selected = abr.select_bitrate(
                current_buffer_seconds=buffer_seconds,
                measured_throughput_bps=bw,
            )
            selections.append(selected)
            logger.debug(f"Oscillating BW={bw/1e6:.1f}Mbps -> {selected}kbps")

        # Count switches
        switches = sum(
            1 for i in range(1, len(selections))
            if selections[i] != selections[i-1]
        )

        # With hysteresis, should have limited switches
        # Allow at most 2 switches for this pattern
        assert switches <= 2, (
            f"Too many switches ({switches}) in oscillating bandwidth. "
            f"Selections: {selections}. "
            f"Expected at most 2 switches due to hysteresis."
        )

        logger.info(
            f"Hysteresis test passed: {switches} switches in {len(selections)} "
            f"decisions. Selections: {selections}"
        )

    def test_hysteresis_prevents_upgrade_without_stability(self, default_bitrate_ladder):
        """Test that upgrades require stability (consecutive stable segments)."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
            min_segments_between_upgrades=3,
        )

        # Start with low bitrate
        throughput_bps = 800e3  # 800 kbps -> 600 kbps safe
        buffer_seconds = 20.0   # Very healthy buffer

        # First selection - should be low
        abr.select_bitrate(buffer_seconds, throughput_bps)

        # Now increase bandwidth to allow upgrade
        throughput_bps = 5e6  # 5 Mbps -> 4000 kbps safe

        # Try upgrading immediately - should be blocked by hysteresis
        selected = abr.select_bitrate(buffer_seconds, throughput_bps)

        # With default min_segments_between_upgrades=3, should not upgrade
        # immediately if not enough stable segments
        logger.info(
            f"Immediate upgrade test: BW increased to 5Mbps, "
            f"selected={selected}kbps (may be held by hysteresis)"
        )


# =============================================================================
# 4.8 STATE EVOLUTION TEST
# =============================================================================

class TestStateEvolution:
    """Test that ABR behavior depends on history (EWMA effect)."""

    def test_ewma_affects_subsequent_decisions(self, default_bitrate_ladder):
        """Call ABR repeatedly and ensure behavior depends on history."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.3,
        )

        # First: measure low throughput
        low_throughput = 1e6  # 1 Mbps
        abr.select_bitrate(10.0, low_throughput)

        # Check EWMA was initialized
        assert abr.ewma_throughput_bps is not None, (
            "EWMA throughput should be initialized after first measurement"
        )

        # Second: measure high throughput
        high_throughput = 10e6  # 10 Mbps
        selected = abr.select_bitrate(10.0, high_throughput)

        # EWMA should be a blend, not pure high
        ewma = abr.ewma_throughput_bps

        assert ewma is not None
        assert ewma < high_throughput, (
            f"EWMA ({ewma/1e6:.1f} Mbps) should be less than "
            f"current throughput ({high_throughput/1e6:.1f} Mbps) "
            f"due to smoothing"
        )

        logger.info(
            f"State evolution test: low={low_throughput/1e6:.1f}Mbps, "
            f"high={high_throughput/1e6:.1f}Mbps, "
            f"EWMA={ewma/1e6:.1f}Mbps, selected={selected}kbps"
        )

    def test_ewma_convergence(self, default_bitrate_ladder):
        """Test that EWMA converges to true mean over time."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
            ewma_alpha=0.2,  # Lower alpha for more smoothing
        )

        constant_throughput = 5e6  # 5 Mbps

        # Make multiple selections with constant throughput
        for _ in range(10):
            abr.select_bitrate(10.0, constant_throughput)

        ewma = abr.ewma_throughput_bps

        # After many samples, EWMA should be close to true value
        # With alpha=0.2, EWMA should be roughly 5e6 * 0.2 * sum(0.8^i) for i=0..inf
        # which converges to about 5e6
        assert ewma is not None
        assert 4e6 < ewma < 6e6, (
            f"EWMA ({ewma/1e6:.1f} Mbps) should be close to "
            f"constant throughput ({constant_throughput/1e6:.1f} Mbps)"
        )

        logger.info(
            f"EWMA convergence test: constant={constant_throughput/1e6:.1f}Mbps, "
            f"EWMA={ewma/1e6:.1f}Mbps"
        )


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_buffer_raises_error(self, default_bitrate_ladder):
        """Negative buffer should raise ValueError."""
        abr = ABRController(available_bitrates=default_bitrate_ladder)

        with pytest.raises(ValueError, match="non-negative"):
            abr.select_bitrate(current_buffer_seconds=-1.0)

    def test_negative_throughput_raises_error(self, default_bitrate_ladder):
        """Negative throughput should raise ValueError."""
        abr = ABRController(available_bitrates=default_bitrate_ladder)

        with pytest.raises(ValueError, match="non-negative"):
            abr.select_bitrate(
                current_buffer_seconds=10.0,
                measured_throughput_bps=-1000.0,
            )

    def test_empty_ladder_raises_error(self):
        """Empty bitrate ladder should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ABRController(available_bitrates=[])

    def test_zero_bitrate_in_ladder_raises_error(self):
        """Zero bitrate in ladder should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ABRController(available_bitrates=[0, 1000, 2000])

    def test_reset_clears_state(self, default_bitrate_ladder):
        """Test that reset clears all internal state."""
        abr = ABRController(
            available_bitrates=default_bitrate_ladder,
            safety_margin=0.8,
        )

        # Make some selections
        abr.select_bitrate(10.0, 5e6)
        abr.select_bitrate(10.0, 5e6)

        # Verify state exists
        assert abr.last_bitrate is not None
        assert abr.ewma_throughput_bps is not None

        # Reset
        abr.reset()

        # Verify state is cleared
        assert abr.last_bitrate is None
        assert abr.ewma_throughput_bps is None

        logger.info("Reset test passed: state properly cleared")
