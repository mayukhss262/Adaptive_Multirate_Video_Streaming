"""
Hybrid Conservative Throughput-Buffer ABR Algorithm

Combines throughput estimation with buffer-based safety logic to select
video rendition bitrates adaptively. Uses EWMA smoothing, safety margins,
and hysteresis to prevent oscillation.
"""

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class ABRController:
    """
    Adaptive Bitrate controller using hybrid throughput-buffer logic.

    The controller maintains state across segment decisions and applies
    conservative policies to avoid rebuffering and quality oscillation.

    Parameters
    ----------
    available_bitrates : list[int]
        Sorted list of available bitrates in kbps.
    safety_margin : float
        Multiplicative safety factor (0 < safety_margin < 1.0).
        Default 0.8 means use 80% of estimated throughput.
    ewma_alpha : float
        Smoothing factor for EWMA throughput estimation (0 < alpha <= 1.0).
        Higher values give more weight to recent measurements.
    buffer_low_threshold : float
        Buffer level (seconds) below which we force downgrade.
    buffer_target : float
        Buffer level (seconds) considered comfortable for upgrades.
    buffer_high_threshold : float
        Buffer level (seconds) above which we allow aggressive upgrades.
    min_segments_between_upgrades : int
        Minimum number of stable segments before allowing an upgrade.
    """

    def __init__(
        self,
        available_bitrates: Optional[list[int]] = None,
        safety_margin: float = 0.8,
        ewma_alpha: float = 0.7,
        buffer_low_threshold: float = config.BUFFER_LOW,
        buffer_target: float = config.BUFFER_TARGET,
        buffer_high_threshold: float = config.BUFFER_MAX,
        min_segments_between_upgrades: int = 1,
    ):
        # Validate and set bitrate ladder
        if available_bitrates is None:
            # Extract bitrates from config.RENDITIONS (label, width, height, bitrate_kbps)
            available_bitrates = [r[3] for r in config.RENDITIONS]

        if not available_bitrates:
            raise ValueError("Available bitrates list cannot be empty")

        if any(b <= 0 for b in available_bitrates):
            raise ValueError("All bitrates must be positive")

        # Sort and deduplicate
        self.available_bitrates: list[int] = sorted(set(available_bitrates))

        # Validate parameters
        if not (0.0 < safety_margin < 1.0):
            raise ValueError(f"Safety margin must be in (0, 1), got {safety_margin}")
        if not (0.0 < ewma_alpha <= 1.0):
            raise ValueError(f"EWMA alpha must be in (0, 1], got {ewma_alpha}")
        if buffer_low_threshold < 0:
            raise ValueError(f"Buffer low threshold must be non-negative, got {buffer_low_threshold}")
        if buffer_target < buffer_low_threshold:
            raise ValueError(f"Buffer target ({buffer_target}) must be >= buffer low threshold ({buffer_low_threshold})")
        if min_segments_between_upgrades < 0:
            raise ValueError(f"Min segments between upgrades must be non-negative, got {min_segments_between_upgrades}")

        self.safety_margin = safety_margin
        self.ewma_alpha = ewma_alpha
        self.buffer_low_threshold = buffer_low_threshold
        self.buffer_target = buffer_target
        self.buffer_high_threshold = buffer_high_threshold
        self.min_segments_between_upgrades = min_segments_between_upgrades

        # Internal state
        self._last_bitrate: Optional[int] = None
        self._ewma_throughput: Optional[float] = None  # in bps
        self._segments_since_switch: int = 0
        self._consecutive_stable: int = 0

        logger.info(
            "ABRController initialized: bitrates=%s kbps, safety_margin=%.2f, "
            "ewma_alpha=%.2f, buffer_low=%.1fs, buffer_target=%.1fs",
            self.available_bitrates, safety_margin, ewma_alpha,
            buffer_low_threshold, buffer_target
        )

    def reset(self) -> None:
        """Reset controller state for a new streaming session."""
        self._last_bitrate = None
        self._ewma_throughput = None
        self._segments_since_switch = 0
        self._consecutive_stable = 0
        logger.debug("ABRController state reset")

    def update_throughput(self, measured_throughput_bps: float) -> None:
        """
        Update the EWMA throughput estimate with a new measurement.

        Parameters
        ----------
        measured_throughput_bps : float
            Measured download throughput in bits per second.
        """
        if measured_throughput_bps < 0:
            raise ValueError(f"Throughput must be non-negative, got {measured_throughput_bps}")

        if self._ewma_throughput is None:
            # First measurement: initialize directly
            self._ewma_throughput = measured_throughput_bps
        else:
            # EWMA update
            self._ewma_throughput = (
                self.ewma_alpha * measured_throughput_bps
                + (1 - self.ewma_alpha) * self._ewma_throughput
            )

        logger.debug(
            "Throughput updated: measured=%.2f Mbps, ewma=%.2f Mbps",
            measured_throughput_bps / 1e6,
            self._ewma_throughput / 1e6
        )

    def select_bitrate(
        self,
        current_buffer_seconds: float,
        measured_throughput_bps: Optional[float] = None,
    ) -> int:
        """
        Select the appropriate bitrate based on throughput and buffer state.

        Parameters
        ----------
        current_buffer_seconds : float
            Current playback buffer occupancy in seconds.
        measured_throughput_bps : float or None
            Most recent measured throughput in bps. If provided, updates
            the EWMA estimate before making the decision.

        Returns
        -------
        int
            Selected bitrate in kbps.
        """
        # Validate inputs
        if current_buffer_seconds < 0:
            raise ValueError(f"Buffer seconds must be non-negative, got {current_buffer_seconds}")

        # Update throughput estimate if measurement provided
        if measured_throughput_bps is not None:
            self.update_throughput(measured_throughput_bps)

        # Determine effective throughput for decision
        if self._ewma_throughput is None or self._ewma_throughput <= 0:
            # No valid throughput estimate: use minimum bitrate
            selected = self.available_bitrates[0]
            reason = "no_throughput_estimate"
            logger.warning(
                "No valid throughput estimate, selecting minimum bitrate: %d kbps",
                selected
            )
            self._update_state(selected, reason)
            return selected

        # Apply safety margin to get safe deliverable throughput
        safe_throughput_bps = self._ewma_throughput * self.safety_margin

        # Find highest bitrate below safe throughput
        throughput_kbps = safe_throughput_bps / 1000.0
        candidate_bitrate = self.available_bitrates[0]
        for bitrate in self.available_bitrates:
            if bitrate <= throughput_kbps:
                candidate_bitrate = bitrate
            else:
                break

        # Apply buffer-based adjustments
        selected = self._apply_buffer_logic(
            candidate_bitrate, current_buffer_seconds, throughput_kbps
        )

        # Apply hysteresis: prevent rapid oscillation
        selected = self._apply_hysteresis(selected, current_buffer_seconds)

        # Log decision
        action = "hold"
        if self._last_bitrate is not None:
            if selected > self._last_bitrate:
                action = "upgrade"
            elif selected < self._last_bitrate:
                action = "downgrade"

        logger.info(
            "Bitrate decision: %d kbps (%s) | buffer=%.1fs | throughput=%.2f Mbps | "
            "safe_throughput=%.2f Mbps | ewma=%.2f Mbps",
            selected, action, current_buffer_seconds,
            throughput_kbps / 1000.0,
            safe_throughput_bps / 1e6,
            self._ewma_throughput / 1e6
        )

        self._update_state(selected, action)
        return selected

    def _apply_buffer_logic(
        self,
        candidate_bitrate: int,
        buffer_seconds: float,
        throughput_kbps: float,
    ) -> int:
        """
        Adjust candidate bitrate based on buffer occupancy.

        If buffer is low, force downgrade. If buffer is high, allow upgrade.
        """
        selected = candidate_bitrate

        # Critical buffer: force minimum safe bitrate
        if buffer_seconds < config.BUFFER_MIN:
            # Panic mode: use minimum bitrate
            selected = self.available_bitrates[0]
            logger.warning(
                "Buffer critical (%.1fs < %.1fs), forcing minimum bitrate: %d kbps",
                buffer_seconds, config.BUFFER_MIN, selected
            )
            return selected

        # Low buffer: force downgrade by at least one step
        if buffer_seconds < self.buffer_low_threshold:
            if self._last_bitrate is not None:
                # Find current position in ladder
                current_idx = self._find_bitrate_index(self._last_bitrate)
                if current_idx > 0:
                    # Downgrade by at least one step
                    downgraded_idx = max(0, current_idx - 1)
                    # Also ensure it's below safe throughput
                    safe_idx = self._find_highest_below(throughput_kbps)
                    selected_idx = min(downgraded_idx, safe_idx)
                    selected = self.available_bitrates[selected_idx]
                    logger.info(
                        "Buffer low (%.1fs < %.1fs), downgrading to %d kbps",
                        buffer_seconds, self.buffer_low_threshold, selected
                    )
                    return selected

        # Healthy buffer: allow upgrade but be conservative
        if buffer_seconds >= self.buffer_high_threshold:
            # Very healthy buffer: allow upgrade if throughput supports it
            if self._last_bitrate is not None:
                current_idx = self._find_bitrate_index(self._last_bitrate)
                # Only upgrade one step at a time
                if current_idx < len(self.available_bitrates) - 1:
                    upgrade_idx = current_idx + 1
                    upgrade_bitrate = self.available_bitrates[upgrade_idx]
                    # Check if upgrade is safe
                    if upgrade_bitrate <= throughput_kbps:
                        # Check if we've been stable long enough
                        if self._consecutive_stable >= self.min_segments_between_upgrades:
                            selected = upgrade_bitrate
                            logger.info(
                                "Buffer healthy (%.1fs >= %.1fs), upgrading to %d kbps",
                                buffer_seconds, self.buffer_high_threshold, selected
                            )
                        else:
                            logger.debug(
                                "Buffer healthy but not enough stable segments (%d < %d)",
                                self._consecutive_stable, self.min_segments_between_upgrades
                            )

        return selected

    def _apply_hysteresis(
        self,
        candidate_bitrate: int,
        buffer_seconds: float,
    ) -> int:
        """
        Apply hysteresis to prevent rapid bitrate oscillation.

        Downgrades are immediate, upgrades require stability.
        """
        if self._last_bitrate is None:
            # First selection: no hysteresis needed
            return candidate_bitrate

        if candidate_bitrate < self._last_bitrate:
            # Downgrade: always allow immediately
            return candidate_bitrate

        if candidate_bitrate > self._last_bitrate:
            # Upgrade: check if we've been stable
            if self._consecutive_stable < self.min_segments_between_upgrades:
                # Not stable enough: hold current bitrate
                logger.debug(
                    "Hysteresis: holding %d kbps (stable=%d < %d)",
                    self._last_bitrate, self._consecutive_stable,
                    self.min_segments_between_upgrades
                )
                return self._last_bitrate

        return candidate_bitrate

    def _update_state(self, selected_bitrate: int, reason: str) -> None:
        """Update internal state after a bitrate decision."""
        if self._last_bitrate is not None:
            if selected_bitrate == self._last_bitrate:
                self._consecutive_stable += 1
            else:
                self._consecutive_stable = 0
                self._segments_since_switch = 0
        else:
            self._consecutive_stable = 1

        self._last_bitrate = selected_bitrate
        self._segments_since_switch += 1

    def _find_bitrate_index(self, bitrate: int) -> int:
        """Find the index of a bitrate in the ladder."""
        try:
            return self.available_bitrates.index(bitrate)
        except ValueError:
            # Bitrate not in ladder: find closest lower
            for i, b in enumerate(self.available_bitrates):
                if b >= bitrate:
                    return max(0, i - 1)
            return len(self.available_bitrates) - 1

    def _find_highest_below(self, throughput_kbps: float) -> int:
        """Find index of highest bitrate below given throughput."""
        idx = 0
        for i, bitrate in enumerate(self.available_bitrates):
            if bitrate <= throughput_kbps:
                idx = i
            else:
                break
        return idx

    @property
    def last_bitrate(self) -> Optional[int]:
        """Return the last selected bitrate in kbps, or None if not yet set."""
        return self._last_bitrate

    @property
    def ewma_throughput_bps(self) -> Optional[float]:
        """Return the current EWMA throughput estimate in bps."""
        return self._ewma_throughput
