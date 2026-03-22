"""
Pytest configuration and shared fixtures for ABR streaming tests.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pytest

import config
from abr_algorithm import ABRController
from network_sim import NetworkSimulator
from streamer import Streamer


# =============================================================================
# BITRATE LADDER FIXTURES
# =============================================================================

@pytest.fixture
def default_bitrate_ladder():
    """Default bitrate ladder in kbps from config."""
    return [r[3] for r in config.RENDITIONS]


@pytest.fixture
def small_bitrate_ladder():
    """Small bitrate ladder for testing."""
    return [300, 800, 1500, 3000]


@pytest.fixture
def single_bitrate_ladder():
    """Single bitrate ladder for edge case testing."""
    return [1000]


@pytest.fixture
def custom_bitrate_ladder():
    """Custom bitrate ladder for specific test scenarios."""
    return [500, 1500, 4000, 8000]


# =============================================================================
# SEGMENT DURATION FIXTURE
# =============================================================================

@pytest.fixture
def segment_duration():
    """Segment duration in seconds from config."""
    return config.CHUNK_DURATION


# =============================================================================
# BUFFER CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def buffer_config():
    """Buffer configuration from config."""
    return {
        "min": config.BUFFER_MIN,
        "low": config.BUFFER_LOW,
        "target": config.BUFFER_TARGET,
        "max": config.BUFFER_MAX,
    }


# =============================================================================
# FAKE NETWORK SIMULATOR
# =============================================================================

class FakeNetworkSimulator:
    """
    Deterministic network simulator for testing.
    
    Supports constant bandwidth, predefined sequences, and zero bandwidth case.
    """

    def __init__(
        self,
        constant_bw: Optional[float] = None,
        sequence: Optional[list[float]] = None,
        zero_at_end: bool = False,
    ):
        """
        Initialize fake network simulator.
        
        Parameters
        ----------
        constant_bw : float, optional
            Constant bandwidth in bps. If provided, always returns this value.
        sequence : list[float], optional
            Predefined sequence of bandwidth values. Cycles through this list.
        zero_at_end : bool
            If True, appends zero bandwidth at the end of sequence.
        """
        self._constant_bw = constant_bw
        self._sequence = sequence.copy() if sequence else []
        self._zero_at_end = zero_at_end
        self._index = 0
        self._time = 0.0

    def get_current_bandwidth(self) -> float:
        """Return the current bandwidth in bps."""
        if self._constant_bw is not None:
            return self._constant_bw
        
        if self._sequence:
            if self._index >= len(self._sequence):
                if self._zero_at_end:
                    return 0.0
                # Cycle back
                self._index = 0
            bw = self._sequence[self._index]
            self._index += 1
            return bw
        
        return 5e6  # Default 5 Mbps

    def advance(self, dt: float) -> None:
        """Advance the internal clock by dt seconds."""
        self._time += dt

    def reset(self) -> None:
        """Reset the simulator state."""
        self._index = 0
        self._time = 0.0

    @property
    def elapsed(self) -> float:
        """Return elapsed time."""
        return self._time


@pytest.fixture
def fake_network_constant():
    """Factory fixture for constant bandwidth network."""
    def _create(bw_bps: float) -> FakeNetworkSimulator:
        return FakeNetworkSimulator(constant_bw=bw_bps)
    return _create


@pytest.fixture
def fake_network_sequence():
    """Factory fixture for predefined bandwidth sequence."""
    def _create(sequence: list[float]) -> FakeNetworkSimulator:
        return FakeNetworkSimulator(sequence=sequence)
    return _create


@pytest.fixture
def fake_network_zero():
    """Factory fixture for zero bandwidth case."""
    def _create(sequence: list[float]) -> FakeNetworkSimulator:
        return FakeNetworkSimulator(sequence=sequence, zero_at_end=True)
    return _create


# =============================================================================
# FAKE ABR CONTROLLER
# =============================================================================

class FakeABRController:
    """
    Fake ABR controller that always returns a fixed bitrate.
    
    Used to isolate streamer unit tests.
    """

    def __init__(self, fixed_bitrate_kbps: int = 1000):
        """
        Initialize fake ABR controller.
        
        Parameters
        ----------
        fixed_bitrate_kbps : int
            Fixed bitrate to return in kbps.
        """
        self.fixed_bitrate_kbps = fixed_bitrate_kbps
        self._call_count = 0
        self._last_buffer = 0.0
        self._last_throughput = None

    def select_bitrate(
        self,
        current_buffer_seconds: float,
        measured_throughput_bps: Optional[float] = None,
    ) -> int:
        """Return the fixed bitrate."""
        self._call_count += 1
        self._last_buffer = current_buffer_seconds
        self._last_throughput = measured_throughput_bps
        return self.fixed_bitrate_kbps

    def reset(self) -> None:
        """Reset controller state."""
        self._call_count = 0
        self._last_buffer = 0.0
        self._last_throughput = None

    @property
    def call_count(self) -> int:
        """Return number of times select_bitrate was called."""
        return self._call_count

    @property
    def last_buffer(self) -> float:
        """Return last buffer value passed to select_bitrate."""
        return self._last_buffer

    @property
    def last_throughput(self) -> Optional[float]:
        """Return last throughput value passed to select_bitrate."""
        return self._last_throughput


@pytest.fixture
def fake_abr():
    """Factory fixture for fake ABR controller."""
    def _create(fixed_bitrate_kbps: int = 1000) -> FakeABRController:
        return FakeABRController(fixed_bitrate_kbps=fixed_bitrate_kbps)
    return _create


# =============================================================================
# REAL NETWORK SIMULATOR (DETERMINISTIC)
# =============================================================================

@pytest.fixture
def deterministic_network():
    """Factory fixture for deterministic network simulator."""
    def _create(
        base_bw: float = 5e6,
        amplitude: float = 0.0,
        period: float = 10.0,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> NetworkSimulator:
        return NetworkSimulator(
            base_bw=base_bw,
            amplitude=amplitude,
            period=period,
            noise_std=noise_std,
            seed=seed,
            use_wall_clock=False,
        )
    return _create


# =============================================================================
# REAL ABR CONTROLLER
# =============================================================================

@pytest.fixture
def real_abr():
    """Factory fixture for real ABR controller."""
    def _create(
        available_bitrates: Optional[list[int]] = None,
        safety_margin: float = 0.8,
        ewma_alpha: float = 0.3,
        min_segments_between_upgrades: int = 3,
    ) -> ABRController:
        return ABRController(
            available_bitrates=available_bitrates,
            safety_margin=safety_margin,
            ewma_alpha=ewma_alpha,
            min_segments_between_upgrades=min_segments_between_upgrades,
        )
    return _create


# =============================================================================
# STREAMER FACTORY
# =============================================================================

@pytest.fixture
def create_streamer():
    """Factory fixture to create a streamer with dependencies."""
    def _create(
        network_sim: NetworkSimulator,
        abr_controller: ABRController,
        chunk_duration: float = config.CHUNK_DURATION,
        buffer_max: float = config.BUFFER_MAX,
        total_segments: int = 10,
    ) -> Streamer:
        return Streamer(
            network_sim=network_sim,
            abr_controller=abr_controller,
            chunk_duration=chunk_duration,
            buffer_max=buffer_max,
            total_segments=total_segments,
        )
    return _create


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@pytest.fixture(autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("numpy").setLevel(logging.WARNING)
