"""
Simulates a fluctuating network connection using a sine wave with
additive Gaussian noise.  Exposes a simple get_current_bandwidth() API.

Usage:
    from network_sim import NetworkSimulator
    sim = NetworkSimulator()
    bw = sim.get_current_bandwidth()   # bits per second
"""

import time
import numpy as np
import config


class NetworkSimulator:
    """
    Bandwidth oscillates as a sine wave around a base value, with additive
    Gaussian noise. Defaults are pulled from config.py.

    Parameters
    ----------
    base_bw   : float – centre bandwidth in bps
    amplitude : float – peak deviation in bps
    period    : float – oscillation period in seconds
    noise_std : float – std-dev of Gaussian noise in bps
    seed      : int | None – RNG seed for reproducibility
    use_wall_clock : bool – if True, time comes from the real clock;
                     if False, advance manually with advance(dt)
    """

    def __init__(
        self,
        base_bw: float = config.NET_BASE_BW,
        amplitude: float = config.NET_AMPLITUDE,
        period: float = config.NET_PERIOD,
        noise_std: float = config.NET_NOISE_STD,
        seed: int | None = config.NET_SEED,
        use_wall_clock: bool = True,
    ):
        self.base_bw = base_bw
        self.amplitude = amplitude
        self.period = period
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._use_wall_clock = use_wall_clock
        self._start_time = time.time() if use_wall_clock else 0.0
        self._manual_time = 0.0

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since the simulator was created."""
        if self._use_wall_clock:
            return time.time() - self._start_time
        return self._manual_time

    def advance(self, dt: float) -> None:
        """Advance the internal clock by *dt* seconds (manual mode)."""
        self._manual_time += dt

    def reset(self) -> None:
        """Reset the internal clock to zero."""
        self._start_time = time.time()
        self._manual_time = 0.0

    def get_current_bandwidth(self) -> float:
        """Return the currently available bandwidth in **bits per second**."""
        t = self.elapsed
        sine = self.amplitude * np.sin(2 * np.pi * t / self.period)
        noise = self._rng.normal(0, self.noise_std)
        return max(self.base_bw + sine + noise, 0.0)


#demo
def _demo() -> None:
    DURATION = 30.0
    DT = 0.25

    sim = NetworkSimulator(seed=42, use_wall_clock=False)
    samples: list[tuple[float, float]] = []
    t = 0.0
    while t <= DURATION:
        samples.append((t, sim.get_current_bandwidth()))
        sim.advance(DT)
        t += DT

    bws = [bw for _, bw in samples]
    print("=" * 60)
    print("Network Bandwidth Simulator — Demo")
    print("=" * 60)
    print(f"  Samples : {len(samples)}")
    print(f"  Min     : {min(bws)/1e6:8.3f} Mbps")
    print(f"  Max     : {max(bws)/1e6:8.3f} Mbps")
    print(f"  Mean    : {np.mean(bws)/1e6:8.3f} Mbps")
    print(f"  Std Dev : {np.std(bws)/1e6:8.3f} Mbps")
    print(f"\n  {'Time (s)':>10}  {'Bandwidth (Mbps)':>18}")
    print(f"  {'-'*10}  {'-'*18}")
    for t_val, bw in samples[:12]:
        print(f"  {t_val:10.2f}  {bw/1e6:18.3f}")
    print(f"  ... ({len(samples) - 12} more samples)")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot([s[0] for s in samples], [s[1]/1e6 for s in samples], lw=1.2)
        plt.xlabel("Time (s)")
        plt.ylabel("Bandwidth (Mbps)")
        plt.title("Simulated Network Bandwidth (Sine + Gaussian Noise)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("bandwidth_demo.png", dpi=150)
        print(f"\n Plot saved to bandwidth_demo.png")
        plt.show()
    except ImportError:
        print("\n  (matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    _demo()
