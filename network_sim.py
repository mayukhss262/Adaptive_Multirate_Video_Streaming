"""
Simulates a fluctuating network connection using configurable bandwidth models.
Exposes a simple get_current_bandwidth() API.

Available models (selected via integer):
    0 — Sine       : sine wave + Gaussian noise (default)
    1 — Constant   : fixed bandwidth
    2 — Ramp       : linear ramp up then down
    3 — Step       : sudden jumps between two levels
    4 — Random Walk: unpredictable random jumps each tick
    5 — Congested  : good bandwidth with periodic congestion drops
    6 — Degrading  : steadily declining bandwidth

Usage:
    from network_sim import NetworkSimulator
    sim = NetworkSimulator(model=0)        # sine (default)
    sim = NetworkSimulator(model=1)        # constant
    bw = sim.get_current_bandwidth()       # bits per second
"""

import time
import logging
import numpy as np
import config

logger = logging.getLogger(__name__)


MODEL_NAMES = {
    0: "Sine",
    1: "Constant",
    2: "Ramp",
    3: "Step",
    4: "Random Walk",
    5: "Congested",
    6: "Degrading",
}

VALID_MODELS = set(MODEL_NAMES.keys())


class NetworkSimulator:
    """
    Bandwidth simulator with pluggable network models.

    Parameters
    ----------
    model     : int – selects the bandwidth model (0-6)
    base_bw   : float – centre bandwidth in bps
    amplitude : float – peak deviation in bps (used by sine, ramp, step)
    period    : float – oscillation / cycle period in seconds
    noise_std : float – std-dev of Gaussian noise in bps
    seed      : int | None – RNG seed for reproducibility
    use_wall_clock : bool – if True, time comes from the real clock;
                     if False, advance manually with advance(dt)
    """

    def __init__(
        self,
        model: int = config.NET_MODEL,
        base_bw: float = config.NET_BASE_BW,
        amplitude: float = config.NET_AMPLITUDE,
        period: float = config.NET_PERIOD,
        noise_std: float = config.NET_NOISE_STD,
        seed: int | None = config.NET_SEED,
        use_wall_clock: bool = True,
    ):
        if model not in VALID_MODELS:
            raise ValueError(
                f"Invalid network model {model}. "
                f"Choose from: {', '.join(f'{k}={v}' for k, v in sorted(MODEL_NAMES.items()))}"
            )

        self.model = model
        self.model_name = MODEL_NAMES[model]
        self.base_bw = base_bw
        self.amplitude = amplitude
        self.period = period
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._use_wall_clock = use_wall_clock
        self._start_time = time.time() if use_wall_clock else 0.0
        self._manual_time = 0.0

        logger.info(
            "NetworkSimulator initialized: model=%d (%s), base_bw=%.1f Mbps, "
            "amplitude=%.1f Mbps, period=%.1fs",
            model, self.model_name, base_bw / 1e6, amplitude / 1e6, period,
        )

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

    # ------------------------------------------------------------------
    # Bandwidth models (each tuned for distinct ABR quality behavior)
    # ABR thresholds (safety_margin=0.8):
    #   400 kbps  needs raw BW >= 0.50 Mbps
    #  1200 kbps  needs raw BW >= 1.50 Mbps
    #  3000 kbps  needs raw BW >= 3.75 Mbps
    #  6000 kbps  needs raw BW >= 7.50 Mbps
    # ------------------------------------------------------------------

    def _model_sine(self, t: float) -> float:
        """Sine wave oscillation around 4 Mbps: swings 2-6 Mbps.
        Crosses 1200/3000 kbps tiers → ABR oscillates between 480p and 720p.
        Periodic quality switches every ~5s with moderate buffer stability."""
        center = 4e6
        amp = 2e6
        sine = amp * np.sin(2 * np.pi * t / self.period)
        noise = self._rng.normal(0, 2e5)
        return max(center + sine + noise, 0.0)

    def _model_constant(self, t: float) -> float:
        """Constant high bandwidth at 8 Mbps with minimal noise.
        Safe throughput = 8 * 0.8 = 6.4 Mbps → ABR consistently selects 1080p (6000 kbps).
        No quality switches, no stalls, maximum quality throughout."""
        noise = self._rng.normal(0, 1e5)
        return max(8e6 + noise, 0.0)

    def _model_ramp(self, t: float) -> float:
        """Monotonic ramp: starts at 1 Mbps, climbs to 9 Mbps over 30s.
        ABR progression: 240p → 480p → 720p → 1080p as bandwidth improves.
        Tests ABR's ability to upgrade quality gradually without overshooting."""
        ramp_duration = 30.0
        start_bw = 1e6
        end_bw = 9e6
        if t >= ramp_duration:
            bw = end_bw
        else:
            frac = t / ramp_duration
            bw = start_bw + frac * (end_bw - start_bw)
        noise = self._rng.normal(0, 1.5e5)
        #return max(bw + noise, 0.0)
        return max(bw, 0.0)

    def _model_step(self, t: float) -> float:
        """Abrupt step function: alternates between 8 Mbps and 1.5 Mbps.
        HIGH phase: safe = 6.4 Mbps → 1080p (6000 kbps)
        LOW phase: safe = 1.2 Mbps → 480p (1200 kbps) or 240p (400 kbps)
        Produces dramatic quality switches every 5s with buffer stress."""
        half = self.period / 2
        in_high_phase = (t % self.period) < half
        if in_high_phase:
            bw = 8e6   # supports 1080p
        else:
            bw = 1.5e6  # supports 480p or 240p
        noise = self._rng.normal(0, 1e5)
        return max(bw + noise, 0.0)

    def _model_random_walk(self, t: float) -> float:
        """Chaotic random walk: unpredictable jumps between 0.5 and 10 Mbps.
        Large variance forces ABR into erratic quality decisions.
        Tests robustness under highly unstable network conditions."""
        step = self._rng.normal(0, 1.5e6)
        if not hasattr(self, '_rw_bw'):
            self._rw_bw = 5e6
        self._rw_bw = float(np.clip(self._rw_bw + step, 0.5e6, 10e6))
        noise = self._rng.normal(0, 1e5)
        return max(self._rw_bw + noise, 0.0)

    def _model_congested(self, t: float) -> float:
        """Congested network: 7 Mbps baseline with periodic severe drops.
        Normal: safe = 5.6 Mbps → 720p/1080p
        Congestion (every 12s for 3s): drops to 0.3 Mbps → forces 240p + stalls
        Tests ABR recovery from severe congestion events."""
        cycle = 12.0
        congestion_duration = 3.0
        phase = t % cycle
        if phase < congestion_duration:
            bw = 0.3e6  # severe congestion: below 400 kbps safe threshold
        else:
            bw = 7e6    # good bandwidth
        noise = self._rng.normal(0, 1.5e5)
        return max(bw + noise, 0.0)

    def _model_degrading(self, t: float) -> float:
        """Progressive degradation: starts at 9 Mbps, declines to 0.8 Mbps over 30s.
        ABR descent: 1080p → 720p → 480p → 240p as bandwidth erodes.
        Tests ABR's ability to gracefully degrade without catastrophic stalls."""
        total_duration = 30.0
        start_bw = 9e6
        end_bw = 0.8e6
        if t >= total_duration:
            bw = end_bw
        else:
            frac = t / total_duration
            bw = start_bw - frac * (start_bw - end_bw)
        noise = self._rng.normal(0, 1.5e5)
        return max(bw + noise, 0.0)

    # ------------------------------------------------------------------
    # Model dispatch table
    # ------------------------------------------------------------------

    _MODELS = {
        0: _model_sine,
        1: _model_constant,
        2: _model_ramp,
        3: _model_step,
        4: _model_random_walk,
        5: _model_congested,
        6: _model_degrading,
    }

    def _compute_bandwidth(self, t: float) -> float:
        """Dispatch to the selected model and clamp to non-negative."""
        model_fn = self._MODELS[self.model]
        return max(model_fn(self, t), 0.0)

    def get_current_bandwidth(self) -> float:
        """Return the currently available bandwidth in **bits per second**."""
        return self._compute_bandwidth(self.elapsed)


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

def _demo() -> None:
    DURATION = 30.0
    DT = 0.25

    print("=" * 60)
    print("Network Bandwidth Simulator — Model Demo")
    print("=" * 60)

    for model_id, model_name in sorted(MODEL_NAMES.items()):
        sim = NetworkSimulator(model=model_id, seed=42, use_wall_clock=False)
        samples: list[tuple[float, float]] = []
        t = 0.0
        while t <= DURATION:
            samples.append((t, sim.get_current_bandwidth()))
            sim.advance(DT)
            t += DT

        bws = [bw for _, bw in samples]
        print(f"\n  Model {model_id}: {model_name}")
        print(f"  {'-' * 40}")
        print(f"    Min     : {min(bws) / 1e6:8.3f} Mbps")
        print(f"    Max     : {max(bws) / 1e6:8.3f} Mbps")
        print(f"    Mean    : {np.mean(bws) / 1e6:8.3f} Mbps")
        print(f"    Std Dev : {np.std(bws) / 1e6:8.3f} Mbps")

    # Plot all models
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(MODEL_NAMES), 1, figsize=(12, 3 * len(MODEL_NAMES)), sharex=True)
        fig.suptitle("Network Bandwidth Models", fontsize=14, fontweight="bold")

        for ax, (model_id, model_name) in zip(axes, sorted(MODEL_NAMES.items())):
            sim = NetworkSimulator(model=model_id, seed=42, use_wall_clock=False)
            times, bws = [], []
            t = 0.0
            while t <= DURATION:
                times.append(t)
                bws.append(sim.get_current_bandwidth() / 1e6)
                sim.advance(DT)
                t += DT
            ax.plot(times, bws, lw=1.2, label=model_name)
            ax.set_ylabel("Bandwidth (Mbps)")
            ax.set_title(f"Model {model_id}: {model_name}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig("network_models_demo.png", dpi=150)
        print(f"\n  Plot saved to network_models_demo.png")
        plt.show()
    except ImportError:
        print("\n  (matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    _demo()
