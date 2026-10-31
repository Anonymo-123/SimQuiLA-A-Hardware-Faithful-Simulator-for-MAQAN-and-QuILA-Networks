"""Link-model contract (DRAFT v0).

Any point-to-point QKD link model -- analytic, statistical or SeQUeNCe
event-level -- implements :class:`LinkModel`. Parameters are deliberately
explicit: there are no hidden physical defaults, because unrealistic defaults
were one of the problems found in the v0.1 baseline review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .timebin import DPSReceiver, PulseFormat


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class DetectorParams:
    """Single-photon detector characteristics (one detector).

    efficiency          Detection efficiency, 0..1.
    dark_count_prob     Dark-count probability per gate (gated detectors, e.g.
                        the time-bin model: one gate per pulse period) or per
                        time slot (free-running models such as CoherentDPSLink).
    dead_time_s         Hold-off / dead time after a click.
    afterpulse_prob     Total afterpulse probability per click (not modelled yet).
    gate_width_s        Gate width; None means "model default" (the time-bin
                        model then gates exactly the port windows).
    timing_jitter_s     RMS timing jitter (Gaussian) of the recorded click time.
    """

    efficiency: float
    dark_count_prob: float
    dead_time_s: float = 0.0
    afterpulse_prob: float = 0.0
    gate_width_s: float | None = None
    timing_jitter_s: float = 0.0

    def __post_init__(self) -> None:
        _check_probability("efficiency", self.efficiency)
        _check_probability("dark_count_prob", self.dark_count_prob)
        _check_probability("afterpulse_prob", self.afterpulse_prob)
        if self.dead_time_s < 0 or self.timing_jitter_s < 0:
            raise ValueError("dead_time_s and timing_jitter_s must be non-negative")


@dataclass(frozen=True)
class LinkParams:
    """Everything needed to evaluate one QKD link.

    protocol            "DPS", "COW", ... (models reject protocols they don't support).
    rep_rate_hz         Pulse repetition rate (MAQAN DPS: one 3-bin pulse every
                        32 ns = 31.25 MHz).
    mean_photon_number  Mean photon number per pulse (MAQAN operates at <= 0.2).
                        For time-bin DPS this is per whole pulse (all its bins).
    channel_loss_db     Measured end-to-end channel loss from Alice's output to
                        Bob's input, including splices, connectors and add-drop
                        multiplexers -- not attenuation x length.
    detector            Detector parameters (both detectors assumed identical
                        unless a model says otherwise).
    receiver_loss_db    Loss inside Bob before the detectors (taps, filters, DLI).
    visibility          Interferometer visibility, 0..1.
    length_km           Fibre length; used for delay and dispersion, not loss.
    pulse_format        Time-bin structure of each pulse (time-bin DPS models);
                        see :class:`quila_sim.core.PulseFormat`.
    receiver            Detector arrangement at Bob (time-bin DPS models);
                        see :class:`quila_sim.core.DPSReceiver`.
    """

    protocol: str
    rep_rate_hz: float
    mean_photon_number: float
    channel_loss_db: float
    detector: DetectorParams
    receiver_loss_db: float = 0.0
    visibility: float = 1.0
    length_km: float | None = None
    pulse_format: PulseFormat | None = None
    receiver: DPSReceiver | None = None

    def __post_init__(self) -> None:
        if self.rep_rate_hz <= 0:
            raise ValueError("rep_rate_hz must be positive")
        if self.mean_photon_number <= 0:
            raise ValueError("mean_photon_number must be positive")
        if self.channel_loss_db < 0 or self.receiver_loss_db < 0:
            raise ValueError("losses must be non-negative (in dB)")
        _check_probability("visibility", self.visibility)

    @property
    def transmittance(self) -> float:
        """Channel plus receiver transmittance (excludes detector efficiency)."""
        return 10.0 ** (-(self.channel_loss_db + self.receiver_loss_db) / 10.0)


@dataclass(frozen=True)
class LinkPerformance:
    """Steady-state performance of a link.

    A secure rate is only meaningful together with the security model that
    produced it, so ``security_model`` is required whenever
    ``secure_rate_bps`` is given.
    """

    raw_rate_bps: float
    sifted_rate_bps: float
    qber: float
    secure_rate_bps: float | None = None
    security_model: str | None = None
    finite_key: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_probability("qber", self.qber)
        if self.secure_rate_bps is not None and not self.security_model:
            raise ValueError("secure_rate_bps requires security_model to be named")


@dataclass
class DetectionRecord:
    """Sampled detection events for one block of pulses.

    slots       Slot index of each sifted event (int64), or time tag in
                picoseconds when ``time_tagged`` is True.
    alice_bits  Alice's key bit for each event (uint8).
    bob_bits    Bob's key bit for each event (uint8).
    n_pulses    Number of pulses Alice sent in this block.
    """

    slots: np.ndarray
    alice_bits: np.ndarray
    bob_bits: np.ndarray
    n_pulses: int
    time_tagged: bool = False

    def __post_init__(self) -> None:
        if not (len(self.slots) == len(self.alice_bits) == len(self.bob_bits)):
            raise ValueError("slots, alice_bits and bob_bits must have equal length")

    @property
    def qber(self) -> float:
        n = len(self.alice_bits)
        return float(np.count_nonzero(self.alice_bits != self.bob_bits)) / n if n else 0.0


@runtime_checkable
class LinkModel(Protocol):
    """Contract for every link model.

    name        Short identifier, e.g. "dps-coherent".
    fidelity    "analytic", "statistical" or "event".
    """

    name: str
    fidelity: str

    def performance(self, params: LinkParams) -> LinkPerformance:
        """Expected steady-state performance (fast; used by planning)."""
        ...

    def sample(
        self, params: LinkParams, n_pulses: int, rng: np.random.Generator
    ) -> DetectionRecord:
        """Draw detection events for ``n_pulses`` pulses (feeds post-processing)."""
        ...
