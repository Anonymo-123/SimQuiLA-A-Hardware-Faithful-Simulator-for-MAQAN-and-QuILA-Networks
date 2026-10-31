"""Time-bin DPS-QKD: pulse format, receiver layout and sifting rule (Model C).

MAQAN / IIT Madras DPS-QKD hardware ("time-bin superposition" DPS; Shaw et al.,
"Equivalence of space and time-bins in DPS-QKD", arXiv:2008.03083,
IEEE Photonics J. 14(5), 2022):

* **Alice.** The intensity modulator (IM) carves one 3 ns pulse every 32 ns
  (31.25 MHz). The phase modulator (PM) writes a phase 0 or pi into each 1 ns
  time bin of that pulse, so every pulse is a block of N = 3 time bins. The
  other 29 ns of each period are empty, apart from light that leaks through
  the IM (finite extinction ratio).
* **Bob's interferometer.** The one-bit-delay interferometer (DLI) has a 1 ns
  delay (1 GHz free spectral range). An N-bin pulse leaves it in N+1 time
  slots: an *early* slot (bin 0 through the short arm only), N-1
  *interference* slots (bins j-1 and j overlap, and their phase difference
  picks the output port), and a *late* slot (bin N-1 through the long arm
  only). For equal bin intensities the slot probabilities are
  1/(2N), 1/N, ..., 1/N, 1/(2N) -- for N = 3: 1/6, 1/3, 1/3, 1/6. Only the
  interference slots carry key bits; Alice's bit for slot j is
  phi_j XOR phi_(j-1).
* **Bob's detector.** One gated InGaAs SPD. DLI port 1 is delayed by 10 ns and
  combined with port 0 in a 2x1 coupler (3 dB, plus a 0.5 dB balancing
  attenuator), so each gate contains two windows of N+1 slots: port 0 at
  0-4 ns and port 1 at 10-14 ns after the pulse arrives. The time stamp (TDC)
  tells Bob the window (his bit) and the slot. MAQAN's Fig. 3 likewise shows a
  single time-stamped SPD.

This module holds what both simulation engines share -- the fast engine
(:mod:`quila_sim.physics.dps_timebin`) and the SeQUeNCe per-photon reference
(``sequence/components/dps_hardware.py``): the pulse format, the receiver
layout and timing, the interferometer's single-photon exit probabilities and
the time-stamp sifting rule.

Units: SI in the dataclasses (``_s``, ``_hz``, ``_db``); inside
:class:`FrameLayout` and the helper functions, times are in units of one time
bin, measured from the arrival of bin 0 of the pulse at Bob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .link import LinkParams

__all__ = [
    "EARLY",
    "GUARD",
    "KEY",
    "KIND_NAMES",
    "LATE",
    "LAYOUTS",
    "MAQAN_DPS",
    "OUTSIDE",
    "DPSReceiver",
    "FrameLayout",
    "PulseFormat",
    "classify_times",
    "exit_probabilities",
    "leak_exit_table",
    "maqan_link_params",
    "pulse_amplitudes",
]

#: Click classification codes returned by :func:`classify_times`.
KEY, EARLY, LATE, GUARD, OUTSIDE = 0, 1, 2, 3, 4
KIND_NAMES = ("key", "early", "late", "guard", "outside")

#: Receiver layouts: one time-multiplexed SPD (MAQAN / IIT Madras) or one SPD per port.
LAYOUTS = ("tdm", "dual")

#: Published values for the IIT Madras time-bin DPS hardware used in MAQAN.
#: Sources: Shaw et al., "Equivalence of space and time-bins in DPS-QKD"
#: (arXiv:2008.03083, IEEE Photonics J. 2022) and Sharma et al., "Developing
#: quantum networks: some key innovations" (Quantum Inf. Process. 25:19, 2026).
#: "Calibrated" entries are chosen so that, switched on alone, the imperfection
#: gives the QBER contribution listed in Shaw et al., Table III (1 ns bins).
MAQAN_DPS: dict[str, Any] = {
    "pulse_rate_hz": 31.25e6,       # one 3 ns IM pulse every 32 ns (Shaw et al. Sect. IV-A)
    "bins_per_pulse": 3,            # phase written every 1 ns inside the 3 ns pulse (Fig. 4)
    "bin_width_s": 1e-9,            # 1 ns bins = DLI delay (DLI free spectral range 1 GHz)
    "mean_photon_number": 0.2,      # per 3-bin pulse; Sharma et al. Sect. 2.1: mu <= 0.2
    "extinction_ratio_db": 14.8,    # calibrated: IM error 1.6 % (Table III); paper: > 14 dB
    "pm_transition_s": 42e-12,      # calibrated: PM rise/fall error 2.1 % = 0.5 * 42 ps / 1 ns
    "visibility": 0.92,             # DLI with 1 GHz FSR; error (1 - V) / 2 = 4.0 % (Table III)
    "layout": "tdm",                # one SPD, time-multiplexed DLI ports (Shaw et al. Sect. IV-A)
    "tdm_delay_s": 10e-9,           # 10 ns fibre delay on one DLI output port
    "combiner_loss_db": 3.0,        # 2x1 coupler before the SPD: half of the photons are lost
    "balance_loss_db": 0.5,         # 0.5 dB attenuator matching the delay-fibre loss
    "detector_efficiency": 0.10,    # gated InGaAs SPD, eta ~ 0.1 (Shaw et al.; Sharma et al.)
    "dark_count_prob": 3e-6,        # per gate window (Sharma et al. Table 2)
    "holdoff_s": 10e-6,             # SPD hold-off time (Shaw et al., tau_H = 10 us)
    "timing_jitter_s": 162e-12,     # rms; calibrated: SPD timing-jitter error 5.0 % (Table III)
    "guard_s": 0.0,                 # temporal filter off; Shaw et al. also report 200 ps
}


# --------------------------------------------------------------------------- #
# Pulse format (Alice)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PulseFormat:
    """Shape of one DPS pulse: N phase-encoded time bins.

    The pulse repetition rate is :attr:`LinkParams.rep_rate_hz`; the pulse
    period must be a whole number of bins with at least one empty bin between
    pulses (for a gap-free continuous train use
    :class:`quila_sim.physics.CoherentDPSLink`).

    bins_per_pulse       N, number of 1-phase time bins per pulse (3 for MAQAN).
    bin_width_s          Width of one time bin; also the DLI delay.
    bin_weights          Relative intensity of each bin (IM pulse shape);
                         None means a flat pulse.
    extinction_ratio_db  IM extinction ratio. Light leaking through the IM
                         between pulses has this much less intensity per bin
                         than the pulse. None means a perfect IM.
    pm_transition_s      Width of the window around each bin edge where the
                         phase modulator is switching and the phase is
                         undefined (rise/fall time). 0 means instantaneous.
    """

    bins_per_pulse: int = 3
    bin_width_s: float = 1e-9
    bin_weights: tuple[float, ...] | None = None
    extinction_ratio_db: float | None = None
    pm_transition_s: float = 0.0

    def __post_init__(self) -> None:
        if self.bins_per_pulse < 2:
            raise ValueError("a DPS pulse needs at least 2 time bins")
        if self.bin_width_s <= 0:
            raise ValueError("bin_width_s must be positive")
        if self.bin_weights is not None:
            if len(self.bin_weights) != self.bins_per_pulse:
                raise ValueError("bin_weights needs one entry per time bin")
            if any(w < 0 for w in self.bin_weights) or sum(self.bin_weights) <= 0:
                raise ValueError("bin_weights must be non-negative and not all zero")
        if self.extinction_ratio_db is not None and self.extinction_ratio_db <= 0:
            raise ValueError("extinction_ratio_db must be positive (or None for a perfect IM)")
        if not 0.0 <= self.pm_transition_s < self.bin_width_s:
            raise ValueError("pm_transition_s must be in [0, bin_width_s)")

    @property
    def weights(self) -> np.ndarray:
        """Normalised intensity of each bin (sums to 1)."""
        if self.bin_weights is None:
            return np.full(self.bins_per_pulse, 1.0 / self.bins_per_pulse)
        w = np.asarray(self.bin_weights, dtype=float)
        return w / w.sum()

    @property
    def leak_ratio(self) -> float:
        """Intensity of one empty bin relative to an average pulse bin."""
        if self.extinction_ratio_db is None:
            return 0.0
        return 10.0 ** (-self.extinction_ratio_db / 10.0)

    def frame_bins(self, rep_rate_hz: float) -> int:
        """Pulse period in bins (32 for MAQAN). Validates the format."""
        exact = 1.0 / (rep_rate_hz * self.bin_width_s)
        frame = int(round(exact))
        if abs(exact - frame) > 1e-6 * max(1.0, exact):
            raise ValueError(
                f"the pulse period ({1e9 / rep_rate_hz:g} ns) must be a whole number of "
                f"time bins ({self.bin_width_s * 1e9:g} ns)"
            )
        if frame < self.bins_per_pulse + 1:
            raise ValueError(
                "the time-bin model needs at least one empty bin between pulses; "
                "for a gap-free continuous pulse train use CoherentDPSLink"
            )
        return frame

    @classmethod
    def maqan(cls) -> PulseFormat:
        """MAQAN / IIT Madras format with published imperfections (see MAQAN_DPS)."""
        return cls(
            bins_per_pulse=MAQAN_DPS["bins_per_pulse"],
            bin_width_s=MAQAN_DPS["bin_width_s"],
            extinction_ratio_db=MAQAN_DPS["extinction_ratio_db"],
            pm_transition_s=MAQAN_DPS["pm_transition_s"],
        )


# --------------------------------------------------------------------------- #
# Receiver layout (Bob)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DPSReceiver:
    """How Bob detects the two DLI output ports.

    layout             "tdm": one SPD; port 1 is delayed by ``tdm_delay_s`` and
                       combined with port 0 (MAQAN / IIT Madras).
                       "dual": one SPD per port, no delay or combiner.
    tdm_delay_s        Delay of port 1 (tdm only); a whole number of bins.
    combiner_loss_db   Loss of the 2x1 combiner, applied to both ports (tdm only).
    balance_loss_db    Balancing attenuator / delay-fibre loss, both ports (tdm only).
    guard_s            Temporal filter: clicks within guard_s/2 of an edge of a
                       key slot are discarded (0 = off). Removes jitter and
                       phase-transition errors at the cost of key rate.
    """

    layout: str = "tdm"
    tdm_delay_s: float = 10e-9
    combiner_loss_db: float = 3.0
    balance_loss_db: float = 0.5
    guard_s: float = 0.0

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")
        if self.tdm_delay_s < 0 or self.combiner_loss_db < 0 or self.balance_loss_db < 0:
            raise ValueError("delays and losses must be non-negative")
        if self.guard_s < 0:
            raise ValueError("guard_s must be non-negative")

    @property
    def n_detectors(self) -> int:
        return 1 if self.layout == "tdm" else 2

    @property
    def port_transmittance(self) -> float:
        """Transmittance from a DLI output port to the detector."""
        if self.layout != "tdm":
            return 1.0
        return 10.0 ** (-(self.combiner_loss_db + self.balance_loss_db) / 10.0)

    @classmethod
    def maqan(cls, guard_s: float = 0.0) -> DPSReceiver:
        return cls(
            layout=MAQAN_DPS["layout"],
            tdm_delay_s=MAQAN_DPS["tdm_delay_s"],
            combiner_loss_db=MAQAN_DPS["combiner_loss_db"],
            balance_loss_db=MAQAN_DPS["balance_loss_db"],
            guard_s=guard_s,
        )


def maqan_link_params(
    channel_loss_db: float,
    *,
    receiver_loss_db: float = 0.0,
    length_km: float | None = None,
    **overrides: Any,
) -> LinkParams:
    """LinkParams for one MAQAN DPS link with the published hardware values.

    ``channel_loss_db`` is the measured link loss (e.g. 7.5 dB for
    IITM-ERNET, see ``scenarios/maqan.yaml``). Any key of :data:`MAQAN_DPS`
    can be overridden, e.g. ``maqan_link_params(7.5, guard_s=200e-12)``.
    """
    from .link import DetectorParams, LinkParams

    unknown = set(overrides) - set(MAQAN_DPS)
    if unknown:
        raise TypeError(f"unknown MAQAN_DPS keys: {sorted(unknown)}")
    v = {**MAQAN_DPS, **overrides}
    return LinkParams(
        protocol="DPS",
        rep_rate_hz=v["pulse_rate_hz"],
        mean_photon_number=v["mean_photon_number"],
        channel_loss_db=channel_loss_db,
        detector=DetectorParams(
            efficiency=v["detector_efficiency"],
            dark_count_prob=v["dark_count_prob"],
            dead_time_s=v["holdoff_s"],
            timing_jitter_s=v["timing_jitter_s"],
        ),
        receiver_loss_db=receiver_loss_db,
        visibility=v["visibility"],
        length_km=length_km,
        pulse_format=PulseFormat(
            bins_per_pulse=v["bins_per_pulse"],
            bin_width_s=v["bin_width_s"],
            extinction_ratio_db=v["extinction_ratio_db"],
            pm_transition_s=v["pm_transition_s"],
        ),
        receiver=DPSReceiver(
            layout=v["layout"],
            tdm_delay_s=v["tdm_delay_s"],
            combiner_loss_db=v["combiner_loss_db"],
            balance_loss_db=v["balance_loss_db"],
            guard_s=v["guard_s"],
        ),
    )


# --------------------------------------------------------------------------- #
# Timing of one pulse period at Bob
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FrameLayout:
    """Timing of one pulse period at Bob, in time bins.

    Bin 0 of the pulse reaches the DLI at t = 0. DLI output slot s (0..N) of
    port p reaches detector ``detector_of_port[p]`` at t = s + port_offset[p].
    Each detector is gated once per period, from gate_start for gate_width.
    """

    n_bins: int
    frame_bins: int
    bin_width_s: float
    layout: str
    port_offset: tuple[int, int]
    detector_of_port: tuple[int, int]
    gate_start: float
    gate_width: float
    guard: float

    @property
    def n_detectors(self) -> int:
        return max(self.detector_of_port) + 1

    @classmethod
    def build(
        cls,
        fmt: PulseFormat,
        receiver: DPSReceiver,
        rep_rate_hz: float,
        gate_width_s: float | None = None,
    ) -> FrameLayout:
        """Validate the format/receiver combination and lay out the period.

        With ``gate_width_s=None`` the gate covers exactly the port windows:
        N+1 slots (dual) or from the port-0 early slot to the port-1 late
        slot (tdm: 14 bins for MAQAN).
        """
        n = fmt.bins_per_pulse
        frame = fmt.frame_bins(rep_rate_hz)
        if receiver.layout == "tdm":
            exact = receiver.tdm_delay_s / fmt.bin_width_s
            delay = int(round(exact))
            if abs(exact - delay) > 1e-6:
                raise ValueError("tdm_delay_s must be a whole number of time bins")
            if delay < n + 1:
                raise ValueError(
                    "tdm_delay_s too short: the port-1 window would overlap the port-0 window"
                )
            if delay + n + 1 > frame:
                raise ValueError(
                    "tdm_delay_s too long: the port-1 window must end within the period"
                )
            offsets, detectors, auto_width = (0, delay), (0, 0), delay + n + 1
        else:
            offsets, detectors, auto_width = (0, 0), (0, 1), n + 1
        width = float(auto_width) if gate_width_s is None else gate_width_s / fmt.bin_width_s
        if not 0.0 < width <= frame:
            raise ValueError("the gate width must be positive and not longer than the pulse period")
        guard = receiver.guard_s / fmt.bin_width_s
        if guard >= 1.0:
            raise ValueError("guard_s must be shorter than one time bin")
        return cls(n, frame, fmt.bin_width_s, receiver.layout, offsets, detectors,
                   0.0, width, guard)

    @classmethod
    def from_params(cls, params: LinkParams) -> FrameLayout:
        if params.pulse_format is None or params.receiver is None:
            raise ValueError("time-bin DPS needs params.pulse_format and params.receiver")
        return cls.build(params.pulse_format, params.receiver, params.rep_rate_hz,
                         params.detector.gate_width_s)

    def blind_gates(self, holdoff_s: float, click_time: float) -> int:
        """Gates a detector misses after a click at ``click_time`` (bins within its gate).

        The detector re-arms at the first gate that starts at or after the end
        of the hold-off, and never within the gate in which it clicked.
        """
        holdoff = holdoff_s / self.bin_width_s
        next_gate = max(math.ceil((click_time + holdoff - self.gate_start) / self.frame_bins), 1)
        return next_gate - 1


# --------------------------------------------------------------------------- #
# Interferometer physics
# --------------------------------------------------------------------------- #

def pulse_amplitudes(phase_bits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Real amplitudes of normalised pulses: sqrt(w_j) * (+1 for phase 0, -1 for pi).

    ``phase_bits`` has shape (..., N); returns the same shape.
    """
    bits = np.asarray(phase_bits)
    return np.sqrt(weights) * np.where(bits.astype(bool), -1.0, 1.0)


def exit_probabilities(amplitudes: np.ndarray, visibility: float) -> np.ndarray:
    """Single-photon exit probabilities of a one-bin-delay interferometer.

    A photon whose (normalised) amplitude in input bin j is c_j leaves output
    slot s through port 0 or 1 with probability

        P[s, 0/1] = ( |c_(s-1)|^2 + |c_s|^2  +/-  2 V Re(c_(s-1)* c_s) ) / 4,

    for s = 0..M (c_(-1) = c_M = 0). Port 0 is constructive for equal phases.
    The probabilities sum to 1; summed over ports, slot s gets
    (|c_(s-1)|^2 + |c_s|^2) / 2 -- 1/(2N), 1/N, ..., 1/N, 1/(2N) for a flat pulse.

    ``amplitudes`` has shape (..., M); the result has shape (..., M+1, 2).
    """
    if not 0.0 <= visibility <= 1.0:
        raise ValueError("visibility must be in [0, 1]")
    c = np.asarray(amplitudes, dtype=complex)
    pad = np.zeros(c.shape[:-1] + (1,), dtype=complex)
    long_arm = np.concatenate([pad, c], axis=-1)    # c_(s-1): delayed by one bin
    short_arm = np.concatenate([c, pad], axis=-1)   # c_s
    base = (np.abs(long_arm) ** 2 + np.abs(short_arm) ** 2) / 4.0
    cross = visibility * np.real(np.conj(long_arm) * short_arm) / 2.0
    return np.stack([base + cross, base - cross], axis=-1)


def leak_exit_table(
    fmt: PulseFormat, layout: FrameLayout, visibility: float, mean_photon_number: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Light leaking through the IM between pulses, as seen within one gate.

    The empty bins of each period (N .. F-1) carry intensity
    ``leak_ratio * mu / N`` each, all at the phase modulator's idle phase, so
    neighbouring empty bins interfere constructively at port 0. Leakage is
    treated as a separate (incoherent) background: it does not interfere with
    the pulse itself, which only affects the discarded early/late slots.

    Returns arrays (slot_start, port, mean): every DLI output slot of leaked
    light that overlaps the gate of a period, with its start time (bins,
    relative to that period) and mean photon number per period at the DLI
    output (before port losses and detection). Light from the previous period
    that falls into this gate is included.
    """
    ratio = fmt.leak_ratio
    if ratio == 0.0 or mean_photon_number == 0.0:
        empty = np.empty(0)
        return empty, empty.astype(np.int8), empty
    n, frame = layout.n_bins, layout.frame_bins
    n_empty = frame - n
    total = ratio * mean_photon_number * n_empty / n   # leaked photons per period
    probs = exit_probabilities(np.full(n_empty, 1.0 / math.sqrt(n_empty)), visibility)
    g0, g1 = layout.gate_start, layout.gate_start + layout.gate_width
    starts, ports, means = [], [], []
    for shift in (-frame, 0):                  # previous period, this period
        for r in range(n_empty + 1):           # output slots of the empty region
            for port in (0, 1):
                t0 = shift + n + r + layout.port_offset[port]
                if t0 + 1.0 > g0 and t0 < g1:
                    starts.append(float(t0))
                    ports.append(port)
                    means.append(total * probs[r, port])
    return np.asarray(starts), np.asarray(ports, dtype=np.int8), np.asarray(means)


# --------------------------------------------------------------------------- #
# Time-stamp sifting rule (Bob's TDC)
# --------------------------------------------------------------------------- #

def classify_times(
    t_bins: np.ndarray, detector: np.ndarray, layout: FrameLayout
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify measured click times, as Bob's sifting does from TDC time stamps.

    ``t_bins``: measured click time relative to the arrival of bin 0 of the
    pulse, in bins. ``detector``: which detector clicked.

    Returns (kind, port, slot): kind is KEY, EARLY, LATE, GUARD or OUTSIDE;
    port is the DLI port of the window the click fell in (Bob's bit for KEY
    clicks, -1 if OUTSIDE); slot is the DLI output slot 0..N (-1 if OUTSIDE).
    A KEY click in slot j gives the key bit for the phase difference between
    bins j-1 and j.
    """
    t = np.asarray(t_bins, dtype=float)
    det = np.asarray(detector, dtype=np.int64)
    kind = np.full(t.shape, OUTSIDE, dtype=np.int8)
    port = np.full(t.shape, -1, dtype=np.int8)
    slot = np.full(t.shape, -1, dtype=np.int16)
    n = layout.n_bins
    for p in (0, 1):
        u = t - layout.port_offset[p]
        j = np.floor(u)
        match = (det == layout.detector_of_port[p]) & (j >= 0) & (j <= n) & (kind == OUTSIDE)
        if not match.any():
            continue
        jj = j[match].astype(np.int64)
        k = np.where(jj == 0, EARLY, np.where(jj == n, LATE, KEY)).astype(np.int8)
        if layout.guard > 0:
            frac = u[match] - jj
            half = layout.guard / 2.0
            k = np.where((k == KEY) & ((frac < half) | (frac >= 1.0 - half)), GUARD, k)
        kind[match] = k
        port[match] = p
        slot[match] = jj
    return kind, port, slot
