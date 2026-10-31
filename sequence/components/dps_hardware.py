"""Hardware components for time-bin DPS-QKD -- Model C, the MAQAN / IIT Madras system.

Event-level (per-photon) reference model on the SeQUeNCe discrete-event kernel.
The fast engine is :class:`quila_sim.physics.TimeBinDPSLink`. Both take the
pulse timing, the interferometer physics and the sifting rule from
:mod:`quila_sim.core.timebin`, and the test suite checks that they agree.

What the hardware does
----------------------
Alice -- :class:`PhasedLightSource`
    The intensity modulator (IM) carves one pulse of N time bins (3 x 1 ns)
    out of a CW laser every pulse period (32 ns); the phase modulator (PM)
    writes phase 0 or pi into each bin; an attenuator sets the mean photon
    number per pulse (MAQAN: mu <= 0.2). The rest of the period is empty,
    except for light leaking through the IM (finite extinction ratio). Each
    pulse is emitted as Poisson(mu) photons in the N-bin state of
    :class:`TimeBinPhoton`; the leaked light as Poisson photons spread over the
    empty bins.

Bob -- :class:`DelayLineInterferometer` and :class:`GatedDetector`
    A one-bin-delay interferometer (DLI). A photon of an N-bin pulse leaves it
    in one of N+1 time slots -- early, N-1 interference slots, late -- through
    port 0 or 1, with the probabilities of
    :func:`quila_sim.core.timebin.exit_probabilities` (1/6, 1/3, 1/3, 1/6 per
    slot for N = 3; the port of an interference slot is set by the phase
    difference of the two bins it overlaps, and the visibility). Within the PM
    transition time of a bin edge the phase difference is undefined and the
    port is random. In the MAQAN receiver ("tdm" layout) port 1 is delayed by
    10 ns and both ports are combined onto ONE gated InGaAs detector (3 dB
    coupler plus a 0.5 dB balancing attenuator), so Bob's bit is the time
    window a click falls in. The "dual" layout has one detector per port.
    The detector is gated once per pulse period, fires on the first photon of
    a gate with efficiency eta, adds dark counts per gate, time-stamps the
    click with Gaussian jitter, and is then blind for the hold-off time.

Times are integer picoseconds, as everywhere in SeQUeNCe.

References
----------
Shaw et al., "Equivalence of space and time-bins in DPS-QKD",
arXiv:2008.03083 (IEEE Photonics J. 2022) -- the IIT Madras time-bin DPS
system. Sharma et al., Quantum Inf. Process. 25:19 (2026) -- MAQAN.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline

from quila_sim.core.timebin import (
    DPSReceiver,
    FrameLayout,
    PulseFormat,
    exit_probabilities,
)

from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils import log
from ..utils.dps_encoding import dps_timebin
from .detector import Detector
from .photon import Photon

#: Signature of the optional interception hook on :class:`PhasedLightSource`:
#: ``hook(pulse_index, phase_bits, n_photons) -> replacement_phase_bits or None``.
#: Returning phase bits means Eve intercepted the pulse and re-sent its photons
#: with those phases.
InterceptHook = Callable[[int, Tuple[int, ...], int], Optional[Sequence[int]]]


# ---------------------------------------------------------------------------
# Photon
# ---------------------------------------------------------------------------

#: Amplitude tuples created and validated by the light source (by id), so that
#: the photons of a pulse do not each re-check the same tuple.
_CHECKED: Dict[int, Tuple[complex, ...]] = {}


def _validated(amplitudes: Sequence[complex]) -> Tuple[complex, ...]:
    """Return a normalised amplitude tuple registered as already checked."""
    amps = tuple(complex(a) for a in amplitudes)
    if not amps or abs(sum(abs(a) ** 2 for a in amps) - 1.0) > 1e-9:
        raise ValueError("TimeBinPhoton amplitudes must be normalised")
    if len(_CHECKED) > 4096:
        _CHECKED.clear()
    _CHECKED[id(amps)] = amps
    return amps


class TimeBinPhoton(Photon):
    """One photon of a time-bin DPS pulse.

    A SeQUeNCe photon holds a qubit, but a MAQAN pulse spans N = 3 time bins.
    The photon's N-bin amplitude vector is therefore kept in ``amplitudes``
    and the base-class qubit state is an unused placeholder -- SeQUeNCe's own
    state checks stay switched on.

    Attributes
    ----------
    amplitudes : tuple[complex, ...]
        Normalised amplitude of the photon in each time bin of its pulse
        (for leaked light: in each empty bin of the period).
    pulse_index : int
        Pulse (period) the photon belongs to, counted within its window.
    kind : str
        "signal" (a pulse photon) or "leak" (light leaking through the IM).
    eve_intercepted : bool
        True if Eve intercepted the pulse and this photon is her re-sent copy.
    """

    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        amplitudes: Sequence[complex],
        *,
        wavelength: float = 1550.0,
        location: Any = None,
        pulse_index: int = 0,
        kind: str = "signal",
        eve_intercepted: bool = False,
    ) -> None:
        super().__init__(name, timeline, wavelength=wavelength, location=location,
                         encoding_type=dps_timebin)
        if _CHECKED.get(id(amplitudes)) is amplitudes:      # tuple already validated
            amps = amplitudes
        else:
            amps = tuple(complex(a) for a in amplitudes)
            if not amps or abs(sum(abs(a) ** 2 for a in amps) - 1.0) > 1e-9:
                raise ValueError("TimeBinPhoton amplitudes must be normalised")
        self.amplitudes: Tuple[complex, ...] = amps
        self.pulse_index: int = pulse_index
        self.kind: str = kind
        self.eve_intercepted: bool = eve_intercepted


# ---------------------------------------------------------------------------
# Alice's transmitter
# ---------------------------------------------------------------------------

class PhasedLightSource(Entity):
    """Alice's time-bin DPS transmitter (CW laser, IM, PM, attenuator).

    Parameters
    ----------
    frequency : float
        Pulse repetition rate in Hz (MAQAN: 31.25 MHz, one pulse per 32 ns).
    wavelength : float
        Wavelength in nm.
    mean_photon_num : float
        Mean photon number per pulse (all N bins together).
    bins_per_pulse : int
        Time bins per pulse, N (MAQAN: 3).
    bin_width_ps : int
        Width of one time bin in ps (MAQAN: 1000); also the DLI delay.
    extinction_ratio_db : float or None
        IM extinction ratio; None means no light between pulses.
    phase_error : float
        Probability that the PM writes the wrong phase into a bin (Alice's
        record keeps the intended phase).

    Attributes
    ----------
    photon_counter : int
        Pulse photons emitted since ``init()`` (leaked light not included).
    leak_photon_counter : int
        Photons of leaked light emitted since ``init()``.
    pulse_counter, pulses_intercepted, photons_intercepted : int
        Pulses sent; pulses Eve intercepted and the photons they carried.
    intercept_hook : callable or None
        See :data:`InterceptHook`.
    """

    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        frequency: float = 31.25e6,
        wavelength: float = 1550.0,
        mean_photon_num: float = 0.2,
        bins_per_pulse: int = 3,
        bin_width_ps: int = 1000,
        extinction_ratio_db: Optional[float] = None,
        phase_error: float = 0.0,
    ) -> None:
        Entity.__init__(self, name, timeline)
        self.frequency: float = float(frequency)
        self.wavelength: float = wavelength
        self.mean_photon_num: float = mean_photon_num
        self.bins_per_pulse: int = int(bins_per_pulse)
        self.bin_width_ps: int = int(bin_width_ps)
        self.extinction_ratio_db: Optional[float] = extinction_ratio_db
        self.phase_error: float = phase_error
        self.encoding_type: dict = dps_timebin
        self.intercept_hook: Optional[InterceptHook] = None
        self.photon_counter = self.leak_photon_counter = self.pulse_counter = 0
        self.pulses_intercepted = self.photons_intercepted = 0
        _ = self.frame_bins  # validate the pulse format now

    def init(self) -> None:
        """Reset the counters. Called by ``Timeline.init()``."""
        self.photon_counter = self.leak_photon_counter = self.pulse_counter = 0
        self.pulses_intercepted = self.photons_intercepted = 0

    @property
    def pulse_format(self) -> PulseFormat:
        return PulseFormat(self.bins_per_pulse, self.bin_width_ps * 1e-12,
                           extinction_ratio_db=self.extinction_ratio_db)

    @property
    def frame_bins(self) -> int:
        """Pulse period in time bins (32 for MAQAN); validates the format."""
        return self.pulse_format.frame_bins(self.frequency)

    @property
    def period_ps(self) -> int:
        return self.frame_bins * self.bin_width_ps

    def leak_photons_per_period(self) -> float:
        """Mean photons leaking through the IM per period (all empty bins)."""
        n = self.bins_per_pulse
        return self.pulse_format.leak_ratio * self.mean_photon_num * (self.frame_bins - n) / n

    def emit(self, phases: Any) -> None:
        """Send one window of pulses, starting now.

        Parameters
        ----------
        phases : array-like
            Phase bits (0 -> phase 0, 1 -> phase pi), shape (K, N): one row of
            N bits per pulse. A flat sequence of K*N bits is also accepted.
            Pulse k leaves at ``now + k * period``.
        """
        n = self.bins_per_pulse
        blocks = np.asarray(phases, dtype=np.uint8)
        if blocks.ndim == 1:
            if blocks.size % n:
                raise ValueError(f"a flat phase list must hold a multiple of {n} bits")
            blocks = blocks.reshape(-1, n)
        if blocks.ndim != 2 or blocks.shape[1] != n:
            raise ValueError(f"phases must have shape (pulses, {n})")
        k_total = blocks.shape[0]
        if k_total == 0:
            return
        now = self.timeline.now()
        period = self.period_ps
        rng = self.get_generator()
        receiver = self._receivers[0]

        sent = blocks
        if self.phase_error > 0:
            sent = blocks ^ (rng.random(blocks.shape) < self.phase_error).astype(np.uint8)
        n_photons = rng.poisson(self.mean_photon_num, size=k_total)
        amp = math.sqrt(1.0 / n)
        hook = self.intercept_hook
        states: Dict[Tuple[int, ...], Tuple[complex, ...]] = {}

        log.logger.info(f"{self.name} emitting {k_total} time-bin DPS pulses")
        for k in range(k_total):
            n_k = int(n_photons[k])
            bits: Sequence[int] = sent[k]
            intercepted = False
            if hook is not None:
                replacement = hook(k, tuple(int(b) for b in bits), n_k)
                if replacement is not None:
                    bits, intercepted = replacement, True
                    self.pulses_intercepted += 1
                    self.photons_intercepted += n_k
            if n_k == 0:
                continue
            key = tuple(int(b) for b in bits)
            amplitudes = states.get(key)
            if amplitudes is None:
                amplitudes = states[key] = _validated(-amp if b else amp for b in key)
            when = now + k * period
            for _ in range(n_k):
                photon = TimeBinPhoton(str(k), self.timeline, amplitudes,
                                       wavelength=self.wavelength, location=self.owner,
                                       pulse_index=k, eve_intercepted=intercepted)
                self.timeline.schedule(Event(when, Process(receiver, "get", [photon])))
            self.photon_counter += n_k
        self.pulse_counter += k_total

        # Light leaking through the IM: idle phase over the empty bins.
        lam = self.leak_photons_per_period()
        if lam > 0:
            n_empty = self.frame_bins - n
            leak_amps = _validated((math.sqrt(1.0 / n_empty),) * n_empty)
            n_leak = rng.poisson(lam, size=k_total)
            for k in np.flatnonzero(n_leak).tolist():
                when = now + k * period + n * self.bin_width_ps
                for _ in range(int(n_leak[k])):
                    photon = TimeBinPhoton(f"leak{k}", self.timeline, leak_amps,
                                           wavelength=self.wavelength, location=self.owner,
                                           pulse_index=k, kind="leak")
                    self.timeline.schedule(Event(when, Process(receiver, "get", [photon])))
                self.leak_photon_counter += int(n_leak[k])

    def get(self, photon: "Photon", **kwargs) -> None:  # pragma: no cover
        """Pass a photon to the first downstream receiver (unused in normal flow)."""
        self._receivers[0].get(photon)

    def update_frequency(self, freq: float) -> None:
        """Change the pulse repetition rate (Hz)."""
        self.frequency = float(freq)

    def update_mean_photon_num(self, mu: float) -> None:
        """Change the mean photon number per pulse."""
        self.mean_photon_num = mu


# ---------------------------------------------------------------------------
# Bob's detector
# ---------------------------------------------------------------------------

class GatedDetector(Detector):
    """Gated single-photon detector (InGaAs SPD in gated mode).

    One gate per pulse period. A photon arriving inside a gate is detected
    with probability ``efficiency``; the detector fires on the first detected
    photon (or dark count) of a gate while it is armed, records the time with
    Gaussian jitter (rounded to ``time_resolution``), and is then blind until
    the first gate that starts at least ``holdoff_ps`` later (never again in
    the same gate). Dark counts occur with probability ``dark_count_prob`` per
    gate, at a uniformly random time inside the gate. Light outside the gates
    is not seen.

    Observers receive ``trigger(detector, {"time", "gate", "true_time"})``.
    """

    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        efficiency: float = 0.1,
        dark_count_prob: float = 0.0,
        holdoff_ps: int = 10_000_000,
        timing_jitter_ps: float = 0.0,
        time_resolution: int = 1,
    ) -> None:
        super().__init__(name, timeline, efficiency=efficiency, dark_count=0.0,
                         count_rate=1e15, time_resolution=time_resolution)
        if not 0.0 <= dark_count_prob <= 1.0:
            raise ValueError("dark_count_prob must be in [0, 1]")
        self.dark_count_prob: float = dark_count_prob
        self.holdoff_ps: int = int(holdoff_ps)
        self.timing_jitter_ps: float = float(timing_jitter_ps)
        self._gate0: Optional[int] = None
        self._period = 1
        self._width = 1
        self._n_gates = 0
        self._window_id = 0
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.photon_counter = 0
        self.ready_time = -1          # absolute time (ps) from which the detector is armed
        self.clicks = 0
        self.holdoff_blocked = 0      # detectable events while blind
        self.outside_gate = 0         # detectable photons arriving outside every gate
        self.dark_events = 0          # dark counts generated (fired or blocked)

    def init(self) -> None:
        """Reset the state. Called by ``Timeline.init()``."""
        self.next_detection_time = -1
        self._reset_counters()

    def set_window(self, first_gate_ps: int, period_ps: int, width_ps: int, n_gates: int) -> None:
        """Gate the detector for one window: gates k = 0..n_gates-1 start at
        ``first_gate_ps + k * period_ps`` and last ``width_ps``. Hold-off state
        carries over from the previous window (time is absolute)."""
        self._gate0 = int(first_gate_ps)
        self._period = int(period_ps)
        self._width = int(width_ps)
        self._n_gates = int(n_gates)
        self._window_id += 1
        if self.dark_count_prob > 0 and self._n_gates > 0:
            self._schedule_dark(-1, self._window_id)

    def gate_of(self, time_ps: int) -> Optional[int]:
        """Index of the gate (in the current window) open at ``time_ps``, or None."""
        if self._gate0 is None:
            return None
        rel = int(time_ps) - self._gate0
        if rel < 0:
            return None
        k = rel // self._period
        if k >= self._n_gates or rel - k * self._period >= self._width:
            return None
        return int(k)

    def get(self, photon: Optional["Photon"] = None, **kwargs) -> None:
        """A photon reaches the detector now."""
        self.photon_counter += 1
        if self.get_generator().random() >= self.efficiency:
            return
        gate = self.gate_of(self.timeline.now())
        if gate is None:
            self.outside_gate += 1
            return
        self._fire(gate)

    def record_detection(self) -> None:
        """SeQUeNCe API: an avalanche now (fires only inside a gate)."""
        gate = self.gate_of(self.timeline.now())
        if gate is None:
            self.outside_gate += 1
            return
        self._fire(gate)

    def _fire(self, gate: int) -> None:
        now = self.timeline.now()
        if now < self.ready_time:
            self.holdoff_blocked += 1
            return
        recorded = float(now)
        if self.timing_jitter_ps > 0:
            recorded += self.get_generator().normal(0.0, self.timing_jitter_ps)
        res = max(1, int(self.time_resolution))
        recorded_ps = int(round(recorded / res)) * res
        self.clicks += 1
        nxt = max(math.ceil((now + self.holdoff_ps - self._gate0) / self._period), gate + 1)
        self.ready_time = self._gate0 + nxt * self._period
        self.notify({"time": recorded_ps, "gate": gate, "true_time": now})

    def _schedule_dark(self, after_gate: int, window_id: int) -> None:
        rng = self.get_generator()
        gate = after_gate + int(rng.geometric(self.dark_count_prob))
        if gate >= self._n_gates:
            return
        when = self._gate0 + gate * self._period + int(rng.random() * self._width)
        self.timeline.schedule(Event(when, Process(self, "_dark_count", [gate, window_id])))

    def _dark_count(self, gate: int, window_id: int) -> None:
        if window_id != self._window_id:        # left over from an earlier window
            return
        self.dark_events += 1
        self._fire(gate)
        self._schedule_dark(gate, window_id)


# ---------------------------------------------------------------------------
# Bob's interferometer and detection optics
# ---------------------------------------------------------------------------

class ClickRecord(NamedTuple):
    """One detector click as logged by :class:`DelayLineInterferometer`."""

    time_ps: int          # time stamp Bob records (with jitter)
    detector: int         # detector index (0 for the single TDM detector)
    gate: int             # gate index within the window
    source: str           # "signal", "leak" or "dark"
    eve: bool             # caused by a photon Eve re-sent
    pulse: int            # pulse the photon came from (-1 for dark counts)
    port: int             # DLI port the photon left through (-1 for dark counts)
    slot: int             # DLI output slot of the photon (-1 for dark counts)
    true_time_ps: int     # avalanche time without jitter


class DelayLineInterferometer(Entity):
    """Bob's one-bin-delay interferometer, port optics and gated detector(s).

    Parameters
    ----------
    frequency : float
        Pulse repetition rate in Hz.
    bins_per_pulse, bin_width_ps : int
        Pulse format (must match Alice's); the DLI delay is one bin.
    visibility : float
        Interference visibility V.
    pm_transition_ps : float
        Width of the window around each bin edge in which Alice's phase
        modulator is switching; a photon there leaves through a random port.
    layout : str
        "tdm": one detector, port 1 delayed by ``tdm_delay_ps`` and combined
        with port 0 (MAQAN). "dual": one detector per port.
    tdm_delay_ps, combiner_loss_db, balance_loss_db
        TDM optics (``tdm`` only): delay of port 1, 2x1 coupler loss and the
        balancing attenuator, both applied to both ports.
    efficiency, dark_count_prob, holdoff_ps, timing_jitter_ps, time_resolution
        Detector parameters (see :class:`GatedDetector`); dark counts per gate.
    gate_width_ps : int or None
        Gate width; None gates exactly the port windows (14 ns for MAQAN).

    Attributes
    ----------
    layout_info : FrameLayout
        Timing of one period at Bob (shared with the fast engine).
    clicks : list[ClickRecord]
        Clicks since the last :meth:`collect`.
    photons_arrived, leak_arrived, port_lost : int
        Pulse photons and leaked photons that reached the DLI; photons lost
        in the port optics (combiner and attenuator).
    """

    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        frequency: float = 31.25e6,
        bins_per_pulse: int = 3,
        bin_width_ps: int = 1000,
        visibility: float = 1.0,
        pm_transition_ps: float = 0.0,
        layout: str = "tdm",
        tdm_delay_ps: int = 10_000,
        combiner_loss_db: float = 3.0,
        balance_loss_db: float = 0.5,
        efficiency: float = 0.1,
        dark_count_prob: float = 0.0,
        holdoff_ps: int = 10_000_000,
        timing_jitter_ps: float = 0.0,
        gate_width_ps: Optional[int] = None,
        time_resolution: int = 1,
    ) -> None:
        Entity.__init__(self, name, timeline)
        if not 0.0 <= visibility <= 1.0:
            raise ValueError("visibility must be in [0, 1]")
        self.frequency: float = float(frequency)
        self.visibility: float = visibility
        self.pulse_format = PulseFormat(int(bins_per_pulse), int(bin_width_ps) * 1e-12,
                                        pm_transition_s=pm_transition_ps * 1e-12)
        self.receiver = DPSReceiver(layout=layout, tdm_delay_s=tdm_delay_ps * 1e-12,
                                    combiner_loss_db=combiner_loss_db,
                                    balance_loss_db=balance_loss_db)
        self.layout_info: FrameLayout = FrameLayout.build(
            self.pulse_format, self.receiver, self.frequency,
            None if gate_width_ps is None else gate_width_ps * 1e-12)
        self.bin_width_ps: int = int(bin_width_ps)
        self.period_ps: int = self.layout_info.frame_bins * self.bin_width_ps
        self.pm_transition_bins: float = pm_transition_ps / self.bin_width_ps
        self.port_transmittance: float = self.receiver.port_transmittance
        self.detectors: List[GatedDetector] = [
            GatedDetector(f"{name}.D{i}", timeline, efficiency=efficiency,
                          dark_count_prob=dark_count_prob, holdoff_ps=holdoff_ps,
                          timing_jitter_ps=timing_jitter_ps, time_resolution=time_resolution)
            for i in range(self.layout_info.n_detectors)
        ]
        self._exit_cache: Dict[Tuple[complex, ...], np.ndarray] = {}
        self._pending: Optional[Tuple[TimeBinPhoton, int, int]] = None
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.clicks: List[ClickRecord] = []
        self.photons_arrived = self.leak_arrived = self.port_lost = 0
        self.window_start_ps: Optional[int] = None

    def init(self) -> None:
        """Observe the detectors and reset the state. Called by ``Timeline.init()``."""
        for det in self.detectors:
            det.owner = self.owner
            if self not in det._observers:
                det.attach(self)
        self._pending = None
        self._reset_counters()

    # -- timing ---------------------------------------------------------------
    def arm(self, window_start_ps: int, num_pulses: int) -> None:
        """Gate the detectors for a window of ``num_pulses`` pulses whose first
        pulse (bin 0) reaches the DLI at ``window_start_ps``."""
        lay = self.layout_info
        self.window_start_ps = int(window_start_ps)
        first_gate = self.window_start_ps + int(round(lay.gate_start * self.bin_width_ps))
        width = int(round(lay.gate_width * self.bin_width_ps))
        for det in self.detectors:
            det.set_window(first_gate, self.period_ps, width, num_pulses)

    # -- optics ---------------------------------------------------------------
    def _exit_cdf(self, amplitudes: Tuple[complex, ...]) -> np.ndarray:
        cdf = self._exit_cache.get(amplitudes)
        if cdf is None:
            probs = exit_probabilities(np.asarray(amplitudes), self.visibility).reshape(-1)
            cdf = np.cumsum(probs)
            cdf /= cdf[-1]
            self._exit_cache[amplitudes] = cdf
        return cdf

    def get(self, photon: "Photon", **kwargs) -> None:
        """A photon from the quantum channel enters the DLI (bin 0 arrives now)."""
        if not isinstance(photon, TimeBinPhoton):
            log.logger.warning(f"{self.name}: received a non-TimeBinPhoton photon - ignored")
            return
        rng = self.get_generator()
        cdf = self._exit_cdf(photon.amplitudes)
        index = min(int(np.searchsorted(cdf, rng.random(), side="right")), cdf.size - 1)
        slot, port = divmod(index, 2)
        tau = rng.random()
        n_bins = len(photon.amplitudes)
        if photon.kind == "signal":
            self.photons_arrived += 1
            half = self.pm_transition_bins / 2.0
            if 1 <= slot <= n_bins - 1 and half > 0 and (tau < half or tau >= 1.0 - half):
                port = int(rng.integers(0, 2))
        else:
            self.leak_arrived += 1
        if self.port_transmittance < 1.0 and rng.random() >= self.port_transmittance:
            self.port_lost += 1
            return
        lay = self.layout_info
        detector = lay.detector_of_port[port]
        # floor, not round: the integer-ps arrival time must stay inside its slot
        arrival = self.timeline.now() + math.floor((slot + tau + lay.port_offset[port])
                                                   * self.bin_width_ps)
        self.timeline.schedule(Event(arrival, Process(self, "_deliver",
                                                      [photon, detector, port, slot])))

    def _deliver(self, photon: TimeBinPhoton, detector: int, port: int, slot: int) -> None:
        self._pending = (photon, port, slot)
        try:
            self.detectors[detector].get(photon)
        finally:
            self._pending = None

    # -- detection ------------------------------------------------------------
    def trigger(self, detector: "GatedDetector", info: Dict[str, Any]) -> None:
        """Called by a detector when it fires (photon or dark count)."""
        index = self.detectors.index(detector)
        if self._pending is None:
            record = ClickRecord(int(info["time"]), index, int(info["gate"]), "dark", False,
                                 -1, -1, -1, int(info["true_time"]))
        else:
            photon, port, slot = self._pending
            record = ClickRecord(int(info["time"]), index, int(info["gate"]), photon.kind,
                                 bool(photon.eve_intercepted), int(photon.pulse_index),
                                 port, slot, int(info["true_time"]))
        self.clicks.append(record)

    def collect(self) -> List[ClickRecord]:
        """Return and clear the clicks recorded since the last call."""
        clicks, self.clicks = self.clicks, []
        return clicks

    @property
    def outside_gate(self) -> int:
        return sum(d.outside_gate for d in self.detectors)

    @property
    def holdoff_blocked(self) -> int:
        return sum(d.holdoff_blocked for d in self.detectors)

    @property
    def click_count(self) -> int:
        return sum(d.clicks for d in self.detectors)

    def update_detector_params(self, detector_id: int, arg_name: str, value: Any) -> None:
        """Set an attribute on one detector (0 for the single TDM detector)."""
        if not 0 <= detector_id < len(self.detectors):
            raise ValueError(f"detector_id must be in 0..{len(self.detectors) - 1}")
        setattr(self.detectors[detector_id], arg_name, value)
