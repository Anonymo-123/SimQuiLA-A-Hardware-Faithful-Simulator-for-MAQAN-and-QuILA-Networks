"""Adapter between the simulation engines and the desktop GUI.

The simulator models the MAQAN / IIT Madras DPS-QKD hardware -- "Model C",
time-bin DPS: one weak coherent pulse of 3 time bins (3 x 1 ns, a phase 0 or
pi in each bin) every 32 ns, Bob's one-bin-delay interferometer, and ONE gated
InGaAs detector shared by both interferometer ports through a 10 ns delay
(time multiplexing). The defaults of :class:`SimulationParameters` are the
published MAQAN values (``quila_sim.core.MAQAN_DPS``) on the IITM-ERNET link.

Engines (``SimulationParameters.engine``), all returning :class:`SimulationResult`:

``"timebin"`` (default)
    :class:`quila_sim.physics.TimeBinDPSLink` -- vectorised Model C; tens of
    millions of pulses in a few seconds. Use this for anything quantitative.

``"event"`` (reference)
    SeQUeNCe's discrete-event kernel with the per-photon Model C hardware in
    ``sequence/components/dps_hardware.py`` and the time-bin sifting protocol
    in ``sequence/qkd/DPSQKD.py``. Slower, limited to
    :data:`EVENT_ENGINE_MAX_PULSES` pulses; it resolves individual photons and
    detector events in time. The tests check that it agrees with "timebin".

``"continuous"`` (comparison only)
    :class:`quila_sim.physics.CoherentDPSLink` -- "Model B", the textbook DPS
    with a gap-free pulse train. This is NOT how MAQAN works. It is mapped to
    the same hardware run as a continuous train: one 1-bin pulse per time bin
    (rate 1/bin width) with the same light per bin (mu / N per pulse), two
    free-running detectors, dark counts per bin from the per-gate value.

Link loss
---------
``channel_loss_db`` is the *measured* end-to-end loss of the link (OTDR or
insertion-loss measurement, or a preset such as MAQAN). When it is ``None``
the loss is ``distance_km * fiber_loss_db_km``. ``distance_km`` is always used
for the propagation delay. ``receiver_loss_db`` is loss inside Bob before the
interferometer outputs (filters, DLI insertion loss); the TDM coupler and
balancing attenuator are separate (``combiner_loss_db``, ``balance_loss_db``).

Eavesdropper
------------
A simple teaching model: each *pulse* is intercepted with probability
``eve_intercept_probability`` and re-sent with random phases, so a fully
intercepted link shows a QBER of about 50 %. Realistic DPS attacks are
roadmap item 13.

Security verdict
----------------
"SECURE" means the QBER is below ``qber_threshold``. The default, 20 %, is the
critical QBER that Sharma et al. (2026) find for 3-pulse DPS against optimal
individual attacks; information-theoretic bounds are lower (4-6 %), and
11 % is the usual BB84 demonstration value (see :data:`SECURITY_THRESHOLDS`).
This is a check, not a security proof; key-rate models are roadmap item 8.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from quila_sim.core import MAQAN_DPS, DetectorParams, LinkParams
from quila_sim.core.timebin import KIND_NAMES, DPSReceiver, FrameLayout, PulseFormat
from quila_sim.physics import SOURCE_NAMES, CoherentDPSLink, TimeBinDPSLink
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.qkd.DPSQKD import alice_bits_for, classify_clicks
from sequence.topology.dps_topo import DPSQKDTopo

ENGINES = ("timebin", "event", "continuous")

#: Names used by earlier versions, mapped to the current engines.
LEGACY_ENGINES = {"coherent": "timebin"}

#: Human-readable model names (shown in the GUI and in exports).
MODEL_NAMES = {
    "timebin": "Model C – time-bin DPS (MAQAN hardware), fast engine",
    "event": "Model C – time-bin DPS (MAQAN hardware), SeQUeNCe per photon",
    "continuous": "Model B – continuous pulse train (comparison only)",
}

#: Pulse limit for the per-photon event engine (keeps the GUI responsive).
EVENT_ENGINE_MAX_PULSES = 2_000_000

_FIBRE_GROUP_VELOCITY_M_S = 2.0e8

#: Security-check thresholds offered in the GUI: (critical QBER, description).
SECURITY_THRESHOLDS = (
    (0.20, "20% \u2013 3-pulse DPS, individual attacks (MAQAN analysis)"),
    (0.11, "11% \u2013 BB84 demonstration value"),
    (0.05, "5% \u2013 information-theoretic DPS bounds (4\u20136%)"),
)


# --------------------------------------------------------------------------- #
# Data records
# --------------------------------------------------------------------------- #

@dataclass
class PulseEvent:
    """One row of the event log.

    Model C engines record one row per pulse that produced a detector click
    (the first ``max_trace_events`` such pulses); the continuous-train
    comparison engine records one row per time slot.
    """

    index: int                          # pulse number (Model B: slot number)
    phases: Tuple[int, ...] = ()        # Alice's phase bits (1 = pi) of the pulse's bins
    detector: Optional[int] = None      # Bob's port window (= his bit) of the click; None if none
    slot: Optional[int] = None          # interferometer output slot, 0 = early ... N = late
    kind: str = ""                      # "key", "early", "late", "guard", "outside"; "" = no click
    source: str = ""                    # "signal", "leak" (IM leakage) or "dark"
    time_ns: float = 0.0                # click time Bob recorded, from the start of the period
    alice_bit: Optional[int] = None     # Alice's bit for a key click
    bob_bit: Optional[int] = None       # Bob's bit for a key click
    n_key: int = 0                      # sifted bits this pulse produced
    eve_intercepted: bool = False       # Eve intercepted this pulse
    dark_count: bool = False            # the click had no photon behind it
    bloch_x: float = 0.0
    bloch_y: float = 0.0
    bloch_z: float = 0.0                # display only: +-V for a key click, 0 otherwise

    @property
    def clicked(self) -> bool:
        return bool(self.kind)

    @property
    def phase(self) -> int:
        """Bit shown for the pulse: Alice's key bit if there is one, else her first phase."""
        if self.alice_bit is not None:
            return int(self.alice_bit)
        return int(self.phases[0]) if self.phases else 0

    @property
    def phase_text(self) -> str:
        return " ".join("π" if b else "0" for b in self.phases)


@dataclass
class SimulationParameters:
    """User-configurable simulation inputs (defaults: MAQAN, IITM-ERNET link).

    Times are in seconds, rates in Hz, losses in dB, as everywhere in
    ``quila_sim``; the GUI converts for display.
    """

    num_pulses: int = 10_000_000
    engine: str = "timebin"
    # -- link ---------------------------------------------------------------
    distance_km: float = 4.8
    fiber_loss_db_km: float = 0.20
    channel_loss_db: Optional[float] = 7.5     # measured; None -> distance x dB/km
    receiver_loss_db: float = 0.0              # inside Bob, before the DLI outputs
    link_name: str = "IITM-ERNET"
    # -- Alice: pulse format and transmitter --------------------------------
    pulse_rate_hz: float = MAQAN_DPS["pulse_rate_hz"]
    bins_per_pulse: int = MAQAN_DPS["bins_per_pulse"]
    bin_width_s: float = MAQAN_DPS["bin_width_s"]
    mean_photon_number: float = MAQAN_DPS["mean_photon_number"]   # per pulse
    extinction_ratio_db: Optional[float] = MAQAN_DPS["extinction_ratio_db"]  # None = perfect IM
    pm_transition_s: float = MAQAN_DPS["pm_transition_s"]
    # -- Bob: interferometer, receiver layout, detector ---------------------
    interferometer_visibility: float = MAQAN_DPS["visibility"]
    receiver_layout: str = MAQAN_DPS["layout"]                     # "tdm" or "dual"
    tdm_delay_s: float = MAQAN_DPS["tdm_delay_s"]
    combiner_loss_db: float = MAQAN_DPS["combiner_loss_db"]
    balance_loss_db: float = MAQAN_DPS["balance_loss_db"]
    detector_efficiency: float = MAQAN_DPS["detector_efficiency"]
    dark_count_prob: float = MAQAN_DPS["dark_count_prob"]           # per gate
    holdoff_s: float = MAQAN_DPS["holdoff_s"]
    timing_jitter_s: float = MAQAN_DPS["timing_jitter_s"]
    gate_width_s: Optional[float] = None       # None -> gate exactly the port windows
    guard_s: float = MAQAN_DPS["guard_s"]       # temporal filter at the key-slot edges
    # -- Eve and the security check -----------------------------------------
    eve_enabled: bool = False
    eve_intercept_probability: float = 0.5
    qber_threshold: float = 0.20               # "SECURE" if QBER below this
    # -- output -------------------------------------------------------------
    max_trace_events: int = 2000               # rows recorded for the event log / replay
    # None => a fresh random seed is drawn for every run. Fix them only for
    # reproducible runs (e.g. tests).
    alice_seed: Optional[int] = None
    bob_seed: Optional[int] = None
    eve_seed: Optional[int] = None
    visibility_seed: Optional[int] = None

    # -- hardware profiles --------------------------------------------------
    @classmethod
    def maqan(cls, **overrides: Any) -> SimulationParameters:
        """Published MAQAN hardware on the IITM-ERNET link (the defaults)."""
        return cls(**overrides)

    @classmethod
    def ideal_hardware(cls, **overrides: Any) -> SimulationParameters:
        """MAQAN pulse format and receiver layout with perfect components:
        V = 1, no jitter, instantaneous phase modulator, perfect IM, no dark
        counts. Useful to see the bare 1/6 : 1/3 : 1/3 : 1/6 structure."""
        base = dict(interferometer_visibility=1.0, timing_jitter_s=0.0, pm_transition_s=0.0,
                    extinction_ratio_db=None, dark_count_prob=0.0)
        base.update(overrides)
        return cls(**base)


#: Keys of earlier configuration files and what they map to.
LEGACY_KEYS = {"frequency_hz": "pulse_rate_hz", "dead_time_s": "holdoff_s"}


def params_from_dict(data: Dict[str, Any]) -> Tuple[SimulationParameters, List[str]]:
    """Build parameters from a saved configuration (JSON dict).

    Accepts files written by earlier versions: ``frequency_hz`` becomes the
    pulse rate, ``dead_time_s`` the hold-off, ``dark_count_hz`` is converted
    to a probability per gate, and engine "coherent" becomes "timebin".
    Returns the parameters and a list of notes about what was converted or
    ignored.
    """
    known = {f.name for f in fields(SimulationParameters)}
    kwargs: Dict[str, Any] = {}
    notes: List[str] = []
    dark_hz = None
    for key, value in data.items():
        if key in known:
            kwargs[key] = value
        elif key in LEGACY_KEYS:
            kwargs.setdefault(LEGACY_KEYS[key], value)
            notes.append(f"'{key}' read as '{LEGACY_KEYS[key]}'")
        elif key == "dark_count_hz":
            dark_hz = float(value)
        else:
            notes.append(f"'{key}' ignored (not a parameter of this version)")
    engine = kwargs.get("engine")
    if engine in LEGACY_ENGINES:
        kwargs["engine"] = LEGACY_ENGINES[engine]
        notes.append(f"engine '{engine}' replaced by '{kwargs['engine']}'")
    params = SimulationParameters(**kwargs)
    if dark_hz is not None and "dark_count_prob" not in data:
        gate_s = frame_layout(params).gate_width * params.bin_width_s
        params.dark_count_prob = -math.expm1(-dark_hz * gate_s)
        notes.append(f"dark_count_hz = {dark_hz:g} converted to {params.dark_count_prob:.3g} "
                     f"per {gate_s * 1e9:g} ns gate")
    return params, notes


@dataclass
class SimulationResult:
    """Everything the GUI needs."""

    events: List[PulseEvent] = field(default_factory=list)
    pulses_sent: int = 0
    photons_emitted: int = 0          # pulse photons Alice sent (IM leakage not included)
    d0_clicks: int = 0                # sifted bits Bob recorded as 0 (port-0 window)
    d1_clicks: int = 0                # sifted bits Bob recorded as 1 (port-1 window)
    dark_counts: int = 0              # detector clicks with no photon behind them
    channel_loss_count: int = 0       # pulses that produced no click at all
    eve_intercepted_count: int = 0    # sifted bits caused by photons Eve re-sent

    photons_intercepted: int = 0      # photons in pulses Eve intercepted
    pulses_intercepted: int = 0
    photons_available_to_eve: int = 0
    photons_remaining: int = 0
    photons_detected: int = 0         # detector clicks caused by pulse photons
    photons_lost: int = 0
    eve_actual_interception_rate: float = 0.0

    alice_key_bits: List[int] = field(default_factory=list)
    bob_key_bits: List[int] = field(default_factory=list)
    sifted_key_bits: int = 0
    bit_errors: int = 0
    qber: float = 0.0
    throughput_bps: float = 0.0       # sifted bits per second of simulated time
    latency_s: float = 0.0            # fibre propagation + classical announcement delay
    security_secure: bool = False
    security_message: str = "pending"

    raw_detector_clicks: int = 0      # every detector click, sifted or not
    discarded_click_count: int = 0    # raw_detector_clicks - sifted bits
    early_late_discards: int = 0      # clicks in the early or late slot (no phase difference)
    double_clicks: int = 0            # key slots where both detectors fired (dual layout)

    # -- Model C detail --------------------------------------------------------
    key_slot_clicks: int = 0          # clicks in an interference slot
    early_clicks: int = 0
    late_clicks: int = 0
    guard_discards: int = 0           # clicks removed by the temporal filter
    outside_window_clicks: int = 0    # clicks between/after the port windows
    leak_clicks: int = 0              # clicks caused by light leaking through the IM
    holdoff_blocked: int = 0          # detectable events lost while the detector was blind
    pulses_with_click: int = 0
    arrival_hist: List[int] = field(default_factory=list)       # Bob's time-stamp histogram
    hist_edges_ns: List[float] = field(default_factory=list)    # its bin edges, ns in the period
    port_windows_ns: List[Tuple[float, float]] = field(default_factory=list)
    expected_click_rate_hz: float = 0.0     # closed form (same model)
    expected_sifted_rate_bps: float = 0.0
    expected_qber: float = 0.0

    engine: str = ""
    model: str = ""
    channel_loss_db_used: float = 0.0
    pulse_rate_hz: float = 0.0
    bins_per_pulse: int = 0
    receiver_layout: str = ""
    duration_s: float = 0.0           # simulated time covered by the pulses
    trace_truncated: bool = False     # events cover only part of the run


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def total_channel_loss_db(params: SimulationParameters) -> float:
    """Channel loss the engines use: the measurement if given, else distance x dB/km."""
    if params.channel_loss_db is not None:
        return float(params.channel_loss_db)
    return params.distance_km * params.fiber_loss_db_km


def pulse_format(params: SimulationParameters) -> PulseFormat:
    er = params.extinction_ratio_db
    return PulseFormat(
        bins_per_pulse=int(params.bins_per_pulse),
        bin_width_s=params.bin_width_s,
        extinction_ratio_db=None if er is None or er <= 0 else float(er),
        pm_transition_s=params.pm_transition_s,
    )


def receiver(params: SimulationParameters) -> DPSReceiver:
    return DPSReceiver(
        layout=params.receiver_layout,
        tdm_delay_s=params.tdm_delay_s,
        combiner_loss_db=params.combiner_loss_db,
        balance_loss_db=params.balance_loss_db,
        guard_s=params.guard_s,
    )


def to_link_params(params: SimulationParameters) -> LinkParams:
    """Translate GUI parameters into the shared :class:`LinkParams` contract (Model C)."""
    return LinkParams(
        protocol="DPS",
        rep_rate_hz=params.pulse_rate_hz,
        mean_photon_number=params.mean_photon_number,
        channel_loss_db=total_channel_loss_db(params),
        receiver_loss_db=params.receiver_loss_db,
        visibility=min(1.0, max(0.0, params.interferometer_visibility)),
        length_km=params.distance_km,
        detector=DetectorParams(
            efficiency=params.detector_efficiency,
            dark_count_prob=params.dark_count_prob,
            dead_time_s=params.holdoff_s,
            gate_width_s=params.gate_width_s,
            timing_jitter_s=params.timing_jitter_s,
        ),
        pulse_format=pulse_format(params),
        receiver=receiver(params),
    )


def frame_layout(params: SimulationParameters) -> FrameLayout:
    """Timing of one pulse period at Bob (validates the pulse format and receiver)."""
    return FrameLayout.build(pulse_format(params), receiver(params), params.pulse_rate_hz,
                             params.gate_width_s)


def to_continuous_link_params(params: SimulationParameters) -> LinkParams:
    """Model B comparison: the same hardware run as a gap-free 1-bin pulse train."""
    lay = frame_layout(params)
    n = int(params.bins_per_pulse)
    dark_per_bin = -math.expm1(math.log1p(-params.dark_count_prob) / lay.gate_width) \
        if params.dark_count_prob < 1 else 1.0
    return LinkParams(
        protocol="DPS",
        rep_rate_hz=1.0 / params.bin_width_s,
        mean_photon_number=params.mean_photon_number / n,
        channel_loss_db=total_channel_loss_db(params),
        receiver_loss_db=params.receiver_loss_db,
        visibility=min(1.0, max(0.0, params.interferometer_visibility)),
        length_km=params.distance_km,
        detector=DetectorParams(
            efficiency=params.detector_efficiency,
            dark_count_prob=dark_per_bin,
            dead_time_s=params.holdoff_s,
        ),
    )


def expected_performance(params: SimulationParameters):
    """Closed-form click rate, sifted rate and QBER of the chosen model."""
    p_eve = params.eve_intercept_probability if params.eve_enabled else 0.0
    if params.engine == "continuous":
        return CoherentDPSLink().performance(to_continuous_link_params(params))
    return TimeBinDPSLink().performance(to_link_params(params), eve_intercept_probability=p_eve)


def _fill_seeds(params: SimulationParameters) -> None:
    for name in ("alice_seed", "bob_seed", "eve_seed", "visibility_seed"):
        if getattr(params, name) is None:
            setattr(params, name, int(np.random.SeedSequence().generate_state(1)[0]))


def _finish(result: SimulationResult, params: SimulationParameters, duration_s: float) -> None:
    """Fill the fields that are computed the same way for every engine."""
    n = min(len(result.alice_key_bits), len(result.bob_key_bits))
    a = np.asarray(result.alice_key_bits[:n], dtype=np.uint8)
    b = np.asarray(result.bob_key_bits[:n], dtype=np.uint8)
    result.sifted_key_bits = n
    result.bit_errors = int(np.count_nonzero(a != b))
    result.qber = result.bit_errors / n if n else 0.0
    result.d0_clicks = int(n - np.count_nonzero(b))
    result.d1_clicks = int(np.count_nonzero(b))
    result.discarded_click_count = max(0, result.raw_detector_clicks - n)
    result.photons_remaining = result.photons_emitted - result.photons_intercepted
    result.photons_lost = max(0, result.photons_emitted - result.photons_detected)
    result.eve_actual_interception_rate = (
        result.photons_intercepted / result.photons_available_to_eve
        if result.photons_available_to_eve else 0.0
    )
    result.duration_s = duration_s
    result.throughput_bps = n / duration_s if duration_s > 0 else 0.0
    result.latency_s = 2.0 * params.distance_km * 1e3 / _FIBRE_GROUP_VELOCITY_M_S
    result.channel_loss_db_used = total_channel_loss_db(params)
    result.pulse_rate_hz = params.pulse_rate_hz
    result.bins_per_pulse = int(params.bins_per_pulse)
    result.receiver_layout = params.receiver_layout
    result.engine = params.engine
    result.model = MODEL_NAMES[params.engine]
    threshold = float(params.qber_threshold)
    result.security_secure = n > 0 and result.qber < threshold
    if n == 0:
        result.security_message = "INCONCLUSIVE / no key material"
    elif result.security_secure:
        result.security_message = f"SECURE (QBER < {threshold:.0%})"
    else:
        result.security_message = f"INSECURE (QBER \u2265 {threshold:.0%})"
    try:
        perf = expected_performance(params)
        result.expected_click_rate_hz = float(perf.raw_rate_bps)
        result.expected_sifted_rate_bps = float(perf.sifted_rate_bps)
        result.expected_qber = float(perf.qber)
    except (ValueError, ZeroDivisionError):  # pragma: no cover - display only
        pass


def _bloch(kind: str, source: str, bob_bit: Optional[int], visibility: float
           ) -> Tuple[float, float, float]:
    """Display-only Bloch vector: +-visibility on z for a photon key click, origin otherwise."""
    if kind != "key" or source == "dark" or bob_bit is None:
        return (0.0, 0.0, 0.0)
    return (0.0, 0.0, visibility if bob_bit == 0 else -visibility)


def _port_windows_ns(lay: FrameLayout) -> List[Tuple[float, float]]:
    bin_ns = lay.bin_width_s * 1e9
    seen = []
    for p in (0, 1):
        start = lay.port_offset[p] * bin_ns
        window = (start, start + (lay.n_bins + 1) * bin_ns)
        if window not in seen:
            seen.append(window)
    return seen


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_simulation(
    params: SimulationParameters,
    on_event: Optional[Callable[[PulseEvent], None]] = None,
    cancel_flag: Optional[threading.Event] = None,
) -> SimulationResult:
    """Run one DPS-QKD simulation with the engine chosen in ``params.engine``.

    ``on_event`` is called for every recorded :class:`PulseEvent`.
    ``cancel_flag`` is accepted for API symmetry; runs complete atomically.
    """
    params.engine = LEGACY_ENGINES.get(params.engine, params.engine)
    if params.engine not in ENGINES:
        raise ValueError(f"unknown engine {params.engine!r}; choose from {ENGINES}")
    if params.num_pulses < 2:
        raise ValueError("num_pulses must be at least 2")
    _fill_seeds(params)
    if params.engine == "timebin":
        result = _run_timebin(params)
    elif params.engine == "event":
        result = _run_event(params)
    else:
        result = _run_continuous(params)
    if on_event is not None:
        for evt in result.events:
            on_event(evt)
    return result


def _rng(params: SimulationParameters) -> np.random.Generator:
    return np.random.default_rng(
        [params.alice_seed, params.bob_seed, params.eve_seed, params.visibility_seed]
    )


# --------------------------------------------------------------------------- #
# Model C, fast engine
# --------------------------------------------------------------------------- #

def _run_timebin(params: SimulationParameters) -> SimulationResult:
    link = to_link_params(params)
    lay = FrameLayout.from_params(link)
    p_eve = params.eve_intercept_probability if params.eve_enabled else 0.0
    run = TimeBinDPSLink().simulate(
        link, params.num_pulses, _rng(params),
        eve_intercept_probability=p_eve, trace_pulses=params.max_trace_events,
    )
    result = SimulationResult()
    result.pulses_sent = params.num_pulses
    result.photons_emitted = run.photons_emitted
    result.photons_intercepted = run.photons_intercepted
    result.pulses_intercepted = run.pulses_intercepted
    result.photons_available_to_eve = run.photons_emitted if params.eve_enabled else 0
    result.photons_detected = run.source_counts["signal"]
    result.leak_clicks = run.source_counts["leak"]
    result.dark_counts = run.source_counts["dark"]
    result.raw_detector_clicks = run.clicks
    result.key_slot_clicks = run.kind_counts["key"]
    result.early_clicks = run.kind_counts["early"]
    result.late_clicks = run.kind_counts["late"]
    result.early_late_discards = result.early_clicks + result.late_clicks
    result.guard_discards = run.kind_counts["guard"]
    result.outside_window_clicks = run.kind_counts["outside"]
    result.double_clicks = run.double_clicks
    result.holdoff_blocked = run.holdoff_blocked
    result.pulses_with_click = run.pulses_with_click
    result.channel_loss_count = params.num_pulses - run.pulses_with_click
    result.eve_intercepted_count = run.eve_key_bits
    result.alice_key_bits = run.alice_bits.tolist()
    result.bob_key_bits = run.bob_bits.tolist()
    bin_ns = lay.bin_width_s * 1e9
    result.arrival_hist = run.arrival_hist.tolist()
    result.hist_edges_ns = (run.hist_edges_bins * bin_ns).tolist()
    result.port_windows_ns = _port_windows_ns(lay)

    vis = link.visibility
    tr = run.trace
    if tr is not None:
        for i in range(tr.pulse.size):
            kind = KIND_NAMES[int(tr.kind[i])]
            source = SOURCE_NAMES[int(tr.source[i])]
            port = int(tr.port[i])
            slot = int(tr.slot[i])
            a_bit = int(tr.alice_bit[i]) if tr.alice_bit[i] >= 0 else None
            b_bit = int(tr.bob_bit[i]) if tr.bob_bit[i] >= 0 else None
            bx, by, bz = _bloch(kind, source, b_bit, vis)
            result.events.append(PulseEvent(
                index=int(tr.pulse[i]), phases=tuple(int(x) for x in tr.phases[i]),
                detector=port if port >= 0 else None, slot=slot if slot >= 0 else None,
                kind=kind, source=source, time_ns=float(tr.time_bins[i]) * bin_ns,
                alice_bit=a_bit, bob_bit=b_bit, n_key=int(tr.n_key[i]),
                eve_intercepted=bool(tr.eve[i]), dark_count=source == "dark",
                bloch_x=bx, bloch_y=by, bloch_z=bz,
            ))
    result.trace_truncated = len(result.events) < run.pulses_with_click
    _finish(result, params, run.duration_s)
    return result


# --------------------------------------------------------------------------- #
# Model C, SeQUeNCe per-photon reference engine
# --------------------------------------------------------------------------- #

def _run_event(params: SimulationParameters) -> SimulationResult:
    if params.num_pulses > EVENT_ENGINE_MAX_PULSES:
        raise ValueError(
            f"the per-photon engine is limited to {EVENT_ENGINE_MAX_PULSES:,} pulses; "
            "use the fast time-bin engine for longer runs"
        )
    link = to_link_params(params)                      # validates everything
    lay = FrameLayout.from_params(link)
    bin_ps = int(round(params.bin_width_s * 1e12))
    if abs(bin_ps - params.bin_width_s * 1e12) > 1e-6:
        raise ValueError("the per-photon engine needs a bin width that is a whole number of ps")
    period_ps = lay.frame_bins * bin_ps
    loss_db = total_channel_loss_db(params) + params.receiver_loss_db
    distance_m = max(params.distance_km * 1e3, 1.0)
    window_s = params.num_pulses * period_ps * 1e-12
    round_trip_s = 2 * distance_m / _FIBRE_GROUP_VELOCITY_M_S
    runtime_ps = (window_s + round_trip_s) * 1e12 * 3 + 1e9
    fmt = link.pulse_format

    topo = DPSQKDTopo.from_nodes(
        "alice", "bob",
        distance=distance_m,
        attenuation=loss_db / distance_m,
        ls_params={
            "frequency": params.pulse_rate_hz,
            "mean_photon_num": params.mean_photon_number,
            "bins_per_pulse": fmt.bins_per_pulse,
            "bin_width_ps": bin_ps,
            "extinction_ratio_db": fmt.extinction_ratio_db,
        },
        dli_params={
            "frequency": params.pulse_rate_hz,
            "bins_per_pulse": fmt.bins_per_pulse,
            "bin_width_ps": bin_ps,
            "visibility": link.visibility,
            "pm_transition_ps": params.pm_transition_s * 1e12,
            "layout": params.receiver_layout,
            "tdm_delay_ps": int(round(params.tdm_delay_s * 1e12)),
            "combiner_loss_db": params.combiner_loss_db,
            "balance_loss_db": params.balance_loss_db,
            "efficiency": params.detector_efficiency,
            "dark_count_prob": params.dark_count_prob,
            "holdoff_ps": int(round(params.holdoff_s * 1e12)),
            "timing_jitter_ps": params.timing_jitter_s * 1e12,
            "gate_width_ps": (None if params.gate_width_s is None
                              else int(round(params.gate_width_s * 1e12))),
        },
        stack_size=1,
        runtime=runtime_ps,
        alice_seed=params.alice_seed,
        bob_seed=params.bob_seed,
    )
    tl = topo.get_timeline()
    alice_node, bob_node = topo.get_node("alice"), topo.get_node("bob")
    alice_proto, bob_proto = alice_node.protocol_stack[0], bob_node.protocol_stack[0]
    ls = alice_node.components[alice_proto.ls_name]
    dli = bob_node.components[bob_proto.dli_name]

    eve_stats = {"pulses": set()}
    if params.eve_enabled and params.eve_intercept_probability > 0:
        eve_rng = np.random.default_rng(params.eve_seed)
        n_bins = fmt.bins_per_pulse

        def eve_hook(pulse_index, phase_bits, n_photons):
            if eve_rng.random() < params.eve_intercept_probability:
                eve_stats["pulses"].add(pulse_index)
                return tuple(int(b) for b in eve_rng.integers(0, 2, size=n_bins))
            return None

        ls.intercept_hook = eve_hook

    tl.schedule(Event(0, Process(alice_proto, "run_single_window", [params.num_pulses])))
    tl.init()
    tl.run()

    result = SimulationResult()
    result.pulses_sent = params.num_pulses
    result.photons_emitted = ls.photon_counter
    result.photons_intercepted = ls.photons_intercepted
    result.pulses_intercepted = ls.pulses_intercepted
    result.photons_available_to_eve = ls.photon_counter if params.eve_enabled else 0
    result.alice_key_bits = list(alice_proto.key_bits)
    result.bob_key_bits = list(bob_proto.key_bits)
    result.holdoff_blocked = dli.holdoff_blocked

    clicks = bob_proto.last_clicks
    blocks = alice_proto.last_phase_blocks
    if blocks is None:
        blocks = np.zeros((params.num_pulses, fmt.bins_per_pulse), np.uint8)
    window_start = bob_proto._window_start
    times = [c.time_ps for c in clicks]
    dets = [c.detector for c in clicks]
    pulse, kind, port, slot, t_bins = classify_clicks(times, dets, window_start, lay,
                                                      params.num_pulses)
    kinds = [KIND_NAMES[int(k)] for k in kind]
    result.raw_detector_clicks = len(clicks)
    result.key_slot_clicks = kinds.count("key")
    result.early_clicks = kinds.count("early")
    result.late_clicks = kinds.count("late")
    result.early_late_discards = result.early_clicks + result.late_clicks
    result.guard_discards = kinds.count("guard")
    result.outside_window_clicks = kinds.count("outside")
    result.photons_detected = sum(1 for c in clicks if c.source == "signal")
    result.leak_clicks = sum(1 for c in clicks if c.source == "leak")
    result.dark_counts = sum(1 for c in clicks if c.source == "dark")
    key_ids: Dict[int, List[int]] = {}
    for i, k in enumerate(kinds):
        if k == "key":
            key_ids.setdefault(int(pulse[i]) * lay.n_bins + int(slot[i]), []).append(i)
    result.double_clicks = sum(1 for v in key_ids.values() if len(v) > 1)
    eve_key = {sid for sid, idx in key_ids.items() if any(clicks[i].eve for i in idx)}
    result.eve_intercepted_count = len(eve_key)
    result.pulses_with_click = len({int(c.gate) for c in clicks})
    result.channel_loss_count = params.num_pulses - result.pulses_with_click

    bin_ns = bin_ps * 1e-3
    edges = np.arange(lay.gate_start - 1.0, lay.gate_start + lay.gate_width + 1.0 + 0.025, 0.05)
    result.arrival_hist = np.histogram(np.asarray(t_bins), bins=edges)[0].tolist()
    result.hist_edges_ns = (edges * bin_ns).tolist()
    result.port_windows_ns = _port_windows_ns(lay)

    # Event log: pulses with a click, in time order (key click shown if a pulse has one)
    vis = link.visibility
    shown: Dict[int, int] = {}
    for i in np.argsort(np.asarray(times, dtype=float), kind="stable").tolist():
        k = int(clicks[i].gate)
        if k not in shown or (kinds[i] == "key" and kinds[shown[k]] != "key"):
            shown[k] = i
    for k in sorted(shown)[: params.max_trace_events]:
        i = shown[k]
        is_key = kinds[i] == "key"
        a_bit = (alice_bits_for(blocks, [int(pulse[i]) * lay.n_bins + int(slot[i])])[0]
                 if is_key else None)
        b_bit = int(port[i]) if is_key else None
        source = clicks[i].source
        bx, by, bz = _bloch(kinds[i], source, b_bit, vis)
        n_key = sum(1 for sid in key_ids if sid // lay.n_bins == k)
        result.events.append(PulseEvent(
            index=k, phases=tuple(int(x) for x in blocks[k]) if 0 <= k < len(blocks) else (),
            detector=int(port[i]) if port[i] >= 0 else None,
            slot=int(slot[i]) if slot[i] >= 0 else None, kind=kinds[i], source=source,
            time_ns=float(t_bins[i]) * bin_ns, alice_bit=a_bit, bob_bit=b_bit, n_key=n_key,
            eve_intercepted=k in eve_stats["pulses"], dark_count=source == "dark",
            bloch_x=bx, bloch_y=by, bloch_z=bz,
        ))
    result.trace_truncated = len(result.events) < len(shown)
    _finish(result, params, window_s)
    return result


# --------------------------------------------------------------------------- #
# Model B, continuous train (comparison only)
# --------------------------------------------------------------------------- #

def _run_continuous(params: SimulationParameters) -> SimulationResult:
    link = to_continuous_link_params(params)
    p_eve = params.eve_intercept_probability if params.eve_enabled else 0.0
    n_slots = int(params.num_pulses) * int(params.bins_per_pulse)   # same number of bins
    run = CoherentDPSLink().simulate(
        link, n_slots, _rng(params),
        eve_intercept_probability=p_eve, trace_slots=params.max_trace_events,
    )
    result = SimulationResult()
    result.pulses_sent = n_slots
    result.photons_emitted = run.photons_emitted
    result.photons_intercepted = run.photons_intercepted
    result.photons_available_to_eve = run.photons_emitted if params.eve_enabled else 0
    result.photons_detected = run.signal_clicks
    result.dark_counts = run.dark_clicks
    result.raw_detector_clicks = run.raw_clicks
    result.key_slot_clicks = run.sifted_bits
    result.early_late_discards = run.edge_clicks
    result.double_clicks = run.double_clicks
    result.holdoff_blocked = run.deadtime_blocked
    result.eve_intercepted_count = run.eve_slots
    result.pulses_with_click = run.sifted_bits
    result.channel_loss_count = (n_slots - 1) - run.sifted_bits
    result.alice_key_bits = run.alice_bits.tolist()
    result.bob_key_bits = run.bob_bits.tolist()
    if run.trace is not None:
        tr = run.trace
        for i in range(tr.slot.size):
            det = int(tr.outcome[i])
            clicked = det >= 0
            source = ("dark" if bool(tr.dark[i]) else "signal") if clicked else ""
            kind = "key" if clicked else ""
            bx, by, bz = _bloch(kind, source, det if clicked else None, link.visibility)
            result.events.append(PulseEvent(
                index=int(tr.slot[i]), detector=det if clicked else None,
                kind=kind, source=source, alice_bit=int(tr.bit[i]),
                bob_bit=det if clicked else None, n_key=1 if clicked else 0,
                eve_intercepted=bool(tr.eve[i]), dark_count=source == "dark",
                bloch_x=bx, bloch_y=by, bloch_z=bz,
            ))
    result.trace_truncated = len(result.events) < n_slots - 1
    _finish(result, params, run.duration_s)
    return result
