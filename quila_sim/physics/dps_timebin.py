"""Time-bin DPS-QKD link engine -- Model C, the MAQAN / IIT Madras hardware.

What is simulated
-----------------
Every pulse period (32 ns for MAQAN) Alice sends one weak coherent pulse of N
time bins (3 x 1 ns) with random phases 0/pi, followed by empty bins. Bob's
one-bin-delay interferometer turns the pulse into N+1 output slots per port
(early, N-1 interference slots, late); with a single time-multiplexed
detector, port 1 arrives 10 ns after port 0. Bob time-stamps each click, and
sifting keeps clicks in interference slots: the window gives Bob's bit, the
slot says which phase difference it belongs to.

The engine works in the photon picture, which is exact for coherent pulses
separated by empty bins (Poisson thinning):

1. Photons per pulse ~ Poisson(mu); Eve (teaching model) intercepts a pulse
   with probability p and re-sends it with random phases.
2. Each photon survives the channel and Bob's optics with T, then the port
   loss and detector efficiency.
3. It leaves the interferometer in slot s through port d with the
   single-photon probabilities of :func:`quila_sim.core.timebin.exit_probabilities`
   -- 1/6, 1/3, 1/3, 1/6 over the slots of a flat 3-bin pulse, the port set by
   the phase difference and the visibility.
4. Its arrival time inside the slot is uniform. Near a bin edge, within the
   phase modulator's transition time, the phase difference is undefined and
   the port is random.
5. Light leaking through the intensity modulator between pulses, and dark
   counts (uniform over the gate), are added as backgrounds.
6. The detector fires on the first event of a gate in which it is armed;
   after a click it is blind for the hold-off time and re-arms at the next
   gate (non-paralysable). The recorded time has Gaussian jitter.
7. Bob classifies the recorded time (:func:`quila_sim.core.timebin.classify_times`):
   key slot, early/late, guard band, or outside the port windows.

Sampling is vectorised over chunks of pulses; only pulses with a detected
event are processed individually, so tens of millions of pulses take seconds.
:meth:`TimeBinDPSLink.performance` gives the same quantities in closed form
(exact per gate, first order in the click probability per gate for
multi-photon gates, renewal approximation for the hold-off).

Calibration (``quila_sim.core.MAQAN_DPS``): with each imperfection switched on
alone, the engine reproduces the QBER budget of Shaw et al. (2022), Table III,
for 1 ns bins -- visibility 4.0 %, SPD timing jitter 5.0 %, PM rise/fall
2.1 %, IM extinction ratio 1.6 % -- and a 200 ps guard band removes ~20 % of
the sifted bits while lowering the QBER by ~2.5 points, as reported there.

Not modelled yet: afterpulsing (1.5 % in Table III; roadmap item 4), slow
drifts of visibility and phase (item 5), realistic attacks (item 13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from quila_sim.core.link import DetectionRecord, LinkParams, LinkPerformance
from quila_sim.core.timebin import (
    KEY,
    KIND_NAMES,
    FrameLayout,
    classify_times,
    exit_probabilities,
    leak_exit_table,
    pulse_amplitudes,
)

__all__ = ["SOURCE_NAMES", "TimeBinDPSLink", "TimeBinRun", "TimeBinTrace"]

#: Origin of a click.
SIGNAL, LEAK, DARK = 0, 1, 2
SOURCE_NAMES = ("signal", "leak", "dark")

#: Width of the arrival-time histogram bins, in time bins (50 ps for 1 ns bins).
HIST_BIN = 0.05

#: Gauss-Legendre order per piece of the gate in the closed form.
_QUAD_ORDER = 12


def _gate_pieces(
    g0: float, g1: float, offsets: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the gate [g0, g1) into pieces on which every intensity is constant.

    Intensities change at whole bins and at the given fractional ``offsets``
    inside each bin (phase-transition zones, guard bands). Returns the piece
    starts and ends, and Gauss-Legendre nodes and weights inside each piece,
    shapes (I,), (I,), (I, Q), (I, Q).
    """
    cuts = {g0, g1}
    for u in range(math.floor(g0), math.ceil(g1) + 1):
        for off in (0.0, *offsets):
            if g0 < u + off < g1:
                cuts.add(u + off)
    edges = np.array(sorted(cuts))
    a, b = edges[:-1], edges[1:]
    real = (b - a) > 1e-12
    a, b = a[real], b[real]
    x, w = np.polynomial.legendre.leggauss(_QUAD_ORDER)
    nodes = a[:, None] + (b - a)[:, None] * (x + 1.0) / 2.0
    weights = (b - a)[:, None] * w / 2.0
    return a, b, nodes, weights


@dataclass
class TimeBinTrace:
    """Details of the pulses that produced a click (first ``trace_pulses`` of them).

    For each such pulse, the click shown is its first key click if it has one,
    otherwise its first click. Times are in bins after bin 0 reached Bob.
    """

    pulse: np.ndarray       # pulse index (int64)
    phases: np.ndarray      # Alice's phase bits, shape (n, N) (uint8)
    eve: np.ndarray         # pulse intercepted by Eve (bool)
    source: np.ndarray      # SIGNAL / LEAK / DARK (int8)
    kind: np.ndarray        # KEY / EARLY / LATE / GUARD / OUTSIDE (int8)
    port: np.ndarray        # port window of the click, -1 if outside (int8)
    slot: np.ndarray        # interferometer slot, -1 if outside (int16)
    time_bins: np.ndarray   # recorded click time (float)
    alice_bit: np.ndarray   # Alice's bit for key clicks, -1 otherwise (int8)
    bob_bit: np.ndarray     # Bob's bit for key clicks, -1 otherwise (int8)
    n_key: np.ndarray       # sifted bits obtained from this pulse (int8)


@dataclass
class TimeBinRun:
    """Everything one engine run produced."""

    n_pulses: int
    pulse_rate_hz: float
    bins_per_pulse: int
    pulses: np.ndarray            # sifted: pulse index (int64)
    slots: np.ndarray             # sifted: interference slot 1..N-1 (int16)
    alice_bits: np.ndarray        # sifted: Alice's bit (uint8)
    bob_bits: np.ndarray          # sifted: Bob's bit (uint8)
    photons_emitted: int = 0      # photons Alice sent (pulses only, not leakage)
    photons_intercepted: int = 0  # Alice's photons in pulses Eve intercepted
    pulses_intercepted: int = 0
    photons_arrived: int = 0      # pulse photons reaching Bob's interferometer
    clicks: int = 0               # clicks the detector registered
    holdoff_blocked: int = 0      # detectable events while the detector was blind
    outside_gate: int = 0         # detected photons next to the gate but outside it
    double_clicks: int = 0        # (dual layout) both detectors in one key slot
    eve_clicks: int = 0           # clicks caused by Eve's re-sent photons
    eve_key_bits: int = 0         # sifted bits caused by Eve's re-sent photons
    pulses_with_click: int = 0
    kind_counts: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    arrival_hist: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    hist_edges_bins: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trace: TimeBinTrace | None = None

    @property
    def sifted_bits(self) -> int:
        return int(self.pulses.size)

    @property
    def bit_errors(self) -> int:
        return int(np.count_nonzero(self.alice_bits != self.bob_bits))

    @property
    def qber(self) -> float:
        return self.bit_errors / self.sifted_bits if self.sifted_bits else 0.0

    @property
    def duration_s(self) -> float:
        return self.n_pulses / self.pulse_rate_hz

    @property
    def sifted_rate_bps(self) -> float:
        return self.sifted_bits / self.duration_s

    @property
    def click_rate_hz(self) -> float:
        return self.clicks / self.duration_s

    @property
    def key_slot_ids(self) -> np.ndarray:
        """Global id of each sifted slot: pulse * N + slot."""
        return self.pulses * self.bins_per_pulse + self.slots


@dataclass(frozen=True)
class _Model:
    """Everything derived from LinkParams that the sampler needs."""

    layout: FrameLayout
    n: int
    mu: float
    survive_channel: float      # channel + common receiver optics
    survive_detect: float       # port loss x detector efficiency
    p_dark: float               # per gate, per detector
    holdoff: float              # bins
    jitter: float               # bins (rms)
    transition: float           # bins (PM phase-undefined window per edge)
    exit: np.ndarray            # (2^N, N+1, 2) exit probabilities per phase pattern
    exit_cdf: np.ndarray        # (2^N, 2(N+1)) cumulative, ordered (slot, port)
    leak_start: np.ndarray      # leaked-light slots inside a gate (bins)
    leak_port: np.ndarray
    leak_mean: np.ndarray       # detected leaked photons per gate, per slot
    offsets: np.ndarray         # port -> detector-time offset (bins)
    det_of_port: np.ndarray     # port -> detector index

    @property
    def leak_total(self) -> float:
        return float(self.leak_mean.sum())


class TimeBinDPSLink:
    """Time-bin DPS link model (Model C) implementing :class:`quila_sim.core.LinkModel`.

    ``params`` must carry ``pulse_format`` and ``receiver``; see
    :func:`quila_sim.core.timebin` and ``MAQAN_DPS`` for the MAQAN values.

    Parameters
    ----------
    chunk_pulses
        Pulses processed per vectorised chunk (memory/speed trade-off).
        Results for a given seed depend on it; keep the default for
        reproducibility.
    """

    name = "dps-timebin"
    fidelity = "statistical"

    def __init__(self, chunk_pulses: int = 1 << 20) -> None:
        if chunk_pulses < 16:
            raise ValueError("chunk_pulses too small")
        self.chunk_pulses = int(chunk_pulses)

    # ------------------------------------------------------------------ #
    # Set-up
    # ------------------------------------------------------------------ #
    @staticmethod
    def _model(params: LinkParams) -> _Model:
        if params.protocol.upper() != "DPS":
            raise ValueError(f"TimeBinDPSLink only models DPS, got {params.protocol!r}")
        layout = FrameLayout.from_params(params)
        fmt, rx, det = params.pulse_format, params.receiver, params.detector
        n = layout.n_bins
        codes = np.arange(2 ** n)
        bits = ((codes[:, None] >> np.arange(n)) & 1).astype(np.uint8)
        exit_p = exit_probabilities(pulse_amplitudes(bits, fmt.weights), params.visibility)
        flat = exit_p.reshape(codes.size, 2 * (n + 1))
        cdf = np.cumsum(flat, axis=1)
        cdf /= cdf[:, -1:]
        survive_detect = det.efficiency * rx.port_transmittance
        survive_channel = params.transmittance
        starts, ports, means = leak_exit_table(fmt, layout, params.visibility,
                                               params.mean_photon_number)
        offsets = np.asarray(layout.port_offset, dtype=float)
        return _Model(
            layout=layout,
            n=n,
            mu=params.mean_photon_number,
            survive_channel=survive_channel,
            survive_detect=survive_detect,
            p_dark=det.dark_count_prob,
            holdoff=det.dead_time_s / layout.bin_width_s,
            jitter=det.timing_jitter_s / layout.bin_width_s,
            transition=fmt.pm_transition_s / layout.bin_width_s,
            exit=exit_p,
            exit_cdf=cdf,
            leak_start=starts,
            leak_port=ports,
            leak_mean=means * survive_channel * survive_detect,
            offsets=offsets,
            det_of_port=np.asarray(layout.detector_of_port, dtype=np.int64),
        )

    # ------------------------------------------------------------------ #
    # Closed form
    # ------------------------------------------------------------------ #
    def performance(
        self, params: LinkParams, *, eve_intercept_probability: float = 0.0
    ) -> LinkPerformance:
        """Expected click rate, sifted rate and QBER, in closed form.

        Within one gate the detected events of a detector form a Poisson
        process whose intensity is piecewise constant (signal slots, leaked
        light, dark counts), so the probability that the gate's *first* event
        is recorded in each key slot is an exact integral, averaged over the
        2^N phase patterns. The hold-off is a renewal process: after a click
        the detector skips a fixed number of gates. Double clicks of the
        "dual" layout (both detectors in one key slot) are neglected.
        """
        m = self._model(params)
        lay, n = m.layout, m.n
        p_eve = float(eve_intercept_probability)
        if not 0.0 <= p_eve <= 1.0:
            raise ValueError("eve_intercept_probability must be in [0, 1]")
        g0, g1 = lay.gate_start, lay.gate_start + lay.gate_width
        half_tr = m.transition / 2.0
        half_g = lay.guard / 2.0
        a, b, nodes, wts = _gate_pieces(g0, g1, (half_tr, 1.0 - half_tr, half_g, 1.0 - half_g))
        mid = (a + b) / 2.0

        n_codes = m.exit.shape[0]
        slot_total = m.exit.sum(axis=2)                                   # (codes, n+1)
        steady = np.divide(m.exit, slot_total[..., None],
                           out=np.full_like(m.exit, 0.5), where=slot_total[..., None] > 0)
        detect_scale = m.mu * m.survive_channel * m.survive_detect
        bits = ((np.arange(n_codes)[:, None] >> np.arange(n)) & 1).astype(np.int64)
        alice_bit = np.full((n_codes, n + 1), -1, dtype=np.int64)
        alice_bit[:, 1:n] = bits[:, 1:] ^ bits[:, :-1]
        # match[c, w, j]: a click recorded in window w, key slot j agrees with Alice
        match = alice_bit[:, None, :] == np.arange(2)[None, :, None]
        lam_dark = (-math.log1p(-m.p_dark) if m.p_dark < 1 else math.inf) / lay.gate_width
        blind_at = np.maximum(
            np.ceil((nodes + params.detector.dead_time_s / lay.bin_width_s - g0)
                    / lay.frame_bins - 1e-9), 1) - 1                      # (I, Q)

        raw = sifted = errors = 0.0
        mean_blind = []
        for d in range(lay.n_detectors):
            # intensity (detected events per bin) on each piece, per phase pattern
            lam = np.full((n_codes, a.size), lam_dark)
            for p in (0, 1):
                if lay.detector_of_port[p] != d:
                    continue
                u = mid - m.offsets[p]
                slot = np.floor(u)
                tau = u - slot
                valid = (slot >= 0) & (slot <= n)
                zone = (slot >= 1) & (slot <= n - 1) & ((tau < half_tr) | (tau >= 1.0 - half_tr))
                si = np.clip(slot, 0, n).astype(np.int64)
                port_prob = np.where(zone[None, :], 0.5, steady[:, si, p])
                lam += np.where(valid[None, :], detect_scale * slot_total[:, si] * port_prob, 0.0)
            for start, port, mean in zip(m.leak_start, m.leak_port, m.leak_mean, strict=True):
                if lay.detector_of_port[int(port)] == d:
                    lam += mean * ((mid >= start) & (mid < start + 1.0))[None, :]
            # survival up to each node, first-event mass at each node
            cum = np.cumsum(lam * (b - a)[None, :], axis=1)
            lam_before = np.concatenate([np.zeros((n_codes, 1)), cum[:, :-1]], axis=1)
            surv = np.exp(-(lam_before[:, :, None] + lam[:, :, None] * (nodes - a[:, None])[None]))
            first = lam[:, :, None] * surv * wts[None]                    # (codes, I, Q)
            q = -np.expm1(-cum[:, -1])                                     # click prob per gate
            # where the first event is recorded: key window w, slot j
            K = np.zeros((n_codes, 2, n + 1))
            for w in (0, 1):
                if lay.detector_of_port[w] != d:
                    continue
                for j in range(1, n):
                    lo = lay.port_offset[w] + j + half_g
                    hi = lay.port_offset[w] + j + 1 - half_g
                    if m.jitter > 0:
                        pr = ndtr((hi - nodes) / m.jitter) - ndtr((lo - nodes) / m.jitter)
                    else:
                        pr = ((nodes >= lo) & (nodes < hi)).astype(float)
                    K[:, w, j] = np.einsum("ciq,iq->c", first, pr)
            key = K.sum(axis=(1, 2))
            ok_own = (K * match).sum(axis=(1, 2))
            ok = (1.0 - p_eve) * ok_own.mean() + p_eve * key.mean() / 2.0
            err = key.mean() - ok
            q_mean = float(q.mean())
            blind = float((first.sum(axis=0) * blind_at).sum() / first.sum()) if q_mean > 0 else 0.0
            mean_blind.append(blind)
            armed = 1.0 / (1.0 + blind * q_mean)
            raw += armed * q_mean
            sifted += armed * key.mean()
            errors += armed * err
        notes = []
        if max(mean_blind, default=0.0) > 0:
            notes.append(f"hold-off: {max(mean_blind):.0f} gates blind after each click")
        rate = params.rep_rate_hz
        return LinkPerformance(
            raw_rate_bps=float(rate * raw),
            sifted_rate_bps=float(rate * sifted),
            qber=float(min(max(errors / sifted, 0.0), 1.0)) if sifted > 0 else 0.0,
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(
        self, params: LinkParams, n_pulses: int, rng: np.random.Generator
    ) -> DetectionRecord:
        run = self.simulate(params, n_pulses, rng)
        return DetectionRecord(run.key_slot_ids, run.alice_bits, run.bob_bits, n_pulses)

    def simulate(
        self,
        params: LinkParams,
        n_pulses: int,
        rng: np.random.Generator,
        *,
        eve_intercept_probability: float = 0.0,
        trace_pulses: int = 0,
    ) -> TimeBinRun:
        """Simulate ``n_pulses`` consecutive pulse periods."""
        m = self._model(params)
        if n_pulses < 1:
            raise ValueError("n_pulses must be at least 1")
        p_eve = float(eve_intercept_probability)
        if not 0.0 <= p_eve <= 1.0:
            raise ValueError("eve_intercept_probability must be in [0, 1]")
        lay, n = m.layout, m.n
        frame = lay.frame_bins
        g0, g1 = lay.gate_start, lay.gate_start + lay.gate_width
        n_det = lay.n_detectors
        weights_pow = (1 << np.arange(n)).astype(np.int64)
        leak_cdf = np.cumsum(m.leak_mean) / m.leak_total if m.leak_total > 0 else None

        run = TimeBinRun(n_pulses=n_pulses, pulse_rate_hz=params.rep_rate_hz, bins_per_pulse=n,
                         pulses=np.empty(0, np.int64), slots=np.empty(0, np.int16),
                         alice_bits=np.empty(0, np.uint8), bob_bits=np.empty(0, np.uint8))
        kind_counts = np.zeros(len(KIND_NAMES), np.int64)
        source_counts = np.zeros(len(SOURCE_NAMES), np.int64)
        edges = np.arange(g0 - 1.0, g1 + 1.0 + HIST_BIN / 2, HIST_BIN)
        hist = np.zeros(edges.size - 1, np.int64)
        ready = [-math.inf] * n_det
        out_pulse, out_slot, out_alice, out_bob = [], [], [], []
        trace_rows: list[tuple] = []
        clicked_pulses = 0

        start = 0
        while start < n_pulses:
            k = min(self.chunk_pulses, n_pulses - start)
            alice = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
            if p_eve > 0:
                eve = rng.random(k) < p_eve
                sent = alice.copy()
                n_eve = int(eve.sum())
                if n_eve:
                    sent[eve] = rng.integers(0, 2, size=(n_eve, n), dtype=np.uint8)
            else:
                eve = np.zeros(k, dtype=bool)
                sent = alice
            code = sent.astype(np.int64) @ weights_pow

            # -- pulse photons -------------------------------------------------
            photons = rng.poisson(m.mu, size=k)
            run.photons_emitted += int(photons.sum())
            if p_eve > 0:
                run.photons_intercepted += int(photons[eve].sum())
                run.pulses_intercepted += int(eve.sum())
            arrived = rng.binomial(photons, m.survive_channel)
            run.photons_arrived += int(arrived.sum())
            detected = rng.binomial(arrived, m.survive_detect)
            hit = np.flatnonzero(detected)
            sig_pulse = np.repeat(hit, detected[hit])
            u = rng.random(sig_pulse.size)
            cat = (m.exit_cdf[code[sig_pulse]] <= u[:, None]).sum(axis=1)
            cat = np.minimum(cat, 2 * n + 1)
            sig_slot = cat // 2
            sig_port = cat % 2
            tau = rng.random(sig_pulse.size)
            if m.transition > 0:
                zone = ((sig_slot >= 1) & (sig_slot <= n - 1)
                        & ((tau < m.transition / 2) | (tau >= 1 - m.transition / 2)))
                n_zone = int(zone.sum())
                if n_zone:
                    sig_port[zone] = rng.integers(0, 2, size=n_zone)
            sig_t = sig_slot + m.offsets[sig_port] + tau
            sig_det = m.det_of_port[sig_port]

            # -- leaked light ---------------------------------------------------
            if leak_cdf is not None:
                n_leak = int(rng.poisson(m.leak_total * k))
                leak_pulse = rng.integers(0, k, size=n_leak)
                which = np.minimum(np.searchsorted(leak_cdf, rng.random(n_leak), side="right"),
                                   leak_cdf.size - 1)
                leak_t = m.leak_start[which] + rng.random(n_leak)
                leak_det = m.det_of_port[m.leak_port[which]]
            else:
                leak_pulse = np.empty(0, np.int64)
                leak_t = np.empty(0)
                leak_det = np.empty(0, np.int64)

            # -- dark counts (uniform over each gate) ---------------------------
            dark_pulse, dark_t, dark_det = [], [], []
            for d in range(n_det):
                n_dark = int(rng.binomial(k, m.p_dark)) if m.p_dark > 0 else 0
                if n_dark:
                    dark_pulse.append(rng.choice(k, size=n_dark, replace=False))
                    dark_t.append(g0 + lay.gate_width * rng.random(n_dark))
                    dark_det.append(np.full(n_dark, d, np.int64))
            dark_pulse = np.concatenate(dark_pulse) if dark_pulse else np.empty(0, np.int64)
            dark_t = np.concatenate(dark_t) if dark_t else np.empty(0)
            dark_det = np.concatenate(dark_det) if dark_det else np.empty(0, np.int64)

            # -- merge, gate, hold-off ------------------------------------------
            pulse = np.concatenate([sig_pulse, leak_pulse, dark_pulse]).astype(np.int64)
            t_true = np.concatenate([sig_t, leak_t, dark_t])
            det = np.concatenate([sig_det, leak_det, dark_det]).astype(np.int64)
            source = np.concatenate([np.full(sig_pulse.size, SIGNAL, np.int8),
                                     np.full(leak_pulse.size, LEAK, np.int8),
                                     np.full(dark_pulse.size, DARK, np.int8)])
            from_eve = np.concatenate([eve[sig_pulse],
                                       np.zeros(leak_pulse.size + dark_pulse.size, bool)])
            inside = (t_true >= g0) & (t_true < g1)
            run.outside_gate += int((~inside).sum())
            pulse, t_true, det = pulse[inside], t_true[inside], det[inside]
            source, from_eve = source[inside], from_eve[inside]
            abs_t = (start + pulse) * float(frame) + t_true
            order = np.argsort(abs_t, kind="stable")
            fired = np.zeros(order.size, dtype=bool)
            for rank, i in enumerate(order.tolist()):
                d = int(det[i])
                t = abs_t[i]
                if t >= ready[d]:
                    fired[rank] = True
                    gate = start + int(pulse[i])
                    nxt = max(math.ceil((t + m.holdoff - g0) / frame), gate + 1)
                    ready[d] = nxt * frame + g0
                else:
                    run.holdoff_blocked += 1
            idx = order[fired]                        # fired events, in time order
            if idx.size == 0:
                start += k
                continue

            f_pulse, f_det, f_src, f_eve = pulse[idx], det[idx], source[idx], from_eve[idx]
            t_rec = t_true[idx]
            if m.jitter > 0:
                t_rec = t_rec + m.jitter * rng.standard_normal(idx.size)
            kind, port, slot = classify_times(t_rec, f_det, lay)
            run.clicks += int(idx.size)
            kind_counts += np.bincount(kind, minlength=len(KIND_NAMES))
            source_counts += np.bincount(f_src, minlength=len(SOURCE_NAMES))
            run.eve_clicks += int(f_eve.sum())
            hist += np.histogram(t_rec, bins=edges)[0]
            run.pulses_with_click += int(np.unique(f_pulse).size)

            # -- sifting --------------------------------------------------------
            is_key = kind == KEY
            kp = f_pulse[is_key]
            kj = slot[is_key].astype(np.int64)
            bob = port[is_key].astype(np.uint8)
            a_bit = alice[kp, kj] ^ alice[kp, kj - 1]
            k_eve = f_eve[is_key]
            keep = np.ones(kp.size, dtype=bool)
            if n_det > 1 and kp.size > 1:
                ids = kp * (n + 1) + kj
                _, first, inverse, counts = np.unique(ids, return_index=True,
                                                      return_inverse=True, return_counts=True)
                keep[:] = False
                keep[first] = True
                double = keep & (counts[inverse] > 1)
                n_double = int(double.sum())
                if n_double:
                    bob[double] = rng.integers(0, 2, size=n_double, dtype=np.uint8)
                    run.double_clicks += n_double
            out_pulse.append(start + kp[keep])
            out_slot.append(kj[keep].astype(np.int16))
            out_alice.append(a_bit[keep].astype(np.uint8))
            out_bob.append(bob[keep])
            run.eve_key_bits += int(k_eve[keep].sum())

            # -- trace ----------------------------------------------------------
            if clicked_pulses < trace_pulses:
                key_bits_per_pulse: dict[int, int] = {}
                for p_ in kp[keep].tolist():
                    key_bits_per_pulse[p_] = key_bits_per_pulse.get(p_, 0) + 1
                shown: dict[int, int] = {}
                for r, p_ in enumerate(f_pulse.tolist()):
                    if p_ not in shown or (kind[r] == KEY and kind[shown[p_]] != KEY):
                        shown[p_] = r
                for p_, r in sorted(shown.items()):
                    if clicked_pulses >= trace_pulses:
                        break
                    is_k = kind[r] == KEY
                    j = int(slot[r])
                    trace_rows.append((
                        start + p_, alice[p_].copy(), bool(eve[p_]), int(f_src[r]),
                        int(kind[r]), int(port[r]), j, float(t_rec[r]),
                        int(alice[p_, j] ^ alice[p_, j - 1]) if is_k else -1,
                        int(port[r]) if is_k else -1,
                        key_bits_per_pulse.get(p_, 0),
                    ))
                    clicked_pulses += 1
            start += k

        run.pulses = np.concatenate(out_pulse) if out_pulse else np.empty(0, np.int64)
        run.slots = np.concatenate(out_slot) if out_slot else np.empty(0, np.int16)
        run.alice_bits = np.concatenate(out_alice) if out_alice else np.empty(0, np.uint8)
        run.bob_bits = np.concatenate(out_bob) if out_bob else np.empty(0, np.uint8)
        run.kind_counts = {name: int(c) for name, c in zip(KIND_NAMES, kind_counts, strict=True)}
        run.source_counts = {name: int(c)
                             for name, c in zip(SOURCE_NAMES, source_counts, strict=True)}
        run.arrival_hist = hist
        run.hist_edges_bins = edges
        if trace_pulses:
            run.trace = _make_trace(trace_rows, n)
        return run


def _make_trace(rows: list[tuple], n: int) -> TimeBinTrace:
    if not rows:
        empty_i = np.empty(0, np.int64)
        return TimeBinTrace(empty_i, np.empty((0, n), np.uint8), np.empty(0, bool),
                            empty_i.astype(np.int8), empty_i.astype(np.int8),
                            empty_i.astype(np.int8), empty_i.astype(np.int16), np.empty(0),
                            empty_i.astype(np.int8), empty_i.astype(np.int8),
                            empty_i.astype(np.int8))
    cols = list(zip(*rows, strict=True))
    return TimeBinTrace(
        pulse=np.asarray(cols[0], np.int64),
        phases=np.asarray(cols[1], np.uint8).reshape(len(rows), n),
        eve=np.asarray(cols[2], bool),
        source=np.asarray(cols[3], np.int8),
        kind=np.asarray(cols[4], np.int8),
        port=np.asarray(cols[5], np.int8),
        slot=np.asarray(cols[6], np.int16),
        time_bins=np.asarray(cols[7], float),
        alice_bit=np.asarray(cols[8], np.int8),
        bob_bit=np.asarray(cols[9], np.int8),
        n_key=np.asarray(cols[10], np.int8),
    )
