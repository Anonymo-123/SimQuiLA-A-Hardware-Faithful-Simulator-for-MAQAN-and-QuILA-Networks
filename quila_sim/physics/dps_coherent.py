"""Continuous-train DPS-QKD link engine -- Model B (for comparison only).

.. note::
   This is **not** how the MAQAN / IIT Madras hardware works. MAQAN sends one
   3-time-bin pulse every 32 ns with empty time between pulses; that is
   Model C, :class:`quila_sim.physics.TimeBinDPSLink`, the simulator default.
   Model B is the textbook DPS of Inoue et al. (a gap-free pulse train, one
   key bit per time slot). It is kept so that the two schemes can be
   compared, e.g. to show what a continuous-train transmitter would gain.

Physics
-------
Alice sends a continuous train of weak coherent pulses with mean photon
number ``mu`` and random phases phi_k in {0, pi}. Bob's one-bit-delay
interferometer (DLI) overlaps pulse k-1 (long arm) with pulse k (short arm),
so time slot k (k = 1 .. n-1) carries Alice's key bit

    b_k = phi_k XOR phi_{k-1}        (0 -> same phase -> detector D0).

Because the DLI outputs are themselves coherent states, the photon numbers
reaching D0 and D1 in slot k are independent Poisson variables with means

    m0 = mu * T * (1 + V s_k) / 2,   m1 = mu * T * (1 - V s_k) / 2,
    s_k = +1 if b_k == 0 else -1,

where T is the channel-plus-receiver transmittance and V the DLI visibility.
Every interior slot can click, with probability ~ mu * T * eta_d in total.
This is exact for coherent states. (The baseline 3-pulse-photon model
discarded half of the photons as early/late, under-predicting the key rate
by 2x.) Only the two ends of the stream have unpaired "early"/"late" light.

Sampling uses the equivalent photon picture: N_k ~ Poisson(mu) photons per
slot (half from each neighbouring pulse), thinned by channel loss, routed to
D0/D1 with probability (1 +/- V s_k)/2, and detected with efficiency eta_d.
Poisson thinning makes this identical in distribution to the coherent-state
description, and it gives exact photon bookkeeping (emitted, intercepted,
lost, detected) for the GUI.

Detectors: per-slot dark-count probability, non-paralysable dead time,
random bit on double clicks. Afterpulsing, gating and jitter are roadmap
item 4.

Eavesdropper: a deliberately simple teaching model (random-phase
intercept-resend per photon, as in the baseline). Realistic DPS attacks are
roadmap item 13.

The stream is processed in chunks with state carried across chunk
boundaries, so there are no measurement windows and no window-edge effects:
long runs behave as continuous time-tagging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from quila_sim.core.link import DetectionRecord, LinkParams, LinkPerformance

__all__ = ["CoherentDPSLink", "DPSRun", "DPSTrace", "closed_form_click_probabilities"]


def _dead_slots(params: LinkParams) -> int:
    """Minimum slot separation between two clicks of one detector (>= 1)."""
    return max(1, math.ceil(params.detector.dead_time_s * params.rep_rate_hz - 1e-9))


def closed_form_click_probabilities(params: LinkParams) -> tuple[float, float]:
    """Exact per-slot probabilities (P(any click), P(error click)) without dead time.

    Averaged over Alice's bit value. Errors include half of all double clicks
    (they get a random bit).
    """
    m = params.mean_photon_number * params.transmittance * params.detector.efficiency
    v = params.visibility
    pd = params.detector.dark_count_prob
    q_right = 1.0 - (1.0 - pd) * math.exp(-m * (1 + v) / 2)   # correct detector clicks
    q_wrong = 1.0 - (1.0 - pd) * math.exp(-m * (1 - v) / 2)   # wrong detector clicks
    p_click = 1.0 - (1.0 - q_right) * (1.0 - q_wrong)
    p_error = q_wrong * (1.0 - q_right) + 0.5 * q_right * q_wrong
    return p_click, p_error


@dataclass
class DPSTrace:
    """Per-slot detail for the first few slots of a run (for display/export).

    slot     Global slot index (1 .. n-1).
    bit      Alice's key bit for the slot (phase difference).
    outcome  Bob's result: -1 no click, 0 or 1.
    dark     True if the click had no photon behind it (dark count only).
    eve      True if a photon resent by Eve was detected in this slot.
    """

    slot: np.ndarray
    bit: np.ndarray
    outcome: np.ndarray
    dark: np.ndarray
    eve: np.ndarray


@dataclass
class DPSRun:
    """Everything one coherent-engine run produced."""

    n_pulses: int
    rep_rate_hz: float
    slots: np.ndarray            # sifted slot indices (int64)
    alice_bits: np.ndarray       # uint8
    bob_bits: np.ndarray         # uint8
    photons_emitted: int = 0
    photons_intercepted: int = 0
    photons_arrived: int = 0     # reached Bob's interferometer
    signal_clicks: int = 0       # detector firings caused by photons (interior slots)
    dark_clicks: int = 0         # detector firings with no photon (interior slots)
    double_clicks: int = 0       # sifted slots where both detectors fired
    deadtime_blocked: int = 0    # firings suppressed by dead time
    edge_photons: int = 0        # unpaired early/late photons at the stream ends
    edge_clicks: int = 0         # firings in the two unpaired edge slots
    eve_slots: int = 0           # sifted slots where an Eve photon was detected
    trace: DPSTrace | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def sifted_bits(self) -> int:
        return int(self.slots.size)

    @property
    def bit_errors(self) -> int:
        return int(np.count_nonzero(self.alice_bits != self.bob_bits))

    @property
    def qber(self) -> float:
        return self.bit_errors / self.sifted_bits if self.sifted_bits else 0.0

    @property
    def duration_s(self) -> float:
        return self.n_pulses / self.rep_rate_hz

    @property
    def raw_clicks(self) -> int:
        """All detector firings, including discarded edge clicks."""
        return self.signal_clicks + self.dark_clicks + self.edge_clicks


def _bernoulli_positions(rng: np.random.Generator, p: float, n: int) -> np.ndarray:
    """Sorted indices in [0, n) where an independent Bernoulli(p) trial succeeds."""
    if p <= 0.0 or n <= 0:
        return np.empty(0, dtype=np.int64)
    k = int(rng.binomial(n, min(p, 1.0)))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)


def _apply_dead_time(slots: np.ndarray, dead: int, last: int) -> tuple[np.ndarray, int]:
    """Non-paralysable dead time on sorted global click slots of one detector.

    Returns a keep-mask and the slot of the last kept click (carried to the
    next chunk). ``dead`` is the minimum separation in slots.
    """
    keep = np.ones(slots.size, dtype=bool)
    if dead <= 1 or slots.size == 0:
        return keep, (int(slots[-1]) if slots.size else last)
    for i, s in enumerate(slots.tolist()):
        if s - last < dead:
            keep[i] = False
        else:
            last = s
    return keep, last


class CoherentDPSLink:
    """Vectorised DPS link model implementing :class:`quila_sim.core.LinkModel`.

    Parameters
    ----------
    chunk_slots
        Slots processed per vectorised chunk (memory/speed trade-off). Results
        for a given seed depend on this value; keep the default for
        reproducibility.
    phase_error_prob
        Probability that Alice's phase modulator puts a pulse in the wrong
        phase. Affects Bob's routing but not Alice's record.
    """

    name = "dps-coherent"
    fidelity = "statistical"

    def __init__(self, chunk_slots: int = 1 << 20, phase_error_prob: float = 0.0) -> None:
        if chunk_slots < 16:
            raise ValueError("chunk_slots too small")
        if not 0.0 <= phase_error_prob <= 1.0:
            raise ValueError("phase_error_prob must be in [0, 1]")
        self.chunk_slots = int(chunk_slots)
        self.phase_error_prob = float(phase_error_prob)

    # ------------------------------------------------------------------ #
    # Closed form
    # ------------------------------------------------------------------ #
    def performance(self, params: LinkParams) -> LinkPerformance:
        """Expected sifted rate and QBER.

        Exact without dead time and without phase errors. With dead time, a
        renewal correction per detector is applied (approximate).
        """
        self._check(params)
        p_click, p_error = closed_form_click_probabilities(params)
        notes: list[str] = []
        dead = _dead_slots(params)
        if dead > 1:
            q_det = p_click / 2.0  # average firing probability per detector
            factor = 1.0 / (1.0 + (dead - 1) * q_det)
            p_click *= factor
            p_error *= factor
            notes.append(f"dead time blocks {dead - 1} slot(s) per click (renewal approx.)")
        if self.phase_error_prob > 0:
            notes.append("phase errors not included in the closed form")
        rate = params.rep_rate_hz * p_click
        qber = p_error / p_click if p_click > 0 else 0.0
        return LinkPerformance(
            raw_rate_bps=rate,
            sifted_rate_bps=rate,  # DPS: every interior click is sifted
            qber=qber,
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(
        self, params: LinkParams, n_pulses: int, rng: np.random.Generator
    ) -> DetectionRecord:
        run = self.simulate(params, n_pulses, rng)
        return DetectionRecord(run.slots, run.alice_bits, run.bob_bits, n_pulses)

    def simulate(
        self,
        params: LinkParams,
        n_pulses: int,
        rng: np.random.Generator,
        *,
        eve_intercept_probability: float = 0.0,
        trace_slots: int = 0,
    ) -> DPSRun:
        """Simulate ``n_pulses`` pulses as one continuous stream."""
        self._check(params)
        if n_pulses < 2:
            raise ValueError("need at least 2 pulses (one interference slot)")
        p_eve = float(eve_intercept_probability)
        if not 0.0 <= p_eve <= 1.0:
            raise ValueError("eve_intercept_probability must be in [0, 1]")

        mu = params.mean_photon_number
        t_ch = params.transmittance
        eta_d = params.detector.efficiency
        vis = params.visibility
        p_dark = params.detector.dark_count_prob
        dead = _dead_slots(params)

        n_slots = n_pulses - 1
        trace_n = max(0, min(int(trace_slots), n_slots))

        slots_out: list[np.ndarray] = []
        alice_out: list[np.ndarray] = []
        bob_out: list[np.ndarray] = []
        tr_bit = np.zeros(trace_n, dtype=np.uint8)
        tr_out = np.full(trace_n, -1, dtype=np.int8)
        tr_dark = np.zeros(trace_n, dtype=bool)
        tr_eve = np.zeros(trace_n, dtype=bool)

        run = DPSRun(n_pulses=n_pulses, rep_rate_hz=params.rep_rate_hz,
                     slots=np.empty(0, np.int64), alice_bits=np.empty(0, np.uint8),
                     bob_bits=np.empty(0, np.uint8))

        last_phase = int(rng.integers(0, 2))          # phi_0
        last_err = bool(rng.random() < self.phase_error_prob) if self.phase_error_prob else False
        last_click = [-(10**18), -(10**18)]           # dead-time state per detector

        start = 0  # chunk-local slot j corresponds to global slot start + j + 1
        while start < n_slots:
            c = min(self.chunk_slots, n_slots - start)
            g = np.arange(start + 1, start + c + 1, dtype=np.int64)  # global slots

            # Alice's phases for pulses start+1 .. start+c; previous pulse carried over
            phases = rng.integers(0, 2, size=c, dtype=np.uint8)
            prev = np.empty(c, dtype=np.uint8)
            prev[0] = last_phase
            prev[1:] = phases[:-1]
            bits = phases ^ prev                       # Alice's key bit per slot
            last_phase = int(phases[-1])

            if self.phase_error_prob > 0:
                err = rng.random(c) < self.phase_error_prob
                prev_err = np.empty(c, dtype=bool)
                prev_err[0] = last_err
                prev_err[1:] = err[:-1]
                routed_bits = bits ^ (err ^ prev_err).astype(np.uint8)
                last_err = bool(err[-1])
            else:
                routed_bits = bits

            # Photons per slot, Eve, channel loss -- on the sparse non-vacuum subset
            n_ph = rng.poisson(mu, size=c)
            run.photons_emitted += int(n_ph.sum())
            idx = np.flatnonzero(n_ph)
            n_k = n_ph[idx]
            e_k = rng.binomial(n_k, p_eve) if p_eve > 0 else np.zeros_like(n_k)
            run.photons_intercepted += int(e_k.sum())
            g_arr = rng.binomial(n_k - e_k, t_ch)
            e_arr = rng.binomial(e_k, t_ch) if p_eve > 0 else np.zeros_like(n_k)
            arrived = g_arr + e_arr
            run.photons_arrived += int(arrived.sum())
            sel = arrived > 0
            idx, g_arr, e_arr = idx[sel], g_arr[sel], e_arr[sel]

            # DLI routing and detection
            s = np.where(routed_bits[idx] == 0, 1.0, -1.0)
            p0 = 0.5 * (1.0 + vis * s)
            g0 = rng.binomial(g_arr, p0)
            g1 = g_arr - g0
            e0 = rng.binomial(e_arr, 0.5) if p_eve > 0 else np.zeros_like(g0)
            e1 = e_arr - e0
            gd0 = rng.binomial(g0, eta_d) > 0
            gd1 = rng.binomial(g1, eta_d) > 0
            ed0 = rng.binomial(e0, eta_d) > 0 if p_eve > 0 else np.zeros_like(gd0)
            ed1 = rng.binomial(e1, eta_d) > 0 if p_eve > 0 else np.zeros_like(gd1)
            sig = [idx[gd0 | ed0], idx[gd1 | ed1]]
            eve_hit = idx[ed0 | ed1]

            # Dark counts, then dead time, per detector
            fired: list[np.ndarray] = []
            for d in (0, 1):
                dark = _bernoulli_positions(rng, p_dark, c)
                all_d = np.union1d(sig[d], dark)
                keep, last_click[d] = _apply_dead_time(g[all_d], dead, last_click[d])
                run.deadtime_blocked += int(np.count_nonzero(~keep))
                kept = all_d[keep]
                is_sig = np.isin(kept, sig[d], assume_unique=True)
                run.signal_clicks += int(np.count_nonzero(is_sig))
                run.dark_clicks += int(np.count_nonzero(~is_sig))
                fired.append(kept)

            # Sifting: every slot with a click; double clicks -> random bit
            c0, c1 = fired
            clicked = np.union1d(c0, c1)
            in0 = np.isin(clicked, c0, assume_unique=True)
            in1 = np.isin(clicked, c1, assume_unique=True)
            both = in0 & in1
            bob = np.where(in1, 1, 0).astype(np.uint8)
            if both.any():
                bob[both] = rng.integers(0, 2, size=int(both.sum()), dtype=np.uint8)
            run.double_clicks += int(both.sum())

            slots_out.append(g[clicked])
            alice_out.append(bits[clicked])
            bob_out.append(bob)
            run.eve_slots += int(np.isin(clicked, eve_hit, assume_unique=True).sum())

            # Trace for the first slots of the run
            if start < trace_n:
                m = min(c, trace_n - start)
                tr_bit[start:start + m] = bits[:m]
                inside = clicked < m
                local = clicked[inside]
                tr_out[start + local] = bob[inside].astype(np.int8)
                sig_any = np.union1d(sig[0], sig[1])
                tr_dark[start + local] = ~np.isin(local, sig_any)
                tr_eve[start + local] = np.isin(local, eve_hit)

            start += c

        # Unpaired light at the two ends of the stream (never sifted)
        for _ in range(2):
            n_edge = int(rng.poisson(mu / 2))
            run.photons_emitted += n_edge
            e_edge = int(rng.binomial(n_edge, p_eve)) if p_eve > 0 else 0
            run.photons_intercepted += e_edge
            arrived_edge = int(rng.binomial(n_edge, t_ch))
            run.edge_photons += arrived_edge
            run.photons_arrived += arrived_edge
            to_d0 = int(rng.binomial(arrived_edge, 0.5))
            for n_det in (to_d0, arrived_edge - to_d0):
                photon_click = rng.binomial(n_det, eta_d) > 0
                dark_click = rng.random() < p_dark
                run.edge_clicks += int(photon_click or dark_click)
        run.slots = np.concatenate(slots_out) if slots_out else np.empty(0, np.int64)
        run.alice_bits = np.concatenate(alice_out) if alice_out else np.empty(0, np.uint8)
        run.bob_bits = np.concatenate(bob_out) if bob_out else np.empty(0, np.uint8)
        if trace_n:
            run.trace = DPSTrace(np.arange(1, trace_n + 1, dtype=np.int64),
                                 tr_bit, tr_out, tr_dark, tr_eve)
        if dead > 1:
            run.notes.append(f"dead time: {dead - 1} slot(s) blocked after each click")
        return run

    # ------------------------------------------------------------------ #
    @staticmethod
    def _check(params: LinkParams) -> None:
        if params.protocol.upper() != "DPS":
            raise ValueError(f"CoherentDPSLink only models DPS, got {params.protocol!r}")
