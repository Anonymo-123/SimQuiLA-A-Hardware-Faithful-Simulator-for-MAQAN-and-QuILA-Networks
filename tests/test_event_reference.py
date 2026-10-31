"""Tests for the SeQUeNCe per-photon reference path of the time-bin DPS model
(Model C, the MAQAN hardware): hardware components, the time-bin sifting
protocol, and agreement with the fast engine."""

import math

import numpy as np
import pytest

from backend.adapter import SimulationParameters, expected_performance, run_simulation
from quila_sim.core import DPSReceiver, FrameLayout, PulseFormat
from sequence.components.dps_hardware import TimeBinPhoton
from sequence.components.photon import Photon
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.qkd.DPSQKD import alice_bits_for, classify_clicks, sift_detections
from sequence.topology.dps_topo import DPSQKDTopo
from sequence.utils.dps_encoding import dps_timebin

SEEDS = dict(alice_seed=11, bob_seed=12, eve_seed=13, visibility_seed=14)


def test_sequence_safety_checks_are_still_active():
    """The N-bin state lives on TimeBinPhoton, so SeQUeNCe's own qubit checks
    stay switched on (the v0.1 baseline had disabled them)."""
    tl = Timeline()
    with pytest.raises(AssertionError):
        Photon("bad", tl, encoding_type=dps_timebin, quantum_state=(1, 0, 0))
    amp = 3 ** -0.5
    photon = TimeBinPhoton("ok", tl, (amp, -amp, amp))
    assert len(photon.amplitudes) == 3 and photon.kind == "signal"
    with pytest.raises(ValueError):
        TimeBinPhoton("unnormalised", tl, (1.0, 1.0, 1.0))


def test_time_stamp_sifting_rule():
    """Bob reads pulse, slot and port window from the time stamp alone."""
    lay = FrameLayout.build(PulseFormat(), DPSReceiver(), 31.25e6)
    t0 = 1_000_000                        # bin 0 of pulse 0 reaches Bob (ps)
    times = [
        t0 + 500,                         # pulse 0, port-0 window, early slot -> no bit
        t0 + 1_500,                       # pulse 0, port 0, slot 1 -> bit 0
        t0 + 32_000 + 12_400,             # pulse 1, port 1, slot 2 -> bit 1
        t0 + 32_000 + 7_000,              # pulse 1, between the windows -> no bit
        t0 + 5 * 32_000 + 1_500,          # pulse 5: outside a 3-pulse window -> no bit
    ]
    pulse, kind, port, slot, _ = classify_clicks(times, [0] * 5, t0, lay, 3)
    assert pulse.tolist() == [0, 0, 1, 1, -1]
    assert slot.tolist() == [0, 1, 2, -1, -1]
    ids, bits = sift_detections(times, [0] * 5, t0, lay, 3, np.random.default_rng(0))
    assert ids == [1, 5] and bits == [0, 1]          # key-slot ids k*3 + j
    blocks = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]], np.uint8)
    assert alice_bits_for(blocks, ids) == [0, 1]     # phi_j XOR phi_(j-1)


def test_double_clicks_get_a_random_bit():
    """With one detector per port, both detectors firing in one key slot give a random bit."""
    lay = FrameLayout.build(PulseFormat(), DPSReceiver(layout="dual"), 31.25e6)
    times, dets = [1_500, 1_600], [0, 1]
    outcomes = {tuple(sift_detections(times, dets, 0, lay, 1, np.random.default_rng(s))[1])
                for s in range(20)}
    assert outcomes == {(0,), (1,)}


def test_event_engine_shows_the_slot_split():
    params = SimulationParameters.ideal_hardware(
        engine="event", num_pulses=150_000, channel_loss_db=0.0, mean_photon_number=0.3,
        detector_efficiency=1.0, holdoff_s=0.0, **SEEDS)
    r = run_simulation(params)
    assert r.bit_errors == 0
    assert r.key_slot_clicks / r.raw_detector_clicks == pytest.approx(2 / 3, abs=0.02)
    h, e = np.array(r.arrival_hist), np.array(r.hist_edges_ns)
    for start in (0.0, 10.0):
        slots = np.array([h[(e[:-1] >= start + s) & (e[:-1] < start + s + 1)].sum()
                          for s in range(4)], float)
        np.testing.assert_allclose(slots / slots.sum(), [1 / 6, 1 / 3, 1 / 3, 1 / 6], atol=0.02)


def test_event_engine_agrees_with_the_closed_form():
    """MAQAN imperfections (jitter, PM transitions, IM leakage, dark counts) on a
    lossless link with a better detector, so that one run has enough key bits."""
    params = SimulationParameters(engine="event", num_pulses=300_000, channel_loss_db=0.0,
                                  detector_efficiency=0.5, holdoff_s=0.2e-6, **SEEDS)
    r = run_simulation(params)
    expected = expected_performance(params)
    bits = expected.sifted_rate_bps * r.duration_s
    assert abs(r.sifted_key_bits - bits) < 4 * math.sqrt(bits)
    q = expected.qber
    assert abs(r.qber - q) < 4 * math.sqrt(q * (1 - q) / r.sifted_key_bits)
    assert len(r.alice_key_bits) == len(r.bob_key_bits)


@pytest.mark.slow
def test_both_engines_give_the_same_physics():
    common = dict(num_pulses=600_000, channel_loss_db=2.0, mean_photon_number=0.4,
                  detector_efficiency=0.3, holdoff_s=1e-6)
    totals = {}
    for engine in ("timebin", "event"):
        bits = errors = 0
        for seed in range(2):
            r = run_simulation(SimulationParameters(
                engine=engine, alice_seed=seed, bob_seed=seed + 10, eve_seed=seed + 20,
                visibility_seed=seed + 30, **common))
            bits += r.sifted_key_bits
            errors += r.bit_errors
        totals[engine] = (bits, errors / bits)
    (b1, q1), (b2, q2) = totals["timebin"], totals["event"]
    assert abs(b1 - b2) < 4 * math.sqrt(b1 + b2)
    assert abs(q1 - q2) < 4 * math.sqrt(q1 * (1 - q1) * (1 / b1 + 1 / b2))


def test_eve_on_the_event_engine():
    r = run_simulation(SimulationParameters(
        engine="event", num_pulses=100_000, channel_loss_db=0.0, detector_efficiency=0.5,
        holdoff_s=0.2e-6, eve_enabled=True, eve_intercept_probability=1.0, **SEEDS))
    assert r.pulses_intercepted == 100_000
    assert r.photons_intercepted == r.photons_emitted
    assert r.qber == pytest.approx(0.5, abs=4 * math.sqrt(0.25 / r.sifted_key_bits))


def test_multi_window_keys_stay_aligned():
    """The DPSQKD protocol over several windows: the detector gates move with
    every window, so Alice's and Bob's keys stay aligned (QBER ~11%)."""
    topo = DPSQKDTopo.from_nodes(
        "alice", "bob", distance=4.8e3, attenuation=2.0 / 4.8e3,
        ls_params={"mean_photon_num": 0.2, "extinction_ratio_db": 14.8},
        dli_params={"visibility": 0.92, "pm_transition_ps": 42, "timing_jitter_ps": 162,
                    "efficiency": 0.1, "dark_count_prob": 3e-6},
        stack_size=1, runtime=5e12, alice_seed=1, bob_seed=2,
    )
    tl = topo.get_timeline()
    alice = topo.get_node("alice").protocol_stack[0]
    tl.schedule(Event(0, Process(alice, "push", [128, 3, math.inf])))
    tl.init()
    tl.run()
    assert len(alice.error_rates) == 3
    assert max(alice.error_rates) < 0.25
    assert np.mean(alice.error_rates) == pytest.approx(0.11, abs=0.06)


def test_dark_counts_only_inside_the_gates_of_the_window():
    """Dark counts are drawn per gate and only for the window's gates; without
    light, every click is a dark count inside a port window or between them."""
    params = SimulationParameters(engine="event", num_pulses=20_000, mean_photon_number=1e-4,
                                  dark_count_prob=0.02, holdoff_s=0.0, **SEEDS)
    r = run_simulation(params)
    expected = 0.02 * 20_000
    assert abs(r.dark_counts - expected) < 4 * math.sqrt(expected)
    assert r.raw_detector_clicks == r.dark_counts + r.photons_detected + r.leak_clicks
    assert r.discarded_click_count < 3 * max(r.dark_counts, 1)


def test_event_engine_pulse_limit():
    with pytest.raises(ValueError):
        run_simulation(SimulationParameters(engine="event", num_pulses=50_000_000))
