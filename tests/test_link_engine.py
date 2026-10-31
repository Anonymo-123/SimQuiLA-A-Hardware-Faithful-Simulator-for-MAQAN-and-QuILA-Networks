"""Physics regression tests for the coherent-state DPS engine (roadmap item 1).

Every expected value comes from the closed-form coherent-state model in
``quila_sim.physics.dps_coherent`` (per-slot click probabilities), or from
elementary probability (renewal theory for dead time, symmetric dark counts).
"""

import math

import numpy as np
import pytest

from quila_sim.core import DetectionRecord, DetectorParams, LinkModel, LinkParams
from quila_sim.physics import CoherentDPSLink, closed_form_click_probabilities


def _params(loss_db=0.0, visibility=1.0, dark=0.0, dead_time_s=0.0, mu=0.2, eta=0.1, rate=1e6):
    return LinkParams(
        protocol="DPS",
        rep_rate_hz=rate,
        mean_photon_number=mu,
        channel_loss_db=loss_db,
        detector=DetectorParams(efficiency=eta, dark_count_prob=dark, dead_time_s=dead_time_s),
        visibility=visibility,
    )


def test_engine_implements_link_model_contract():
    engine = CoherentDPSLink()
    assert isinstance(engine, LinkModel)
    record = engine.sample(_params(), 10_000, np.random.default_rng(0))
    assert isinstance(record, DetectionRecord)
    assert record.n_pulses == 10_000


@pytest.mark.parametrize("loss_db", [0.0, 2.0, 5.0])
def test_sifted_rate_matches_coherent_state_theory(loss_db):
    """Every interior slot clicks with probability ~mu*T*eta.

    The baseline 3-pulse photon model gave about half of this.
    """
    n = 2_000_000
    params = _params(loss_db)
    run = CoherentDPSLink().simulate(params, n, np.random.default_rng(1))
    p_click, _ = closed_form_click_probabilities(params)
    expected = p_click * (n - 1)
    assert abs(run.sifted_bits - expected) < 5 * math.sqrt(expected)
    assert run.sifted_bits / expected == pytest.approx(1.0, abs=0.03)
    assert run.qber == 0.0  # perfect visibility, no dark counts


def test_qber_matches_theory_with_visibility_and_dark_counts():
    params = _params(loss_db=7.5, visibility=0.95, dark=2e-4)
    run = CoherentDPSLink().simulate(params, 3_000_000, np.random.default_rng(2))
    p_click, p_error = closed_form_click_probabilities(params)
    q = p_error / p_click
    sigma = math.sqrt(q * (1 - q) / run.sifted_bits)
    assert abs(run.qber - q) < 4 * sigma


def test_performance_agrees_with_sampling():
    params = _params(loss_db=7.5, visibility=0.98, dark=2e-4)
    engine = CoherentDPSLink()
    perf = engine.performance(params)
    run = engine.simulate(params, 3_000_000, np.random.default_rng(3))
    assert run.sifted_bits / run.duration_s == pytest.approx(perf.sifted_rate_bps, rel=0.03)
    assert run.qber == pytest.approx(perf.qber, abs=0.01)
    assert perf.secure_rate_bps is None  # secure rates come with item 8


def test_double_clicks_get_random_bits():
    """With dark counts only, Bob's bits must be balanced.

    The baseline kept detector 0 on double clicks, giving ~59% zeros here.
    """
    params = _params(loss_db=60.0, dark=0.3)
    run = CoherentDPSLink().simulate(params, 400_000, np.random.default_rng(4))
    ones = float(run.bob_bits.mean())
    assert abs(ones - 0.5) < 0.01
    assert run.double_clicks == pytest.approx(0.09 * 399_999, rel=0.05)


def test_dead_time_follows_renewal_theory():
    """Non-paralysable dead time of D slots: firing rate q / (1 + (D-1) q)."""
    q, dead_slots, n = 0.05, 10, 400_000
    params = _params(loss_db=60.0, dark=q, dead_time_s=(dead_slots - 0.5) / 1e6)
    run = CoherentDPSLink().simulate(params, n, np.random.default_rng(5))
    expected_per_detector = (n - 1) * q / (1 + (dead_slots - 1) * q)
    assert run.dark_clicks / 2 == pytest.approx(expected_per_detector, rel=0.03)
    assert run.deadtime_blocked > 0


def test_eavesdropper_teaching_model():
    params = _params(loss_db=0.0)
    for p, expected_qber in ((0.5, 0.25), (1.0, 0.5)):
        run = CoherentDPSLink().simulate(params, 1_000_000, np.random.default_rng(6),
                                         eve_intercept_probability=p)
        assert run.photons_intercepted / run.photons_emitted == pytest.approx(p, abs=0.01)
        assert run.qber == pytest.approx(expected_qber, abs=0.02)


def test_stream_is_continuous_across_chunks():
    """Small chunks must give the same statistics and well-formed slot indices."""
    params = _params(loss_db=3.0, visibility=0.97, dark=1e-4)
    n = 500_000
    small = CoherentDPSLink(chunk_slots=4096).simulate(params, n, np.random.default_rng(7))
    large = CoherentDPSLink().simulate(params, n, np.random.default_rng(8))
    for run in (small, large):
        assert np.all(np.diff(run.slots) > 0)
        assert run.slots[0] >= 1 and run.slots[-1] <= n - 1
    assert small.sifted_bits == pytest.approx(large.sifted_bits, rel=0.05)
    assert small.qber == pytest.approx(large.qber, abs=0.01)


def test_same_seed_same_result():
    params = _params(loss_db=2.0, dark=1e-4, visibility=0.97)
    a = CoherentDPSLink().simulate(params, 200_000, np.random.default_rng(9))
    b = CoherentDPSLink().simulate(params, 200_000, np.random.default_rng(9))
    assert np.array_equal(a.slots, b.slots)
    assert np.array_equal(a.bob_bits, b.bob_bits)


def test_trace_matches_sifted_key():
    params = _params(loss_db=0.0, mu=0.5, eta=0.5)
    run = CoherentDPSLink().simulate(params, 50_000, np.random.default_rng(10), trace_slots=500)
    tr = run.trace
    clicked = tr.outcome >= 0
    in_range = run.slots <= 500
    assert np.array_equal(tr.slot[clicked], run.slots[in_range])
    assert np.array_equal(tr.outcome[clicked].astype(np.uint8), run.bob_bits[in_range])
    assert np.array_equal(tr.bit[clicked], run.alice_bits[in_range])


def test_rejects_other_protocols():
    with pytest.raises(ValueError):
        CoherentDPSLink().performance(
            LinkParams("COW", 1e6, 0.2, 1.0, DetectorParams(0.1, 0.0))
        )
