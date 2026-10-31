"""Tests for the time-bin DPS model ("Model C") of the MAQAN / IIT Madras hardware.

One pulse of 3 time bins every 32 ns; Bob's one-bin-delay interferometer
gives 4 output slots in the ratio 1/6 : 1/3 : 1/3 : 1/6; one time-multiplexed
gated detector. Reference: Shaw et al., arXiv:2008.03083 (IEEE Photonics J.
2022), and Sharma et al., Quantum Inf. Process. 25:19 (2026).
"""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from quila_sim.core import (
    MAQAN_DPS,
    DetectionRecord,
    DPSReceiver,
    FrameLayout,
    LinkModel,
    PulseFormat,
    maqan_link_params,
)
from quila_sim.core.timebin import (
    EARLY,
    GUARD,
    KEY,
    LATE,
    OUTSIDE,
    classify_times,
    exit_probabilities,
    leak_exit_table,
    pulse_amplitudes,
)
from quila_sim.physics import TimeBinDPSLink

ENGINE = TimeBinDPSLink()

#: Every imperfection switched off (MAQAN pulse format and receiver layout kept).
IDEAL = dict(visibility=1.0, extinction_ratio_db=None, pm_transition_s=0.0,
             timing_jitter_s=0.0, dark_count_prob=0.0)


def _phase_patterns(n=3):
    codes = np.arange(2 ** n)
    return ((codes[:, None] >> np.arange(n)) & 1).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Interferometer physics and timing
# --------------------------------------------------------------------------- #

def test_slot_split_is_one_sixth_one_third_one_third_one_sixth():
    bits = _phase_patterns()
    p = exit_probabilities(pulse_amplitudes(bits, np.full(3, 1 / 3)), visibility=1.0)
    assert p.shape == (8, 4, 2)
    np.testing.assert_allclose(p.sum(axis=2), np.tile([1 / 6, 1 / 3, 1 / 3, 1 / 6], (8, 1)))
    np.testing.assert_allclose(p.sum(axis=(1, 2)), 1.0)
    for row, b in zip(p, bits, strict=True):
        for j in (1, 2):                      # interference slots: port = phase difference
            diff = b[j] ^ b[j - 1]
            assert row[j, diff] == pytest.approx(1 / 3)
            assert row[j, 1 - diff] == pytest.approx(0.0)
        np.testing.assert_allclose(row[0], [1 / 12, 1 / 12])   # early/late: no interference
        np.testing.assert_allclose(row[3], [1 / 12, 1 / 12])


def test_visibility_sends_one_minus_v_over_two_to_the_wrong_port():
    p = exit_probabilities(pulse_amplitudes(np.zeros(3, np.uint8), np.full(3, 1 / 3)), 0.92)
    assert p[1, 1] / p[1].sum() == pytest.approx((1 - 0.92) / 2)


def test_maqan_frame_layout():
    lay = FrameLayout.build(PulseFormat.maqan(), DPSReceiver.maqan(), MAQAN_DPS["pulse_rate_hz"])
    assert lay.frame_bins == 32
    assert lay.port_offset == (0, 10) and lay.detector_of_port == (0, 0)
    assert lay.gate_width == 14.0              # port-0 window 0-4 ns, port-1 window 10-14 ns
    assert lay.blind_gates(10e-6, 7.0) == 312  # 10 us hold-off at 32 ns per gate
    assert DPSReceiver.maqan().port_transmittance == pytest.approx(10 ** -0.35)


def test_invalid_formats_are_rejected():
    fmt = PulseFormat()
    with pytest.raises(ValueError, match="continuous pulse train"):
        fmt.frame_bins(1e9)                            # no empty bin: that is Model B
    with pytest.raises(ValueError, match="whole number"):
        fmt.frame_bins(30e6)                           # 33.3 ns is not a whole number of bins
    with pytest.raises(ValueError, match="too short"):
        FrameLayout.build(fmt, DPSReceiver(tdm_delay_s=3e-9), 31.25e6)
    with pytest.raises(ValueError, match="too long"):
        FrameLayout.build(fmt, DPSReceiver(tdm_delay_s=10e-9), 1 / 12e-9)


def test_time_stamp_classification():
    lay = FrameLayout.build(PulseFormat(), DPSReceiver(guard_s=200e-12), 31.25e6)
    t = np.array([0.5, 1.5, 2.95, 3.5, 7.0, 10.5, 11.5, 12.05, 13.5, 14.5])
    kind, port, slot = classify_times(t, np.zeros(t.size, int), lay)
    assert kind.tolist() == [EARLY, KEY, GUARD, LATE, OUTSIDE, EARLY, KEY, GUARD, LATE, OUTSIDE]
    assert port.tolist() == [0, 0, 0, 0, -1, 1, 1, 1, 1, -1]
    assert slot.tolist() == [0, 1, 2, 3, -1, 0, 1, 2, 3, -1]


def test_im_leakage_lands_in_the_port1_key_slots():
    lay = FrameLayout.build(PulseFormat(), DPSReceiver(), 31.25e6)
    fmt = PulseFormat(extinction_ratio_db=15.0)
    starts, ports, means = leak_exit_table(fmt, lay, 1.0, 0.2)
    per_bin = 10 ** -1.5 * 0.2 / 3             # leaked photons per empty bin
    in_port1_key_slots = means[(ports == 0) & ((starts == 11) | (starts == 12))]
    np.testing.assert_allclose(in_port1_key_slots, per_bin)
    # V = 1: leaked light interferes constructively at port 0, so none of it
    # reaches the key slots through port 1
    in_key_slots = np.isin(starts, [1, 2, 11, 12])
    assert means[(ports == 1) & in_key_slots].sum() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Engine: physics, closed form, calibration
# --------------------------------------------------------------------------- #

def test_ideal_hardware_shows_the_slot_split_and_two_key_bits_per_three():
    params = maqan_link_params(0.0, mean_photon_number=0.05, detector_efficiency=1.0,
                               holdoff_s=0.0, **IDEAL)
    run = ENGINE.simulate(params, 2_000_000, np.random.default_rng(1))
    assert run.qber == 0.0
    assert run.kind_counts["key"] / run.clicks == pytest.approx(2 / 3, abs=0.01)
    h, e = run.arrival_hist, run.hist_edges_bins
    for start in (0, 10):
        slots = np.array([h[(e[:-1] >= start + s) & (e[:-1] < start + s + 1)].sum()
                          for s in range(4)], float)
        np.testing.assert_allclose(slots / slots.sum(), [1 / 6, 1 / 3, 1 / 3, 1 / 6], atol=0.012)


@pytest.mark.parametrize("label, overrides, expected", [
    ("visibility 0.92", dict(visibility=0.92), 0.040),
    ("SPD timing jitter", dict(timing_jitter_s=MAQAN_DPS["timing_jitter_s"]), 0.050),
    ("PM rise/fall time", dict(pm_transition_s=MAQAN_DPS["pm_transition_s"]), 0.021),
    ("IM extinction ratio", dict(extinction_ratio_db=MAQAN_DPS["extinction_ratio_db"]), 0.016),
])
def test_each_imperfection_reproduces_shaw_table_iii(label, overrides, expected):
    """Shaw et al. (2022), Table III, 1 ns bins: the QBER caused by each
    imperfection alone. MAQAN_DPS is calibrated to these values."""
    settings = {**IDEAL, **overrides}
    params = maqan_link_params(0.0, holdoff_s=0.0, **settings)
    assert ENGINE.performance(params).qber == pytest.approx(expected, abs=5e-4), label


def test_temporal_filter_matches_shaw():
    """Shaw et al.: a 200 ps guard band discards ~20% of the key bits and
    lowers the QBER by ~3 points (0.12 -> 0.09; afterpulsing not modelled here)."""
    plain = ENGINE.performance(maqan_link_params(7.5))
    guarded = ENGINE.performance(maqan_link_params(7.5, guard_s=200e-12))
    assert guarded.sifted_rate_bps / plain.sifted_rate_bps == pytest.approx(0.8, abs=0.03)
    assert 0.02 < plain.qber - guarded.qber < 0.035


def test_maqan_iitm_ernet_prediction():
    """IITM-ERNET (7.5 dB measured): ~21 kbps sifted at ~11% QBER; MAQAN
    reports a raw key rate of about 19-21 kbps on this link."""
    perf = ENGINE.performance(maqan_link_params(7.5))
    assert 19e3 < perf.sifted_rate_bps < 23e3
    assert 0.10 < perf.qber < 0.12


def test_closed_form_matches_sampling_at_maqan_settings():
    params = maqan_link_params(7.5)
    perf = ENGINE.performance(params)
    run = ENGINE.simulate(params, 31_250_000, np.random.default_rng(2))    # one second
    assert run.click_rate_hz == pytest.approx(perf.raw_rate_bps, rel=0.03)
    assert run.sifted_rate_bps == pytest.approx(perf.sifted_rate_bps, rel=0.03)
    sigma = math.sqrt(perf.qber * (1 - perf.qber) / run.sifted_bits)
    assert abs(run.qber - perf.qber) < 4 * sigma


@pytest.mark.parametrize("layout", ["tdm", "dual"])
def test_closed_form_matches_sampling_at_high_click_rates(layout):
    """Many photons per gate and a short hold-off: the closed form is exact per
    gate (first event wins), not just first order."""
    params = maqan_link_params(0.0, mean_photon_number=1.5, detector_efficiency=0.6,
                               holdoff_s=150e-9, layout=layout)
    perf = ENGINE.performance(params)
    run = ENGINE.simulate(params, 1_000_000, np.random.default_rng(3))
    assert run.click_rate_hz == pytest.approx(perf.raw_rate_bps, rel=0.01)
    assert run.sifted_rate_bps == pytest.approx(perf.sifted_rate_bps, rel=0.015)
    assert run.qber == pytest.approx(perf.qber, abs=0.01)


def test_full_intercept_resend_gives_fifty_percent_errors():
    params = maqan_link_params(7.5)
    assert ENGINE.performance(params, eve_intercept_probability=1.0).qber == pytest.approx(0.5)
    run = ENGINE.simulate(params, 10_000_000, np.random.default_rng(4),
                          eve_intercept_probability=1.0)
    assert run.qber == pytest.approx(0.5, abs=4 * math.sqrt(0.25 / run.sifted_bits))
    assert run.pulses_intercepted == 10_000_000


def test_holdoff_limits_the_click_rate():
    """After each click the detector skips 312 gates: rate = R q / (1 + 312 q)."""
    params = maqan_link_params(0.0, mean_photon_number=1.0, detector_efficiency=1.0, **IDEAL)
    free = ENGINE.performance(maqan_link_params(0.0, mean_photon_number=1.0,
                                                detector_efficiency=1.0, holdoff_s=0.0,
                                                **IDEAL))
    q = free.raw_rate_bps / MAQAN_DPS["pulse_rate_hz"]
    held = ENGINE.performance(params)
    assert held.raw_rate_bps == pytest.approx(MAQAN_DPS["pulse_rate_hz"] * q / (1 + 312 * q))
    assert held.raw_rate_bps < 1 / MAQAN_DPS["holdoff_s"]


def test_link_model_contract_and_key_slot_ids():
    assert isinstance(ENGINE, LinkModel)
    rec = ENGINE.sample(maqan_link_params(0.0), 200_000, np.random.default_rng(5))
    assert isinstance(rec, DetectionRecord) and rec.n_pulses == 200_000
    j = rec.slots % 3
    assert set(np.unique(j)) <= {1, 2}         # only interference slots carry key bits
    assert np.all(np.diff(rec.slots) > 0)      # time order, no slot twice


def test_maqan_hardware_facts_match_the_scenario_file():
    """scenarios/maqan.yaml lists the same published DPS hardware values."""
    path = Path(__file__).resolve().parents[1] / "scenarios" / "maqan.yaml"
    data = yaml.safe_load(path.read_text())
    hardware = data["dps_hardware"]
    assert set(hardware) == set(MAQAN_DPS)
    for key, value in MAQAN_DPS.items():
        entry = hardware[key]
        if isinstance(value, float):
            assert float(entry["value"]) == pytest.approx(value), key
        else:
            assert entry["value"] == value, key
        assert entry["source"], key
