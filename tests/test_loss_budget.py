"""Tests for measured link loss and the MAQAN preset (roadmap item 3)."""

import pytest

from backend.adapter import (
    SimulationParameters,
    run_simulation,
    to_link_params,
    total_channel_loss_db,
)
from quila_sim.core import LossBudget, LossEvent, load_preset
from quila_sim.physics import TimeBinDPSLink


def test_itemised_and_measured_totals():
    budget = LossBudget.from_counts(
        4.8, 0.2, splices=6, splice_loss_db=0.1, connectors=4, connector_loss_db=0.3,
        components={"adm": 1.2}, measured_total_db=7.5,
    )
    assert budget.fibre_db == pytest.approx(0.96)
    assert budget.itemised_db == pytest.approx(0.96 + 0.6 + 1.2 + 1.2)
    assert budget.total_db == pytest.approx(7.5)           # measurement wins
    assert budget.unaccounted_db == pytest.approx(7.5 - 3.96)
    assert budget.by_kind()["connector"] == pytest.approx(1.2)


def test_itemised_total_used_without_measurement():
    budget = LossBudget(2.0, 0.25, (LossEvent("splice", 0.05), LossEvent("connector", 0.4)))
    assert budget.total_db == pytest.approx(0.95)
    assert budget.unaccounted_db is None
    assert budget.transmittance == pytest.approx(10 ** (-0.095))


def test_budget_validation():
    with pytest.raises(ValueError):
        LossBudget.from_counts(1.0, 0.2, splices=3)        # loss per splice missing
    with pytest.raises(ValueError):
        LossBudget(measured_total_db=-1.0)
    with pytest.raises(ValueError):
        LossEvent("teleporter", 1.0)
    # OTDR "gainers" are allowed as long as the total is not negative
    assert LossBudget(1.0, 0.2, (LossEvent("splice", -0.05),)).total_db == pytest.approx(0.15)


def test_budget_dict_round_trip():
    budget = LossBudget(5.6, 0.2, (LossEvent("adm", 1.1, 0.0, "ADM at SETS"),), 3.2, "test")
    assert LossBudget.from_dict(budget.to_dict()) == budget


def test_maqan_preset():
    sc = load_preset("maqan")
    assert len(sc.nodes) == 5
    measured = {
        lk.name: lk.effective_loss_db for lk in sc.links if lk.effective_loss_db is not None
    }
    assert measured == {"IITM-ERNET": 7.5, "ERNET-SETS": 2.0, "SETS-NIC": 3.2}
    iitm = sc.link("IITM-ERNET")
    assert iitm.loss_budget.unaccounted_db == pytest.approx(7.5 - 4.8 * 0.2)
    assert sc.link("SETS-INTRA").effective_loss_db is None   # unknown: never guessed


def test_measured_loss_overrides_distance_times_attenuation():
    params = SimulationParameters(distance_km=4.8, fiber_loss_db_km=0.2, channel_loss_db=7.5)
    assert total_channel_loss_db(params) == pytest.approx(7.5)
    params.channel_loss_db = None
    assert total_channel_loss_db(params) == pytest.approx(0.96)


def test_simulation_uses_measured_loss():
    """MAQAN IITM-ERNET: 4.8 km of fibre (0.96 dB at 0.2 dB/km) but 7.5 dB measured."""
    params = SimulationParameters(
        num_pulses=31_250_000, distance_km=4.8, fiber_loss_db_km=0.2, channel_loss_db=7.5,
        alice_seed=1, bob_seed=2, eve_seed=3, visibility_seed=4,
    )
    r = run_simulation(params)
    assert r.channel_loss_db_used == pytest.approx(7.5)
    assert to_link_params(params).channel_loss_db == pytest.approx(7.5)
    expected = TimeBinDPSLink().performance(to_link_params(params)).sifted_rate_bps
    assert r.throughput_bps == pytest.approx(expected, rel=0.04)
    assert r.duration_s == pytest.approx(1.0)                  # 31.25 M pulses, 32 ns apart
    assert r.throughput_bps == pytest.approx(r.sifted_key_bits / 1.0, rel=1e-9)
    params.channel_loss_db = None          # 0.96 dB: ~2x the key (the hold-off limits it)
    assert TimeBinDPSLink().performance(to_link_params(params)).sifted_rate_bps > 1.8 * expected
