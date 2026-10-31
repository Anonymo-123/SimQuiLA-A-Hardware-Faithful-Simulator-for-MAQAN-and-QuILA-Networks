"""Tests for the shared contracts in quila_sim.core (DRAFT v0)."""

from pathlib import Path

import numpy as np
import pytest

from quila_sim.core import (
    DetectionRecord,
    DetectorParams,
    InMemoryKeyPool,
    KeyPool,
    LinkModel,
    LinkParams,
    LinkPerformance,
    load_scenario,
)

ROOT = Path(__file__).resolve().parents[1]


def _params(**overrides):
    base = dict(
        protocol="DPS",
        rep_rate_hz=1e9,
        mean_photon_number=0.2,
        channel_loss_db=7.5,
        detector=DetectorParams(efficiency=0.1, dark_count_prob=1e-6),
    )
    base.update(overrides)
    return LinkParams(**base)


def test_maqan_scenario_loads():
    sc = load_scenario(ROOT / "scenarios" / "maqan.yaml")
    assert len(sc.nodes) == 5
    assert sc.link("IITM-ERNET").loss_db == pytest.approx(7.5)
    assert sc.defaults["mean_photon_number_max"] == pytest.approx(0.2)


def test_scenario_rejects_unknown_node():
    from quila_sim.core import Scenario

    with pytest.raises(ValueError):
        Scenario.from_dict(
            {"name": "bad", "nodes": [{"name": "A"}], "links": [{"name": "AB", "a": "A", "b": "B"}]}
        )


def test_link_params_validation_and_transmittance():
    p = _params(channel_loss_db=10.0)
    assert p.transmittance == pytest.approx(0.1)
    with pytest.raises(ValueError):
        DetectorParams(efficiency=1.5, dark_count_prob=0.0)
    with pytest.raises(ValueError):
        _params(visibility=1.2)


def test_secure_rate_needs_security_model():
    with pytest.raises(ValueError):
        LinkPerformance(raw_rate_bps=1, sifted_rate_bps=1, qber=0.01, secure_rate_bps=0.5)


class _ToyLink:
    """Illustrates the LinkModel contract; not a physics model."""

    name = "toy"
    fidelity = "analytic"

    def performance(self, params):
        rate = params.rep_rate_hz * params.mean_photon_number * params.transmittance
        return LinkPerformance(raw_rate_bps=rate, sifted_rate_bps=rate, qber=0.0)

    def sample(self, params, n_pulses, rng):
        slots = np.arange(3, dtype=np.int64)
        bits = np.zeros(3, dtype=np.uint8)
        return DetectionRecord(slots, bits, bits.copy(), n_pulses)


def test_toy_link_satisfies_contract():
    model = _ToyLink()
    assert isinstance(model, LinkModel)
    rec = model.sample(_params(), 100, np.random.default_rng(0))
    assert rec.qber == 0.0


def test_in_memory_key_pool():
    pool = InMemoryKeyPool(endpoints=("IITM", "ERNET"), capacity_bits=1000)
    assert isinstance(pool, KeyPool)
    assert pool.deposit(800, t_s=1.0) == 800
    assert pool.deposit(500, t_s=2.0) == 200  # capped at capacity
    assert pool.withdraw(256, t_s=3.0, consumer="aes-rekey")
    assert pool.level_bits == 744
    assert not pool.withdraw(10_000, t_s=4.0)
    assert pool.denied_requests == 1 and pool.discarded_bits == 300
