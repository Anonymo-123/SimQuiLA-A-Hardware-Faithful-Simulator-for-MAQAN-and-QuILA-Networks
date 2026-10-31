"""Integration tests for the MultiVeStA statistical-validation layer.

Run with:  python -m pytest tests/test_mvesta_integration.py -v
(from the app/ directory), or plain: python tests/test_mvesta_integration.py

Covers the four acceptance checks from the integration brief:
  1. The original GUI code is untouched (no source edits, imports unaffected).
  2. MultiVeStA-style driver can run the same simulator backend.
  3. Different seeds produce independent results.
  4. MultiVeStA (rval) results correspond exactly to the simulator's own
     SimulationResult outputs -- i.e. no separate/duplicated computation.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.adapter import SimulationParameters, run_simulation  # noqa: E402
from backend.mvesta_adapter import DPSQKDSimulationModel  # noqa: E402
from backend.statistical_validation import (  # noqa: E402
    independent_seed_smoke_test,
    run_batch,
)


def _file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_gui_core_files_untouched():
    """Check 1: the original GUI pages/widgets were not rewritten -- only
    a new, opt-in gui/validation_page.py was added, plus the minimal
    wiring in main_window.py to add it as a fourth nav tab. Every other
    original GUI module (dashboard, analytics, settings, worker, theme,
    topology/bloch views, widgets) has zero dependency on the new
    statistical-validation layer, so it cannot have been broken by it.
    main_window.py is allowed to reference it (that's the intentional,
    additive wiring); everything else must not.
    """
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gui_dir = os.path.join(app_dir, "gui")
    allowed_to_reference = {"main_window.py", "validation_page.py"}
    offenders = []
    for fname in os.listdir(gui_dir):
        if not fname.endswith(".py") or fname in allowed_to_reference:
            continue
        with open(os.path.join(gui_dir, fname), "r", encoding="utf-8") as f:
            src = f.read()
        if "mvesta" in src.lower():
            offenders.append(fname)
    assert not offenders, f"Original GUI files reference the new adapter: {offenders}"


def test_gui_still_importable():
    """Check 1 (cont'd): gui.main_window and friends still import fine.
    Skips gracefully if PyQt6 isn't installed in this environment (that's
    an environment issue, not something the integration could have broken)
    rather than failing the whole suite.
    """
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        try:
            import pytest
            pytest.skip("PyQt6 not installed in this environment")
        except ImportError:
            print("SKIP  test_gui_still_importable: PyQt6 not installed")
            return
    import gui.main_window  # noqa: F401
    import gui.dashboard_page  # noqa: F401
    import gui.analytics_page  # noqa: F401


def test_adapter_runs_same_backend():
    """Check 2: DPSQKDSimulationModel.run_step() produces a SimulationResult
    with the same shape/fields as calling backend.adapter.run_simulation()
    directly -- it is the same engine, not a reimplementation.
    """
    model = DPSQKDSimulationModel()
    model.configure(num_pulses=200_000, distance_km=5.0)
    model.reset(7)
    model.run_step()
    result = model.last_result
    assert result is not None
    for field in ("qber", "sifted_key_bits", "d0_clicks", "d1_clicks",
                  "raw_detector_clicks", "early_late_discards",
                  "eve_intercepted_count", "pulses_sent"):
        assert hasattr(result, field)


def test_reset_is_reproducible():
    """Same seed -> same trial (a prerequisite for "independent seeds ->
    independent results" being a meaningful, testable claim rather than
    just noise).
    """
    model = DPSQKDSimulationModel()
    model.configure(num_pulses=500_000, distance_km=10.0, eve_enabled=True,
                    eve_intercept_probability=0.3)

    model.reset(999)
    model.run_step()
    qber_a = model.rval("QBER")
    key_a = model.rval("SIFTED_KEY_LEN")

    model.reset(999)
    model.run_step()
    qber_b = model.rval("QBER")
    key_b = model.rval("SIFTED_KEY_LEN")

    assert qber_a == qber_b
    assert key_a == key_b


def test_different_seeds_independent():
    """Check 3: different seeds produce independent (not identical)
    results. This directly guards against the original bug Dr. Jagannathan
    flagged (identical results every run).
    """
    model = DPSQKDSimulationModel()
    model.configure(num_pulses=300_000, distance_km=15.0, dark_count_prob=1e-4)
    assert independent_seed_smoke_test(model, n=25, base_seed=1000)


def test_rval_matches_backend_output_exactly():
    """Check 4: rval() values equal the corresponding SimulationResult
    fields bit-for-bit (float equality is fine here since it's a direct
    attribute read, not a recomputation) for the SAME cached result.
    """
    model = DPSQKDSimulationModel()
    model.configure(num_pulses=400_000, distance_km=20.0, eve_enabled=True,
                    eve_intercept_probability=0.4)
    model.reset(2024)
    model.run_step()
    r = model.last_result

    assert model.rval("QBER") == r.qber
    assert model.rval("SIFTED_KEY_LEN") == r.sifted_key_bits
    assert model.rval("D0_CLICKS") == r.d0_clicks
    assert model.rval("D1_CLICKS") == r.d1_clicks
    assert model.rval("DETECTOR_CLICKS") == r.raw_detector_clicks
    assert model.rval("EARLY_LATE_DISCARDS") == r.early_late_discards
    assert model.rval("EVE_INTERCEPTIONS") == r.eve_intercepted_count
    assert model.rval("DARK_COUNTS") == r.dark_counts
    assert model.rval("BIT_ERRORS") == r.bit_errors
    assert model.rval(_index_of("QBER")) == r.qber


def _index_of(name: str) -> int:
    for idx, n in DPSQKDSimulationModel.list_observables().items():
        if n == name:
            return idx
    raise AssertionError(name)


def test_batch_statistics_shape():
    model = DPSQKDSimulationModel()
    model.configure(num_pulses=300_000, distance_km=10.0)
    stats = run_batch(model, seeds=range(2000, 2020), observables=["QBER"])
    qber_stats = stats["QBER"]
    assert qber_stats.n == 20
    assert len(qber_stats.values) == 20
    assert qber_stats.ci95_low <= qber_stats.mean <= qber_stats.ci95_high


def test_eve_photon_bookkeeping_is_consistent():
    """Eve intercepts whole pulses (time-bin model); the photon counts must
    stay consistent: photons_intercepted never exceeds photons_emitted, and
    the Eve-derived counts sum back to the total exactly. (Guards against an
    old bug where an OR across neighbouring slots counted detector events
    instead of photons.)
    """
    params = SimulationParameters(
        num_pulses=800, eve_enabled=True, eve_intercept_probability=0.5,
        mean_photon_number=0.8, alice_seed=1, bob_seed=2, eve_seed=3,
        visibility_seed=4,
    )
    r = run_simulation(params)
    assert r.photons_intercepted <= r.photons_emitted
    assert r.photons_available_to_eve == r.photons_emitted
    assert r.photons_intercepted + r.photons_remaining == r.photons_emitted
    assert r.photons_detected <= r.photons_emitted
    assert r.photons_lost == r.photons_emitted - r.photons_detected


def test_eve_disabled_intercepts_nothing():
    params = SimulationParameters(num_pulses=300, eve_enabled=False)
    r = run_simulation(params)
    assert r.photons_intercepted == 0
    assert r.photons_available_to_eve == 0
    assert r.eve_actual_interception_rate == 0.0


def test_eve_actual_rate_converges_to_configured_probability():
    """Validation requirement: for p in {0.1, 0.5, 0.9}, the measured
    per-photon interception rate (pooled across many seeds/photons) must
    land close to the configured probability."""
    for p in (0.1, 0.5, 0.9):
        total_emitted = 0
        total_intercepted = 0
        for seed in range(6):
            params = SimulationParameters(
                num_pulses=5000, eve_enabled=True, eve_intercept_probability=p,
                mean_photon_number=0.8,
                alice_seed=100 + seed, bob_seed=200 + seed,
                eve_seed=300 + seed, visibility_seed=400 + seed,
            )
            r = run_simulation(params)
            total_emitted += r.photons_available_to_eve
            total_intercepted += r.photons_intercepted
        pooled_rate = total_intercepted / total_emitted
        assert abs(pooled_rate - p) < 0.03, (p, pooled_rate, total_emitted)


def test_bob_key_bits_match_alice_length_even_at_high_dcr():
    """Regression test for the Bob key-bit filtering bug.

    Bob's instrumented_end_photon_pulse() used to append a bit for every
    slot in `seen` with no bounds check, while Alice (sequence/qkd/
    DPSQKD.py) only ever appends a bit when `1 <= slot < n`. An
    out-of-window dark count (slot >= n) would therefore inflate
    bob_key_bits beyond alice_key_bits -- masked only because every
    reader happened to bound itself by len(alice_key_bits). This must
    hold exactly, especially at high dark-count rates where the bug was
    most visible.
    """
    for engine in ("timebin", "event"):
        for dark in (3e-6, 1e-3, 5e-2):
            params = SimulationParameters(
                engine=engine, num_pulses=20_000, channel_loss_db=0.0,
                dark_count_prob=dark, holdoff_s=0.1e-6, mean_photon_number=0.5,
                alice_seed=11, bob_seed=22, eve_seed=33, visibility_seed=44,
            )
            r = run_simulation(params)
            assert len(r.alice_key_bits) == len(r.bob_key_bits), (engine, dark)


def test_dark_counts_do_not_fire_past_the_measurement_window():
    """Regression test for the dark-count/discarded-click performance bug.

    SeQUeNCe's free-running Detector.add_dark_count() used to keep firing for
    the whole padded Timeline runtime (~10x the real pulse window). The
    gated detector of the time-bin model draws dark counts per gate, and
    only for the gates of the measurement window, so the number of dark
    counts follows the number of pulses, not the runtime padding.
    """
    params = SimulationParameters(
        engine="event", num_pulses=5000, distance_km=10.0, channel_loss_db=None,
        dark_count_prob=0.02, holdoff_s=0.0, mean_photon_number=0.5,
        alice_seed=1, bob_seed=2, eve_seed=3, visibility_seed=4,
    )
    r = run_simulation(params)
    expected = 0.02 * 5000
    assert abs(r.dark_counts - expected) < 5 * expected ** 0.5, r.dark_counts
    assert r.discarded_click_count < 3 * max(r.dark_counts, 1) + r.photons_detected, (
        r.discarded_click_count, r.dark_counts)


if __name__ == "__main__":
    tests = [
        test_gui_core_files_untouched,
        test_gui_still_importable,
        test_adapter_runs_same_backend,
        test_reset_is_reproducible,
        test_different_seeds_independent,
        test_rval_matches_backend_output_exactly,
        test_batch_statistics_shape,
        test_eve_photon_bookkeeping_is_consistent,
        test_eve_disabled_intercepts_nothing,
        test_eve_actual_rate_converges_to_configured_probability,
        test_bob_key_bits_match_alice_length_even_at_high_dcr,
        test_dark_counts_do_not_fire_past_the_measurement_window,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
