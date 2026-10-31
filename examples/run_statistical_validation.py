"""First statistical-validation experiment: run me from the app/ directory:

    python examples/run_statistical_validation.py

Produces:
  1. QBER mean + 95% CI over 50 independent seeds at a fixed operating point.
  2. Distance vs. detector-clicks sweep (mean +/- 95% CI per distance).
  3. Interferometer visibility vs. QBER sweep.
  4. Eve intercept probability vs. QBER sweep.
  5. Mean photon number vs. detection/key statistics sweep.
  6. Early/late discard statistics.

The model is the MAQAN time-bin DPS hardware (Model C); see backend/adapter.py.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.mvesta_adapter import DPSQKDSimulationModel  # noqa: E402
from backend.statistical_validation import (  # noqa: E402
    early_late_discard_stats,
    run_batch,
    sweep_distance_vs_detector_clicks,
    sweep_eve_probability_vs_qber,
    sweep_mean_photon_number,
    sweep_visibility_vs_qber,
)

SEEDS_QUICK = range(1, 51)          # 50 independent trials
SEEDS_SWEEP = range(1, 21)          # 20 independent trials per sweep point


def main() -> None:
    model = DPSQKDSimulationModel()
    # MAQAN time-bin DPS hardware on the IITM-ERNET link (the defaults of
    # backend.adapter.SimulationParameters); 3 million pulses = 96 ms of
    # operation per trial.
    model.configure(num_pulses=3_000_000, eve_enabled=False)

    print("=== 1. QBER mean / 95% CI over 50 independent seeds ===")
    stats = run_batch(model, SEEDS_QUICK, ["QBER", "SIFTED_KEY_LEN"])
    q = stats["QBER"]
    print(f"QBER: mean={q.mean:.4f}  95% CI=[{q.ci95_low:.4f}, {q.ci95_high:.4f}]  n={q.n}")
    k = stats["SIFTED_KEY_LEN"]
    print(f"Sifted key length: mean={k.mean:.1f}  95% CI=[{k.ci95_low:.1f}, {k.ci95_high:.1f}]")

    print("\n=== 2. Distance vs. Detector Clicks ===")
    for d, s in sweep_distance_vs_detector_clicks(
        model, [5, 10, 20, 40, 60], SEEDS_SWEEP
    ).items():
        print(f"  {d:>5} km -> clicks mean={s.mean:.1f}  95% CI=[{s.ci95_low:.1f}, {s.ci95_high:.1f}]")
    model.configure(distance_km=4.8, channel_loss_db=7.5)  # restore the measured link

    print("\n=== 3. Interferometer visibility vs. QBER ===")
    for v, s in sweep_visibility_vs_qber(
        model, [0.99, 0.96, 0.92, 0.88, 0.80], SEEDS_SWEEP
    ).items():
        print(f"  visibility={v:.2f} -> QBER mean={s.mean:.4f}  95% CI=[{s.ci95_low:.4f}, {s.ci95_high:.4f}]")
    model.configure(interferometer_visibility=0.92)

    print("\n=== 4. Eve intercept probability vs. QBER ===")
    for p, s in sweep_eve_probability_vs_qber(
        model, [0.0, 0.1, 0.25, 0.5, 1.0], SEEDS_SWEEP
    ).items():
        print(f"  Eve p={p:.2f} -> QBER mean={s.mean:.4f}  95% CI=[{s.ci95_low:.4f}, {s.ci95_high:.4f}]")
    model.configure(eve_enabled=False, eve_intercept_probability=0.5)

    print("\n=== 5. Mean photon number vs. detection/key statistics ===")
    for mu, per_obs in sweep_mean_photon_number(
        model, [0.05, 0.1, 0.2, 0.3, 0.5], SEEDS_SWEEP
    ).items():
        clicks = per_obs["DETECTOR_CLICKS"]
        key = per_obs["SIFTED_KEY_LEN"]
        qber = per_obs["QBER"]
        print(f"  mu={mu:.2f} -> clicks mean={clicks.mean:.1f}, "
              f"sifted key mean={key.mean:.1f}, QBER mean={qber.mean:.4f}")
    model.configure(mean_photon_number=0.2)

    print("\n=== 6. Early/late discard statistics (50 seeds) ===")
    el = early_late_discard_stats(model, SEEDS_QUICK)
    print(f"  mean={el.mean:.2f}  95% CI=[{el.ci95_low:.2f}, {el.ci95_high:.2f}]  n={el.n}")


if __name__ == "__main__":
    main()
