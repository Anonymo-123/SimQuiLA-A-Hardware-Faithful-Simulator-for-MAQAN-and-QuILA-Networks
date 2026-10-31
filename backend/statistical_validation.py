"""Batch statistical-validation experiments on top of ``mvesta_adapter``.

This is the layer that actually runs "N independent seeds and analyze the
distribution", i.e. what a real MultiVeStA/py4j session would drive from the
Java side via reset()/run_step()/rval() one seed at a time. It is provided
so the same experiments can be run without standing up MultiVeStA itself
(see ``mvesta_integrator.py`` for the optional real py4j gateway).

Nothing here touches backend/adapter.py, sequence/, or gui/. It only calls
DPSQKDSimulationModel, which itself only calls the existing
backend.adapter.run_simulation().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from backend.mvesta_adapter import DPSQKDSimulationModel


@dataclass
class BatchStats:
    observable: str
    n: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    values: List[float]


def _summarize(name: str, values: Sequence[float]) -> BatchStats:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else float("nan")
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    # Normal-approximation 95% CI on the mean (n independent trials).
    half_width = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return BatchStats(
        observable=name,
        n=n,
        mean=mean,
        std=std,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
        values=list(arr),
    )


def run_batch(
    model: DPSQKDSimulationModel,
    seeds: Sequence[int],
    observables: Sequence[str] = ("QBER", "SIFTED_KEY_LEN"),
) -> Dict[str, BatchStats]:
    """Run one independent trial per seed and summarize each observable.

    ``model`` must already have configure() called with the fixed physical
    parameters for this batch. Returns one BatchStats per observable name.
    """
    collected: Dict[str, List[float]] = {obs: [] for obs in observables}
    for seed in seeds:
        model.reset(seed)
        model.run_step()
        for obs in observables:
            collected[obs].append(model.rval(obs))
    return {obs: _summarize(obs, vals) for obs, vals in collected.items()}


def independent_seed_smoke_test(model: DPSQKDSimulationModel, n: int = 20,
                                 base_seed: int = 0) -> bool:
    """Sanity check: n different seeds must not all produce identical
    results (this is the specific bug Dr. Jagannathan originally flagged,
    so it is worth guarding against regressing).
    """
    qbers = []
    for i in range(n):
        model.reset(base_seed + i)
        model.run_step()
        qbers.append(model.rval("QBER"))
    return len(set(qbers)) > 1


# --------------------------------------------------------------------------- #
# Parameter sweeps (mirrors the four analytics-page sweeps in gui/analytics
# _page.py, but run through the MultiVeStA-style seed-averaged interface
# instead of a single-seed sample per x-value)
# --------------------------------------------------------------------------- #

def sweep_distance_vs_detector_clicks(
    model: DPSQKDSimulationModel,
    distances_km: Sequence[float],
    seeds: Sequence[int],
) -> Dict[float, BatchStats]:
    """Loss follows the distance (distance x dB/km): a measured link loss set
    with configure(channel_loss_db=...) is switched off for this sweep."""
    out = {}
    for d in distances_km:
        model.configure(distance_km=d, channel_loss_db=None)
        out[d] = run_batch(model, seeds, ["DETECTOR_CLICKS"])["DETECTOR_CLICKS"]
    return out


def sweep_visibility_vs_qber(
    model: DPSQKDSimulationModel,
    visibilities: Sequence[float],
    seeds: Sequence[int],
) -> Dict[float, BatchStats]:
    out = {}
    for v in visibilities:
        model.configure(interferometer_visibility=v)
        out[v] = run_batch(model, seeds, ["QBER"])["QBER"]
    return out


def sweep_eve_probability_vs_qber(
    model: DPSQKDSimulationModel,
    eve_probabilities: Sequence[float],
    seeds: Sequence[int],
) -> Dict[float, BatchStats]:
    out = {}
    model.configure(eve_enabled=True)
    for p in eve_probabilities:
        model.configure(eve_intercept_probability=p)
        out[p] = run_batch(model, seeds, ["QBER"])["QBER"]
    return out


def sweep_eve_probability_vs_actual_rate(
    model: DPSQKDSimulationModel,
    eve_probabilities: Sequence[float],
    seeds: Sequence[int],
) -> Dict[float, BatchStats]:
    """For each configured ``eve_intercept_probability``, run one
    independent trial per seed and summarize the *measured* photon-level
    interception rate (``EVE_ACTUAL_RATE`` = photons_intercepted /
    photons_available_to_eve). Used to validate that the photon-level Eve
    model converges to the configured probability as photon counts grow
    (see requirements: p=0.1/0.5/0.9 should each yield a mean actual rate
    close to p, with the 95% CI narrowing as num_pulses/seeds increase).
    """
    out = {}
    model.configure(eve_enabled=True)
    for p in eve_probabilities:
        model.configure(eve_intercept_probability=p)
        out[p] = run_batch(model, seeds, ["EVE_ACTUAL_RATE"])["EVE_ACTUAL_RATE"]
    return out


def sweep_mean_photon_number(
    model: DPSQKDSimulationModel,
    mean_photon_numbers: Sequence[float],
    seeds: Sequence[int],
) -> Dict[float, Dict[str, BatchStats]]:
    out = {}
    for mu in mean_photon_numbers:
        model.configure(mean_photon_number=mu)
        out[mu] = run_batch(
            model, seeds,
            ["DETECTOR_CLICKS", "SIFTED_KEY_LEN", "QBER"],
        )
    return out


def early_late_discard_stats(
    model: DPSQKDSimulationModel,
    seeds: Sequence[int],
) -> BatchStats:
    return run_batch(model, seeds, ["EARLY_LATE_DISCARDS"])["EARLY_LATE_DISCARDS"]
