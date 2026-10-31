"""Statistical-validation adapter: DPS-QKD backend  <->  MultiVeStA-style driver.

This module is purely additive. It imports ``backend.adapter`` and calls its
public ``run_simulation()`` function exactly the way ``gui/worker.py``
already does -- nothing in ``backend/adapter.py``, ``sequence/``, or ``gui/``
is modified, subclassed-and-monkeypatched, or duplicated. If this file is
deleted, the GUI is completely unaffected.

Why this design, and how it maps to the MultiVeStA-QNetworks reference repo
----------------------------------------------------------------------------
The reference integration (anubhavsrivastavaa/multivesta-qnetworks) drives a
*long-lived, multi-node* SeQUeNCe simulation that runs for a large number of
discrete events (entanglement generation/swap attempts over seconds of
simulated time). There, ``run_step()`` genuinely means "execute one event
from Timeline's event queue and return control", because MultiVeStA wants to
evaluate queries (e.g. "are a and c entangled *yet*?") at many intermediate
points along a single long trajectory.

The DPS-QKD backend here is architecturally different: ``run_simulation()``
already IS one atomic trial -- a fixed number of time-bin pulses, the sifted
key and the QBER, produced by one of the engines described in
``backend/adapter.py`` (by default the fast Model C engine; with
``engine="event"`` a SeQUeNCe topology built fresh via
``DPSQKDTopo.from_nodes(...)`` and torn down again). There is no meaningful
intermediate state to expose mid-run: the
GUI itself only ever reads the fully-populated ``SimulationResult`` after
the window closes. Exposing SeQUeNCe's raw per-event ``Timeline.run_step()``
here would require unwinding that atomic-window design and would risk
exactly the kind of engine changes this integration was meant to avoid.

So the natural unit of "one step" for THIS simulator is one complete,
independently-seeded trial -- which is also exactly the unit the requested
statistical experiments operate over (QBER confidence intervals across
seeds, distance-vs-clicks sweeps, etc.). The adapter therefore implements:

    reset(seed)   -> derive alice/bob/eve/visibility seeds from one int,
                     store pending SimulationParameters, do NOT run yet
    run_step()    -> execute exactly one run_simulation() trial with those
                     seeds/parameters, cache the SimulationResult, and mark
                     the "step" complete (mirrors MultiVeStA's expectation
                     that after run_step() the model is in a new state that
                     rval() can query)
    rval(obs)     -> read an observable out of the cached SimulationResult

If a real Timeline-level run_step() is ever wanted (e.g. a future multi-node
DPS-QKD topology that runs long enough to have meaningful mid-trajectory
state), it is a strict extension of this file, not a change to it.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Dict, Optional, Union

import numpy as np

from backend.adapter import (
    LEGACY_ENGINES,
    LEGACY_KEYS,
    SimulationParameters,
    SimulationResult,
    params_from_dict,
    run_simulation,
)


# --------------------------------------------------------------------------- #
# Observable registry
# --------------------------------------------------------------------------- #
# Every entry is a function SimulationResult -> float. Keys double as the
# integer-observable table (position in this list) and as string directives
# ("QBER", "SIFTED_KEY_LEN", ...), following the reference repo's pattern of
# supporting both int and string observations from rval().

def _security_secure_as_float(r: SimulationResult) -> float:
    return 1.0 if r.security_secure else 0.0


_OBSERVABLES: Dict[str, "callable[[SimulationResult], float]"] = {
    "QBER":                 lambda r: float(r.qber),
    "D0_CLICKS":            lambda r: float(r.d0_clicks),
    "D1_CLICKS":            lambda r: float(r.d1_clicks),
    "DETECTOR_CLICKS":      lambda r: float(r.raw_detector_clicks),
    "DARK_COUNTS":          lambda r: float(r.dark_counts),
    "CHANNEL_LOSS_COUNT":   lambda r: float(r.channel_loss_count),
    "PHOTON_LOSS_RATE":     lambda r: (
        float(r.channel_loss_count) / float(r.pulses_sent)
        if r.pulses_sent else 0.0
    ),
    "EARLY_LATE_DISCARDS":  lambda r: float(r.early_late_discards),
    "DISCARDED_CLICKS":     lambda r: float(r.discarded_click_count),
    "SIFTED_KEY_LEN":       lambda r: float(r.sifted_key_bits),
    "BIT_ERRORS":           lambda r: float(r.bit_errors),
    "EVE_INTERCEPTIONS":    lambda r: float(r.eve_intercepted_count),
    "PULSES_SENT":          lambda r: float(r.pulses_sent),
    "PHOTONS_EMITTED":      lambda r: float(r.photons_emitted),
    "THROUGHPUT_BPS":       lambda r: float(r.throughput_bps),
    "LATENCY_S":            lambda r: float(r.latency_s),
    "SECURE":               _security_secure_as_float,

    # -- Photon-level Eve observables (added; positions after these are
    # appended, so existing integer-index directives above are unchanged) --
    # NOTE: EVE_INTERCEPTIONS above counts *sifted bits* caused by photons
    # Eve re-sent. The observables below count photons: Eve intercepts
    # whole pulses, and EVE_ACTUAL_RATE is the fraction of Alice's photons
    # that were in intercepted pulses (it converges to the configured
    # probability).
    "EVE_PHOTONS_INTERCEPTED": lambda r: float(r.photons_intercepted),
    "EVE_PHOTONS_AVAILABLE":   lambda r: float(r.photons_available_to_eve),
    "EVE_ACTUAL_RATE":         lambda r: float(r.eve_actual_interception_rate),
    "PHOTONS_REMAINING":       lambda r: float(r.photons_remaining),
    "PHOTONS_DETECTED":        lambda r: float(r.photons_detected),
    "PHOTONS_LOST":            lambda r: float(r.photons_lost),

    # -- Model C (time-bin DPS, MAQAN hardware) observables (appended in
    # v0.3, so the integer indices above are unchanged) --
    "KEY_SLOT_CLICKS":         lambda r: float(r.key_slot_clicks),
    "GUARD_DISCARDS":          lambda r: float(r.guard_discards),
    "OUTSIDE_WINDOW_CLICKS":   lambda r: float(r.outside_window_clicks),
    "LEAK_CLICKS":             lambda r: float(r.leak_clicks),
    "HOLDOFF_BLOCKED":         lambda r: float(r.holdoff_blocked),
    "PULSES_INTERCEPTED":      lambda r: float(r.pulses_intercepted),
    "PULSES_WITH_CLICK":       lambda r: float(r.pulses_with_click),
    "EXPECTED_QBER":           lambda r: float(r.expected_qber),
    "EXPECTED_SIFTED_RATE_BPS": lambda r: float(r.expected_sifted_rate_bps),
}

# Integer-observable table: position == index, mirroring the reference
# repo's rval(0)/rval(3)-style integer directives, for callers (e.g. a
# future py4j gateway) that prefer int observations over strings.
_OBSERVABLE_BY_INDEX = list(_OBSERVABLES.keys())


#: Parameters that reset(seed) controls; configure() must not set them.
_SEED_FIELDS = ("alice_seed", "bob_seed", "eve_seed", "visibility_seed")


def _config_fields() -> set:
    return {f.name for f in fields(SimulationParameters)} - set(_SEED_FIELDS)


class PendingConfig:
    """Non-seed simulation parameters held fixed across a MultiVeStA batch.

    reset(seed) only ever touches the four RNG seeds; every other parameter
    of :class:`backend.adapter.SimulationParameters` (link loss, pulse format,
    detector, Eve, ...) is set once via configure() before the batch starts,
    matching how a MultiVeStA experiment fixes the model under test and only
    varies the seed between independent runs. Defaults are those of
    SimulationParameters (the MAQAN hardware on the IITM-ERNET link), except
    that no per-pulse trace is recorded.
    """

    def __init__(self) -> None:
        self.params = SimulationParameters(max_trace_events=0)

    def __getattr__(self, name: str) -> Any:
        if name == "params":
            raise AttributeError(name)
        return getattr(self.params, name)

    def set(self, **kwargs: Any) -> None:
        """Set parameters by name; names of earlier versions are translated."""
        allowed = _config_fields()
        updates: Dict[str, Any] = {}
        legacy: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in allowed:
                updates[key] = value
            elif key in LEGACY_KEYS or key == "dark_count_hz":
                legacy[key] = value
            else:
                raise ValueError(f"Unknown simulation parameter: {key!r}")
        if "engine" in updates:
            updates["engine"] = LEGACY_ENGINES.get(updates["engine"], updates["engine"])
        if legacy:
            data = {f.name: getattr(self.params, f.name) for f in fields(SimulationParameters)}
            data.update(updates)
            data.update(legacy)
            if "dark_count_hz" in legacy:
                data.pop("dark_count_prob", None)
            for old, new in LEGACY_KEYS.items():
                if old in legacy:
                    data.pop(new, None)
            self.params, _notes = params_from_dict(data)
        else:
            self.params = replace(self.params, **updates)


class DPSQKDSimulationModel:
    """MultiVeStA-facing model: reset(seed) / run_step() / rval(observable).

    Usage
    -----
    >>> model = DPSQKDSimulationModel()
    >>> model.configure(distance_km=25.0, eve_enabled=True)
    >>> model.reset(12345)
    >>> model.run_step()
    >>> model.rval("QBER")
    0.0734...
    >>> model.is_step_complete()
    True

    Every reset(seed) + run_step() pair is one statistically independent
    trial: the four backend RNG streams (alice/bob/eve/visibility) are all
    deterministically derived from the single integer seed via
    ``numpy.random.SeedSequence(seed).spawn(4)``, so the *same* seed always
    reproduces the *same* trial, and *different* seeds are guaranteed
    independent draws (this is what SeedSequence.spawn is for -- it is not
    just e.g. seed, seed+1, seed+2, seed+3, which could correlate for some
    generators).
    """

    def __init__(self) -> None:
        self._config = PendingConfig()
        self._pending_params: Optional[SimulationParameters] = None
        self._result: Optional[SimulationResult] = None
        self._step_complete: bool = False
        self._last_seed: Optional[int] = None

    # -- configuration (non-seed parameters; call before a batch) -------- #

    def configure(self, **kwargs) -> None:
        """Set physical parameters held fixed across a batch of seeds.

        Accepts any field of :class:`backend.adapter.SimulationParameters`
        except the seeds, e.g. configure(distance_km=25,
        interferometer_visibility=0.9). Names used by earlier versions
        (frequency_hz, dead_time_s, dark_count_hz) are translated.
        """
        self._config.set(**kwargs)

    # -- MultiVeStA-facing API -------------------------------------------- #

    def reset(self, seed: int) -> None:
        """Start a new, independent simulation trial from ``seed``.

        Does not run anything yet (mirrors the reference repo's
        ``setSimulatorForNewSimulation``, which re-initializes the model
        but does not itself advance time).
        """
        self._last_seed = int(seed)
        seed_seq = np.random.SeedSequence(int(seed))
        alice_seed, bob_seed, eve_seed, visibility_seed = (
            int(s.generate_state(1)[0]) for s in seed_seq.spawn(4)
        )

        self._pending_params = replace(
            self._config.params,
            alice_seed=alice_seed,
            bob_seed=bob_seed,
            eve_seed=eve_seed,
            visibility_seed=visibility_seed,
        )
        self._result = None
        self._step_complete = False

    def run_step(self) -> None:
        """Execute exactly one trial (see module docstring for why "one
        step" == "one complete run_simulation() call" for this backend).
        """
        if self._pending_params is None:
            raise RuntimeError("run_step() called before reset(seed)")
        self._result = run_simulation(self._pending_params)
        self._step_complete = True

    def run(self) -> None:
        """Alias for run_step(): API symmetry with the reference repo's
        performWholeSimulation(), since one step IS the whole trial here.
        """
        self.run_step()

    def is_step_complete(self) -> bool:
        return self._step_complete

    # MultiVeStA calls this "getTime"; there is no meaningful intra-trial
    # simulated time to report here (see module docstring), so we report
    # 0.0 before the trial and 1.0 after -- consistent with treating one
    # trial as one discrete time step.
    def get_time(self) -> float:
        return 1.0 if self._step_complete else 0.0

    def rval(self, observable: Union[int, str]) -> float:
        """Return the value of ``observable`` for the last completed trial.

        Accepts either an integer index into the observable table (see
        list_observables()) or a string name such as "QBER",
        "EARLY_LATE_DISCARDS", "EVE_INTERCEPTIONS", etc.
        """
        if self._result is None:
            raise RuntimeError("rval() called before a trial was run "
                                "(call reset(seed) then run_step() first)")
        if isinstance(observable, int):
            try:
                name = _OBSERVABLE_BY_INDEX[observable]
            except IndexError as exc:
                raise ValueError(
                    f"Unknown integer observable {observable}; valid range "
                    f"is 0..{len(_OBSERVABLE_BY_INDEX) - 1}"
                ) from exc
        else:
            name = observable.strip().upper()
            if name not in _OBSERVABLES:
                raise ValueError(
                    f"Unknown observable {observable!r}. Valid names: "
                    f"{sorted(_OBSERVABLES)}"
                )
        return _OBSERVABLES[name](self._result)

    @property
    def last_result(self) -> Optional[SimulationResult]:
        """Escape hatch: the full SimulationResult for the last trial, for
        callers (e.g. the statistical driver) that want more than one
        scalar observable without re-running the trial.
        """
        return self._result

    @staticmethod
    def list_observables() -> Dict[int, str]:
        return dict(enumerate(_OBSERVABLE_BY_INDEX))
