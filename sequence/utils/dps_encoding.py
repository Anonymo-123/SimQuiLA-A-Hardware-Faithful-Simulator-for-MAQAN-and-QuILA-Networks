"""DPS-QKD encoding definitions.

``dps_timebin`` (used by the current hardware model)
    Time-bin DPS as built at IIT Madras for MAQAN ("Model C"): every pulse is a
    block of N time bins (N = 3, 1 ns each) with a phase 0 or pi written into
    each bin. One photon of such a pulse is in the superposition

        |psi> = sum_j c_j |bin j>,   c_j = sqrt(w_j) * (+1 for phase 0, -1 for pi),

    over the N bins of its pulse. Bob's one-bin-delay interferometer turns this
    into N+1 output slots (early, N-1 interference slots, late); see
    :func:`quila_sim.core.timebin.exit_probabilities`. The N amplitudes are kept
    on :class:`~sequence.components.dps_hardware.TimeBinPhoton`, so SeQUeNCe's
    own qubit-state checks stay switched on.

``dps_phase`` (kept for reference)
    The two-time-bin qubit of a gap-free continuous pulse train (textbook DPS,
    "Model B"). Not used by the hardware model any more.
"""

dps_timebin = {
    "name": "dps_timebin",
    "bins_per_pulse": 3,          # N; the actual value travels with each photon
    "bin_separation": 1000,       # ps; the actual value is set by the light source
}

dps_phase = {
    "name": "dps_phase",
    "bases": [
        (
            (complex(1), complex(0)),   # phase 0  ->  |0>
            (complex(0), complex(1)),   # phase pi ->  |1>
        )
    ],
    "num_phases": 2,
    "bin_separation": 0,          # set dynamically from the source frequency
}
