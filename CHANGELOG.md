# Changelog

## v0.3.0 - the MAQAN time-bin DPS hardware ("Model C")

The DPS-QKD hardware used in MAQAN does not send a continuous
pulse train. It sends one 3 ns pulse every 32 ns, with a phase written into
each 1 ns time bin, and the rest of the period is empty. v0.2 modelled a
continuous train; v0.3 models the real hardware.

**Physics (Model C, now the default everywhere)**

- Alice: one pulse of N = 3 time bins (1 ns each) every 32 ns (31.25 MHz);
  mu per pulse (default 0.2); light leaking through the intensity modulator
  between pulses (extinction ratio).
- Bob: one-bin-delay interferometer; each pulse leaves in N + 1 slots in the
  ratio 1/6 : 1/3 : 1/3 : 1/6 (early, two interference slots, late); key bits
  come only from the interference slots (phi_j XOR phi_(j-1)), i.e. 2/3 of
  the detected photons.
- One gated InGaAs detector shared by both interferometer ports (port 1
  delayed 10 ns, 3 dB coupler + 0.5 dB attenuator); Bob's bit is the time
  window of a click. Gate per period, dark counts per gate, 10 us hold-off,
  timing jitter, temporal filter (guard band). A "dual" layout (one detector
  per port) is available for comparison.
- Phase-modulator transition time and IM extinction ratio included.
- Published MAQAN values in `quila_sim.core.MAQAN_DPS` and
  `scenarios/maqan.yaml` (with sources). Each imperfection alone reproduces
  its line of the QBER budget in Shaw et al. (2022), Table III; a 200 ps
  temporal filter behaves as reported there. Predicted IITM-ERNET
  performance: 21.3 kbps sifted at 11.1 % QBER (afterpulsing, 1.5 % in
  Table III, is not modelled yet).
- New fast engine `quila_sim.physics.TimeBinDPSLink` (~15 million pulses/s)
  with an exact per-gate closed form (`performance()`).
- SeQUeNCe per-photon reference rewritten for Model C: `TimeBinPhoton`,
  `PhasedLightSource`, `DelayLineInterferometer`, `GatedDetector`, and a
  time-bin sifting protocol (key-slot ids k*N + j). Pulse limit raised to
  2 million.
- The continuous-train model (v0.2) is kept as "Model B" for comparison only
  (`engine="continuous"`).
- Eve (teaching model) intercepts whole pulses.

**App**

- New controls: pulse period, time bins per pulse, bin width, IM extinction
  ratio, PM transition time, detector layout, dark counts per gate, hold-off,
  timing jitter, temporal filter, Bob's internal loss; "MAQAN hardware" and
  "Ideal components" buttons. Defaults: MAQAN on the IITM-ERNET link.
- New statistics: key-slot, early/late, guard-band and outside-window clicks,
  IM-leakage clicks, clicks lost in hold-off, and the closed-form expected
  rate and QBER next to the simulated ones.
- Event log: one row per pulse with a click (phases, port window, slot, time).
- Analytics: Bob's time-stamp histogram (shows the 1:2:2:1 slots in both
  port windows), click classification, and new sweeps (distance and mu vs
  key rate, visibility and temporal filter vs QBER) with closed-form curves.
- Security check threshold is selectable (default 20 %, the critical QBER of
  3-pulse DPS against individual attacks in the MAQAN analysis; 11 % and 5 %
  also offered). Still a check, not a security proof (roadmap item 8).
- Configuration files from v0.2 still load (old keys are converted).

**MultiVeStA**

- `configure()` accepts every `SimulationParameters` field; new observables
  appended at the end (existing integer indices unchanged).

**Tests**

- New `tests/test_timebin_model.py`; `tests/test_event_reference.py`
  rewritten for Model C; MultiVeStA and loss tests updated. 68 tests
  (1 marked slow).

## v0.2.0 - link model corrected and matched to MAQAN (roadmap items 1-3)

**Physics**

- New default engine `quila_sim.physics.CoherentDPSLink`: exact for weak
  coherent pulses, continuous stream (no windows), ~10-20 million pulses/s.
  Sifted rate now matches theory; the v0.1 model gave about half.
- SeQUeNCe per-photon engine rewritten with the same physics and kept as a
  reference (choose "Reference" in the app; limited to 20,000 pulses).
- Interferometer visibility is part of the backend; double clicks give a
  random bit instead of always 0.

**Bug fixes**

- Multi-window runs of the stock protocol no longer desynchronise Alice's and
  Bob's keys when there are dark counts (error was 45-51%, now as expected).
- Sifted key rate (`throughput_bps`) is computed; it was always 0.
- SeQUeNCe's photon/state safety checks are active again (v0.1 had switched
  them off).

**Link loss**

- Loss can be entered as a measured value in dB, with an itemised loss budget
  (fibre, splices, connectors, ADMs). `quila_sim.core.LossBudget`.
- MAQAN preset: IITM-ERNET 7.5 dB, ERNET-SETS 2.0 dB, SETS-NIC 3.2 dB.

**App**

- Engine selector, link presets, "Use measured link loss" switch, "Loss
  budget..." dialog, Channel Loss and Sifted Key Rate statistics.
- Pulse count up to 10 million (log-scale slider); default 100,000.
- The replay, event log and key display show the first 2,000 slots; all
  statistics use the whole run.
- "Total Loss" statistic renamed "Slots Without Click".

**Tests**

- 26 new tests (engine vs theory, multi-window fix, double clicks, dead time,
  loss budgets, MAQAN preset). 44 tests in total.

## v0.1.0 - baseline

DPS-QKD desktop simulator as imported (tag `v0.1.0-baseline`).
