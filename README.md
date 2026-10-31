# SimQuiLA

A simulator for QKD links and networks for the **QuILA** project, starting
from the Chennai **MAQAN** testbed. It is built on
[SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe) 0.8.5 and currently
simulates DPS-QKD links through a desktop app.

> **Status: v0.3.** The DPS link is modelled exactly as the MAQAN
> hardware works ("Model C", time-bin DPS): one 3-time-bin pulse every 32 ns,
> a one-bin-delay interferometer with a 1/6 : 1/3 : 1/3 : 1/6 slot split, and one
> time-multiplexed gated detector, with the published imperfections (jitter,
> phase-modulator transitions, IM leakage, dark counts, hold-off). Fast engine
> plus a SeQUeNCe per-photon reference; measured link loss with a MAQAN preset.

## Get started

Needs Python 3.11 or 3.12.

```bash
git clone <repository URL>
cd quila-sim
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest           # checks everything works (should say "passed")
python main.py   # opens the desktop app
```

## What's where

| Folder                | Contents                                                                              |
| --------------------- | ------------------------------------------------------------------------------------- |
| `main.py`, `gui/` | Desktop app                                                                           |
| `backend/`          | Connects the app to the simulation engine; Cascade error correction; statistics tools |
| `sequence/`         | SeQUeNCe 0.8.5 with our DPS-QKD additions (changes listed in `PATCHES.md`)          |
| `quila_sim/`        | New code for the QuILA work, one folder per work package                              |
| `scenarios/`        | Network descriptions, e.g. `maqan.yaml`                                             |
| `tests/`            | Automatic tests                                                                       |

## Licence and credits

- Our code: see [`LICENSE`](LICENSE).
- `sequence/` comes from SeQUeNCe (© UChicago Argonne, LLC) and is covered by
  [`LICENSE-SeQUeNCe`](LICENSE-SeQUeNCe); keep that file.

If you publish results from this simulator, please cite SeQUeNCe:
X. Wu et al., *Quantum Sci. Technol.* **6**, 045027 (2021).
