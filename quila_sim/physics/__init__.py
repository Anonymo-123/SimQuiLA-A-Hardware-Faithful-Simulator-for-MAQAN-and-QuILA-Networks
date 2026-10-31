"""WP1 - Link physics: DPS link engines, measured loss budgets,
gated-detector and SNSPD models, DLI visibility and phase tracking.

Two DPS link models are provided:

* :class:`TimeBinDPSLink` (Model C) -- the MAQAN / IIT Madras hardware:
  disjoint 3-time-bin pulses, one every 32 ns, a one-bin-delay interferometer
  with a 1/6 : 1/3 : 1/3 : 1/6 slot split, and a single time-multiplexed gated
  detector. This is the default everywhere in the simulator.
* :class:`CoherentDPSLink` (Model B) -- a gap-free continuous pulse train with
  one key bit per slot (the textbook DPS of Inoue et al.). Kept for
  comparison; it is *not* how the MAQAN hardware works.

Issues: label ``wp1-link-physics`` (roadmap items 1-6). Implement
:class:`quila_sim.core.LinkModel`.
"""

from .dps_coherent import CoherentDPSLink, DPSRun, DPSTrace, closed_form_click_probabilities
from .dps_timebin import SOURCE_NAMES, TimeBinDPSLink, TimeBinRun, TimeBinTrace

__all__ = [
    "SOURCE_NAMES",
    "CoherentDPSLink",
    "DPSRun",
    "DPSTrace",
    "TimeBinDPSLink",
    "TimeBinRun",
    "TimeBinTrace",
    "closed_form_click_probabilities",
]
