"""Shared contracts for all work packages (DRAFT v0).

These interfaces let the network layer, planning tools and GUI swap link
models of different fidelity without code changes. They are a draft: review
and freeze them in the Phase 0 issue "Define and freeze shared interfaces"
before feature work starts.

Unit convention for everything in ``quila_sim``: SI units with the unit in
the name (``_hz``, ``_s``, ``_km``, ``_db``, ``_bps``). SeQUeNCe internals
use picoseconds and dB/m; convert at the boundary, never inside models.
"""

from .keypool import InMemoryKeyPool, KeyPool
from .link import DetectionRecord, DetectorParams, LinkModel, LinkParams, LinkPerformance
from .loss import LossBudget, LossEvent
from .scenario import Link, Node, Scenario, load_preset, load_scenario
from .timebin import MAQAN_DPS, DPSReceiver, FrameLayout, PulseFormat, maqan_link_params

__all__ = [
    "MAQAN_DPS",
    "DPSReceiver",
    "DetectionRecord",
    "DetectorParams",
    "FrameLayout",
    "InMemoryKeyPool",
    "KeyPool",
    "Link",
    "LinkModel",
    "LinkParams",
    "LinkPerformance",
    "LossBudget",
    "LossEvent",
    "Node",
    "PulseFormat",
    "Scenario",
    "load_preset",
    "load_scenario",
    "maqan_link_params",
]
