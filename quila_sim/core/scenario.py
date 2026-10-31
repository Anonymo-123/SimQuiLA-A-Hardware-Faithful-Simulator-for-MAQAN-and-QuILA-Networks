"""Network scenario description (DRAFT v0).

A scenario lists nodes and fibre links with *measured* properties. Loaders
accept YAML (``.yaml``/``.yml``) or JSON. See ``scenarios/maqan.yaml``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loss import LossBudget

#: Folder holding the scenario presets shipped with the repository.
PRESETS_DIR = Path(__file__).resolve().parents[2] / "scenarios"


@dataclass(frozen=True)
class Node:
    name: str
    site: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Link:
    """A fibre link between two nodes.

    loss_db is the measured end-to-end loss in dB (the ground truth).
    loss_budget optionally itemises it (fibre, splices, connectors, ADMs);
    see :mod:`quila_sim.core.loss`. length_km is used for propagation delay.
    Any of these may be None when not yet measured; models must refuse to
    run on a link whose loss is unknown rather than guess.
    """

    name: str
    a: str
    b: str
    length_km: float | None = None
    loss_db: float | None = None
    protocols: tuple[str, ...] = ()
    notes: str = ""
    loss_budget: LossBudget | None = None

    @property
    def effective_loss_db(self) -> float | None:
        """Measured loss if given, otherwise the loss budget's total, otherwise None."""
        if self.loss_db is not None:
            return self.loss_db
        if self.loss_budget is not None:
            return self.loss_budget.total_db
        return None


@dataclass
class Scenario:
    name: str
    nodes: list[Node]
    links: list[Link]
    source: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate node names in scenario {self.name!r}")
        known = set(names)
        for link in self.links:
            for end in (link.a, link.b):
                if end not in known:
                    raise ValueError(f"link {link.name!r} refers to unknown node {end!r}")

    def node(self, name: str) -> Node:
        return next(n for n in self.nodes if n.name == name)

    def link(self, name: str) -> Link:
        return next(lk for lk in self.links if lk.name == name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        nodes = [Node(**n) for n in data.get("nodes", [])]
        links = []
        for raw in data.get("links", []):
            raw = dict(raw)
            raw["protocols"] = tuple(raw.get("protocols", ()) or ())
            if raw.get("loss_budget") is not None:
                raw["loss_budget"] = LossBudget.from_dict(raw["loss_budget"])
            links.append(Link(**raw))
        return cls(
            name=data["name"],
            nodes=nodes,
            links=links,
            source=data.get("source", ""),
            defaults=dict(data.get("defaults", {})),
        )


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # PyYAML, listed in requirements.txt

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return Scenario.from_dict(data)


def load_preset(name: str) -> Scenario:
    """Load a scenario shipped in the repository's ``scenarios/`` folder, e.g. ``"maqan"``."""
    for suffix in (".yaml", ".yml", ".json"):
        path = PRESETS_DIR / f"{name}{suffix}"
        if path.exists():
            return load_scenario(path)
    raise FileNotFoundError(f"no preset named {name!r} in {PRESETS_DIR}")
