"""Measured link-loss budgets (roadmap item 3).

Field fibre loss is dominated by splices, connectors, patch panels and
add-drop multiplexers, not by the fibre's attenuation per kilometre. For
example, MAQAN's IITM-ERNET link loses ~7.5 dB over 4.8 km (~1.5 dB/km),
about seven times what 0.2 dB/km fibre alone would give. A link's loss
should therefore be entered as a *measured* number, optionally with an
itemised breakdown that explains it.

Rules
-----
* ``measured_total_db`` (insertion-loss or OTDR end-to-end measurement) is
  the ground truth whenever it is given.
* The itemised part (fibre attenuation x length + listed events) is an
  explanation. If both exist, ``unaccounted_db`` shows the difference, which
  is useful for spotting a dirty connector or a missing event.
* Enter each physical feature once: a splice seen on the OTDR trace goes in
  ``events`` once, not also as a separate splice count.
* OTDR traces can show apparent *gains* at splices between different fibres
  (a backscatter artefact). Negative event losses are therefore accepted, but
  the total must not be negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Recognised event kinds. Anything else should use "other" with a label.
EVENT_KINDS = ("splice", "connector", "adm", "wdm", "filter", "bend", "patch-panel", "other")


@dataclass(frozen=True)
class LossEvent:
    """One localised loss: a splice, connector, add-drop multiplexer, etc.

    kind         One of :data:`EVENT_KINDS`.
    loss_db      Insertion loss in dB (may be negative only for OTDR gainers).
    position_km  Distance from the transmitter, if known (e.g. from an OTDR trace).
    label        Free text, e.g. "patch panel, ERNET rack 3".
    """

    kind: str
    loss_db: float
    position_km: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown loss event kind {self.kind!r}; use one of {EVENT_KINDS}")
        if self.position_km is not None and self.position_km < 0:
            raise ValueError("position_km must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "loss_db": self.loss_db}
        if self.position_km is not None:
            out["position_km"] = self.position_km
        if self.label:
            out["label"] = self.label
        return out


@dataclass(frozen=True)
class LossBudget:
    """Loss of one fibre link from Alice's output to Bob's input, in dB.

    fibre_length_km              Fibre length.
    fibre_attenuation_db_per_km  Intrinsic fibre attenuation (e.g. 0.2 at 1550 nm).
    events                       Localised losses (splices, connectors, ADMs, ...).
    measured_total_db            Measured end-to-end loss; overrides the itemised sum.
    source                       Where the numbers come from (OTDR file, date, paper).
    """

    fibre_length_km: float = 0.0
    fibre_attenuation_db_per_km: float = 0.0
    events: tuple[LossEvent, ...] = field(default_factory=tuple)
    measured_total_db: float | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if self.fibre_length_km < 0 or self.fibre_attenuation_db_per_km < 0:
            raise ValueError("fibre length and attenuation must be non-negative")
        if self.measured_total_db is not None and self.measured_total_db < 0:
            raise ValueError("measured_total_db must be non-negative")
        if self.measured_total_db is None and self.itemised_db < 0:
            raise ValueError("itemised loss is negative; check the event list")

    # -- totals ---------------------------------------------------------------
    @property
    def fibre_db(self) -> float:
        return self.fibre_length_km * self.fibre_attenuation_db_per_km

    @property
    def events_db(self) -> float:
        return sum(e.loss_db for e in self.events)

    @property
    def itemised_db(self) -> float:
        return self.fibre_db + self.events_db

    @property
    def total_db(self) -> float:
        """Loss used by the simulator: the measurement if present, else the itemised sum."""
        return self.measured_total_db if self.measured_total_db is not None else self.itemised_db

    @property
    def unaccounted_db(self) -> float | None:
        """Measured minus itemised loss, or None if there is nothing to compare."""
        if self.measured_total_db is None or (self.fibre_db == 0 and not self.events):
            return None
        return self.measured_total_db - self.itemised_db

    @property
    def transmittance(self) -> float:
        return 10.0 ** (-self.total_db / 10.0)

    def by_kind(self) -> dict[str, float]:
        """Itemised loss grouped by kind, with the fibre as its own entry."""
        out: dict[str, float] = {}
        if self.fibre_db:
            out["fibre"] = self.fibre_db
        for e in self.events:
            out[e.kind] = out.get(e.kind, 0.0) + e.loss_db
        return out

    def summary(self) -> str:
        """One-line human-readable description."""
        parts = [f"{self.total_db:.2f} dB total"]
        if self.measured_total_db is not None:
            parts.append("measured")
        items = ", ".join(f"{k} {v:.2f}" for k, v in self.by_kind().items())
        if items:
            parts.append(f"itemised: {items}")
        if self.unaccounted_db is not None:
            parts.append(f"unaccounted {self.unaccounted_db:+.2f} dB")
        return "; ".join(parts)

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_counts(
        cls,
        fibre_length_km: float,
        fibre_attenuation_db_per_km: float,
        *,
        splices: int = 0,
        splice_loss_db: float | None = None,
        connectors: int = 0,
        connector_loss_db: float | None = None,
        components: dict[str, float] | None = None,
        measured_total_db: float | None = None,
        source: str = "",
    ) -> LossBudget:
        """Build a budget from counts, e.g. 4 splices x 0.1 dB + 6 connectors x 0.3 dB.

        Per-item losses must be given explicitly whenever the count is non-zero.
        ``components`` maps a kind from :data:`EVENT_KINDS` (e.g. ``"adm"``)
        to its insertion loss.
        """
        events: list[LossEvent] = []
        for count, loss, kind in ((splices, splice_loss_db, "splice"),
                                  (connectors, connector_loss_db, "connector")):
            if count < 0:
                raise ValueError(f"{kind} count must be non-negative")
            if count and loss is None:
                raise ValueError(f"give {kind}_loss_db for {count} {kind}(s)")
            events.extend(LossEvent(kind, float(loss)) for _ in range(count))
        for kind, loss in (components or {}).items():
            events.append(LossEvent(kind, float(loss)))
        return cls(fibre_length_km, fibre_attenuation_db_per_km, tuple(events),
                   measured_total_db, source)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LossBudget:
        events = tuple(LossEvent(**e) for e in data.get("events", []) or [])
        return cls(
            fibre_length_km=float(data.get("fibre_length_km", 0.0) or 0.0),
            fibre_attenuation_db_per_km=float(data.get("fibre_attenuation_db_per_km", 0.0) or 0.0),
            events=events,
            measured_total_db=(None if data.get("measured_total_db") is None
                               else float(data["measured_total_db"])),
            source=str(data.get("source", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "fibre_length_km": self.fibre_length_km,
            "fibre_attenuation_db_per_km": self.fibre_attenuation_db_per_km,
            "events": [e.to_dict() for e in self.events],
        }
        if self.measured_total_db is not None:
            out["measured_total_db"] = self.measured_total_db
        if self.source:
            out["source"] = self.source
        return out
