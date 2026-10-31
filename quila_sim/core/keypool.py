"""Key-pool contract (DRAFT v0).

A key pool holds secret key shared by one pair of nodes. QKD links deposit
key; applications and trusted-relay forwarding withdraw it. WP5 builds key
management, relaying and routing on top of this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class KeyPool(Protocol):
    """Shared-key buffer between two nodes."""

    endpoints: tuple[str, str]

    @property
    def level_bits(self) -> int:
        """Bits currently available."""
        ...

    def deposit(self, n_bits: int, t_s: float) -> int:
        """Add key; returns the number of bits actually stored (capacity limited)."""
        ...

    def withdraw(self, n_bits: int, t_s: float, consumer: str = "") -> bool:
        """Remove key if enough is available; returns False (and removes nothing) otherwise."""
        ...


@dataclass
class InMemoryKeyPool:
    """Minimal reference implementation, useful for tests and early WP5 work."""

    endpoints: tuple[str, str]
    capacity_bits: int | None = None
    _level: int = 0
    deposited_bits: int = 0
    withdrawn_bits: int = 0
    discarded_bits: int = 0
    denied_requests: int = 0
    last_update_s: float = 0.0

    @property
    def level_bits(self) -> int:
        return self._level

    def deposit(self, n_bits: int, t_s: float) -> int:
        if n_bits < 0:
            raise ValueError("n_bits must be non-negative")
        room = n_bits if self.capacity_bits is None else max(0, self.capacity_bits - self._level)
        stored = min(n_bits, room)
        self._level += stored
        self.deposited_bits += stored
        self.discarded_bits += n_bits - stored
        self.last_update_s = t_s
        return stored

    def withdraw(self, n_bits: int, t_s: float, consumer: str = "") -> bool:
        if n_bits < 0:
            raise ValueError("n_bits must be non-negative")
        self.last_update_s = t_s
        if n_bits > self._level:
            self.denied_requests += 1
            return False
        self._level -= n_bits
        self.withdrawn_bits += n_bits
        return True
