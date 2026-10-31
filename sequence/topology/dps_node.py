"""DPSQKDNode -- network node for time-bin DPS-QKD (Model C, MAQAN hardware).

This module defines ``DPSQKDNode``, the DPS-QKD analogue of
:class:`~sequence.topology.node.QKDNode`. Alice and Bob are both
``DPSQKDNode`` objects; the role decides which hardware is used.

Node hardware
-------------
Every ``DPSQKDNode`` has **both** a
:class:`~sequence.components.dps_hardware.PhasedLightSource` (Alice's IM + PM
transmitter: one 3-bin pulse every 32 ns for MAQAN) **and** a
:class:`~sequence.components.dps_hardware.DelayLineInterferometer` (Bob's
one-bin-delay interferometer with its gated detector -- one time-multiplexed
SPD for MAQAN).

Protocol stack
--------------
    Layer 1  ->  Cascade (error correction)       [optional, stack_size >= 2]
    Layer 0  ->  DPSQKD  (time-bin sifting)       [always installed]

Photon routing
--------------
* Outgoing (Alice): ``PhasedLightSource.emit()`` schedules
  ``DPSQKDNode.get(photon)``, which sends the photon into the quantum channel.
* Incoming (Bob): ``Node.receive_qubit()`` forwards the photon to the first
  component, the DLI.

Classical messages are dispatched by ``msg.protocol_type`` (as in QKDNode).
"""

from __future__ import annotations

from math import inf
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    from ..components.photon import Photon
    from ..message import Message

import numpy as np

from .node import Node
from ..components.dps_hardware import PhasedLightSource, DelayLineInterferometer
from ..qkd.DPSQKD import DPSQKD, pair_dpsqkd_protocols
from ..qkd.cascade import Cascade, pair_cascade_protocols
from ..utils import log


class DPSQKDNode(Node):
    """Network node for Differential Phase Shift QKD.

    Analogous to :class:`~sequence.topology.node.QKDNode` but configured for
    DPS-QKD hardware and the DPS-QKD protocol stack.

    Hardware installed in every node:

    * ``<name>.lightsource`` — :class:`~sequence.components.dps_hardware.PhasedLightSource`
      (used when this node acts as Alice).
    * ``<name>.dli`` — :class:`~sequence.components.dps_hardware.DelayLineInterferometer`
      (used when this node acts as Bob; default first component for incoming
      qubits).

    Protocol stack (slot indices match QKDNode):

    * ``protocol_stack[0]`` — :class:`~sequence.qkd.DPSQKD.DPSQKD`  (sifting)
    * ``protocol_stack[1]`` — :class:`~sequence.qkd.cascade.Cascade` (error correction)

    Parameters
    ----------
    name : str
        Unique node identifier.
    timeline : Timeline
        Simulation timeline.
    stack_size : int
        Number of protocol-stack layers to install (default 5; 1 = DPSQKD
        only, 2 = DPSQKD + Cascade).
    seed : int, optional
        RNG seed for reproducibility.
    ls_params : dict, optional
        Keyword arguments forwarded to :class:`PhasedLightSource`:
        ``frequency``, ``wavelength``, ``mean_photon_num``, ``bins_per_pulse``,
        ``bin_width_ps``, ``extinction_ratio_db``, ``phase_error``.
    dli_params : dict, optional
        Keyword arguments forwarded to :class:`DelayLineInterferometer`:
        ``frequency``, ``bins_per_pulse``, ``bin_width_ps``, ``visibility``,
        ``pm_transition_ps``, ``layout``, ``tdm_delay_ps``, ``combiner_loss_db``,
        ``balance_loss_db``, ``efficiency``, ``dark_count_prob`` (per gate),
        ``holdoff_ps``, ``timing_jitter_ps``, ``gate_width_ps``,
        ``time_resolution``. Pulse rate and format must match Alice's.

    Attributes
    ----------
    destination : str or None
        Name of the far node; set by the DPSQKD protocol before each emit.
    protocol_stack : list[Protocol or None]
        Five-slot list: slot 0 = DPSQKD, slot 1 = Cascade, rest None.
    """

    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        stack_size: int = 5,
        seed: Optional[int] = None,
        ls_params: Optional[dict] = None,
        dli_params: Optional[dict] = None,
    ) -> None:
        """Construct a DPSQKDNode.

        Parameters
        ----------
        name : str
            Node name.
        timeline : Timeline
            Simulation timeline.
        stack_size : int
            How many protocol-stack layers to create (default 5).
            Use ``stack_size=1`` for raw DPS-QKD without Cascade.
        seed : int, optional
            Random-number-generator seed.
        ls_params : dict, optional
            Overrides for :class:`PhasedLightSource` constructor keyword
            arguments (e.g. ``{"mean_photon_num": 0.1}``).
        dli_params : dict, optional
            Overrides for :class:`DelayLineInterferometer` constructor keyword
            arguments (e.g. ``{"efficiency": 0.1, "dark_count_prob": 3e-6}``).
        """
        super().__init__(name, timeline, seed)
        ls_params  = ls_params  or {}
        dli_params = dli_params or {}

        self.destination: Optional[str] = None

        # ── PhasedLightSource (Alice's transmitter) ───────────────────
        ls_name      = name + ".lightsource"
        lightsource  = PhasedLightSource(ls_name, timeline, **ls_params)
        self.add_component(lightsource)
        lightsource.add_receiver(self)      # photon flow: LS → node → QC

        # ── DelayLineInterferometer (Bob's receiver) ──────────────────
        dli_name = name + ".dli"
        dli      = DelayLineInterferometer(dli_name, timeline, **dli_params)
        self.add_component(dli)
        self.set_first_component(dli_name)  # incoming qubits → DLI

        # ── Protocol stack ────────────────────────────────────────────
        self.protocol_stack = [None] * 5

        if stack_size > 0:
            dpsqkd = DPSQKD(self, name + ".DPSQKD", ls_name, dli_name)
            self.protocol_stack[0] = dpsqkd
            self.protocols.append(dpsqkd)

        if stack_size > 1:
            cascade = Cascade(self, name + ".cascade")
            self.protocol_stack[1] = cascade
            self.protocols.append(cascade)
            # Wire stack: DPSQKD ↕ Cascade
            self.protocol_stack[0].upper_protocols.append(cascade)
            cascade.lower_protocols.append(self.protocol_stack[0])

    # ------------------------------------------------------------------
    # Protocol stack helpers
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Validate that roles have been set before simulation start."""
        super().init()
        if self.protocol_stack[0] is not None:
            assert self.protocol_stack[0].role != -1, (
                f"Node '{self.name}': DPSQKD role not set. "
                "Call pair_dpsqkd_protocols() before tl.init()."
            )

    def set_protocol_layer(self, layer: int, protocol: Any) -> None:
        """Replace a protocol in the stack (mirrors QKDNode API).

        Parameters
        ----------
        layer : int
            Stack layer index (0–4).
        protocol : StackProtocol
            New protocol to install.
        """
        if not (0 <= layer <= 4):
            raise ValueError(f"layer must be 0–4; got {layer}")

        if self.protocol_stack[layer] is not None:
            self.protocols.remove(self.protocol_stack[layer])
        self.protocol_stack[layer] = protocol
        self.protocols.append(protocol)

        if layer > 0 and self.protocol_stack[layer - 1] is not None:
            self.protocol_stack[layer - 1].upper_protocols.append(protocol)
            protocol.lower_protocols.append(self.protocol_stack[layer - 1])

        if layer < 4 and self.protocol_stack[layer + 1] is not None:
            protocol.upper_protocols.append(self.protocol_stack[layer + 1])
            self.protocol_stack[layer + 1].lower_protocols.append(protocol)

    # ------------------------------------------------------------------
    # Hardware parameter helpers (mirrors QKDNode API)
    # ------------------------------------------------------------------

    def update_lightsource_params(self, arg_name: str, value: Any) -> None:
        """Set an attribute on the PhasedLightSource component.

        Parameters
        ----------
        arg_name : str
            Attribute name (``"frequency"``, ``"mean_photon_num"``, etc.).
        value : Any
            New value.
        """
        ls = self.components[self.name + ".lightsource"]
        ls.__setattr__(arg_name, value)

    def update_detector_params(
        self, detector_id: int, arg_name: str, value: Any
    ) -> None:
        """Set an attribute on one of the DLI's internal detectors.

        Parameters
        ----------
        detector_id : int
            0 for the single time-multiplexed detector ("tdm" layout);
            0 or 1 for the "dual" layout.
        arg_name : str
            Attribute name (``"efficiency"``, ``"dark_count_prob"``, ``"holdoff_ps"``, ...).
        value : Any
            New value.
        """
        dli = self.components[self.name + ".dli"]
        dli.update_detector_params(detector_id, arg_name, value)

    # ------------------------------------------------------------------
    # Photon routing overrides
    # ------------------------------------------------------------------

    def get(self, photon: "Photon", **kwargs) -> None:
        """Forward photon emitted by PhasedLightSource to the quantum channel.

        Called by the light source (via its ``_receivers[0]``).
        Mirrors ``QKDNode.get()``.

        Parameters
        ----------
        photon : Photon
            Newly created DPS photon.
        """
        self.send_qubit(self.destination, photon)

    # ------------------------------------------------------------------
    # Classical-message routing override
    # ------------------------------------------------------------------

    def receive_message(self, src: str, msg: "Message") -> None:
        """Route a classical message to the matching protocol.

        Dispatches by ``msg.protocol_type`` (same strategy as QKDNode).

        Parameters
        ----------
        src : str
            Name of the sending node.
        msg : Message
            Received message.
        """
        log.logger.debug(
            f"{self.name} receive_message from {src}: {msg.msg_type}"
        )
        for protocol in self.protocols:
            if type(protocol) == msg.protocol_type:
                protocol.received_message(src, msg)
                return

        raise Exception(
            f"Node '{self.name}': no protocol matched "
            f"msg.protocol_type='{msg.protocol_type}'"
        )
