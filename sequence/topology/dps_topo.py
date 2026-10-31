"""DPSQKDTopo — topology manager for Differential Phase Shift QKD networks.

This module defines :class:`DPSQKDTopo`, the DPS-QKD counterpart of
:class:`~sequence.topology.qkd_topo.QKDTopo`.  It loads a complete two-node
(or multi-node) DPS-QKD network from a JSON configuration file or a Python
``dict``, wiring up nodes, channels, and protocol pairs automatically.

JSON schema
-----------
The configuration dictionary / file follows the base
:class:`~sequence.topology.topology.Topology` schema with the addition of a
custom ``"DPSQKDNode"`` node type and optional per-node component templates:

.. code-block:: json

    {
        "stop_time": 1e12,
        "nodes": [
            {
                "name":  "alice",
                "type":  "DPSQKDNode",
                "seed":  0,
                "stack_size": 1,
                "ls_params":  {"frequency": 31.25e6, "mean_photon_num": 0.2},
                "dli_params": {}
            },
            {
                "name":  "bob",
                "type":  "DPSQKDNode",
                "seed":  1,
                "stack_size": 1,
                "ls_params":  {},
                "dli_params": {"efficiency": 0.1, "dark_count_prob": 3e-6,
                               "visibility": 0.92}
            }
        ],
        "qchannels": [
            {"source": "alice", "destination": "bob",
             "distance": 1000, "attenuation": 0.0002,
             "polarization_fidelity": 1.0}
        ],
        "cchannels": [
            {"source": "alice", "destination": "bob", "distance": 1000},
            {"source": "bob",   "destination": "alice", "distance": 1000}
        ],
        "pairs": [
            {"alice": "alice", "bob": "bob"}
        ]
    }

A ``"pairs"`` list in the configuration (or the positional pairing heuristic)
determines which DPSQKD protocol instances are paired via
:func:`~sequence.qkd.DPSQKD.pair_dpsqkd_protocols` and, if the Cascade layer
is installed, via
:func:`~sequence.qkd.cascade.pair_cascade_protocols`.

The class also provides the convenience factory method
:meth:`DPSQKDTopo.from_nodes` for programmatic (no-JSON) network creation.

Example (no JSON)
-----------------
.. code-block:: python

    from sequence.topology.dps_topo import DPSQKDTopo

    topo = DPSQKDTopo.from_nodes(
        "alice", "bob",
        distance=10e3,
        attenuation=2e-4,
        ls_params={"frequency": 31.25e6, "mean_photon_num": 0.2},
        dli_params={"efficiency": 0.1, "dark_count_prob": 3e-6, "visibility": 0.92},
        stack_size=1,
        runtime=2e12,
    )
    tl = topo.get_timeline()
    alice_proto = topo.get_protocol("alice")
    bob_proto   = topo.get_protocol("bob")
"""

from __future__ import annotations

import json
import copy
from collections import defaultdict
from typing import Optional, Union

from .node import Node
from .dps_node import DPSQKDNode
from .topology import Topology
from ..kernel.timeline import Timeline
from ..components.optical_channel import QuantumChannel, ClassicalChannel
from ..qkd.DPSQKD import DPSQKD, pair_dpsqkd_protocols
from ..qkd.cascade import Cascade, pair_cascade_protocols
from ..utils import log


class DPSQKDTopo(Topology):
    """Topology manager for Differential Phase Shift QKD networks.

    Subclasses :class:`~sequence.topology.topology.Topology` and overrides
    ``_load`` to support the ``"DPSQKDNode"`` node type and the ``"pairs"``
    section for automatic protocol wiring.

    Attributes
    ----------
    nodes : dict[str, list[DPSQKDNode]]
        Mapping of node-type string → list of node objects.
    qchannels : list[QuantumChannel]
        All quantum channels in the network.
    cchannels : list[ClassicalChannel]
        All classical channels in the network.
    tl : Timeline
        Shared simulation timeline.
    """

    DPS_QKD_NODE = "DPSQKDNode"

    # ------------------------------------------------------------------
    # Topology._load implementation
    # ------------------------------------------------------------------

    def _load(self, config_source: Union[str, dict]) -> None:
        """Parse configuration and build the network.

        Parameters
        ----------
        config_source : str or dict
            Path to a JSON file (str) or a configuration dictionary (dict).
        """
        if isinstance(config_source, str):
            with open(config_source) as fh:
                config = json.load(fh)
        elif isinstance(config_source, dict):
            config = copy.deepcopy(config_source)
        else:
            raise TypeError(
                f"config_source must be str or dict, got {type(config_source)}"
            )

        self._get_templates(config)
        self._add_timeline(config)
        self._add_nodes(config)
        self._add_qchannels(config)
        self._add_cchannels(config)
        self._add_cconnections(config)
        self._add_pairs(config)

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _add_timeline(self, config: dict) -> None:
        stop_time = config.get(Topology.STOP_TIME, float("inf"))
        self.tl   = Timeline(stop_time)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _add_nodes(self, config: dict) -> None:
        for node_cfg in config.get(Topology.ALL_NODE, []):
            node_type  = node_cfg[Topology.TYPE]
            node_name  = node_cfg[Topology.NAME]
            seed       = node_cfg.get(Topology.SEED, None)
            stack_size = node_cfg.get("stack_size", 5)
            ls_params  = node_cfg.get("ls_params",  {})
            dli_params = node_cfg.get("dli_params", {})

            if node_type == self.DPS_QKD_NODE:
                node = DPSQKDNode(
                    node_name,
                    self.tl,
                    stack_size = stack_size,
                    seed       = seed,
                    ls_params  = ls_params,
                    dli_params = dli_params,
                )
            else:
                raise NotImplementedError(
                    f"DPSQKDTopo: unknown node type '{node_type}'"
                )

            if node_type in self.nodes:
                self.nodes[node_type].append(node)
            else:
                self.nodes[node_type] = [node]

    # ------------------------------------------------------------------
    # Protocol pairing
    # ------------------------------------------------------------------

    def _add_pairs(self, config: dict) -> None:
        """Wire DPSQKD (and optionally Cascade) protocol pairs.

        Reads an optional ``"pairs"`` list from the configuration:

        .. code-block:: json

            "pairs": [
                {"alice": "node_name_alice", "bob": "node_name_bob"}
            ]

        If no ``"pairs"`` key is present, the method pairs the first
        ``DPSQKDNode`` listed with the second (positional fallback for the
        common two-node case).
        """
        dps_nodes = self.nodes.get(self.DPS_QKD_NODE, [])

        pairs_cfg = config.get("pairs", None)

        if pairs_cfg:
            node_map = {n.name: n for n in dps_nodes}
            for pair in pairs_cfg:
                alice_node = node_map[pair["alice"]]
                bob_node   = node_map[pair["bob"]]
                self._wire_pair(alice_node, bob_node)
        else:
            # Positional fallback: pair list[0] with list[1]
            if len(dps_nodes) >= 2:
                self._wire_pair(dps_nodes[0], dps_nodes[1])

    def _wire_pair(
        self,
        alice_node: "DPSQKDNode",
        bob_node: "DPSQKDNode",
    ) -> None:
        """Pair the DPSQKD (and Cascade if present) protocols on two nodes.

        Parameters
        ----------
        alice_node : DPSQKDNode
            The sending-side node (will be assigned role = 0).
        bob_node : DPSQKDNode
            The receiving-side node (will be assigned role = 1).
        """
        alice_dps = alice_node.protocol_stack[0]
        bob_dps   = bob_node.protocol_stack[0]

        if alice_dps is None or bob_dps is None:
            raise RuntimeError(
                "Cannot pair: one or both nodes have no DPSQKD protocol "
                f"(alice_stack[0]={alice_dps}, bob_stack[0]={bob_dps})"
            )

        pair_dpsqkd_protocols(alice_dps, bob_dps)

        # Cascade layer
        alice_cas = alice_node.protocol_stack[1]
        bob_cas   = bob_node.protocol_stack[1]
        if alice_cas is not None and bob_cas is not None:
            pair_cascade_protocols(alice_cas, bob_cas)

    # ------------------------------------------------------------------
    # Convenience factory (no JSON required)
    # ------------------------------------------------------------------

    @classmethod
    def from_nodes(
        cls,
        alice_name: str,
        bob_name: str,
        distance: float = 1e3,
        attenuation: float = 2e-4,
        polarization_fidelity: float = 1.0,
        classical_delay: int = 0,
        ls_params: Optional[dict] = None,
        dli_params: Optional[dict] = None,
        stack_size: int = 5,
        runtime: float = float("inf"),
        alice_seed: int = 0,
        bob_seed: int = 1,
    ) -> "DPSQKDTopo":
        """Build a two-node DPS-QKD topology without a JSON file.

        Parameters
        ----------
        alice_name : str
            Name for the Alice node.
        bob_name : str
            Name for the Bob node.
        distance : float
            Quantum-channel length in metres (default 1 km).
        attenuation : float
            Fibre attenuation in dB/m (default 2 × 10⁻⁴ dB/m ≈ 0.2 dB/km).
        polarization_fidelity : float
            Polarization-fidelity of the quantum channel (default 1.0).
        classical_delay : int
            Additional classical-channel delay in ps (default 0).
        ls_params : dict, optional
            Keyword arguments for :class:`PhasedLightSource`.
        dli_params : dict, optional
            Keyword arguments for :class:`DelayLineInterferometer`.
        stack_size : int
            Number of protocol-stack layers (default 5).
        runtime : float
            Timeline stop time in ps (default inf).
        alice_seed : int
            RNG seed for Alice (default 0).
        bob_seed : int
            RNG seed for Bob (default 1).

        Returns
        -------
        DPSQKDTopo
            Fully wired topology ready for ``tl.init()`` / ``tl.run()``.
        """
        config = {
            Topology.STOP_TIME: runtime,
            Topology.ALL_NODE: [
                {
                    Topology.NAME: alice_name,
                    Topology.TYPE: cls.DPS_QKD_NODE,
                    Topology.SEED: alice_seed,
                    "stack_size": stack_size,
                    "ls_params":  ls_params  or {},
                    "dli_params": dli_params or {},
                },
                {
                    Topology.NAME: bob_name,
                    Topology.TYPE: cls.DPS_QKD_NODE,
                    Topology.SEED: bob_seed,
                    "stack_size": stack_size,
                    "ls_params":  ls_params  or {},
                    "dli_params": dli_params or {},
                },
            ],
            Topology.ALL_Q_CHANNEL: [
                {
                    Topology.SRC: alice_name,
                    Topology.DST: bob_name,
                    Topology.DISTANCE: distance,
                    Topology.ATTENUATION: attenuation,
                    "polarization_fidelity": polarization_fidelity,
                }
            ],
            Topology.ALL_C_CONNECT: [
                {
                    Topology.CONNECT_NODE_1: alice_name,
                    Topology.CONNECT_NODE_2: bob_name,
                    Topology.DISTANCE: distance,
                    Topology.DELAY: classical_delay,
                }
            ],
            "pairs": [
                {"alice": alice_name, "bob": bob_name}
            ],
        }
        return cls(config)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_protocol(self, node_name: str) -> Optional["DPSQKD"]:
        """Return the DPSQKD protocol installed on the named node.

        Parameters
        ----------
        node_name : str
            Node name.

        Returns
        -------
        DPSQKD or None
        """
        for node in self.nodes.get(self.DPS_QKD_NODE, []):
            if node.name == node_name:
                return node.protocol_stack[0]
        return None

    def get_node(self, node_name: str) -> Optional["DPSQKDNode"]:
        """Return a named DPSQKDNode.

        Parameters
        ----------
        node_name : str
            Node name.

        Returns
        -------
        DPSQKDNode or None
        """
        for node in self.nodes.get(self.DPS_QKD_NODE, []):
            if node.name == node_name:
                return node
        return None

    def get_dps_nodes(self):
        """Return all DPSQKDNode instances in the topology."""
        return list(self.nodes.get(self.DPS_QKD_NODE, []))
