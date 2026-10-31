"""QuILA-Sim: QKD link and network simulation for MAQAN and QuILA.

New code for the QuILA work packages lives in this package. The v0.1
desktop simulator (``gui/``, ``backend/``) and the vendored SeQUeNCe engine
(``sequence/``) are outside it.

Subpackages
-----------
core            Shared contracts every work package builds against (WP0).
physics         Link physics: coherent-state engine, detectors, loss, DLI (WP1).
ingest          Importers for measured MAQAN data (WP2).
postprocessing  Key-rate models, Cascade/LDPC, privacy amplification (WP3).
protocols       COW, DPS-MDI, DPS twin-field link models (WP4).
attacks         Attack models and detection statistics (WP4).
network         Multi-port nodes, key pools, trusted relays, routing (WP5).
planning        Fibre routes, relay placement, cost models (WP6).
interfaces      ETSI GS QKD 014/015 emulation, service API (WP7).
"""

__version__ = "0.1.0.dev0"
