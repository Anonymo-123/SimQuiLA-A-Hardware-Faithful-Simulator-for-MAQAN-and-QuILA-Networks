"""DPS-QKD sifting protocol for time-bin DPS (Model C, the MAQAN / IIT Madras system).

Layer-0 protocol of the DPS-QKD stack, parallel to ``BB84`` in
``sequence.qkd.BB84``.

How the time-bin protocol works
-------------------------------
Alice sends one pulse of N time bins (N = 3, 1 ns each) every pulse period
(32 ns), with a random phase 0 or pi in each bin. Bob's one-bin-delay
interferometer overlaps bin j-1 with bin j, so a photon of pulse k that is
detected in interference slot j (j = 1 .. N-1) carries the bit

    b(k, j) = phi(k, j) XOR phi(k, j-1)      (0 -> port 0, 1 -> port 1).

Bob time-stamps his clicks. From the time stamp alone he knows the pulse k,
the slot j, and -- with MAQAN's single time-multiplexed detector -- the DLI
port, which is his bit. Clicks in the early slot (j = 0), the late slot
(j = N), a guard band, or outside the port windows carry no bit. Bob
announces only the key-slot ids k*N + j; Alice computes her bits from her
phase record.

Protocol stack interface
------------------------
``DPSQKD`` is a :class:`~sequence.protocol.StackProtocol` at layer 0: it gets
key-length requests from layer 1 (typically Cascade) via :meth:`push` and
hands finished keys up via ``_pop``.

Classical messages (per window of pulses)
-----------------------------------------
::

    Alice                                          Bob
      |-- BEGIN_PHOTON_PULSE ---------------------->|  window start, pulse
      |   (frequency, light_time, start_time,        |  format; Bob gates
      |    wavelength, mean_photon_num, num_pulses,  |  his detector
      |    bins_per_pulse, bin_width_ps)             |
      |== quantum channel: 3-bin pulses ===========>|
      |<-- DETECTION_TIMES (key-slot ids k*N+j) -----|  Bob's sifting

Functions
---------
pair_dpsqkd_protocols(alice, bob)
    Wire two DPSQKD instances together (before any ``push()``).
classify_clicks(...), sift_detections(...)
    Bob's time-stamp sifting rule, shared with the GUI adapter.

References
----------
Inoue, Waks, Yamamoto, PRL 89, 037902 (2002) -- DPS-QKD.
Shaw et al., arXiv:2008.03083 (2022) -- time-bin DPS at IIT Madras.
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy

if TYPE_CHECKING:
    from ..topology.node import Node

from quila_sim.core.timebin import KEY, OUTSIDE, FrameLayout, classify_times

from ..kernel.event import Event
from ..kernel.process import Process
from ..message import Message
from ..protocol import StackProtocol
from ..utils import log


# ---------------------------------------------------------------------------
# Pairing helper
# ---------------------------------------------------------------------------

def pair_dpsqkd_protocols(sender: "DPSQKD", receiver: "DPSQKD") -> None:
    """Wire two DPSQKD protocol instances together before simulation start.

    Sets ``another`` on both and assigns ``role`` 0 (Alice) / 1 (Bob).
    """
    sender.another = receiver
    receiver.another = sender
    sender.role = 0
    receiver.role = 1


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class DPSQKDMsgType(Enum):
    """Classical message types.

    BEGIN_PHOTON_PULSE  Alice -> Bob: a window of pulses starts; timing and
                        pulse format so Bob can gate his detector.
    DETECTION_TIMES     Bob -> Alice: key-slot ids (k*N + j) of his sifted
                        clicks in the window just finished.
    """
    BEGIN_PHOTON_PULSE = auto()
    DETECTION_TIMES = auto()


class DPSQKDMessage(Message):
    """Classical message between paired DPSQKD instances.

    BEGIN_PHOTON_PULSE fields: frequency (Hz), light_time (s), start_time (ps,
    Alice's emission of the window's first pulse), wavelength (nm),
    mean_photon_num (per pulse), num_pulses, bins_per_pulse, bin_width_ps.

    DETECTION_TIMES fields: detection_slots (key-slot ids k*N + j);
    detector_ids (optional, diagnostics only -- a real Bob never announces
    his bits).
    """

    def __init__(self, msg_type: DPSQKDMsgType, receiver: str, **kwargs) -> None:
        Message.__init__(self, msg_type, receiver)
        self.protocol_type = DPSQKD           # for node.receive_message routing
        if msg_type is DPSQKDMsgType.BEGIN_PHOTON_PULSE:
            self.frequency = kwargs["frequency"]
            self.light_time = kwargs["light_time"]
            self.start_time = kwargs["start_time"]
            self.wavelength = kwargs["wavelength"]
            self.mean_photon_num = kwargs["mean_photon_num"]
            self.num_pulses = kwargs["num_pulses"]
            self.bins_per_pulse = kwargs.get("bins_per_pulse", 3)
            self.bin_width_ps = kwargs.get("bin_width_ps", 1000)
        elif msg_type is DPSQKDMsgType.DETECTION_TIMES:
            self.detection_slots = kwargs["detection_slots"]
            self.detector_ids = kwargs.get("detector_ids", [])
        else:
            raise ValueError(f"DPSQKDMessage: unknown msg_type '{msg_type}'")


# ---------------------------------------------------------------------------
# Sifting rule (shared by the protocol and the GUI adapter)
# ---------------------------------------------------------------------------

def classify_clicks(
    times_ps: Sequence[int],
    detectors: Sequence[int],
    window_start_ps: int,
    layout: FrameLayout,
    num_pulses: int,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Read Bob's time stamps the way his TDC-based sifting does.

    ``window_start_ps`` is the arrival time at Bob's DLI of bin 0 of the
    window's first pulse. Each click is assigned to the pulse whose gate it is
    nearest (assignment windows are centred on the gates), then classified
    within that period by :func:`quila_sim.core.timebin.classify_times`.

    Returns (pulse, kind, port, slot, t_bins): pulse index in the window (-1
    if outside it); KEY/EARLY/LATE/GUARD/OUTSIDE; DLI port of the time window
    (-1 if outside); DLI output slot (-1 if outside); time within the period
    in bins.
    """
    det = numpy.asarray(detectors, dtype=numpy.int64)
    bin_ps = round(layout.bin_width_s * 1e12, 6)
    period_ps = layout.frame_bins * bin_ps
    # Work in picoseconds relative to the window first, and divide last, so a
    # click exactly on a slot edge is not moved across it by rounding.
    t_rel = numpy.asarray(times_ps, dtype=float) - float(window_start_ps)
    shift = (layout.gate_start - (layout.frame_bins - layout.gate_width) / 2.0) * bin_ps
    pulse = numpy.floor((t_rel - shift) / period_ps).astype(numpy.int64)
    t_bins = (t_rel - pulse * period_ps) / bin_ps
    kind, port, slot = classify_times(t_bins, det, layout)
    outside = (pulse < 0) | (pulse >= num_pulses)
    if outside.any():
        kind[outside] = OUTSIDE
        port[outside] = -1
        slot[outside] = -1
        pulse[outside] = -1
    return pulse, kind, port, slot, t_bins


def sift_detections(
    times_ps: Sequence[int],
    detectors: Sequence[int],
    window_start_ps: int,
    layout: FrameLayout,
    num_pulses: int,
    rng,
) -> Tuple[List[int], List[int]]:
    """Turn one window of clicks into Bob's sifted key.

    Only clicks in an interference slot (KEY) carry a bit: Bob's bit is the
    DLI port of the time window, and the key-slot id is ``k * N + j``. With
    one detector per port, a key slot in which both detectors fired gets a
    uniformly random bit (``rng``). Alice applies the same ids, so the keys
    stay aligned.

    Returns (key-slot ids in increasing order, Bob's bits).
    """
    pulse, kind, port, slot, _ = classify_clicks(times_ps, detectors, window_start_ps,
                                                 layout, num_pulses)
    n = layout.n_bins
    fired: dict = {}
    is_key = kind == KEY
    for k, j, b in zip(pulse[is_key].tolist(), slot[is_key].tolist(), port[is_key].tolist(),
                       strict=True):
        fired.setdefault(k * n + j, set()).add(b)
    ids = sorted(fired)
    bits = [int(rng.integers(0, 2)) if len(fired[i]) > 1 else next(iter(fired[i])) for i in ids]
    return ids, bits


def alice_bits_for(phase_blocks: numpy.ndarray, slot_ids: Sequence[int]) -> List[int]:
    """Alice's bits for announced key-slot ids: phi(k, j) XOR phi(k, j-1)."""
    blocks = numpy.asarray(phase_blocks)
    n_pulses, n = blocks.shape
    bits = []
    for sid in slot_ids:
        k, j = divmod(int(sid), n)
        if 0 <= k < n_pulses and 1 <= j <= n - 1:
            bits.append(int(blocks[k, j] ^ blocks[k, j - 1]))
    return bits


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class DPSQKD(StackProtocol):
    """Time-bin DPS-QKD sifting protocol (one instance on Alice, one on Bob).

    **Alice (role 0)** -- ``start_protocol()`` announces a window and
    ``begin_photon_pulse()`` sends windows of random 3-bin phase blocks via
    :meth:`~sequence.components.dps_hardware.PhasedLightSource.emit`.

    **Bob (role 1)** -- on BEGIN_PHOTON_PULSE gates his detector
    (:meth:`~sequence.components.dps_hardware.DelayLineInterferometer.arm`)
    and at the end of every window sifts his time stamps
    (:func:`sift_detections`) and announces the key-slot ids.

    **Alice (DETECTION_TIMES)** -- computes her bits for the announced ids.

    Both sides accumulate bits until ``key_lengths[0]`` bits are available,
    then call ``set_key()`` and ``_pop()``.

    Attributes
    ----------
    ls_name, dli_name : str
        Names of the light source (Alice) and DLI (Bob) components.
    role : int
        0 = Alice, 1 = Bob (set by :func:`pair_dpsqkd_protocols`).
    light_time : float
        Window duration (s).
    phase_lists : list[numpy.ndarray]
        Alice: phase blocks (pulses x N) of the windows not yet answered.
    key_bits : list[int]
        Sifted bits accumulated so far.
    error_rates : list[float]
        Fraction of differing bits per finished key (diagnostics only).
    """

    def __init__(self, owner: "Node", name: str, lightsource: str, dli: str,
                 role: int = -1) -> None:
        if owner is None:        # allow construction without a node for tests
            return
        super().__init__(owner, name)
        self.ls_name: str = lightsource
        self.dli_name: str = dli
        self.role: int = role

        self.working: bool = False
        self.ready: bool = True
        self.light_time: float = 0.0
        self.ls_freq: float = 0.0
        self.start_time: int = 0
        self._num_pulses: int = 0
        self._window_start: int = 0
        self._window_ps: int = 0

        self.phase_lists: List[numpy.ndarray] = []
        self.key_bits: List[int] = []
        self.key: int = 0
        self.another: Optional["DPSQKD"] = None

        self.key_lengths: List[int] = []
        self.keys_left_list: List[int] = []
        self.end_run_times: List[int] = []

        self.latency: float = 0.0
        self.last_key_time: int = 0
        self.throughputs: List[float] = []
        self.error_rates: List[float] = []

        # single-window mode (see run_single_window)
        self._single_window: bool = False
        self.last_clicks: list = []                      # Bob: clicks of the last window
        self.last_phase_blocks: Optional[numpy.ndarray] = None   # Alice: phases of the last window

    # ------------------------------------------------------------------
    # StackProtocol interface
    # ------------------------------------------------------------------

    def push(self, length: int, key_num: int, run_time: float = math.inf) -> None:
        """Request ``key_num`` keys of ``length`` bits (call on Alice)."""
        if self.role != 0:
            raise AssertionError("push() must be called on the Alice instance (role=0)")
        log.logger.info(f"{self.name} push: keylen={length}, keynum={key_num}")
        self.key_lengths.append(length)
        self.another.key_lengths.append(length)
        self.keys_left_list.append(key_num)
        now = self.owner.timeline.now()
        end_run_time = math.inf if math.isinf(run_time) else int(run_time) + now
        self.end_run_times.append(end_run_time)
        self.another.end_run_times.append(end_run_time)
        if self.ready:
            self.ready = False
            self.working = True
            self.another.working = True
            self.start_protocol()

    def pop(self, info: int) -> None:
        """Layer-0 protocol: nothing below it pops keys up to it."""
        pass

    # ------------------------------------------------------------------
    # Alice
    # ------------------------------------------------------------------

    def _window_pulses(self, ls) -> int:
        """Pulses per window: enough for about ``key_length`` photons."""
        return max(1, round(self.key_lengths[0] / ls.mean_photon_num))

    def start_protocol(self) -> None:
        """Start a key-generation batch (Alice only)."""
        log.logger.debug(f"{self.name} start_protocol")
        if not self.key_lengths:
            self.ready = True
            return
        self.phase_lists, self.another.phase_lists = [], []
        self.key_bits, self.another.key_bits = [], []
        self.latency = self.another.latency = 0.0
        self.working = self.another.working = True
        self._single_window = self.another._single_window = False
        ls = self.owner.components[self.ls_name]
        self._announce(self._window_pulses(ls))

    def run_single_window(self, num_pulses: int) -> None:
        """Send exactly one window of ``num_pulses`` pulses and sift it (Alice only).

        No key-length bookkeeping: afterwards both sides hold the window's
        sifted bits in ``key_bits``, Alice its phases in ``last_phase_blocks``
        and Bob its clicks in ``last_clicks``. Used by the GUI's per-photon
        reference engine.
        """
        if self.role != 0:
            raise AssertionError("run_single_window() must be called on Alice (role=0)")
        if num_pulses < 1:
            raise ValueError("num_pulses must be at least 1")
        for proto in (self, self.another):
            proto.phase_lists, proto.key_bits = [], []
            proto.key_lengths, proto.keys_left_list = [], []
            proto.end_run_times = [math.inf]
            proto.working = True
            proto._single_window = True
        self.ready = False
        self._announce(int(num_pulses))

    def _announce(self, num_pulses: int) -> None:
        """Tell Bob a window is coming and schedule its emission (Alice)."""
        ls = self.owner.components[self.ls_name]
        self.ls_freq = ls.frequency
        self._num_pulses = num_pulses
        self.light_time = num_pulses * ls.period_ps * 1e-12
        cc_delay = self.owner.cchannels[self.another.owner.name].delay
        self.start_time = int(self.owner.timeline.now()) + round(cc_delay)
        msg = DPSQKDMessage(
            DPSQKDMsgType.BEGIN_PHOTON_PULSE, self.another.name,
            frequency=self.ls_freq, light_time=self.light_time, start_time=self.start_time,
            wavelength=ls.wavelength, mean_photon_num=ls.mean_photon_num,
            num_pulses=num_pulses, bins_per_pulse=ls.bins_per_pulse,
            bin_width_ps=ls.bin_width_ps,
        )
        self.owner.send_message(self.another.owner.name, msg)
        self.owner.timeline.schedule(Event(self.start_time,
                                           Process(self, "begin_photon_pulse", [])))
        self.last_key_time = self.owner.timeline.now()

    def begin_photon_pulse(self) -> None:
        """Send one window of pulses (Alice only), or end the batch."""
        log.logger.debug(f"{self.name} begin_photon_pulse")
        if self.working and self.owner.timeline.now() < self.end_run_times[0]:
            self.owner.destination = self.another.owner.name
            ls = self.owner.components[self.ls_name]
            blocks = self.owner.get_generator().integers(
                0, 2, size=(self._num_pulses, ls.bins_per_pulse), dtype=numpy.uint8)
            ls.emit(blocks)
            self.phase_lists.append(blocks)
            self.last_phase_blocks = blocks
            self.start_time = self.owner.timeline.now()
            if self._single_window:
                return
            next_start = self.start_time + self._num_pulses * ls.period_ps
            self.owner.timeline.schedule(Event(next_start, Process(self, "begin_photon_pulse", [])))
        else:
            self.working = self.another.working = False
            self.key_lengths.pop(0)
            self.keys_left_list.pop(0)
            self.end_run_times.pop(0)
            self.another.key_lengths.pop(0)
            self.another.end_run_times.pop(0)
            drain_time = (self.owner.timeline.now()
                          + self.owner.qchannels[self.another.owner.name].delay + 1)
            self.owner.timeline.schedule(Event(drain_time, Process(self, "start_protocol", [])))

    # ------------------------------------------------------------------
    # Bob
    # ------------------------------------------------------------------

    def end_photon_pulse(self) -> None:
        """Sift the finished window and announce the key-slot ids (Bob only)."""
        log.logger.debug(f"{self.name} end_photon_pulse")
        if not (self.working and self.owner.timeline.now() < self.end_run_times[0]):
            return
        dli = self.owner.components[self.dli_name]
        clicks = dli.collect()
        slot_ids, bits = sift_detections(
            [c.time_ps for c in clicks], [c.detector for c in clicks],
            self._window_start, dli.layout_info, self._num_pulses, self.owner.get_generator())
        self.key_bits.extend(bits)
        self.last_clicks = clicks
        log.logger.info(f"{self.name} window produced {len(bits)} key bits; "
                        f"total = {len(self.key_bits)}")

        if not self._single_window:
            # next window: windows follow each other without a gap
            self._window_start += self._window_ps
            self.start_time = self._window_start
            dli.arm(self._window_start, self._num_pulses)
            self.owner.timeline.schedule(Event(self._window_start + self._window_ps - 1,
                                               Process(self, "end_photon_pulse", [])))
        msg = DPSQKDMessage(DPSQKDMsgType.DETECTION_TIMES, self.another.name,
                            detection_slots=slot_ids)
        self.owner.send_message(self.another.owner.name, msg)

    # ------------------------------------------------------------------
    # Classical messages
    # ------------------------------------------------------------------

    def received_message(self, src: str, msg: "Message") -> bool:
        """Dispatch an incoming classical message."""
        if not self.working or self.owner.timeline.now() >= self.end_run_times[0]:
            return True

        if msg.msg_type is DPSQKDMsgType.BEGIN_PHOTON_PULSE:             # Bob
            self.ls_freq = msg.frequency
            self.light_time = msg.light_time
            self._num_pulses = msg.num_pulses
            dli = self.owner.components[self.dli_name]
            if (dli.layout_info.n_bins != msg.bins_per_pulse
                    or dli.bin_width_ps != msg.bin_width_ps
                    or abs(dli.frequency - msg.frequency) > 1e-6 * msg.frequency):
                raise ValueError(f"{self.name}: Bob's interferometer does not match Alice's "
                                 "pulse format (bins per pulse, bin width, pulse rate)")
            qc_delay = self.another.owner.qchannels[self.owner.name].delay
            self.start_time = int(msg.start_time) + qc_delay
            self._window_start = self.start_time
            self._window_ps = self._num_pulses * dli.period_ps
            dli.arm(self._window_start, self._num_pulses)
            self.owner.timeline.schedule(Event(self._window_start + self._window_ps - 1,
                                               Process(self, "end_photon_pulse", [])))

        elif msg.msg_type is DPSQKDMsgType.DETECTION_TIMES:              # Alice
            self._process_detection_times(msg.detection_slots)

        return True

    # ------------------------------------------------------------------
    # Alice: key extraction
    # ------------------------------------------------------------------

    def _process_detection_times(self, slot_ids: Sequence[int], *_unused) -> None:
        """Alice's bits for Bob's announced key-slot ids (oldest window first)."""
        if not self.phase_lists:
            log.logger.warning(f"{self.name} no phase list queued - ignoring")
            return
        blocks = self.phase_lists.pop(0)
        self.key_bits.extend(alice_bits_for(blocks, slot_ids))
        log.logger.info(f"{self.name} extracted bits from window; total = {len(self.key_bits)}")

        if self.key_lengths and len(self.key_bits) >= self.key_lengths[0]:
            throughput = (self.key_lengths[0] * 1e12
                          / max(1, self.owner.timeline.now() - self.last_key_time))
            while (self.key_lengths and len(self.key_bits) >= self.key_lengths[0]
                   and self.keys_left_list[0] > 0):
                log.logger.info(f"{self.name} generated a valid key")
                self.set_key()
                self._pop(info=self.key)
                self.another.set_key()
                self.another._pop(info=self.another.key)
                if self.latency == 0.0:
                    self.latency = (self.owner.timeline.now() - self.last_key_time) * 1e-12
                self.throughputs.append(throughput)
                num_errors = bin(self.key ^ self.another.key).count("1")
                self.error_rates.append(num_errors / max(1, self.key_lengths[0]))
                self.keys_left_list[0] -= 1
            self.last_key_time = self.owner.timeline.now()
            if self.keys_left_list and self.keys_left_list[0] < 1:
                self.working = self.another.working = False

    # ------------------------------------------------------------------
    # Key formatting
    # ------------------------------------------------------------------

    def set_key(self) -> None:
        """Move the first ``key_lengths[0]`` bits into ``self.key`` (an int)."""
        key_len = self.key_lengths[0]
        key_bits = self.key_bits[:key_len]
        del self.key_bits[:key_len]
        self.key = int("".join(str(b) for b in key_bits), 2)
