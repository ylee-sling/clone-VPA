# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Bit-exact codecs of the three message types of Table 1.

All messages are little-endian, ``struct``-packed, and end with a CRC-16 over
every preceding byte (CRC-16/CCITT-FALSE: polynomial ``0x1021``, initial value
``0xFFFF``, no reflection, no final XOR; check value ``0x29B1`` for
``b"123456789"``). Sizes are payload bytes, excluding transport framing:

=====================  ==========================================================  =====
message                layout (``struct`` format)                                  bytes
=====================  ==========================================================  =====
uplink (clone->master) ``<B H I 24s B B`` + ``<H`` CRC                             35
downlink (master->cl.) ``<B H I B B 24s B`` + ``<H`` CRC                           36
safety broadcast       ``<B B I 3h 3h B B`` + ``<H`` CRC                           22
=====================  ==========================================================  =====

Field semantics
---------------
* ``timestamp``: microseconds modulo ``2^32`` (wraps every 4294.97 s);
  :func:`unwrap_timestamp` maps it back to the receiver's clock.
* Uplink ``uncertainty``: ``sigma_bar_{t+1}`` in steps of ``1/64``, rounded up
  (a receiver never sees less uncertainty than the sender had), saturating at 255.
* Uplink ``flags``: :data:`UP_FALLBACK`, :data:`UP_BRAKING`, :data:`UP_STALE_PLAN`,
  :data:`UP_NULL_PLAN`.
* Safety broadcast: position int16 at 1 cm, velocity int16 at 1 cm/s (rounded to
  nearest, saturating at +-327.67), effective radius uint8 cm rounded **up**,
  sequence number modulo 256, ``flags`` = executor flags (fallback, braking).

Downlink layout and the 36-byte total
-------------------------------------
Table 1 lists exactly eight downlink fields totalling 36 bytes: agent id (1),
sequence (2), timestamp (4), mode token (1), milestone index (1), plan RVQ code
(24), "plan uncertainty and validity" (1), CRC (2). A separate status-flags byte
would make the message 37 bytes and break Proposition 4(a). The flags are
therefore carried in the upper nibble of the mode-token byte: the clone mode
table has 16 rows (Table 3, "mode table 16 x 256"), so a local mode needs only
4 bits. The "plan uncertainty and validity" byte packs ``sigma_bar_M`` (upper
nibble, steps of :data:`SIGMA_M_STEP`, rounded up) and the validity (lower
nibble, in units of :data:`VALIDITY_UNIT` seconds from the timestamp; ``0``
means "no valid plan", which makes the clone use the null plan token).

Every decoder checks the exact length first and the CRC second and raises
:class:`PacketLengthError` / :class:`PacketCRCError` (both :class:`PacketError`).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch

__all__ = [
    "PacketError", "PacketLengthError", "PacketCRCError", "PacketFieldError",
    "crc16_ccitt",
    "UPLINK_SIZE", "DOWNLINK_SIZE", "SAFETY_SIZE",
    "UP_FALLBACK", "UP_BRAKING", "UP_STALE_PLAN", "UP_NULL_PLAN",
    "DN_NEW_EPOCH", "DN_MILESTONE_ADVANCED", "DN_DEGRADED", "DN_RESERVED",
    "SIGMA_UP_STEP", "SIGMA_M_STEP", "VALIDITY_UNIT",
    "UplinkPacket", "DownlinkPacket", "SafetyPacket",
    "timestamp_us", "unwrap_timestamp", "SequenceFilter",
    "encode_safety_broadcast", "decode_safety_broadcast", "SafetyUnits", "TABLE1_UNITS", "ARM_UNITS",
    "sphere_agent_id", "encode_sphere_broadcast", "decode_sphere_broadcast",
]


# ---------------------------------------------------------------------------
# errors and CRC
# ---------------------------------------------------------------------------
class PacketError(ValueError):
    """Base class of all decoding errors."""


class PacketLengthError(PacketError):
    """The byte string does not have the exact message length."""


class PacketCRCError(PacketError):
    """The CRC-16 does not match the payload."""


class PacketFieldError(PacketError):
    """A field value cannot be represented in its wire format."""


def _make_table() -> List[int]:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
        table.append(crc)
    return table


_CRC_TABLE = _make_table()


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE of ``data`` (poly 0x1021, init 0xFFFF, MSB first, no final XOR)."""
    crc = init
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc


def _seal(body: bytes) -> bytes:
    return body + struct.pack("<H", crc16_ccitt(body))


def _open(data: bytes, size: int, name: str) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise PacketError(f"{name}: expected bytes, got {type(data).__name__}")
    data = bytes(data)
    if len(data) != size:
        raise PacketLengthError(f"{name}: expected {size} bytes, got {len(data)}")
    body, (crc,) = data[:-2], struct.unpack("<H", data[-2:])
    if crc16_ccitt(body) != crc:
        raise PacketCRCError(f"{name}: CRC mismatch (got 0x{crc:04X}, expected 0x{crc16_ccitt(body):04X})")
    return body


def _u(value: int, bits: int, name: str) -> int:
    value = int(value)
    if not 0 <= value < (1 << bits):
        raise PacketFieldError(f"{name}={value} does not fit in {bits} unsigned bits")
    return value


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
_UP_FMT = "<BHI24sBB"
_DN_FMT = "<BHIBB24sB"
_SF_FMT = "<BBI3h3hBB"
UPLINK_SIZE = struct.calcsize(_UP_FMT) + 2      # 35
DOWNLINK_SIZE = struct.calcsize(_DN_FMT) + 2    # 36
SAFETY_SIZE = struct.calcsize(_SF_FMT) + 2      # 22
assert (UPLINK_SIZE, DOWNLINK_SIZE, SAFETY_SIZE) == (35, 36, 22)

UP_FALLBACK = 0x01       # executor in fallback (primary cause this tick)
UP_BRAKING = 0x02        # braking profile or position hold active
UP_STALE_PLAN = 0x04     # the last plan has expired (clone runs on the null plan)
UP_NULL_PLAN = 0x08      # no plan received yet in this mission

DN_NEW_EPOCH = 0x1       # the global mode u_g was resampled in this cycle
DN_MILESTONE_ADVANCED = 0x2
DN_DEGRADED = 0x4        # the master runs with a partial uplink set
DN_RESERVED = 0x8

SIGMA_UP_STEP = 1.0 / 64.0   # uplink uncertainty resolution
SIGMA_M_STEP = 0.125          # downlink plan-uncertainty resolution (4 bits)
VALIDITY_UNIT = 0.25          # downlink validity resolution [s] (4 bits, 0 = invalid)

_CM_LIMIT = 32767


@dataclass(frozen=True)
class SafetyUnits:
    """Resolution of the int16 position / velocity and uint8 radius fields of the safety packet.

    :data:`TABLE1_UNITS` (1 cm, 1 cm/s, 1 cm) is the aerial format of Table 1.
    :data:`ARM_UNITS` (1 mm, 1 mm/s, 1 mm) keeps the same 22-byte layout for the
    collision spheres of an arm (Task T2), whose collision distances are about
    0.1 m: positions up to +-32.767 m, speeds up to +-32.767 m/s, radii up to 0.255 m.
    """

    pos: float = 0.01
    vel: float = 0.01
    radius: float = 0.01


TABLE1_UNITS = SafetyUnits()
ARM_UNITS = SafetyUnits(0.001, 0.001, 0.001)


def _quant_up(x: float, step: float, max_code: int) -> int:
    if not math.isfinite(x):
        return max_code                      # unknown uncertainty is reported as maximal
    if x <= 0:
        return 0
    return min(max_code, int(math.ceil(x / step - 1e-9)))


def _cm(x: float, res: float = 0.01) -> int:
    return max(-_CM_LIMIT, min(_CM_LIMIT, int(round(x / res))))


# ---------------------------------------------------------------------------
# message types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UplinkPacket:
    """Clone -> master (35 bytes): RVQ code of ``P_bus z_t`` and ``sigma_bar_{t+1}``."""

    agent_id: int
    seq: int
    timestamp: int
    code: bytes
    uncertainty: int
    flags: int = 0

    @staticmethod
    def build(agent_id: int, seq: int, t_s: float, code: Sequence[int], sigma_bar: float,
              flags: int = 0) -> "UplinkPacket":
        """Quantize physical values (``t_s`` seconds, ``sigma_bar`` dimensionless)."""
        return UplinkPacket(agent_id, seq % 65536, timestamp_us(t_s), bytes(bytearray(int(c) for c in code)),
                            _quant_up(float(sigma_bar), SIGMA_UP_STEP, 255), flags)

    @property
    def sigma_bar(self) -> float:
        return self.uncertainty * SIGMA_UP_STEP

    def pack(self) -> bytes:
        if len(self.code) != 24:
            raise PacketFieldError(f"RVQ code must be 24 bytes, got {len(self.code)}")
        return _seal(struct.pack(_UP_FMT, _u(self.agent_id, 8, "agent_id"), _u(self.seq, 16, "seq"),
                                 _u(self.timestamp, 32, "timestamp"), bytes(self.code),
                                 _u(self.uncertainty, 8, "uncertainty"), _u(self.flags, 8, "flags")))

    @staticmethod
    def unpack(data: bytes) -> "UplinkPacket":
        a, s, t, code, unc, fl = struct.unpack(_UP_FMT, _open(data, UPLINK_SIZE, "uplink"))
        return UplinkPacket(a, s, t, code, unc, fl)


@dataclass(frozen=True)
class DownlinkPacket:
    """Master -> clone (36 bytes): mode token, milestone index and the plan's RVQ code.

    ``mode`` (0..15) and ``status`` (0..15) share the mode-token byte;
    ``sigma_code`` (0..15) and ``validity_code`` (0..15) share the
    uncertainty/validity byte (see the module docstring).
    """

    agent_id: int
    seq: int
    timestamp: int
    mode: int
    milestone: int
    code: bytes
    sigma_code: int
    validity_code: int
    status: int = 0

    @staticmethod
    def build(agent_id: int, seq: int, t_s: float, mode: int, milestone: int, code: Sequence[int],
              sigma_M: float, validity_s: float, status: int = 0) -> "DownlinkPacket":
        """Quantize physical values; ``validity_s <= 0`` encodes "no valid plan"."""
        v = 0 if validity_s <= 0 else max(1, min(15, int(math.floor(validity_s / VALIDITY_UNIT + 1e-9))))
        return DownlinkPacket(agent_id, seq % 65536, timestamp_us(t_s), mode, milestone,
                              bytes(bytearray(int(c) for c in code)),
                              _quant_up(float(sigma_M), SIGMA_M_STEP, 15), v, status)

    @property
    def sigma_M(self) -> float:
        return self.sigma_code * SIGMA_M_STEP

    @property
    def validity_s(self) -> float:
        return self.validity_code * VALIDITY_UNIT

    @property
    def plan_valid(self) -> bool:
        return self.validity_code > 0

    def pack(self) -> bytes:
        if len(self.code) != 24:
            raise PacketFieldError(f"RVQ code must be 24 bytes, got {len(self.code)}")
        mode_byte = (_u(self.status, 4, "status") << 4) | _u(self.mode, 4, "mode")
        uv_byte = (_u(self.sigma_code, 4, "sigma_code") << 4) | _u(self.validity_code, 4, "validity_code")
        return _seal(struct.pack(_DN_FMT, _u(self.agent_id, 8, "agent_id"), _u(self.seq, 16, "seq"),
                                 _u(self.timestamp, 32, "timestamp"), mode_byte,
                                 _u(self.milestone, 8, "milestone"), bytes(self.code), uv_byte))

    @staticmethod
    def unpack(data: bytes) -> "DownlinkPacket":
        a, s, t, mb, ms, code, uv = struct.unpack(_DN_FMT, _open(data, DOWNLINK_SIZE, "downlink"))
        return DownlinkPacket(a, s, t, mb & 0x0F, ms, code, uv >> 4, uv & 0x0F, mb >> 4)


@dataclass(frozen=True)
class SafetyPacket:
    """Agent -> all (22 bytes), 100 Hz: quantized state, effective radius and flags."""

    agent_id: int
    seq: int
    timestamp: int
    pos_cm: tuple
    vel_cms: tuple
    radius_cm: int
    flags: int = 0

    @staticmethod
    def build(agent_id: int, seq: int, t_s: float, pos: Sequence[float], vel: Sequence[float],
              radius: float, flags: int = 0, units: SafetyUnits = TABLE1_UNITS) -> "SafetyPacket":
        """Quantize metres, m/s and the radius (rounded up, never under-reported).

        The integer fields hold counts of ``units`` (centimetres for Table 1, the
        default; the field names keep the Table-1 units for readability).
        """
        r = min(255, max(0, int(math.ceil(float(radius) / units.radius - 1e-9))))
        return SafetyPacket(agent_id, seq % 256, timestamp_us(t_s), tuple(_cm(p, units.pos) for p in pos),
                            tuple(_cm(v, units.vel) for v in vel), r, flags)

    @property
    def pos(self) -> tuple:
        return self.position(TABLE1_UNITS)

    @property
    def vel(self) -> tuple:
        return self.velocity(TABLE1_UNITS)

    @property
    def radius(self) -> float:
        return self.radius_m(TABLE1_UNITS)

    def position(self, units: SafetyUnits = TABLE1_UNITS) -> tuple:
        return tuple(p * units.pos for p in self.pos_cm)

    def velocity(self, units: SafetyUnits = TABLE1_UNITS) -> tuple:
        return tuple(v * units.vel for v in self.vel_cms)

    def radius_m(self, units: SafetyUnits = TABLE1_UNITS) -> float:
        return self.radius_cm * units.radius

    def pack(self) -> bytes:
        if len(self.pos_cm) != 3 or len(self.vel_cms) != 3:
            raise PacketFieldError("position and velocity need three components")
        for v in (*self.pos_cm, *self.vel_cms):
            if not -32768 <= int(v) <= 32767:
                raise PacketFieldError(f"int16 overflow: {v}")
        return _seal(struct.pack(_SF_FMT, _u(self.agent_id, 8, "agent_id"), _u(self.seq, 8, "seq"),
                                 _u(self.timestamp, 32, "timestamp"), *map(int, self.pos_cm),
                                 *map(int, self.vel_cms), _u(self.radius_cm, 8, "radius_cm"),
                                 _u(self.flags, 8, "flags")))

    @staticmethod
    def unpack(data: bytes) -> "SafetyPacket":
        f = struct.unpack(_SF_FMT, _open(data, SAFETY_SIZE, "safety"))
        return SafetyPacket(f[0], f[1], f[2], tuple(f[3:6]), tuple(f[6:9]), f[9], f[10])


# ---------------------------------------------------------------------------
# time stamps and sequence numbers
# ---------------------------------------------------------------------------
_WRAP = 1 << 32


def timestamp_us(t_s: float) -> int:
    """Wire timestamp: ``round(t * 1e6) mod 2^32``."""
    return int(round(float(t_s) * 1e6)) % _WRAP


def unwrap_timestamp(ts_us: int, now_s: float) -> float:
    """Receiver time [s] congruent to ``ts_us`` modulo ``2^32`` us and nearest to ``now_s``."""
    now_us = int(round(now_s * 1e6))
    delta = (int(ts_us) - now_us) % _WRAP
    if delta >= _WRAP // 2:
        delta -= _WRAP
    return (now_us + delta) * 1e-6


class SequenceFilter:
    """Accept only messages newer than the last accepted one per sender (RFC 1982 arithmetic).

    ``bits`` is 16 for the latent bus and 8 for the safety broadcast; a gap
    larger than half the sequence space is treated as old (duplicate or
    reordered) and rejected.
    """

    def __init__(self, bits: int = 16):
        self.mod = 1 << bits
        self.last: dict = {}

    def accept(self, sender: int, seq: int) -> bool:
        prev = self.last.get(sender)
        if prev is not None:
            d = (seq - prev) % self.mod
            if d == 0 or d >= self.mod // 2:
                return False
        self.last[sender] = seq
        return True

    def reset(self, sender: Optional[int] = None) -> None:
        if sender is None:
            self.last.clear()
        else:
            self.last.pop(sender, None)


# ---------------------------------------------------------------------------
# wiring to the executor's SafetyBroadcast
# ---------------------------------------------------------------------------
def encode_safety_broadcast(bc, units: SafetyUnits = TABLE1_UNITS) -> List[bytes]:
    """Pack every agent of an executor :class:`~cma_vpa.safety.executor.SafetyBroadcast`.

    Returns one 22-byte string per agent, in batch order.
    """
    ids = bc.agent_id.tolist()
    seq = bc.seq.tolist()
    st = bc.stamp.tolist()
    pos = bc.pos.tolist()
    vel = bc.vel.tolist()
    rad = bc.radius.tolist()
    fl = bc.flags.tolist()
    return [SafetyPacket.build(ids[i], seq[i], st[i], pos[i], vel[i], rad[i], fl[i], units).pack()
            for i in range(len(ids))]


def decode_safety_broadcast(packets: Iterable[bytes], now_s: float, *, dtype: torch.dtype = torch.float64,
                            device: Optional[torch.device] = None, units: SafetyUnits = TABLE1_UNITS):
    """Decode 22-byte packets into a :class:`~cma_vpa.safety.executor.SafetyBroadcast`.

    Packets that fail the length or CRC check are dropped (the sender then
    counts as not heard, which Lemma 15 covers beyond ``R_link`` and the stale
    check covers within it). Timestamps are unwrapped against ``now_s``.
    """
    from ..safety.executor import SafetyBroadcast   # local import: comm does not depend on safety otherwise

    good: List[SafetyPacket] = []
    for raw in packets:
        try:
            good.append(SafetyPacket.unpack(raw))
        except PacketError:
            continue
    n = len(good)
    if n == 0:
        z = torch.zeros(0, dtype=dtype, device=device)
        return SafetyBroadcast(agent_id=torch.zeros(0, dtype=torch.long, device=device),
                               seq=torch.zeros(0, dtype=torch.long, device=device), stamp=z,
                               pos=z.reshape(0, 3), vel=z.reshape(0, 3), radius=z,
                               flags=torch.zeros(0, dtype=torch.uint8, device=device))
    return SafetyBroadcast(
        agent_id=torch.tensor([p.agent_id for p in good], dtype=torch.long, device=device),
        seq=torch.tensor([p.seq for p in good], dtype=torch.long, device=device),
        stamp=torch.tensor([unwrap_timestamp(p.timestamp, now_s) for p in good], dtype=dtype, device=device),
        pos=torch.tensor([p.position(units) for p in good], dtype=dtype, device=device),
        vel=torch.tensor([p.velocity(units) for p in good], dtype=dtype, device=device),
        radius=torch.tensor([p.radius_m(units) for p in good], dtype=dtype, device=device),
        flags=torch.tensor([p.flags for p in good], dtype=torch.uint8, device=device),
    )


# ---------------------------------------------------------------------------
# arms: one 22-byte packet per collision sphere (Task T2)
# ---------------------------------------------------------------------------
def sphere_agent_id(arm: int, sphere: int) -> int:
    """Wire ``agent_id`` of sphere ``sphere`` (0..7) of arm ``arm`` (0..31): ``arm * 8 + sphere``."""
    if not (0 <= arm < 32 and 0 <= sphere < 8):
        raise PacketFieldError(f"sphere id out of range: arm {arm}, sphere {sphere}")
    return arm * 8 + sphere


def encode_sphere_broadcast(bc, units: SafetyUnits = ARM_UNITS) -> List[bytes]:
    """Pack a :class:`~cma_vpa.safety.arm_barrier.SphereBroadcast`: one packet per sphere of every arm.

    The arm's flags (fallback, braking) are repeated in every sphere packet so
    that a receiver that loses some packets still sees them.
    """
    B, S = bc.pos.shape[:2]
    ids = bc.arm_id.tolist()
    seq = bc.seq.tolist()
    st = bc.stamp.tolist()
    pos = bc.pos.tolist()
    vel = bc.vel.tolist()
    rad = bc.radius.tolist()
    fl = bc.flags.tolist()
    return [SafetyPacket.build(sphere_agent_id(ids[b], k), seq[b], st[b], pos[b][k], vel[b][k], rad[b][k], fl[b],
                               units).pack() for b in range(B) for k in range(S)]


def decode_sphere_broadcast(packets: Iterable[bytes], now_s: float, units: SafetyUnits = ARM_UNITS) -> dict:
    """Decode sphere packets into ``{arm: {sphere: (pos, vel, radius, flags, stamp)}}``.

    Packets failing the length or CRC check are dropped.
    """
    out: dict = {}
    for raw in packets:
        try:
            p = SafetyPacket.unpack(raw)
        except PacketError:
            continue
        arm, k = divmod(p.agent_id, 8)
        out.setdefault(arm, {})[k] = (p.position(units), p.velocity(units), p.radius_m(units), p.flags,
                                      unwrap_timestamp(p.timestamp, now_s))
    return out

