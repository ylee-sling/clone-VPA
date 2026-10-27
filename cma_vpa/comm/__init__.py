# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Latent bus and safety channel (Definition 3, Table 1, Lemma 4, Propositions 4-5)."""

from .rvq import BUS_DIM, CODEBOOK_SIZE, N_STAGES, ResidualVQ, RVQOutput, gaussian_rd_floor, rd_floor
from .packet import (ARM_UNITS, TABLE1_UNITS, SafetyUnits, decode_sphere_broadcast, encode_sphere_broadcast,
                     sphere_agent_id)
from .packet import (DOWNLINK_SIZE, SAFETY_SIZE, UPLINK_SIZE, DownlinkPacket, PacketCRCError, PacketError,
                     PacketFieldError, PacketLengthError, SafetyPacket, SequenceFilter, UplinkPacket,
                     crc16_ccitt, decode_safety_broadcast, encode_safety_broadcast, timestamp_us,
                     unwrap_timestamp)

__all__ = [
    "ARM_UNITS", "TABLE1_UNITS", "SafetyUnits", "decode_sphere_broadcast", "encode_sphere_broadcast", "sphere_agent_id",
    "BUS_DIM", "CODEBOOK_SIZE", "N_STAGES", "ResidualVQ", "RVQOutput", "gaussian_rd_floor", "rd_floor",
    "DOWNLINK_SIZE", "SAFETY_SIZE", "UPLINK_SIZE", "DownlinkPacket", "PacketCRCError", "PacketError",
    "PacketFieldError", "PacketLengthError", "SafetyPacket", "SequenceFilter", "UplinkPacket",
    "crc16_ccitt", "decode_safety_broadcast", "encode_safety_broadcast", "timestamp_us", "unwrap_timestamp",
]
