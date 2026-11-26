"""splatscan live-stream protocol, version 1 (see docs/protocol.md for the normative spec).

One protocol message per WebSocket *binary* message. Everything is little-endian.

Header (12 bytes)::

    u32 magic = 0x31435353 ("SSC1")   u8 type   u8 flags (0)   u16 reserved (0)   u32 seq

Types::

    0x01 HELLO   u16 version, u8 sh_degree, u8 up_axis (1 = +y, 2 = +z), u32 capacity_hint, f32 scene_radius
    0x02 ADD     u32 n, then n splat records
    0x03 UPDATE  u32 n, then n splat records (replace every field of an existing id)
    0x04 REMOVE  u32 n, then n x u32 id
    0x05 FRAME   f64 t, u32 frame_index, u32 splat_count, f32 R_wc[9] (row-major), f32 p_wc[3],
                 f32 fx, fy, cx, cy, u16 width, u16 height, f32 psnr (NaN = unknown)
    0x06 RESET   (no payload) - drop every splat

Splat record (``4 + 4 * (11 + 3K)`` bytes, K = (sh_degree + 1)^2; 96 bytes for degree 1)::

    u32 id, f32 pos[3], f32 scale[3] (linear std-dev, metres), f32 rot[4] (unit w,x,y,z),
    f32 opacity (0..1, sigmoid applied), f32 sh[K][3] (k-major, rgb interleaved)

Poses use the computer-vision camera convention (x right, y down, z forward) and the scan's
world frame (z up for splatscan). A recorded stream (``.sslog``) is the 8-byte file magic
``b"SSCLOG01"`` followed by ``u32 length`` + message bytes, repeated.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x31435353
VERSION = 1
HEADER = struct.Struct("<IBBHI")
HELLO, ADD, UPDATE, REMOVE, FRAME, RESET = 1, 2, 3, 4, 5, 6
TYPE_NAMES = {HELLO: "HELLO", ADD: "ADD", UPDATE: "UPDATE", REMOVE: "REMOVE", FRAME: "FRAME", RESET: "RESET"}
HELLO_BODY = struct.Struct("<HBBIf")
FRAME_BODY = struct.Struct("<dII9f3f4fHHf")
LOG_MAGIC = b"SSCLOG01"
MAX_RECORDS_PER_MESSAGE = 4096


def record_floats(sh_degree: int) -> int:
    return 11 + 3 * (sh_degree + 1) ** 2


def record_bytes(sh_degree: int) -> int:
    return 4 + 4 * record_floats(sh_degree)


def _header(kind: int, seq: int) -> bytes:
    return HEADER.pack(MAGIC, kind, 0, 0, seq & 0xFFFFFFFF)


def encode_hello(seq: int, sh_degree: int, up_axis: int = 2, capacity_hint: int = 0,
                 scene_radius: float = 5.0) -> bytes:
    return _header(HELLO, seq) + HELLO_BODY.pack(VERSION, sh_degree, up_axis, capacity_hint, scene_radius)


def encode_splats(kind: int, seq: int, ids, pos, scale, rot, opacity, sh) -> bytes:
    """ADD or UPDATE. Arrays: ids (n,), pos (n,3), scale (n,3), rot (n,4), opacity (n,), sh (n,3K)."""
    if kind not in (ADD, UPDATE):
        raise ValueError("kind must be ADD or UPDATE")
    n = len(ids)
    nf = 11 + sh.shape[1]
    rec = np.empty((n, 1 + nf), np.float32)
    rec[:, 0] = np.asarray(ids, np.uint32).view(np.float32)
    rec[:, 1:4] = pos
    rec[:, 4:7] = scale
    rec[:, 7:11] = rot
    rec[:, 11] = opacity
    rec[:, 12:] = sh
    return _header(kind, seq) + struct.pack("<I", n) + rec.astype("<f4").tobytes()


def encode_remove(seq: int, ids) -> bytes:
    ids = np.asarray(ids, "<u4")
    return _header(REMOVE, seq) + struct.pack("<I", len(ids)) + ids.tobytes()


def encode_frame(seq: int, t: float, frame_index: int, splat_count: int, R_wc, p_wc, intr,
                 psnr: float = float("nan")) -> bytes:
    R = np.asarray(R_wc, np.float64).reshape(9)
    p = np.asarray(p_wc, np.float64).reshape(3)
    return _header(FRAME, seq) + FRAME_BODY.pack(t, frame_index, splat_count, *R, *p, intr.fx, intr.fy,
                                                 intr.cx, intr.cy, intr.width, intr.height, psnr)


def encode_reset(seq: int) -> bytes:
    return _header(RESET, seq)


@dataclass
class Message:
    kind: int
    seq: int
    body: dict


def decode(buf: bytes, sh_degree: int | None = None) -> Message:
    """Decode one message. ADD/UPDATE need ``sh_degree`` (from the HELLO) to size records."""
    if len(buf) < HEADER.size:
        raise ValueError("short message")
    magic, kind, _flags, _res, seq = HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")
    o = HEADER.size
    if kind == HELLO:
        v, deg, up, cap, rad = HELLO_BODY.unpack_from(buf, o)
        if v != VERSION:
            raise ValueError(f"unsupported protocol version {v}")
        return Message(kind, seq, dict(version=v, sh_degree=deg, up_axis=up, capacity_hint=cap, scene_radius=rad))
    if kind in (ADD, UPDATE):
        if sh_degree is None:
            raise ValueError("ADD/UPDATE before HELLO")
        (n,) = struct.unpack_from("<I", buf, o)
        nf = record_floats(sh_degree)
        expected = o + 4 + n * 4 * (1 + nf)
        if len(buf) != expected:
            raise ValueError(f"record size mismatch: {len(buf)} bytes, expected {expected}")
        rec = np.frombuffer(buf, "<f4", count=n * (1 + nf), offset=o + 4).reshape(n, 1 + nf)
        return Message(kind, seq, dict(ids=rec[:, 0].copy().view(np.uint32), pos=rec[:, 1:4], scale=rec[:, 4:7],
                                       rot=rec[:, 7:11], opacity=rec[:, 11], sh=rec[:, 12:]))
    if kind == REMOVE:
        (n,) = struct.unpack_from("<I", buf, o)
        if len(buf) != o + 4 + 4 * n:
            raise ValueError("REMOVE size mismatch")
        return Message(kind, seq, dict(ids=np.frombuffer(buf, "<u4", count=n, offset=o + 4).copy()))
    if kind == FRAME:
        f = FRAME_BODY.unpack_from(buf, o)
        return Message(kind, seq, dict(t=f[0], frame_index=f[1], splat_count=f[2],
                                       R_wc=np.array(f[3:12]).reshape(3, 3), p_wc=np.array(f[12:15]),
                                       fx=f[15], fy=f[16], cx=f[17], cy=f[18], width=f[19], height=f[20],
                                       psnr=f[21]))
    if kind == RESET:
        return Message(kind, seq, {})
    raise ValueError(f"unknown message type {kind}")


class DeltaEncoder:
    """Turns successive map snapshots into ADD / UPDATE / REMOVE messages.

    A splat is re-sent (UPDATE) only when it changed visibly since the last time it was sent:
    position by more than ``pos_tol`` metres, scale by more than ``scale_rel`` (relative),
    opacity or any SH coefficient by more than ``attr_tol``, or rotation by more than
    ``rot_tol`` (|1 - |<q, q'>||). ``snapshot()`` re-sends everything (for late joiners).
    """

    def __init__(self, sh_degree: int, pos_tol=2e-3, scale_rel=0.03, attr_tol=0.02, rot_tol=1e-3):
        self.sh_degree = sh_degree
        self.tol = (pos_tol, scale_rel, attr_tol, rot_tol)
        self.seq = 0
        self.sent: dict[str, np.ndarray] | None = None     # sorted by id

    def _next(self):
        self.seq += 1
        return self.seq

    def hello(self, capacity_hint=0, scene_radius=5.0, up_axis=2) -> bytes:
        return encode_hello(self._next(), self.sh_degree, up_axis, capacity_hint, scene_radius)

    def frame(self, t, frame_index, splat_count, R_wc, p_wc, intr, psnr=float("nan")) -> bytes:
        return encode_frame(self._next(), t, frame_index, splat_count, R_wc, p_wc, intr, psnr)

    def _chunks(self, kind, a, idx):
        out = []
        for s in range(0, len(idx), MAX_RECORDS_PER_MESSAGE):
            j = idx[s:s + MAX_RECORDS_PER_MESSAGE]
            out.append(encode_splats(kind, self._next(), a["ids"][j], a["pos"][j], a["scale"][j], a["rot"][j],
                                     a["opacity"][j], a["sh"][j]))
        return out

    def diff(self, snap: dict) -> tuple[list, dict]:
        """Messages bringing a client from the last sent state to ``snap`` (from GaussianMap.activated())."""
        order = np.argsort(snap["ids"], kind="stable")
        cur = {k: v[order] for k, v in snap.items()}
        msgs = []
        counts = dict(added=0, updated=0, removed=0)
        if self.sent is None:
            idx = np.arange(len(cur["ids"]))
            msgs += self._chunks(ADD, cur, idx)
            counts["added"] = len(idx)
        else:
            prev = self.sent
            in_prev = np.isin(cur["ids"], prev["ids"], assume_unique=True)
            removed = prev["ids"][~np.isin(prev["ids"], cur["ids"], assume_unique=True)]
            for s in range(0, len(removed), 65536):
                msgs.append(encode_remove(self._next(), removed[s:s + 65536]))
            add_idx = np.nonzero(~in_prev)[0]
            both = np.nonzero(in_prev)[0]
            pj = np.searchsorted(prev["ids"], cur["ids"][both])
            pt, sr, at, rt = self.tol
            ch = np.linalg.norm(cur["pos"][both] - prev["pos"][pj], axis=1) > pt
            ch |= (np.abs(cur["scale"][both] / np.maximum(prev["scale"][pj], 1e-9) - 1) > sr).any(1)
            ch |= np.abs(cur["opacity"][both] - prev["opacity"][pj]) > at
            ch |= (np.abs(cur["sh"][both] - prev["sh"][pj]) > at).any(1)
            ch |= (1 - np.abs((cur["rot"][both] * prev["rot"][pj]).sum(1))) > rt
            upd_idx = both[ch]
            msgs += self._chunks(ADD, cur, add_idx)
            msgs += self._chunks(UPDATE, cur, upd_idx)
            counts.update(added=len(add_idx), updated=len(upd_idx), removed=len(removed))
            # what the client now holds: unchanged rows keep their previously sent values
            held = {k: v.copy() for k, v in cur.items()}
            stale = both[~ch]
            for k in held:
                held[k][stale] = prev[k][pj[~ch]]
            cur = held
        self.sent = cur
        return msgs, counts

    def snapshot(self) -> list:
        if self.sent is None:
            return []
        return [encode_reset(self._next())] + self._chunks(ADD, self.sent, np.arange(len(self.sent["ids"])))


def write_log_header(f):
    f.write(LOG_MAGIC)


def append_log(f, msg: bytes):
    f.write(struct.pack("<I", len(msg)))
    f.write(msg)


def read_log(path: str):
    with open(path, "rb") as f:
        if f.read(8) != LOG_MAGIC:
            raise ValueError("not a splatscan stream log")
        while True:
            h = f.read(4)
            if len(h) < 4:
                return
            (n,) = struct.unpack("<I", h)
            yield f.read(n)
