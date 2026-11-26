"""Minimal RFC 6455 WebSocket framing (server and client) on asyncio streams.

Just what the splat stream needs: the opening handshake, binary/close/ping/pong frames,
unfragmented messages from us, fragmented messages accepted from peers, no extensions.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA
MAX_MESSAGE = 64 * 1024 * 1024


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def encode_frame(payload: bytes, opcode: int = OP_BIN, mask: bool = False) -> bytes:
    n = len(payload)
    head = bytearray([0x80 | opcode])
    mbit = 0x80 if mask else 0
    if n < 126:
        head.append(mbit | n)
    elif n < 65536:
        head.append(mbit | 126)
        head += struct.pack(">H", n)
    else:
        head.append(mbit | 127)
        head += struct.pack(">Q", n)
    if mask:
        key = os.urandom(4)
        head += key
        payload = _xor(payload, key)
    return bytes(head) + payload


def _xor(data: bytes, key: bytes) -> bytes:
    if not data:
        return data
    k = (key * (len(data) // 4 + 1))[: len(data)]
    return (int.from_bytes(data, "little") ^ int.from_bytes(k, "little")).to_bytes(len(data), "little")


async def read_frame(reader: asyncio.StreamReader):
    """Returns (fin, opcode, payload)."""
    b0, b1 = await reader.readexactly(2)
    fin, opcode = bool(b0 & 0x80), b0 & 0x0F
    masked, n = bool(b1 & 0x80), b1 & 0x7F
    if n == 126:
        (n,) = struct.unpack(">H", await reader.readexactly(2))
    elif n == 127:
        (n,) = struct.unpack(">Q", await reader.readexactly(8))
    if n > MAX_MESSAGE:
        raise ValueError("frame too large")
    key = await reader.readexactly(4) if masked else None
    payload = await reader.readexactly(n)
    if key:
        payload = _xor(payload, key)
    return fin, opcode, payload


class Connection:
    """A WebSocket endpoint. ``is_client`` decides whether outgoing frames are masked."""

    def __init__(self, reader, writer, is_client: bool):
        self.reader, self.writer, self.is_client = reader, writer, is_client
        self.closed = False

    async def send(self, data: bytes, opcode: int = OP_BIN):
        self.writer.write(encode_frame(data, opcode, mask=self.is_client))
        await self.writer.drain()

    async def recv(self):
        """Next complete data message (bytes or str), or None once the peer closes."""
        buf, op0 = bytearray(), None
        while True:
            try:
                fin, op, payload = await read_frame(self.reader)
            except (asyncio.IncompleteReadError, ConnectionError):
                self.closed = True
                return None
            if op == OP_PING:
                await self.send(payload, OP_PONG)
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                if not self.closed:
                    self.closed = True
                    try:
                        await self.send(payload[:2], OP_CLOSE)
                    except ConnectionError:
                        pass
                return None
            if op != OP_CONT:
                op0 = op
            buf += payload
            if len(buf) > MAX_MESSAGE:
                raise ValueError("message too large")
            if fin:
                return buf.decode() if op0 == OP_TEXT else bytes(buf)

    async def close(self, code: int = 1000):
        if not self.closed:
            self.closed = True
            try:
                await self.send(struct.pack(">H", code), OP_CLOSE)
            except ConnectionError:
                pass
        self.writer.close()


async def server_handshake(reader, writer) -> Connection | None:
    """Reads the HTTP upgrade request; answers 101 (or 426 for plain HTTP) and returns a Connection."""
    raw = await reader.readuntil(b"\r\n\r\n")
    lines = raw.decode("latin-1").split("\r\n")
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    if headers.get("upgrade", "").lower() != "websocket" or "sec-websocket-key" not in headers:
        body = b"splatscan stream server: connect with a WebSocket client (e.g. splatweb ?live=ws://host:port)\n"
        writer.write(b"HTTP/1.1 426 Upgrade Required\r\nContent-Type: text/plain\r\nUpgrade: websocket\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
        await writer.drain()
        writer.close()
        return None
    resp = ("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key(headers['sec-websocket-key'])}\r\n\r\n")
    writer.write(resp.encode())
    await writer.drain()
    return Connection(reader, writer, is_client=False)


async def connect(host: str, port: int, path: str = "/") -> Connection:
    reader, writer = await asyncio.open_connection(host, port)
    key = base64.b64encode(os.urandom(16)).decode()
    req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
    writer.write(req.encode())
    await writer.drain()
    raw = await reader.readuntil(b"\r\n\r\n")
    status = raw.split(b"\r\n", 1)[0]
    if b" 101 " not in status + b" ":
        raise ConnectionError(f"handshake failed: {status!r}")
    if accept_key(key).encode() not in raw:
        raise ConnectionError("bad Sec-WebSocket-Accept")
    return Connection(reader, writer, is_client=True)
