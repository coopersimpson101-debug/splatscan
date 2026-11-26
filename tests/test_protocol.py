"""Protocol v1 encode/decode, the delta encoder, the hand-written WebSocket layer and the server."""
import asyncio
import os
import struct

import numpy as np
import pytest

from splatscan import protocol as P
from splatscan import ws
from splatscan.server import StreamServer, _replay_client
from splatscan.vio_source import Intrinsics

INTR = Intrinsics(160, 120, 100.0, 100.0, 79.5, 59.5)


def snap(ids, seed=0, deg=1):
    rng = np.random.default_rng(seed)
    n = len(ids)
    q = rng.normal(size=(n, 4))
    return dict(ids=np.asarray(ids, np.uint32), pos=rng.normal(size=(n, 3)).astype(np.float32),
                scale=rng.uniform(0.01, 0.2, (n, 3)).astype(np.float32),
                rot=(q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32),
                opacity=rng.uniform(0.1, 0.9, n).astype(np.float32),
                sh=rng.normal(size=(n, 3 * (deg + 1) ** 2)).astype(np.float32))


class Client:
    """Applies decoded messages - what any receiver (e.g. splatweb) must do."""

    def __init__(self):
        self.deg, self.state, self.frames = None, {}, []

    def apply(self, buf):
        m = P.decode(buf, self.deg)
        if m.kind == P.HELLO:
            self.deg = m.body["sh_degree"]
        elif m.kind in (P.ADD, P.UPDATE):
            b = m.body
            for i, sid in enumerate(b["ids"]):
                self.state[int(sid)] = (b["pos"][i].copy(), b["opacity"][i], b["sh"][i].copy())
        elif m.kind == P.REMOVE:
            for sid in m.body["ids"]:
                self.state.pop(int(sid), None)
        elif m.kind == P.RESET:
            self.state.clear()
        elif m.kind == P.FRAME:
            self.frames.append(m.body)
        return m


def test_record_size_and_header_layout():
    assert P.record_bytes(1) == 96 and P.record_bytes(0) == 4 + 4 * 14 and P.record_bytes(3) == 4 + 4 * 59
    s = snap([5, 6])
    buf = P.encode_splats(P.ADD, 9, s["ids"], s["pos"], s["scale"], s["rot"], s["opacity"], s["sh"])
    magic, kind, flags, res, seq = struct.unpack_from("<IBBHI", buf)
    assert (magic, kind, flags, res, seq) == (0x31435353, P.ADD, 0, 0, 9)
    assert buf[:4] == b"SSC1"
    assert len(buf) == 12 + 4 + 2 * 96
    assert len(P.encode_frame(1, 0.0, 0, 0, np.eye(3), np.zeros(3), INTR)) == 12 + 88


def test_roundtrip_every_message_type():
    s = snap(range(7))
    m = P.decode(P.encode_hello(1, 1, 2, 77, 4.5))
    assert m.body == dict(version=1, sh_degree=1, up_axis=2, capacity_hint=77, scene_radius=4.5)
    m = P.decode(P.encode_splats(P.UPDATE, 2, s["ids"], s["pos"], s["scale"], s["rot"], s["opacity"], s["sh"]), 1)
    assert m.kind == P.UPDATE
    for k in ("pos", "scale", "rot", "opacity", "sh"):
        np.testing.assert_array_equal(m.body[k], s[k])
    np.testing.assert_array_equal(m.body["ids"], s["ids"])
    assert list(P.decode(P.encode_remove(3, [1, 2, 3])).body["ids"]) == [1, 2, 3]
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    f = P.decode(P.encode_frame(4, 12.25, 245, 1000, R, [1, 2, 3], INTR, 27.5)).body
    np.testing.assert_allclose(f["R_wc"], R)
    assert (f["t"], f["frame_index"], f["splat_count"], f["width"], f["height"]) == (12.25, 245, 1000, 160, 120)
    assert f["psnr"] == pytest.approx(27.5)
    assert P.decode(P.encode_reset(5)).kind == P.RESET


def test_decode_rejects_malformed():
    with pytest.raises(ValueError, match="magic"):
        P.decode(b"XXXX" + bytes(8))
    s = snap([1])
    buf = P.encode_splats(P.ADD, 1, s["ids"], s["pos"], s["scale"], s["rot"], s["opacity"], s["sh"])
    with pytest.raises(ValueError, match="HELLO"):
        P.decode(buf)
    with pytest.raises(ValueError, match="size"):
        P.decode(buf, sh_degree=0)
    with pytest.raises(ValueError, match="short"):
        P.decode(b"SSC1")


def test_delta_encoder_converges_client_to_snapshot():
    enc = P.DeltaEncoder(1)
    c = Client()
    c.apply(enc.hello())
    s = snap(range(100), seed=1)
    msgs, counts = enc.diff(s)
    assert counts == dict(added=100, updated=0, removed=0)
    for m in msgs:
        c.apply(m)
    # small changes below tolerance are not sent; big ones are
    s2 = {k: v.copy() for k, v in s.items()}
    s2["pos"][:10] += 1e-4                  # below 2 mm
    s2["pos"][10:20] += 0.05                # above
    s2["opacity"][20] += 0.1
    keep = np.r_[0:90]
    s2 = {k: v[keep] for k, v in s2.items()}
    extra = snap(range(1000, 1005), seed=2)
    s2 = {k: np.concatenate([s2[k], extra[k]]) for k in s2}
    msgs, counts = enc.diff(s2)
    assert counts == dict(added=5, updated=11, removed=10)
    for m in msgs:
        c.apply(m)
    assert set(c.state) == set(int(i) for i in s2["ids"])
    order = {int(i): j for j, i in enumerate(s2["ids"])}
    for sid, (pos, op, sh) in c.state.items():
        j = order[sid]
        assert np.abs(pos - s2["pos"][j]).max() <= 2e-3 + 1e-7
        assert abs(op - s2["opacity"][j]) <= 0.02 + 1e-7
    # a late joiner gets exactly the held state from snapshot()
    late = Client()
    late.apply(enc.hello())
    for m in enc.snapshot():
        late.apply(m)
    assert set(late.state) == set(c.state)
    for sid in c.state:
        np.testing.assert_array_equal(late.state[sid][0], c.state[sid][0])


def test_large_adds_are_chunked():
    enc = P.DeltaEncoder(0)
    msgs, counts = enc.diff(snap(range(P.MAX_RECORDS_PER_MESSAGE * 2 + 5), deg=0))
    assert len(msgs) == 3 and counts["added"] == P.MAX_RECORDS_PER_MESSAGE * 2 + 5


def test_ws_frame_encoding_lengths_and_masking():
    for n in (0, 5, 125, 126, 65535, 65536, 200_000):
        payload = os.urandom(n)
        for mask in (False, True):
            buf = ws.encode_frame(payload, ws.OP_BIN, mask=mask)

            async def read(buf=buf):
                r = asyncio.StreamReader()
                r.feed_data(buf)
                r.feed_eof()
                return await ws.read_frame(r)

            fin, op, got = asyncio.run(read())
            assert fin and op == ws.OP_BIN and got == payload


def test_accept_key_rfc6455_example():
    assert ws.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_live_server_hello_snapshot_and_deltas(tmp_path):
    srv = StreamServer(port=0, sh_degree=1, log_path=str(tmp_path / "s.sslog")).start()
    try:
        srv.publish_snapshot(snap(range(50), seed=3), frame=dict(t=0.5, frame_index=10, splat_count=50, R_wc=np.eye(3),
                                                                 p_wc=np.zeros(3), intr=INTR))

        async def run():
            c = Client()
            conn = await ws.connect("127.0.0.1", srv.port)
            got = []
            while len(c.state) < 50:
                got.append(c.apply(await conn.recv()).kind)
            assert got[0] == P.HELLO and P.RESET in got
            s2 = snap(range(10, 60), seed=4)
            await asyncio.get_running_loop().run_in_executor(None, lambda: srv.publish_snapshot(s2))
            while set(c.state) != set(range(10, 60)):
                c.apply(await asyncio.wait_for(conn.recv(), 5))
            await conn.close()
            return c

        c = asyncio.run(run())
        assert len(c.state) == 50
        assert srv.stats["clients_total"] == 1
    finally:
        srv.stop()
    # the log replays into the same final state
    c2 = Client()
    for m in P.read_log(str(tmp_path / "s.sslog")):
        c2.apply(m)
    assert set(c2.state) == set(range(10, 60)) and len(c2.frames) == 1


def test_plain_http_gets_426():
    srv = StreamServer(port=0).start()
    try:
        async def run():
            r, w = await asyncio.open_connection("127.0.0.1", srv.port)
            w.write(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            await w.drain()
            data = await r.read()
            w.close()
            return data

        assert asyncio.run(run()).startswith(b"HTTP/1.1 426")
    finally:
        srv.stop()


def test_replay_server(tmp_path):
    path = tmp_path / "r.sslog"
    enc = P.DeltaEncoder(1)
    with open(path, "wb") as f:
        P.write_log_header(f)
        P.append_log(f, enc.hello())
        for m in enc.diff(snap(range(20)))[0]:
            P.append_log(f, m)
        P.append_log(f, enc.frame(0.0, 0, 20, np.eye(3), np.zeros(3), INTR))

    async def run():
        srv = await asyncio.start_server(lambda r, w: _replay_client(r, w, str(path), 0, False), "127.0.0.1", 0)
        port = srv.sockets[0].getsockname()[1]
        conn = await ws.connect("127.0.0.1", port)
        c = Client()
        while not c.frames:
            c.apply(await asyncio.wait_for(conn.recv(), 5))
        await conn.close()
        srv.close()
        return c

    c = asyncio.run(run())
    assert len(c.state) == 20
