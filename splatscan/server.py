"""WebSocket stream server: live (from the mapper) or replaying a recorded ``.sslog``.

Live use, from the mapping thread::

    srv = StreamServer(port=8765, sh_degree=1).start()      # serves in a background thread
    srv.publish_snapshot(gmap.activated(), frame=...)       # after each keyframe

Every new client first receives HELLO + a full snapshot of what other clients already have,
then the same deltas as everyone else - registration and publishing share one lock and each
client has its own FIFO, so a late joiner can never miss or double-apply a delta.

Replay a recording (each client gets the whole stream from the start, paced by FRAME times)::

    python -m splatscan.server --replay runs/scan.sslog --port 8765 --speed 1
"""
from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import time

from . import protocol as P
from .ws import server_handshake

_CLOSE = object()


class StreamServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, sh_degree: int = 1, log_path: str | None = None,
                 scene_radius: float = 5.0):
        self.host, self.port = host, port
        self.enc = P.DeltaEncoder(sh_degree)
        self.scene_radius = scene_radius
        self.lock = threading.Lock()
        self.clients: list[queue.SimpleQueue] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.ready = threading.Event()
        self.log = open(log_path, "wb") if log_path else None
        if self.log:
            P.write_log_header(self.log)
            P.append_log(self.log, self.enc.hello(scene_radius=scene_radius))
        self.stats = dict(messages=0, bytes=0, clients_total=0)

    # ------------------------------------------------------------------ producer side
    def publish_snapshot(self, snap: dict, frame: dict | None = None) -> dict:
        """Diff ``snap`` against the last published state and broadcast the deltas (+ a FRAME)."""
        with self.lock:
            msgs, counts = self.enc.diff(snap)
            if frame is not None:
                msgs.append(self.enc.frame(**frame))
            self._broadcast(msgs)
        counts["bytes"] = sum(len(m) for m in msgs)
        return counts

    def publish_frame(self, **frame):
        with self.lock:
            self._broadcast([self.enc.frame(**frame)])

    def _broadcast(self, msgs):
        for m in msgs:
            if self.log:
                P.append_log(self.log, m)
            for q in self.clients:
                q.put(m)
            self.stats["messages"] += 1
            self.stats["bytes"] += len(m)

    # ------------------------------------------------------------------ network side
    async def _client(self, reader, writer):
        conn = await server_handshake(reader, writer)
        if conn is None:
            return
        q: queue.SimpleQueue = queue.SimpleQueue()
        with self.lock:
            q.put(self.enc.hello(scene_radius=self.scene_radius))
            for m in self.enc.snapshot():
                q.put(m)
            self.clients.append(q)
            self.stats["clients_total"] += 1
        loop = asyncio.get_running_loop()

        async def pump():
            while True:
                m = await loop.run_in_executor(None, q.get)
                if m is _CLOSE:
                    break
                await conn.send(m)

        sender = asyncio.ensure_future(pump())
        try:
            while await conn.recv() is not None:   # ignore client messages, wait for close
                pass
        finally:
            with self.lock:
                if q in self.clients:
                    self.clients.remove(q)
            q.put(_CLOSE)
            await asyncio.gather(sender, return_exceptions=True)
            await conn.close()

    def start(self) -> "StreamServer":
        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            srv = self.loop.run_until_complete(asyncio.start_server(self._client, self.host, self.port))
            self.port = srv.sockets[0].getsockname()[1]
            self.ready.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=run, daemon=True, name="splatscan-ws")
        self.thread.start()
        self.ready.wait(5)
        return self

    def stop(self):
        with self.lock:
            for q in self.clients:
                q.put(_CLOSE)
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.log:
            self.log.close()
            self.log = None


# ---------------------------------------------------------------------- replay
async def _replay_client(reader, writer, path: str, speed: float, loop_forever: bool):
    conn = await server_handshake(reader, writer)
    if conn is None:
        return

    async def drain_incoming():
        while await conn.recv() is not None:
            pass

    rx = asyncio.ensure_future(drain_incoming())
    try:
        while True:
            t0_wall, t0_seq = time.perf_counter(), None
            for m in P.read_log(path):
                if conn.closed:
                    return
                kind = m[4]
                if kind == P.FRAME and speed > 0:
                    t = P.decode(m).body["t"]
                    t0_seq = t if t0_seq is None else t0_seq
                    wait = (t - t0_seq) / speed - (time.perf_counter() - t0_wall)
                    if wait > 0:
                        await asyncio.sleep(wait)
                await conn.send(m)
            if not loop_forever:
                break
            await asyncio.sleep(2.0)
            await conn.send(P.encode_reset(0))
        while not conn.closed:          # keep the connection open; the client closes
            await asyncio.sleep(0.5)
    except ConnectionError:
        pass
    finally:
        rx.cancel()
        await conn.close()


def serve_replay(path: str, host: str, port: int, speed: float, loop_forever: bool):
    async def main():
        srv = await asyncio.start_server(lambda r, w: _replay_client(r, w, path, speed, loop_forever), host, port)
        print(f"replaying {path} on ws://{host}:{srv.sockets[0].getsockname()[1]} (speed x{speed})", flush=True)
        async with srv:
            await srv.serve_forever()

    asyncio.run(main())


def main():
    ap = argparse.ArgumentParser(description="splatscan WebSocket stream server (replay mode)")
    ap.add_argument("--replay", required=True, help="a .sslog written by scripts/run_scan.py --log")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed; 0 = as fast as possible")
    ap.add_argument("--loop", action="store_true", help="restart the stream after it ends")
    a = ap.parse_args()
    serve_replay(a.replay, a.host, a.port, a.speed, a.loop)


if __name__ == "__main__":
    main()
