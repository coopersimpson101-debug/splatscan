# splatscan live-stream protocol, version 1

Normative description of what `splatscan/protocol.py` writes and splatweb's `src/io/live.ts`
reads. Transport: WebSocket (RFC 6455), **one protocol message per binary WebSocket message**.
All integers and floats are little-endian. Text messages are ignored by receivers.

## Header (12 bytes)

| offset | field | type | value |
|---:|---|---|---|
| 0 | magic | `u32` | `0x31435353` (bytes `S S C 1`) |
| 4 | type | `u8` | 1-6, below |
| 5 | flags | `u8` | 0 (reserved) |
| 6 | reserved | `u16` | 0 |
| 8 | seq | `u32` | sender's running message counter (informational, wraps) |

## Messages

| type | name | payload after the header |
|---:|---|---|
| 1 | HELLO | `u16 version` (=1), `u8 sh_degree`, `u8 up_axis` (1 = +y, 2 = +z), `u32 capacity_hint`, `f32 scene_radius` |
| 2 | ADD | `u32 n`, then `n` splat records |
| 3 | UPDATE | `u32 n`, then `n` splat records; every field of the splat is replaced |
| 4 | REMOVE | `u32 n`, then `n × u32 id` |
| 5 | FRAME | `f64 t`, `u32 frame_index`, `u32 splat_count`, `f32 R_wc[9]` (row-major), `f32 p_wc[3]`, `f32 fx, fy, cx, cy`, `u16 width, height`, `f32 psnr` (NaN = unknown); 88 bytes |
| 6 | RESET | empty; drop every splat |

### Splat record

`4 + 4 × (11 + 3K)` bytes with `K = (sh_degree + 1)²` (96 bytes at degree 1):

| field | type | meaning |
|---|---|---|
| id | `u32` | stable splat id, unique for the session |
| pos | `f32[3]` | mean, world frame, metres |
| scale | `f32[3]` | standard deviations along the local axes (linear, not log) |
| rot | `f32[4]` | unit quaternion `w, x, y, z` (local → world) |
| opacity | `f32` | 0..1 (sigmoid already applied) |
| sh | `f32[K][3]` | real SH coefficients, coefficient-major, RGB interleaved; colour = `max(0, Σ_k Y_k(d) c_k + 0.5)` with the 3DGS constants |

### Frames and axes

`FRAME` carries the tracked camera pose **camera → world** in the computer-vision camera
convention (x right, y down, z forward) and its intrinsics at mapping resolution. splatscan's
world is **z-up** (`up_axis = 2`); splatweb maps it to its +y-up convention with
`upAxisModelMatrix(2)`, i.e. `(x, y, z) → (x, z, −y)`.

## Sender behaviour (`DeltaEncoder`)

After each keyframe the mapper's whole map is diffed against what was last sent:

* new ids → ADD; ids no longer present (pruned, or split parents) → REMOVE;
* surviving ids → UPDATE only if a visible change exceeds a tolerance: position 2 mm, scale 3 %
  (relative), opacity or any SH coefficient 0.02, rotation `1 − |⟨q, q'⟩| > 10⁻³`;
* the encoder remembers what each client now holds (sent values, not the true current values),
  so tolerances never accumulate drift;
* records are chunked at 4096 per message.

A client that connects mid-session receives HELLO, then `RESET` + ADDs of the held state
(`snapshot()`), then the same deltas as everyone else. Registration and broadcasting share one
lock and every client has its own FIFO, so a late joiner cannot miss a delta.

## Receiver rules

* ADD for a known id behaves as UPDATE; UPDATE for an unknown id behaves as ADD; REMOVE for an
  unknown id is ignored. (Makes snapshot + delta overlap harmless.)
* ADD/UPDATE before HELLO, a wrong magic, a record-size mismatch or `version ≠ 1` are errors.

## Recorded streams (`.sslog`)

`b"SSCLOG01"` followed by `u32 length` + message bytes, repeated. `scripts/run_scan.py` writes one
per run; `python -m splatscan.server --replay file.sslog` serves it (each client from the start,
paced by FRAME times, `--speed 0` = as fast as possible).

## Bandwidth (measured, see README Results)

Splats are sent as f32 (96 B at SH degree 1). Most of the traffic after the first keyframes is
UPDATEs of splats that are still being optimised. f16 packing (as splatweb uses on the GPU)
would halve it; see the roadmap.
