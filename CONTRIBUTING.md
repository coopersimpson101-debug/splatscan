# Contributing

Thanks for your interest! splatscan is a learning-oriented project: the core algorithms
(rasteriser and its backward pass, optimiser, mapper, protocol, WebSocket framing) are written by
hand on purpose, so please keep new dependencies to I/O, plotting and test tooling.

## Development setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,figures]"
python -m pytest -q
```

## Ground rules

- Every gradient you add to the rasteriser needs a test against `rasterize_reference`
  (autograd) **and** a finite-difference check.
- Numbers in the README must come from a script in this repo, with machine and date. Anything
  that needs hardware goes in the "Hardware validation" table with status ⏳.
- Protocol changes bump the version and must update `docs/protocol.md`, the encoder and
  splatweb's decoder together (`scripts/make_protocol_fixture.py` keeps them honest).

## Pull requests

Keep them focused, describe how you tested, and make sure `python -m pytest -q` passes.
