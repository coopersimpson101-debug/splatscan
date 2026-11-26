# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-26

First public release.

### Added
- Tile-based differentiable Gaussian-splatting rasteriser in PyTorch with a hand-derived
  backward pass (means, scale/rotation covariance, opacity, SH degree 0-3, rendered depth),
  verified against autograd on a dense reference and against finite differences
- Online mapper: keyframe selection from VIO poses, silhouette-driven growth from depth,
  sliding-window optimisation, clone/split/prune, per-splat Adam
- Pose source: insight-vio's stereo MSCKF on a synthetic camera + IMU lab walk
  (procedurally textured room ray-cast in numpy)
- Sparse depth from insight-vio's stereo matches (DLT triangulation) for sensor-free seeding
- Live-stream protocol v1 (WebSocket), hand-written RFC 6455 server, delta encoder, replay server
- splatweb `feat/live-stream` companion: in-place live loader and `?live=ws://...`
- Jetson deployment config, systemd unit and device-source wiring (not validated on hardware)

[0.1.0]: https://github.com/coopersimpson101-debug/splatscan/releases/tag/v0.1.0
