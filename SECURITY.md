# Security policy

## Supported versions

Only the latest release on `main` receives fixes.

## Scope notes

- The stream server binds to `127.0.0.1` by default. The Jetson config binds `0.0.0.0` so LAN
  viewers can connect: the protocol has **no authentication or encryption**. Run it only on a
  trusted network, or put it behind a TLS reverse proxy (`wss://`) with access control.
- The server never executes or deserialises code from clients; it ignores client messages and
  rejects frames larger than 64 MiB.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting ("Security" tab, "Report a vulnerability")
on this repository instead of opening a public issue.
