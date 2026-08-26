# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.x (latest minor) | ✅ |
| older 0.x | ❌ |

## Reporting a Vulnerability

Email **dev@codecora.dev** with details and a repro if possible. Do not open a public issue for vulnerabilities.

- You will get an acknowledgment within 72 hours.
- Fixes for accepted reports ship in a patch release, credited in the CHANGELOG unless you prefer otherwise.

## Scope

vecq is an embedded library, not a networked service. In scope: malformed file parsing (`from_bytes`), out-of-bounds reads in the SIMD paths, panics on crafted input. Out of scope: how you store or transmit `.vecq` files.
