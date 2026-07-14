"""Hashing helpers used by canonical-form hashing and chunk-file fingerprints."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def sha256_hex(data: bytes | str) -> str:
    """Return the lowercase 64-char sha256 hex digest of ``data``."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_chained(parts: Iterable[bytes | str]) -> str:
    """Return sha256 over the concatenation of ``parts``.

    Each part is length-prefixed (4-byte big-endian) before being fed in, so
    ``["a", "bc"]`` and ``["ab", "c"]`` produce different digests.
    """
    h = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8") if isinstance(part, str) else part
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
    return h.hexdigest()
