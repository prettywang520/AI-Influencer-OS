"""Internal helper: deterministic content-addressed IDs via sha256
over canonical JSON. Same convention used by every content-hash ID
computed elsewhere in this codebase -- sorted keys, compact
separators, no whitespace, timestamps/volatile fields excluded by the
caller before this is invoked.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def content_hash(payload: dict[str, Any], *, length: int = 16) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]
