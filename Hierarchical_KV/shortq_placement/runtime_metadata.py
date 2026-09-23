"""Runtime placement metadata format.

The persisted runtime file contains *only* ``chunk_hash_hex -> L0/L1/L2``.
Experimental diagnostics belong in separate files.
"""

from __future__ import annotations

import json
from pathlib import Path

VALID_TIERS = {"L0", "L1", "L2"}


def write_runtime_metadata(path: str | Path, placements: dict[str, str]) -> None:
    bad = {k: v for k, v in placements.items() if v not in VALID_TIERS}
    if bad:
        raise ValueError(f"invalid runtime placement values: {list(bad.items())[:3]}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(sorted(placements.items())), indent=2) + "\n", encoding="utf-8")
