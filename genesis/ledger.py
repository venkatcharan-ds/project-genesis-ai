"""Append-only, hash-chained evidence ledger (JSON Lines).

Every entry stores the SHA-256 of the previous entry, so any edit or deletion
of past evidence is detectable with `verify()`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


class EvidenceLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def _last_hash(self) -> str:
        entries = self.entries()
        return entries[-1]["hash"] if entries else GENESIS_HASH

    def append(self, run_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "kind": kind,
            "payload": payload,
            "prev_hash": self._last_hash(),
        }
        entry["hash"] = _entry_hash(entry)
        with self.path.open("a") as f:
            f.write(_canonical(entry) + "\n")
        return entry

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS_HASH
        for i, e in enumerate(self.entries()):
            if e.get("prev_hash") != prev:
                return False, f"entry {i}: broken chain (prev_hash mismatch)"
            if _entry_hash(e) != e.get("hash"):
                return False, f"entry {i}: content hash mismatch (entry was modified)"
            prev = e["hash"]
        return True, f"ok ({len(self.entries())} entries)"
