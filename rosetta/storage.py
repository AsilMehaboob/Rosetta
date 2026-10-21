"""File-backed store for learned artifacts.

The hackathon stand-in for the metadata database. Learn and Answer modules only
call this interface, so swapping in PostgreSQL later changes this file alone.
Output is canonical (sorted keys, fixed indent) so identical content writes
byte-identical files.
"""

import json
from pathlib import Path
from typing import Any


class Storage:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.json"

    def write(self, kind: str, key: str, value: Any) -> Path:
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def read(self, kind: str, key: str) -> Any | None:
        path = self._path(kind, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def keys(self, kind: str) -> list[str]:
        folder = self.root / kind
        if not folder.exists():
            return []
        return sorted(p.stem for p in folder.glob("*.json"))
