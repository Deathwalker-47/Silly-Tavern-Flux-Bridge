#!/usr/bin/env python3
"""LoRA dictionary format: shared fields on the entry, one file per architecture under `models`.

    python lora_dict.py [master_lora_dict.json]

converts an older flat dictionary in place, keeping a .bak copy beside it.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

ARCHITECTURES = ("flux", "zimage")
# Fields that belong to one LoRA file rather than to the entry.
FILE_FIELDS = ("url", "weight", "trigger_words", "prepend_prompt", "append_prompt", "negative_prompt")


def lora_for(entry: Dict, arch: str) -> Optional[Dict]:
    """The LoRA file for this architecture, or None until it's populated."""
    return (entry.get("models") or {}).get(arch)


def flat_entry(entry: Dict, arch: str) -> Optional[Dict]:
    """Shared fields plus this architecture's file: the shape the image pipeline reads."""
    model = lora_for(entry, arch)
    if not model:
        return None
    return {**{k: v for k, v in entry.items() if k != "models"}, **model}


def migrate_entry(entry: Dict) -> Dict:
    """Flat entry to per-architecture entry. An entry that already has `models` is unchanged."""
    if "models" in entry:
        return entry
    shared = {k: v for k, v in entry.items() if k not in FILE_FIELDS}
    flux = {k: entry[k] for k in FILE_FIELDS if k in entry}
    return {**shared, "models": {"flux": flux, "zimage": None}}


def migrate_dict(data: Dict) -> Dict:
    return {**data, "loras": {k: migrate_entry(v) for k, v in (data.get("loras") or {}).items()}}


def check_unique_urls(data: Dict) -> None:
    """A file has one architecture, so the same URL can't be listed under two."""
    seen: Dict[str, str] = {}
    for lora_id, entry in (data.get("loras") or {}).items():
        for arch in ARCHITECTURES:
            url = (lora_for(entry, arch) or {}).get("url")
            if not url:
                continue
            if seen.setdefault(url, arch) != arch:
                raise ValueError(f"LoRA '{lora_id}': {url} is listed under both {seen[url]} and {arch}")


def migrate_file(path) -> bool:
    """Convert the file in place, keeping a .bak copy. False if it was already converted."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    migrated = migrate_dict(data)
    if migrated == data:
        return False
    check_unique_urls(migrated)
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(json.dumps(migrated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "master_lora_dict.json"
    print(f"{target}: {'converted' if migrate_file(target) else 'already converted'}")
