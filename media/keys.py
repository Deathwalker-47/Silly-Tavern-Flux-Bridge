"""API keys saved from the settings panel. They sit in front of .env until cleared.

Keys only ever come in: nothing here sends one back, and telemetry never sees one.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List

from .catalog import PROVIDER_LABELS
from .config import MediaConfig

logger = logging.getLogger(__name__)

FIELDS = {"runware": "RUNWARE_API_KEY", "nanogpt": "NANOGPT_API_KEY", "sogni": "SOGNI_API_KEY",
          "replicate": "REPLICATE_API_TOKEN", "pixeldojo": "PIXELDOJO_API_KEY"}

_changed: Callable[[], None] = lambda: None


def on_change(callback: Callable[[], None]) -> None:
    """The bridge's old txt2img clients keep their own copy of a key."""
    global _changed
    _changed = callback


def _path() -> Path:
    return MediaConfig.DATA_DIR / "keys.json"


def _read() -> Dict[str, str]:
    try:
        return json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"🔑 Could not read saved keys: {type(e).__name__}")
        return {}


def _write(keys: Dict[str, str]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part")
    with os.fdopen(os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        json.dump(keys, f)
    temp.replace(path)


def apply() -> None:
    """Puts saved keys in front of .env. Runs at startup and after every change."""
    saved = _read()
    for provider, field in FIELDS.items():
        setattr(MediaConfig, field, saved.get(provider) or os.getenv(field, ""))
    MediaConfig.HELPER_API_KEY = os.getenv("HELPER_API_KEY", "") or MediaConfig.NANOGPT_API_KEY
    _changed()


def save(provider: str, key: str) -> None:
    _write({**_read(), provider: key})
    apply()


def clear(provider: str) -> None:
    keys = _read()
    keys.pop(provider, None)
    _write(keys)
    apply()


def status(hints: bool = True) -> List[Dict[str, str]]:
    """Where each key comes from, with its last four characters at most."""
    saved = _read()
    rows = []
    for provider, field in FIELDS.items():
        value = getattr(MediaConfig, field)
        rows.append({"id": provider, "label": PROVIDER_LABELS.get(provider, provider),
                     "source": "settings" if saved.get(provider) else "env" if value else "none",
                     "hint": value[-4:] if hints and len(value) > 12 else ""})
    return rows
