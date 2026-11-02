"""A structured record of every pipeline step, kept so the pipeline can be improved.

Plain log lines can't answer "which model gets the best ratings" or "which step is
slow". Each step here is a row in `events` with its timing, outcome and facts, tied to
its generation, so a whole run can be replayed (`trace`) and runs can be compared
(`insights.py`).

Two levels of detail:
  data_json  numbers and short facts, kept for good
  raw_json   full prompts and LLM replies, cleared after TELEMETRY_RAW_DAYS

Nothing here can reach a key or an image: `scrub` turns image data into a size note,
and request headers are never passed in. Recording never raises, because a failed log
write must not cost a finished render.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from . import db

logger = logging.getLogger("media.telemetry")

RAW_DAYS = int(os.getenv("TELEMETRY_RAW_DAYS", 90))

# The generation a step belongs to, so provider code can record without passing ids around.
current_generation: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("media_generation", default=None)
current_chat: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("media_chat", default=None)

_DATA_URI = re.compile(r"^data:[\w/+.-]+;base64,", re.I)
_SECRET_NAME = re.compile(r"(api[_-]?key|authorization|secret|password|(^|[_-])token$)", re.I)
_BASE64ISH = re.compile(r"^[A-Za-z0-9+/=_-]{512,}$")


def scrub(value: Any) -> Any:
    """A copy that is safe to store: image data becomes a size note, secrets are dropped."""
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        if _DATA_URI.match(value) or _BASE64ISH.match(value):
            return f"<image data, {len(value)} chars>"
        return value
    if isinstance(value, dict):
        return {str(k): ("<hidden>" if _SECRET_NAME.search(str(k)) else scrub(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [scrub(v) for v in value]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)


def _dump(value: Any) -> Optional[str]:
    return None if value in (None, {}, []) else json.dumps(scrub(value), ensure_ascii=False, default=str)


def record(stage: str, event: str, *, ok: Optional[bool] = None, ms: Optional[float] = None, raw: Any = None,
           generation_id: Optional[str] = None, chat_id: Optional[str] = None, **data: Any) -> None:
    """Stores one step. Keyword arguments become its queryable facts."""
    try:
        db.add_event(time.time(), generation_id or current_generation.get(), chat_id or current_chat.get(),
                     stage, event, ok, None if ms is None else int(ms), _dump(data), _dump(raw))
    except Exception as e:
        logger.warning(f"telemetry write failed for {stage}/{event}: {e}")


@contextmanager
def step(stage: str, event: str, **data: Any) -> Iterator[Dict[str, Any]]:
    """Times a step and records how it ended. The block can add facts to the yielded dict.

    A `raw` key in that dict goes to raw_json.
    """
    facts: Dict[str, Any] = dict(data)
    started = time.perf_counter()
    try:
        yield facts
    except BaseException as e:
        raw = facts.pop("raw", None)
        record(stage, event, ok=False, ms=(time.perf_counter() - started) * 1000, raw=raw,
               error=str(e)[:500], error_type=e.__class__.__name__, **facts)
        raise
    raw = facts.pop("raw", None)
    record(stage, event, ok=True, ms=(time.perf_counter() - started) * 1000, raw=raw, **facts)


def trace(generation_id: str) -> Dict[str, Any]:
    """The full story of one generation, in order, with time since the job started."""
    rows = db.events_for(generation_id)
    start = rows[0]["at"] if rows else 0
    steps = []
    for row in rows:
        entry = {"t": round(row["at"] - start, 2), "stage": row["stage"], "event": row["event"],
                 "ok": None if row["ok"] is None else bool(row["ok"]), "ms": row["ms"]}
        for key in ("data_json", "raw_json"):
            if row[key]:
                entry[key[:-5]] = json.loads(row[key])
        steps.append(entry)
    return {"generation": db.generation(generation_id), "steps": steps}


def prune() -> None:
    db.clear_raw_events(time.time() - RAW_DAYS * 86400)
