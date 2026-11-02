"""SQLite store: assets, generations and route results. One file to back up.

WAL mode, one connection, short statements. Calls take a millisecond or so, so the
async job code uses this directly.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import MediaConfig

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  id TEXT PRIMARY KEY,              -- sha256 prefix of the uploaded file
  path TEXT NOT NULL, mime TEXT NOT NULL, bytes INTEGER NOT NULL,
  purpose TEXT NOT NULL DEFAULT 'reference',  -- reference | frame
  created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS asset_remote (   -- what each provider calls this file, so it's uploaded once
  asset_id TEXT NOT NULL, provider TEXT NOT NULL, handle TEXT NOT NULL, expires_at INTEGER,
  PRIMARY KEY (asset_id, provider)
);
CREATE TABLE IF NOT EXISTS generations (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,               -- image | video
  status TEXT NOT NULL,             -- queued | planning | keyframe | rendering | done | failed | cancelled
  message TEXT,
  chat_id TEXT, message_id INTEGER, message_key TEXT, group_id TEXT, folder TEXT,
  provider TEXT, model_id TEXT, family TEXT, mode TEXT,
  provider_task_id TEXT,            -- lets the bridge resume polling after a restart
  prompt TEXT, negative_prompt TEXT,
  plan_json TEXT,                   -- full director output
  params_json TEXT,                 -- exact request sent, without keys or image data
  refs_json TEXT,                   -- [{asset, person, slot}]
  loras_json TEXT,
  seed INTEGER, aspect_ratio TEXT, resolution TEXT, seconds REAL,
  cost_usd REAL, cost_credits REAL,
  parent_id TEXT,                   -- keyframe or Best Option step 1 this came from
  file_path TEXT, mime TEXT,
  st_url TEXT,                      -- set when the extension has saved it into SillyTavern
  error TEXT,
  rating INTEGER CHECK (rating BETWEEN 1 AND 5),
  comment TEXT,
  created_at INTEGER NOT NULL, finished_at INTEGER, acked_at INTEGER, rated_at INTEGER
);
CREATE TABLE IF NOT EXISTS generation_people (   -- a 3-person image shows in all 3 galleries
  generation_id TEXT NOT NULL, person_key TEXT NOT NULL, role TEXT NOT NULL,
  PRIMARY KEY (generation_id, person_key)
);
CREATE TABLE IF NOT EXISTS route_status (
  provider TEXT NOT NULL, model_id TEXT NOT NULL,
  status TEXT NOT NULL,             -- allowed | banned
  note TEXT, updated_at INTEGER NOT NULL,
  PRIMARY KEY (provider, model_id)
);
CREATE TABLE IF NOT EXISTS events (          -- one row per pipeline step, see telemetry.py
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at REAL NOT NULL,
  generation_id TEXT, chat_id TEXT,
  stage TEXT NOT NULL,              -- llm | director | guard | refs | loras | provider | job | client | voice | group ...
  event TEXT NOT NULL,
  ok INTEGER, ms INTEGER,
  data_json TEXT,                   -- numbers and short facts, kept for good
  raw_json TEXT                     -- full prompts and replies, cleared after TELEMETRY_RAW_DAYS
);
CREATE INDEX IF NOT EXISTS idx_events_gen ON events(generation_id);
CREATE INDEX IF NOT EXISTS idx_events_stage ON events(stage, event, at);
CREATE INDEX IF NOT EXISTS idx_gen_chat ON generations(chat_id, status);
CREATE INDEX IF NOT EXISTS idx_gen_key ON generations(message_key, kind);
CREATE INDEX IF NOT EXISTS idx_people ON generation_people(person_key);
"""

OPEN_STATUSES = ("queued", "planning", "keyframe", "rendering")

_conn: Optional[sqlite3.Connection] = None
_lock = threading.RLock()


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """Opens the database, once. Tests pass their own path."""
    global _conn
    with _lock:
        if path is not None and _conn is not None:
            close()
        if _conn is None:
            target = Path(path or MediaConfig.DATA_DIR / "media.db")
            target.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(str(target), check_same_thread=False, isolation_level=None)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")
            _conn.executescript(SCHEMA)
            _conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        return _conn


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def _run(sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
    with _lock:
        return connect().execute(sql, tuple(args))


def _rows(sql: str, args: Iterable[Any] = ()) -> List[Dict[str, Any]]:
    return [dict(r) for r in _run(sql, args).fetchall()]


def _one(sql: str, args: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
    row = _run(sql, args).fetchone()
    return dict(row) if row else None


# ---- generations ----

def insert_generation(**fields: Any) -> None:
    fields.setdefault("created_at", int(time.time()))
    names = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    _run(f"INSERT INTO generations ({names}) VALUES ({marks})", fields.values())


def update_generation(generation_id: str, **fields: Any) -> None:
    if fields:
        sets = ", ".join(f"{name} = ?" for name in fields)
        _run(f"UPDATE generations SET {sets} WHERE id = ?", [*fields.values(), generation_id])


def generation(generation_id: str) -> Optional[Dict[str, Any]]:
    return _one("SELECT * FROM generations WHERE id = ?", (generation_id,))


def _open_marks() -> str:
    return ", ".join("?" for _ in OPEN_STATUSES)


def open_for_message(message_key: str, kind: str) -> Optional[Dict[str, Any]]:
    """The running job for this message, so a double tap can't pay twice."""
    return _one(f"SELECT * FROM generations WHERE message_key = ? AND kind = ? AND status IN ({_open_marks()})",
                (message_key, kind, *OPEN_STATUSES))


def open_generations() -> List[Dict[str, Any]]:
    return _rows(f"SELECT * FROM generations WHERE status IN ({_open_marks()})", OPEN_STATUSES)


def open_for_chat(chat_id: str) -> List[Dict[str, Any]]:
    """Unfinished jobs, and finished ones the extension hasn't saved yet."""
    return _rows(f"SELECT * FROM generations WHERE chat_id = ? AND (status IN ({_open_marks()}) "
                 "OR (status = 'done' AND st_url IS NULL)) ORDER BY created_at", (chat_id, *OPEN_STATUSES))


def add_people(generation_id: str, people: Iterable[Dict[str, str]]) -> None:
    for person in people:
        _run("INSERT OR REPLACE INTO generation_people (generation_id, person_key, role) VALUES (?, ?, ?)",
             (generation_id, person["key"], person.get("role") or "other"))


def list_generations(person: str = "", kind: str = "", limit: int = 60, offset: int = 0) -> List[Dict[str, Any]]:
    where, args = ["g.status = 'done'"], []
    join = ""
    if person:
        join = "JOIN generation_people p ON p.generation_id = g.id"
        where.append("p.person_key = ?")
        args.append(person)
    if kind:
        where.append("g.kind = ?")
        args.append(kind)
    steps = "FROM generations c WHERE c.parent_id = g.id AND c.message_key IS NULL"    # keyframes, Best Option step 1
    return _rows(f"SELECT g.*, (SELECT SUM(c.cost_usd) {steps}) AS steps_usd, (SELECT SUM(c.cost_credits) {steps}) AS steps_credits "
                 f"FROM generations g {join} WHERE {' AND '.join(where)} "
                 "ORDER BY g.created_at DESC LIMIT ? OFFSET ?", [*args, limit, offset])


def unrated_count(person: str) -> int:
    row = _one("SELECT COUNT(*) AS n FROM generations g JOIN generation_people p ON p.generation_id = g.id "
               "WHERE p.person_key = ? AND g.status = 'done' AND g.rating IS NULL", (person,))
    return row["n"] if row else 0


def expired_files(saved_before: int, unsaved_before: int) -> List[Dict[str, Any]]:
    return _rows("SELECT * FROM generations WHERE file_path IS NOT NULL AND "
                 "((acked_at IS NOT NULL AND acked_at < ?) OR (acked_at IS NULL AND finished_at < ?))",
                 (saved_before, unsaved_before))


# ---- assets ----

def asset(asset_id: str) -> Optional[Dict[str, Any]]:
    return _one("SELECT * FROM assets WHERE id = ?", (asset_id,))


def put_asset(asset_id: str, path: str, mime: str, size: int, purpose: str) -> None:
    _run("INSERT OR REPLACE INTO assets (id, path, mime, bytes, purpose, created_at) VALUES (?, ?, ?, ?, ?, ?)",
         (asset_id, path, mime, size, purpose, int(time.time())))


def promote_asset(asset_id: str) -> None:
    _run("UPDATE assets SET purpose = 'reference' WHERE id = ?", (asset_id,))


def stale_frames(before: int) -> List[Dict[str, Any]]:
    return _rows("SELECT * FROM assets WHERE purpose = 'frame' AND created_at < ?", (before,))


def delete_asset(asset_id: str) -> None:
    _run("DELETE FROM assets WHERE id = ?", (asset_id,))
    _run("DELETE FROM asset_remote WHERE asset_id = ?", (asset_id,))


def remote_handle(asset_id: str, provider: str) -> Optional[str]:
    row = _one("SELECT handle, expires_at FROM asset_remote WHERE asset_id = ? AND provider = ?", (asset_id, provider))
    if not row or (row["expires_at"] and row["expires_at"] <= time.time()):
        return None
    return row["handle"]


def save_remote(asset_id: str, provider: str, handle: str, expires_at: Optional[int]) -> None:
    _run("INSERT OR REPLACE INTO asset_remote (asset_id, provider, handle, expires_at) VALUES (?, ?, ?, ?)",
         (asset_id, provider, handle, expires_at))


# ---- events ----

def add_event(at: float, generation_id: Optional[str], chat_id: Optional[str], stage: str, event: str,
              ok: Optional[bool], ms: Optional[int], data_json: Optional[str], raw_json: Optional[str]) -> None:
    _run("INSERT INTO events (at, generation_id, chat_id, stage, event, ok, ms, data_json, raw_json) "
         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
         (at, generation_id, chat_id, stage, event, None if ok is None else int(ok), ms, data_json, raw_json))


def events_for(generation_id: str) -> List[Dict[str, Any]]:
    return _rows("SELECT * FROM events WHERE generation_id = ? ORDER BY id", (generation_id,))


def events_since(since: float, stage: str = "", limit: int = 100000) -> List[Dict[str, Any]]:
    if stage:
        return _rows("SELECT * FROM events WHERE at >= ? AND stage = ? ORDER BY id LIMIT ?", (since, stage, limit))
    return _rows("SELECT * FROM events WHERE at >= ? ORDER BY id LIMIT ?", (since, limit))


def generations_since(since: int) -> List[Dict[str, Any]]:
    return _rows("SELECT * FROM generations WHERE created_at >= ? ORDER BY created_at", (since,))


def people_for(generation_ids: Iterable[str]) -> Dict[str, List[str]]:
    ids = list(generation_ids)
    out: Dict[str, List[str]] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ", ".join("?" for _ in chunk)
        for row in _rows(f"SELECT * FROM generation_people WHERE generation_id IN ({marks})", chunk):
            out.setdefault(row["generation_id"], []).append(row["person_key"])
    return out


def clear_raw_events(before: float) -> None:
    _run("UPDATE events SET raw_json = NULL WHERE at < ? AND raw_json IS NOT NULL", (before,))


# ---- routes ----

def route_override(provider: str, model_id: str) -> Optional[str]:
    row = _one("SELECT status FROM route_status WHERE provider = ? AND model_id = ?", (provider, model_id))
    return row["status"] if row else None


def set_route(provider: str, model_id: str, status: str, note: str = "") -> None:
    _run("INSERT OR REPLACE INTO route_status (provider, model_id, status, note, updated_at) VALUES (?, ?, ?, ?, ?)",
         (provider, model_id, status, note, int(time.time())))


def route_results() -> List[Dict[str, Any]]:
    return _rows("SELECT * FROM route_status")
