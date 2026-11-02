"""Turns the event record into answers: what works, what's slow, what fails, what to change.

`report` powers the settings page and dev/insights.py. Each section joins what happened
(events) with what it cost and how you rated it (generations), so a change to a guide,
a default model or a setting can be judged on real results.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import db

MIN_RATED = 3           # ratings needed before a comparison means anything
SLOW_STEP_MS = 60_000


def _percentile(values: List[float], share: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * share))]


def _mean(values: Iterable[float]) -> Optional[float]:
    items = [v for v in values if v is not None]
    return round(statistics.mean(items), 3) if items else None


def _json(text: Optional[str]) -> Any:
    return json.loads(text) if text else None


def _plan(row: Dict[str, Any]) -> Dict[str, Any]:
    return _json(row.get("plan_json")) or {}


def _rating_table(rows: List[Dict[str, Any]], key: Callable[[Dict[str, Any]], Any]) -> List[Dict[str, Any]]:
    """Runs and average rating per value of `key`, best rated first."""
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = key(row)
        for item in value if isinstance(value, list) else [value]:
            if item not in (None, ""):
                groups[item].append(row)
    table = [{"value": value, "runs": len(items), "rated": sum(1 for r in items if r.get("rating")),
              "rating": _mean(r.get("rating") for r in items)} for value, items in groups.items()]
    return sorted(table, key=lambda t: (t["rating"] is None, -(t["rating"] or 0), -t["runs"]))


def routes_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["kind"], row.get("provider"), row.get("model_id"), row.get("mode"))].append(row)
    table = []
    for (kind, provider, model, mode), items in groups.items():
        done = [r for r in items if r["status"] == "done"]
        failed = [r for r in items if r["status"] == "failed"]
        seconds = [r["finished_at"] - r["created_at"] for r in done if r.get("finished_at")]
        errors = Counter((r.get("error") or "unknown")[:120] for r in failed)
        table.append({
            "kind": kind, "provider": provider, "model": model, "mode": mode, "runs": len(items), "done": len(done),
            "failed": len(failed), "cancelled": sum(1 for r in items if r["status"] == "cancelled"),
            "success_rate": round(len(done) / len(items), 3) if items else None,
            "median_seconds": _percentile(seconds, 0.5), "p90_seconds": _percentile(seconds, 0.9),
            "avg_usd": _mean(r.get("cost_usd") for r in done), "total_usd": round(sum(r.get("cost_usd") or 0 for r in done), 4),
            "avg_credits": _mean(r.get("cost_credits") for r in done), "total_credits": round(sum(r.get("cost_credits") or 0 for r in done), 2),
            "rated": sum(1 for r in done if r.get("rating")), "rating": _mean(r.get("rating") for r in done),
            "low_ratings": sum(1 for r in done if (r.get("rating") or 5) <= 2), "top_errors": errors.most_common(3),
        })
    return sorted(table, key=lambda t: -t["runs"])


def steps_table(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("ms") is not None or event.get("ok") is not None:
            groups[(event["stage"], event["event"])].append(event)
    table = []
    for (stage, name), items in groups.items():
        times = [e["ms"] for e in items if e.get("ms") is not None]
        marked = [e for e in items if e.get("ok") is not None]
        table.append({"stage": stage, "event": name, "count": len(items),
                      "failure_rate": round(sum(1 for e in marked if not e["ok"]) / len(marked), 3) if marked else None,
                      "median_ms": _percentile(times, 0.5), "p90_ms": _percentile(times, 0.9), "max_ms": max(times) if times else None})
    return sorted(table, key=lambda t: (t["stage"], -(t["median_ms"] or 0)))


def _facts(events: List[Dict[str, Any]], stage: str, name: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for event in events:
        if event["stage"] == stage and (name is None or event["event"] == name):
            out.append({**(_json(event.get("data_json")) or {}), "ok": event.get("ok"), "ms": event.get("ms"), "event": event["event"]})
    return out


def director_section(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    plans = _facts(events, "director", "plan")
    decides = _facts(events, "director", "decide")
    guards = _facts(events, "guard")
    return {
        "plans": len(plans), "failed": sum(1 for p in plans if not p["ok"]), "retries": sum(1 for p in plans if (p.get("attempt") or 1) > 1),
        "median_ms": _percentile([p["ms"] for p in plans if p.get("ms") is not None], 0.5),
        "over_prompt_limit": sum(1 for p in plans if p.get("over_limit")),
        "blocked": sum(1 for g in guards if g["ok"] == 0), "block_reasons": Counter(g.get("reason") for g in guards if g["ok"] == 0).most_common(5),
        "decide": dict(Counter(d.get("action") for d in decides)),
        "aspect_overridden": sum(1 for a in _facts(events, "pipeline", "aspect") if a.get("wanted") and a.get("wanted") != a.get("used")),
    }


def client_section(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = _facts(events, "client")
    return {"events": dict(Counter(c["event"] for c in client)),
            "skips": Counter(c.get("reason") for c in client if c["event"] == "auto_skip").most_common(8)}


def voice_section(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    calls = _facts(events, "llm", "voice_tags")
    tags = _facts(events, "voice", "tags")
    return {"calls": len(calls), "failed": sum(1 for c in calls if not c["ok"]),
            "median_ms": _percentile([c["ms"] for c in calls if c.get("ms") is not None], 0.5),
            "p90_ms": _percentile([c["ms"] for c in calls if c.get("ms") is not None], 0.9),
            "rejected": sum(1 for t in tags if t.get("rejected")), "changed": sum(1 for t in tags if t.get("changed"))}


def group_section(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    speakers = _facts(events, "group", "next_speaker")
    rounds = _facts(events, "client", "group_round")
    return {"speaker_calls": len(speakers), "user_turns": sum(1 for s in speakers if s.get("speaker") == "user"),
            "long_replies": sum(1 for s in speakers if s.get("long")),
            "round_stops": dict(Counter(r.get("stopped_by") for r in rounds)), "avg_round_messages": _mean(r.get("messages") for r in rounds)}


def suggestions(routes: List[Dict[str, Any]], steps: List[Dict[str, Any]], director: Dict[str, Any],
                voice: Dict[str, Any], tables: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Plain findings from the numbers. Each one names something concrete to look at."""
    notes: List[str] = []
    for route in routes:
        name = f"{route['model']} on {route['provider']}"
        if route["runs"] >= 3 and (route["success_rate"] or 0) < 0.7:
            reason = route["top_errors"][0][0] if route["top_errors"] else "unknown"
            notes.append(f"{name} succeeds only {route['success_rate']:.0%} of the time ({route['failed']} failed). Most common: {reason}")
        if route["rated"] >= MIN_RATED and route["low_ratings"] / route["rated"] >= 0.4:
            notes.append(f"{name} gets low ratings on {route['low_ratings']} of {route['rated']} rated runs. Its guide or route needs a look.")
    for kind in ("image", "video"):
        rated = [r for r in routes if r["kind"] == kind and r["rated"] >= MIN_RATED]
        if len(rated) >= 2:
            best, worst = max(rated, key=lambda r: r["rating"]), min(rated, key=lambda r: r["rating"])
            if best["rating"] - worst["rating"] >= 0.8:
                notes.append(f"For {kind}s, {best['model']} rates {best['rating']} against {worst['rating']} for {worst['model']}.")
    for step in steps:
        if step["stage"] != "provider" and (step["median_ms"] or 0) > SLOW_STEP_MS:
            notes.append(f"{step['stage']}/{step['event']} is slow: median {step['median_ms'] / 1000:.0f} s.")
    if director["plans"] >= 5 and director["retries"] / director["plans"] > 0.1:
        notes.append(f"The director needed a retry on {director['retries']} of {director['plans']} plans. Check its raw replies in a trace.")
    if director["over_prompt_limit"]:
        notes.append(f"The director went over a model's prompt limit {director['over_prompt_limit']} times, so those prompts were cut.")
    if voice["calls"] >= 10 and voice["rejected"] / voice["calls"] > 0.15:
        notes.append(f"Voice tags changed the words on {voice['rejected']} of {voice['calls']} lines, so those were spoken untagged.")
    if (voice["p90_ms"] or 0) > 3000:
        notes.append(f"Voice tags are slow: p90 {voice['p90_ms'] / 1000:.1f} s against a 1.5 s first-sentence budget.")
    for name, table in tables.items():
        rated = [row for row in table if row["rated"] >= MIN_RATED]
        if len(rated) >= 2 and rated[0]["rating"] - rated[-1]["rating"] >= 1:
            notes.append(f"By {name}: {rated[0]['value']} rates {rated[0]['rating']}, {rated[-1]['value']} rates {rated[-1]['rating']}.")
    return notes


def report(days: int = 30) -> Dict[str, Any]:
    since = time.time() - days * 86400
    everything = db.generations_since(int(since))
    rows = [r for r in everything if r.get("chat_id") or r.get("message_key") or r["status"] != "done"]   # no keyframes or steps
    events = db.events_since(since)
    done = [r for r in rows if r["status"] == "done"]
    people = db.people_for(r["id"] for r in done)

    def count_refs(row: Dict[str, Any]) -> str:
        count = len(_json(row.get("refs_json")) or [])
        return "none" if not count else "1-3" if count <= 3 else "4-9" if count <= 9 else "10+"

    def prompt_size(row: Dict[str, Any]) -> str:
        size = len(row.get("prompt") or "")
        return "under 300" if size < 300 else "300-800" if size < 800 else "800-2000" if size < 2000 else "2000+"

    tables = {
        "aspect ratio": _rating_table(done, lambda r: r.get("aspect_ratio")),
        "reference count": _rating_table(done, count_refs),
        "prompt length": _rating_table(done, prompt_size),
        "POV": _rating_table(done, lambda r: "pov" if _plan(r).get("pov") else "third person" if r.get("plan_json") else None),
        "LoRA": _rating_table(done, lambda r: [l.get("id") for l in _json(r.get("loras_json")) or []]),
        "person": _rating_table(done, lambda r: people.get(r["id"]) or None),
    }
    routes = routes_table(rows)
    steps = steps_table(events)
    director = director_section(events)
    voice = voice_section(events)
    return {
        "days": days,
        "totals": {"generations": len(rows), "done": len(done), "failed": sum(1 for r in rows if r["status"] == "failed"),
                   "usd": round(sum(r.get("cost_usd") or 0 for r in everything), 4),
                   "credits": round(sum(r.get("cost_credits") or 0 for r in everything), 2),
                   "rated": sum(1 for r in done if r.get("rating")), "rating": _mean(r.get("rating") for r in done)},
        "routes": routes, "steps": steps, "director": director, "voice": voice,
        "groups": group_section(events), "client": client_section(events), "tables": tables,
        "comments": [{"id": r["id"], "model": r.get("model_id"), "rating": r.get("rating"), "comment": r.get("comment")}
                     for r in done if r.get("comment")][-50:],
        "suggestions": suggestions(routes, steps, director, voice, tables),
    }


def export_lines(days: int = 30) -> Iterable[str]:
    """Every generation and event as JSON lines, for analysis outside the bridge."""
    since = time.time() - days * 86400
    for row in db.generations_since(int(since)):
        yield json.dumps({"type": "generation", **row}, ensure_ascii=False)
    for event in db.events_since(since):
        yield json.dumps({"type": "event", **event}, ensure_ascii=False)
