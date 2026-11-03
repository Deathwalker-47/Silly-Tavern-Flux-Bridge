"""The Media Studio HTTP API. Everything except /media/files/{id} needs the bridge token;
file ids are random, so <video> tags work without headers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from . import assets, catalog, db, director, insights, keys, lora_files, loras, telemetry
from .catalog import IMAGE_FAMILIES, PROVIDER_LABELS, Route, VIDEO_FAMILIES
from .config import MediaConfig
from .helper import helper
from .jobs import no_cancel_reason, public_job, runner
from .providers import PROVIDERS, provider_configured
from .references import DEFAULT_ORDER

logger = logging.getLogger("media.routes")

router = APIRouter()

MODE_TEXT = {
    "reference": {"label": "Reference images", "text": "Builds the clip straight from each character's reference photos. No image is made first. Use it when the scene has no image yet. Faces can vary more than with a first frame."},
    "first": {"label": "First frame", "text": "Starts from one still: this message's image, or a new one from Seedream 4.5 (Flux when LoRA is on with one character). Strongest likeness. The model decides how the scene ends."},
    "first_last": {"label": "First + last frame", "text": "Starts and ends on two stills, so you control where the scene lands. Best for a clear change, such as walking from the door to the window. Costs up to two extra images."},
}

_check_token: Optional[Callable[[Any], None]] = None
_token_set: Callable[[], bool] = lambda: False


def use_token_check(check: Callable[[Any], None], token_set: Callable[[], bool]) -> None:
    """The bridge's own check, so there's one copy of the token logic."""
    global _check_token, _token_set
    _check_token, _token_set = check, token_set


def public_catalog(kind: str) -> Dict[str, Any]:
    """Provider -> families -> options for the pickers. Failed routes are hidden."""
    labels = IMAGE_FAMILIES if kind == "image" else VIDEO_FAMILIES
    providers = []
    for name, families in catalog.tables(kind).items():
        public_families = []
        for family, options in families.items():
            shown = [o.to_public() for o in options if not o.hidden and catalog.route_status(o) is not Route.BANNED]
            if shown:
                public_families.append({"id": family, "label": labels.get(family, family), "variants": shown})
        providers.append({"id": name, "label": PROVIDER_LABELS.get(name, name), "configured": provider_configured(name),
                          "families": public_families})
    return {"families": labels, "providers": providers}


def failed_routes() -> List[Dict[str, str]]:
    """What you failed, so a mistaken Fail can be put back."""
    labels = {(name, o.id): o.label for kind in ("image", "video") for name, families in catalog.tables(kind).items()
              for options in families.values() for o in options}
    return [{"provider": r["provider"], "model": r["model_id"],
             "label": f"{PROVIDER_LABELS.get(r['provider'], r['provider'])}: {labels.get((r['provider'], r['model_id']), r['model_id'])}"}
            for r in db.route_results() if r["status"] == "banned"]


@router.get("/media/capabilities")
async def capabilities(request: Request = None):
    _check_token(request)
    return {
        "image": public_catalog("image"),
        "video": {**public_catalog("video"), "sogni_network": (MediaConfig.SOGNI_NETWORK or "fast").lower()},
        "failed_routes": failed_routes(),
        "modes": MODE_TEXT,
        "helper": {"configured": helper.configured, "backend": helper.backend()[3], "default_model": helper.backend()[2]},
        "reference_window": assets.WINDOW,
        "slots": DEFAULT_ORDER,
    }


async def balance_of(name: str) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"id": name, "label": PROVIDER_LABELS.get(name, name), "configured": True}
    try:
        entry.update(await asyncio.wait_for(PROVIDERS[name].balance(), timeout=20) or {"amount": None})
    except Exception as e:
        entry.update(amount=None, error=str(e)[:200])
    return entry


@router.get("/media/balances")
async def balances(request: Request = None):
    _check_token(request)
    names = [n for n in PROVIDERS if n != "mock" and provider_configured(n)]
    results = await asyncio.gather(*(balance_of(n) for n in names))
    for entry in results:
        telemetry.record("account", "balance", provider=entry["id"], amount=entry.get("amount"), unit=entry.get("unit"))
    return {"balances": results}


def _key_writes(request: Any, provider: str) -> None:
    """No loopback exception: behind a reverse proxy every request looks local."""
    _check_token(request)
    if not _token_set():
        raise HTTPException(status_code=403, detail="Set EDIT_API_TOKEN on the bridge to change keys from settings.")
    if provider not in keys.FIELDS:
        raise HTTPException(status_code=404, detail="Unknown provider")


@router.get("/media/keys")
async def list_keys(request: Request):
    _check_token(request)
    writable = _token_set()
    return {"keys": keys.status(hints=writable), "writable": writable}


@router.put("/media/keys/{provider}")
async def save_key(provider: str, request: Request):
    _key_writes(request, provider)
    key = (await request.json() or {}).get("key")
    if not isinstance(key, str) or not 8 <= len(key) <= 512 or any(c.isspace() for c in key):
        raise HTTPException(status_code=400, detail="That doesn't look like an API key.")
    keys.save(provider, key)
    telemetry.record("account", "key_saved", provider=provider)
    checked = await (lora_files.check_token() if provider == "huggingface" else balance_of(provider))
    return {"saved": True, "balance": checked.get("amount"), "unit": checked.get("unit"),
            "error": (checked.get("error") or "").replace(key, "") or None}


@router.delete("/media/keys/{provider}")
async def clear_key(provider: str, request: Request):
    _key_writes(request, provider)
    keys.clear(provider)
    telemetry.record("account", "key_cleared", provider=provider)
    return {"cleared": True}


@router.post("/assets")
async def upload_asset(request: Request, purpose: str = "reference"):
    """A raw image body. Hashes it, checks the window, strips metadata, stores it once."""
    _check_token(request)
    if purpose not in ("reference", "frame"):
        raise HTTPException(status_code=400, detail="purpose must be reference or frame")
    declared = int((getattr(request, "headers", None) or {}).get("content-length") or 0)
    if purpose == "reference" and declared > assets.MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"Image is {declared // (1024 * 1024)} MB. The maximum is 10 MB.")
    data = await request.body()
    try:
        with telemetry.step("assets", "store", purpose=purpose, bytes=len(data)) as facts:
            asset, existed = await asyncio.to_thread(assets.store, data, purpose)
            facts.update(existed=existed, stored_bytes=asset.bytes)
    except assets.AssetRejected as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"asset_id": asset.id, "existed": existed, "bytes": asset.bytes}


@router.get("/assets/{asset_id}/exists")
async def asset_exists(asset_id: str, request: Request = None):
    _check_token(request)
    return {"exists": assets.exists(asset_id)}


@router.post("/media/lora-status")
async def lora_status(request: Request):
    _check_token(request)
    body = await request.json()
    prompts = body.get("character_prompts") or {}
    return {"characters": {n: bool(loras.character_lora(n, prompts.get(n, ""))) for n in body.get("names") or []}}


# ---- director ----

@router.post("/director/decide")
async def director_decide(request: Request):
    """Whether to send nothing, an image or a video this turn."""
    _check_token(request)
    try:
        return await director.decide(await request.json())
    except Exception as e:
        logger.error(f"🧠 [Director] decide failed: {e}")
        raise HTTPException(status_code=502, detail=f"Director failed: {e}")


@router.post("/director/next-speaker")
async def director_next_speaker(request: Request):
    """Group loop: the next speaker, or the user's turn."""
    _check_token(request)
    try:
        return await director.next_speaker(await request.json())
    except Exception as e:
        logger.error(f"🧠 [Director] next speaker failed: {e}")
        raise HTTPException(status_code=502, detail=f"Director failed: {e}")


# ---- jobs ----

def _start(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        row = runner.create(body)
    except (ValueError, KeyError, catalog.RouteNotAllowed) as e:
        raise HTTPException(status_code=400, detail=str(e).strip("'\""))
    return {"job_id": row["id"], "status": row["status"]}


@router.post("/media/jobs")
async def create_job(request: Request):
    _check_token(request)
    return _start(await request.json())


@router.get("/media/jobs")
async def list_jobs(chat_id: str = "", open: int = 1, request: Request = None):
    """Unfinished or unsaved jobs in this chat, for picking up after a reload."""
    _check_token(request)
    return {"jobs": [public_job(r) for r in db.open_for_chat(chat_id)]}


@router.get("/media/jobs/{job_id}")
async def get_job(job_id: str, request: Request = None):
    _check_token(request)
    row = db.generation(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown job")
    return public_job(row)


@router.post("/media/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request = None):
    _check_token(request)
    row = db.generation(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown job")
    refusal = no_cancel_reason(row)
    if refusal:
        raise HTTPException(status_code=409, detail=refusal)
    return public_job(runner.cancel(job_id) or row)


@router.post("/media/jobs/{job_id}/ack")
async def ack_job(job_id: str, request: Request):
    """The file is saved into SillyTavern, so the bridge's copy can expire."""
    _check_token(request)
    if not db.generation(job_id):
        raise HTTPException(status_code=404, detail="Unknown job")
    body = await request.json()
    db.update_generation(job_id, st_url=body.get("st_url") or "saved", acked_at=int(time.time()))
    telemetry.record("client", "acked", generation_id=job_id, attached=bool(body.get("attached")), reason=body.get("reason"),
                     wait_seconds=body.get("wait_seconds"))
    return {"ok": True}


@router.get("/media/files/{file_id}")
async def media_file(file_id: str):
    from fastapi.responses import FileResponse  # imported lazily; not part of the test stubs

    row = db.generation(file_id) if file_id.isalnum() else None
    if not row or not row.get("file_path") or not Path(row["file_path"]).exists():
        raise HTTPException(status_code=404, detail="File not found or expired")
    return FileResponse(row["file_path"], media_type=row.get("mime") or "application/octet-stream")


@router.api_route("/lora-files/{link}/{name}", methods=["GET", "HEAD"])
async def lora_file(link: str, name: str, request: Request):
    """A private LoRA for a provider. No token: the random link is the key, and it expires."""
    url = lora_files.source(link)
    if not url:
        raise HTTPException(status_code=404, detail="Link expired")
    return await lora_files.stream(url, request.method, request.headers.get("range"))


# ---- gallery ----

def public_generation(row: Dict[str, Any]) -> Dict[str, Any]:
    def parsed(name: str) -> Any:
        return json.loads(row[name]) if row.get(name) else None

    def cost(name: str, steps: str) -> Any:
        """What the whole generation cost, not only its last render."""
        spent = [v for v in (row.get(name), row.get(steps)) if v is not None]
        return round(sum(spent), 4) if spent else None

    return {
        "id": row["id"], "kind": row["kind"], "provider": row.get("provider"), "provider_label": PROVIDER_LABELS.get(row.get("provider") or "", row.get("provider")),
        "model_id": row.get("model_id"), "family": row.get("family"), "mode": row.get("mode"), "prompt": row.get("prompt"),
        "seed": row.get("seed"), "aspect_ratio": row.get("aspect_ratio"), "resolution": row.get("resolution"), "seconds": row.get("seconds"),
        "cost_usd": cost("cost_usd", "steps_usd"), "cost_credits": cost("cost_credits", "steps_credits"), "rating": row.get("rating"), "comment": row.get("comment"),
        "url": row.get("st_url") if (row.get("st_url") or "").startswith("/") else None,
        "file_url": f"/media/files/{row['id']}" if row.get("file_path") else None,
        "refs": parsed("refs_json") or [], "loras": parsed("loras_json") or [], "created_at": row["created_at"],
    }


@router.get("/media/generations")
async def list_generations(person: str = "", kind: str = "", limit: int = 60, offset: int = 0, request: Request = None):
    _check_token(request)
    rows = db.list_generations(person, kind, min(limit, 200), offset)
    return {"generations": [public_generation(r) for r in rows], "unrated": db.unrated_count(person) if person else 0}


@router.get("/media/generations/{generation_id}/trace")
async def generation_trace(generation_id: str, request: Request = None):
    """Every step of one generation in order: prompts, references, timing, errors."""
    _check_token(request)
    if not db.generation(generation_id):
        raise HTTPException(status_code=404, detail="Unknown generation")
    return telemetry.trace(generation_id)


@router.patch("/media/generations/{generation_id}")
async def rate_generation(generation_id: str, request: Request):
    _check_token(request)
    row = db.generation(generation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown generation")
    body = await request.json()
    fields: Dict[str, Any] = {}
    if "rating" in body:
        rating = body["rating"]
        if rating is not None and (not isinstance(rating, int) or not 1 <= rating <= 5):
            raise HTTPException(status_code=400, detail="rating must be 1 to 5")
        fields.update(rating=rating, rated_at=int(time.time()))
    if "comment" in body:
        fields["comment"] = str(body["comment"] or "")
    db.update_generation(generation_id, **fields)
    telemetry.record("client", "rated", generation_id=generation_id, rating=fields.get("rating"), commented=bool(fields.get("comment")),
                     provider=row.get("provider"), model=row.get("model_id"))
    return {"ok": True}


# ---- routes under test ----

@router.post("/media/routes/test")
async def test_route(request: Request):
    """Runs your own prompt on any route at the lowest setting, shortest clip."""
    _check_token(request)
    body = await request.json()
    if not str(body.get("prompt") or "").strip():
        raise HTTPException(status_code=400, detail="Write a prompt to test with")
    return _start({**body, "test": True, "origin": "route_test"})


@router.put("/media/routes/{provider}/{model:path}")
async def save_route(provider: str, model: str, request: Request):
    """Pass or Fail from Test this route. Your result wins over the catalog."""
    _check_token(request)
    body = await request.json()
    status = body.get("status")
    if status not in ("allowed", "banned"):
        raise HTTPException(status_code=400, detail="status must be allowed or banned")
    db.set_route(provider, model, status, str(body.get("note") or ""))
    telemetry.record("client", "route_result", provider=provider, model=model, status=status)
    return {"ok": True}


# ---- telemetry ----

@router.post("/media/events")
async def client_events(request: Request):
    """What only the browser knows: attached or missed, auto skips, reloads, round endings."""
    _check_token(request)
    body = await request.json()
    for event in (body.get("events") or [])[:50]:
        name = str(event.get("event") or "unknown")[:60]
        facts = {k: v for k, v in event.items() if k not in ("event", "generation_id", "chat_id", "ok", "ms", "stage")}
        telemetry.record("client", name, generation_id=event.get("generation_id"), chat_id=event.get("chat_id"),
                         ok=event.get("ok"), ms=event.get("ms"), **facts)
    return {"ok": True}


@router.get("/media/insights")
async def media_insights(days: int = 30, request: Request = None):
    """What works, what's slow, what fails and what to change, from real runs and your ratings."""
    _check_token(request)
    return await asyncio.to_thread(insights.report, days)


@router.get("/media/insights/export")
async def media_export(days: int = 30, request: Request = None):
    from fastapi.responses import PlainTextResponse  # imported lazily; not part of the test stubs

    _check_token(request)
    return PlainTextResponse("\n".join(insights.export_lines(days)), media_type="application/x-ndjson")
