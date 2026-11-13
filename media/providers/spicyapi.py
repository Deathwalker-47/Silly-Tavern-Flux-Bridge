"""SpicyAPI (spicyapi.ai): images and video through one async job API, with no content filter.

POST /jobs/createTask {"model", "input"}, then GET /jobs/recordInfo?taskId=. Every answer is an
envelope {"code", "msg", "data"}. A SpicyAPI model id is one task (".../reference-to-video",
".../image-to-video", ".../edit", ".../text-to-image"), so a catalog entry lists its ids in
extra["tasks"]. Unknown input fields are refused (400), so a request holds only that model's fields.

Pictures go up once through the upload flow and come back as spicy://f/ handles that last a day.
A data URL would skip that, but the whole request is capped at 2 MiB. Output links last 20
minutes, so a finished file is downloaded at once. The charge is held at the quote and settles
when the task ends; the poll reads it. The public API has no cancel.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from .. import assets, db, lora_files
from ..config import MediaConfig
from .base import Provider, RenderRequest, RenderResult, check_key, download, error_text, send, sniff_mime

URL_LIFETIME = 23 * 3600         # spicy://f/ uploads last a day; stop reusing one before it dies
UPLOAD_EDGE, UPLOAD_QUALITY = 2560, 90
MAX_LORAS = 3
FINISHED = ("succeeded", "failed", "canceled", "expired")


def _usd(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fail(resp: httpx.Response, payload: Any) -> RuntimeError:
    code = payload.get("code") if isinstance(payload, dict) else None
    if resp.status_code == 402 or code == 40201:
        return RuntimeError("SpicyAPI: out of credit")
    message = payload.get("msg") if isinstance(payload, dict) and payload.get("msg") else error_text(resp)
    return RuntimeError(f"SpicyAPI {resp.status_code}: {message}")


def unwrap(resp: httpx.Response) -> Dict[str, Any]:
    """The envelope's data, or an error with SpicyAPI's own message."""
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    code = payload.get("code") if isinstance(payload, dict) else None
    if resp.status_code >= 300 or (code is not None and code != 200):
        raise _fail(resp, payload)
    return (payload.get("data") if isinstance(payload, dict) else None) or {}


def task_model(req: RenderRequest) -> str:
    tasks = req.option.extra["tasks"]
    if req.option.kind == "video":
        return tasks["reference" if req.mode == "reference" else "first"]
    if req.mode == "lora":
        return tasks["lora"]
    if req.refs:
        return tasks["reference"]
    if "text" not in tasks:
        raise RuntimeError(f"{req.option.label} needs at least one reference image")
    return tasks["text"]


def _seed(inp: Dict[str, Any], req: RenderRequest) -> None:
    if req.seed is not None and not req.option.extra.get("no_seed"):
        inp["seed"] = int(req.seed)


def lora_list(req: RenderRequest) -> List[Dict[str, Any]]:
    """The character LoRA first, then concepts, up to the three a model takes."""
    return [{"path": l["url"], "scale": float(l.get("weight", 1))} for l in req.loras if l.get("url")][:MAX_LORAS]


def seedream(req: RenderRequest, refs: List[str]) -> Dict[str, Any]:
    inp: Dict[str, Any] = {"prompt": req.prompt, "aspect_ratio": req.aspect_ratio,
                           "resolution": req.option.extra["resolution"]}
    if req.option.extra.get("output_format"):          # 5.5 sends PNG unless told otherwise
        inp["output_format"] = req.option.extra["output_format"]
    if refs:
        inp["image_urls"] = refs
    return inp


def qwen_edit(req: RenderRequest, refs: List[str]) -> Dict[str, Any]:
    inp: Dict[str, Any] = {"prompt": req.prompt, "image_urls": refs, "aspect_ratio": req.aspect_ratio}
    _seed(inp, req)
    return inp


def qwen_spicy(req: RenderRequest, refs: List[str]) -> Dict[str, Any]:
    inp: Dict[str, Any] = {"prompt": req.prompt, "image_url": refs[0]}
    _seed(inp, req)
    return inp


def lora_image(req: RenderRequest, refs: List[str]) -> Dict[str, Any]:
    extra = req.option.extra
    inp: Dict[str, Any] = {"prompt": req.prompt, "aspect_ratio": req.aspect_ratio, "output_format": "jpeg"}
    loras = lora_list(req)
    if loras:
        inp["loras"] = loras
    if extra.get("steps"):
        inp["steps"] = extra["steps"]
    if extra.get("cfg"):
        inp["guidance_scale"] = extra["cfg"]
    _seed(inp, req)
    return inp


def sized_text(req: RenderRequest, refs: List[str]) -> Dict[str, Any]:
    width, height = req.option.extra["sizes"][req.aspect_ratio]
    inp: Dict[str, Any] = {"prompt": req.prompt, "width": width, "height": height}
    _seed(inp, req)
    return inp


def video(req: RenderRequest, refs: List[str], first: Optional[str], last: Optional[str]) -> Dict[str, Any]:
    extra = req.option.extra
    inp: Dict[str, Any] = {"prompt": req.prompt, "duration_seconds": req.seconds, "resolution": req.resolution}
    if extra.get("audio_switch"):
        inp["generate_audio"] = bool(req.audio)     # SpicyAPI adds sound unless told not to
    _seed(inp, req)
    if req.mode == "reference":
        inp["reference_image_urls"] = refs
        inp["aspect_ratio"] = req.aspect_ratio
        return inp
    inp["image_url"] = first
    if last and req.mode == "first_last":
        inp["last_image_url"] = last
    shape = extra.get("i2v_aspect")                  # most follow the frame; Wan 3.0 falls back to 16:9 without one
    if shape:
        inp["aspect_ratio"] = req.aspect_ratio if shape == "frame" else shape
    return inp


IMAGE_BUILDERS = {"seedream": seedream, "qwen_edit": qwen_edit, "qwen_spicy": qwen_spicy, "lora": lora_image,
                  "sized_text": sized_text}
HIDDEN = ("image_urls", "image_url", "last_image_url", "reference_image_urls")


def output_url(record: Dict[str, Any], kind: str) -> Optional[str]:
    """The finished file's link. None while a link is still being made."""
    files = (record.get("output") or {}).get("assets") or []
    ready = [a for a in files if a.get("url") and not a.get("pending") and not a.get("unavailable")]
    wanted = [a for a in ready if (a.get("role") == kind) or str(a.get("mime") or "").startswith(kind)]
    pick = (wanted or ready or [None])[0]
    if pick:
        return pick["url"]
    if any(a.get("pending") for a in files):
        return None
    raise RuntimeError("SpicyAPI finished without a file")


class SpicyAPI(Provider):
    name = "spicyapi"
    poll_seconds = 4.0

    @property
    def configured(self) -> bool:
        return bool(MediaConfig.SPICYAPI_API_KEY)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {MediaConfig.SPICYAPI_API_KEY}"}

    async def ensure_asset(self, asset: Any) -> str:
        handle = db.remote_handle(asset.id, self.name)
        if handle:
            return handle
        data = assets.to_jpeg(asset, UPLOAD_EDGE, UPLOAD_QUALITY)
        base = MediaConfig.SPICYAPI_BASE
        async with httpx.AsyncClient(timeout=120) as client:
            ticket = unwrap(await send(client, "POST", f"{base}/common/upload-url", headers=self._headers(),
                                       json={"contentType": "image/jpeg", "bytes": len(data)}))
            put = await client.request(ticket.get("method") or "PUT", ticket["uploadUrl"], content=data,
                                       headers=ticket.get("headers") or {})
            if put.status_code >= 300:
                raise RuntimeError(f"SpicyAPI upload {put.status_code}")
            committed = unwrap(await send(client, "POST", f"{base}/files/{ticket['fileId']}/commit", headers=self._headers()))
        uri = committed.get("uri")
        if not uri:
            raise RuntimeError("SpicyAPI upload gave no file handle")
        db.save_remote(asset.id, self.name, uri, int(time.time()) + URL_LIFETIME)
        return uri

    async def build(self, req: RenderRequest) -> Dict[str, Any]:
        refs = [await self.ensure_asset(a) for a in req.refs]
        if req.option.kind == "image":
            if req.option.extra.get("needs_ref") and not refs:
                raise RuntimeError(f"{req.option.label} needs at least one reference image")
            return IMAGE_BUILDERS[req.option.extra["build"]](req, refs)
        first = await self.ensure_asset(req.first) if req.first else None
        last = await self.ensure_asset(req.last) if req.last else None
        return video(req, refs, first, last)

    async def submit(self, req: RenderRequest) -> str:
        model = task_model(req)
        inp = await self.build(req)
        req.params = {"model": model, **{k: v for k, v in inp.items() if k not in HIDDEN}, "references": len(req.refs)}
        if "loras" in inp:
            req.params["loras"] = [l["path"] for l in inp["loras"]]     # the history keeps the Hugging Face links
            inp["loras"] = [{**l, "path": lora_files.provider_url(l["path"])} for l in inp["loras"]]
        headers = {**self._headers(), "Idempotency-Key": str(uuid.uuid4())}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await send(client, "POST", f"{MediaConfig.SPICYAPI_BASE}/jobs/createTask",
                              json={"model": model, "input": inp}, headers=headers)
        task = unwrap(resp)
        req.cost_usd = _usd(task.get("estimatedCost"))
        if not task.get("taskId"):
            raise RuntimeError(f"SpicyAPI returned no task id: {resp.text[:200]}")
        return task["taskId"]

    async def poll(self, task_id: str, req: RenderRequest) -> Optional[RenderResult]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await send(client, "GET", f"{MediaConfig.SPICYAPI_BASE}/jobs/recordInfo", params={"taskId": task_id},
                              headers=self._headers())
        if resp.status_code == 404:
            raise RuntimeError("SpicyAPI: the task expired before it could be collected")
        if resp.status_code >= 300:
            return None                     # a blip; the job's deadline ends the wait if it lasts
        record = unwrap(resp)
        state = record.get("state")
        if state in FINISHED and state != "succeeded":
            reason = record.get("errorMessage") or record.get("errorCode") or state
            raise RuntimeError(f"SpicyAPI {state}: {reason}")
        if state != "succeeded":
            return None
        url = output_url(record, req.option.kind)
        if not url:
            return None
        data = await download(url)
        return RenderResult(data=data, mime=sniff_mime(data), seed=(record.get("input") or {}).get("seed"),
                            cost_usd=_usd(record.get("cost")), task_id=task_id)

    async def balance(self) -> Optional[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{MediaConfig.SPICYAPI_BASE}/chat/credit", headers=self._headers())
        check_key(resp)
        if resp.status_code != 200:
            return None
        amount = _usd(((resp.json() or {}).get("data") or {}).get("available"))
        return {"amount": amount, "unit": "USD"} if amount is not None else None
