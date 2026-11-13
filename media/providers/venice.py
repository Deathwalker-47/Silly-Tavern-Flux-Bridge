"""Venice: video on POST /video/queue then /video/retrieve, images on /image/generate and /image/multi-edit.

Pay as you go on an API key, no subscription. Frames and references go in as data URLs, so
nothing is uploaded anywhere first. Venice rejects unknown fields (400), so each request holds
only fields that model takes. Every Venice model is its own id per task, so a catalog entry
lists them in extra["tasks"]: "reference" and "first" for video. A first-and-last clip uses
the "first" model with an end frame.

The video price comes from /video/quote (free) just before the clip is queued, so the record
holds the real charge. An inference key can't read /billing/balance (it needs an admin key);
/api_keys/rate_limits reports the same USD balance and works with any key.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .. import assets
from ..config import MediaConfig
from .base import Provider, RenderRequest, RenderResult, check_key, error_text, send, sniff_mime, to_data_uri

MAX_REQUEST = 8_000_000          # base64 total for one request; the bigger ones get 413
MAX_SEED = 999_999_999


def _fail(resp: httpx.Response) -> RuntimeError:
    if resp.status_code == 402:
        return RuntimeError("Venice: out of credit")
    if resp.status_code == 422:
        return RuntimeError(f"Venice refused this under its content policy: {error_text(resp)}")
    if resp.status_code == 503:
        return RuntimeError("Venice: the model is at capacity, try again shortly")
    return RuntimeError(f"Venice {resp.status_code}: {error_text(resp)}")


def video_model(req: RenderRequest) -> str:
    return req.option.extra["tasks"]["reference" if req.mode == "reference" else "first"]


def build_video(req: RenderRequest, first: Optional[str], last: Optional[str], refs: List[str]) -> Dict[str, Any]:
    extra = req.option.extra
    body: Dict[str, Any] = {"model": video_model(req), "prompt": req.prompt, "duration": f"{req.seconds}s",
                            "resolution": req.resolution}
    if extra.get("audio_switch"):
        body["audio"] = bool(req.audio)     # Venice adds sound unless told not to
    if req.mode == "reference":
        body["reference_image_urls"] = refs
        body["aspect_ratio"] = req.aspect_ratio
    else:
        body["image_url"] = first
        if last and req.mode == "first_last":
            body["end_image_url"] = last
        shape = extra.get("i2v_aspect")     # most take the frame's shape on their own; LTX needs it named
        if shape:
            body["aspect_ratio"] = req.aspect_ratio if shape == "frame" else shape
    return body


def build_image(req: RenderRequest, refs: List[str]) -> Tuple[str, Dict[str, Any]]:
    """The path and body: an edit of the references, or a new picture from text alone."""
    extra = req.option.extra
    if refs:
        body: Dict[str, Any] = {"modelId": extra["edit"], "prompt": req.prompt, "images": refs,
                                "aspect_ratio": req.aspect_ratio, "safe_mode": False}
        if extra.get("resolution"):
            body["resolution"] = extra["resolution"]
        return "/image/multi-edit", body
    if not extra.get("generate"):
        raise RuntimeError(f"{req.option.label} needs at least one reference image")
    body = {"model": extra["generate"], "prompt": req.prompt, "safe_mode": False, "format": "jpeg",
            "return_binary": True, "hide_watermark": True, "aspect_ratio": req.aspect_ratio}
    if extra.get("resolution"):
        body["resolution"] = extra["resolution"]
    if req.seed is not None:
        body["seed"] = int(req.seed) % MAX_SEED
    return "/image/generate", body


def split_task(task_id: str) -> Tuple[str, str]:
    """A queue that answers with its own download link keeps it in the task id, so a restart can still collect."""
    queue_id, _, url = task_id.partition("|")
    return queue_id, url


class Venice(Provider):
    name = "venice"
    poll_seconds = 5.0

    def __init__(self) -> None:
        self._done: Dict[str, RenderResult] = {}

    @property
    def configured(self) -> bool:
        return bool(MediaConfig.VENICE_API_KEY)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {MediaConfig.VENICE_API_KEY}"}

    # ---- images ----

    async def _image(self, req: RenderRequest) -> str:
        if req.option.extra.get("needs_ref") and not req.refs:
            raise RuntimeError(f"{req.option.label} needs at least one reference image")
        refs = [to_data_uri(r) for r in assets.fit_refs(req.refs, MAX_REQUEST)] if req.refs else []
        path, body = build_image(req, refs)
        req.params = {k: v for k, v in body.items() if k != "images"}
        req.params["references"] = len(refs)
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await send(client, "POST", f"{MediaConfig.VENICE_BASE}{path}", json=body, headers=self._headers())
        if resp.status_code >= 300:
            raise _fail(resp)
        if resp.headers.get("x-venice-is-blurred", "").lower() == "true":
            raise RuntimeError("Venice blurred the picture even with safe mode off")
        data = resp.content
        if not sniff_mime(data).startswith("image/"):
            raise RuntimeError(f"Venice returned no image: {resp.text[:200]}")
        task_id = uuid.uuid4().hex
        self._done[task_id] = RenderResult(data=data, mime=sniff_mime(data), seed=body.get("seed"), task_id=task_id)
        return task_id

    # ---- video ----

    async def _quote(self, client: httpx.AsyncClient, body: Dict[str, Any]) -> Optional[float]:
        fields = {k: body[k] for k in ("model", "duration", "resolution", "aspect_ratio", "audio") if k in body}
        try:
            resp = await client.post(f"{MediaConfig.VENICE_BASE}/video/quote", json=fields, headers=self._headers())
            return float(resp.json()["quote"]) if resp.status_code == 200 else None
        except (ValueError, KeyError, TypeError, httpx.HTTPError):
            return None

    async def _video(self, req: RenderRequest) -> str:
        images = assets.fit_refs([a for a in (req.first, req.last, *req.refs) if a], MAX_REQUEST)
        first = to_data_uri(images.pop(0)) if req.first else None
        last = to_data_uri(images.pop(0)) if req.last else None
        body = build_video(req, first, last, [to_data_uri(i) for i in images])
        hidden = ("image_url", "end_image_url", "reference_image_urls")
        req.params = {k: v for k, v in body.items() if k not in hidden}
        req.params["references"] = len(req.refs)
        async with httpx.AsyncClient(timeout=180) as client:
            req.cost_usd = await self._quote(client, body)
            resp = await send(client, "POST", f"{MediaConfig.VENICE_BASE}/video/queue", json=body, headers=self._headers())
        if resp.status_code >= 300:
            raise _fail(resp)
        queued = resp.json()
        queue_id = queued.get("queue_id")
        if not queue_id:
            raise RuntimeError(f"Venice returned no queue id: {resp.text[:200]}")
        return f"{queue_id}|{queued['download_url']}" if queued.get("download_url") else queue_id

    async def submit(self, req: RenderRequest) -> str:
        return await (self._image(req) if req.option.kind == "image" else self._video(req))

    async def poll(self, task_id: str, req: RenderRequest) -> Optional[RenderResult]:
        if req.option.kind == "image":
            result = self._done.pop(task_id, None)
            if not result:
                raise RuntimeError("Interrupted by a bridge restart")
            return result
        queue_id, download_url = split_task(task_id)
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
            resp = await send(client, "POST", f"{MediaConfig.VENICE_BASE}/video/retrieve",
                              json={"queue_id": queue_id, "delete_media_on_completion": False}, headers=self._headers())
            if resp.status_code == 404:
                raise RuntimeError("Venice: the clip expired or was deleted before it could be collected")
            if resp.status_code in (402, 422):
                raise _fail(resp)
            if resp.status_code >= 500 and resp.status_code != 503:
                raise _fail(resp)
            if resp.status_code >= 300:
                return None                 # a blip or capacity; the job's deadline ends the wait if it lasts
            if "json" in resp.headers.get("content-type", ""):
                if str((resp.json() or {}).get("status", "")).upper() != "COMPLETED" or not download_url:
                    return None
                clip_resp = await client.get(download_url, headers=self._headers() if "venice.ai" in download_url else {})
                if clip_resp.status_code != 200:
                    return None
                clip = clip_resp.content
            else:
                clip = resp.content
            if not sniff_mime(clip).startswith("video/"):
                raise RuntimeError("Venice finished without a video file")
            try:                            # the clip is ours now, so Venice can drop its copy
                await client.post(f"{MediaConfig.VENICE_BASE}/video/complete", json={"queue_id": queue_id}, headers=self._headers())
            except httpx.HTTPError:
                pass
        return RenderResult(data=clip, mime=sniff_mime(clip), task_id=task_id)

    async def balance(self) -> Optional[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{MediaConfig.VENICE_BASE}/api_keys/rate_limits", headers=self._headers())
        check_key(resp)
        if resp.status_code != 200:
            return None
        payload = resp.json() or {}
        usd = ((payload.get("data") or payload).get("balances") or {}).get("USD")
        return {"amount": float(usd), "unit": "USD"} if usd is not None else None
