"""PixelDojo: high-quality images and video, plus MiniMax H3 and H3 Turbo.

An async job API: POST /models/{id}/run, then GET /jobs/{id}. Outputs last one hour, so
they're downloaded at once. Only fields documented for that model are sent, because
PixelDojo ignores unknown fields with no error and still bills the job. There is no
Relax option in the API (a job sent with every likely field was billed normally).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from .. import assets, db
from ..config import MediaConfig
from .base import Provider, RenderRequest, RenderResult, download, error_text, send, sniff_mime

URL_LIFETIME = 23 * 3600        # uploads last 24 h; stop reusing a link before it dies


def _portrait_or_landscape(aspect: str) -> str:
    return {"9:16": "portrait_9_16", "3:4": "portrait_3_4", "16:9": "landscape_16_9", "4:3": "landscape_4_3"}[aspect]


def _seed(body: Dict[str, Any], req: RenderRequest) -> None:
    if req.seed is not None:
        body["seed"] = req.seed


def seedream_4(req: RenderRequest, refs: List[str]) -> Dict[str, Any]:
    body: Dict[str, Any] = {"image_size": _portrait_or_landscape(req.aspect_ratio)}
    _seed(body, req)
    if refs:
        body["image_urls"] = refs
    return body


def seedream_5(req: RenderRequest, refs: List[str]) -> Dict[str, Any]:
    # auto_1K picks its own shape and is square from a text prompt, so a set aspect
    # ratio needs custom dimensions. 768x1344 stays in the 1K price class.
    width, height = req.option.extra["sizes"][req.aspect_ratio]
    body: Dict[str, Any] = {"image_size": "custom", "customWidth": width, "customHeight": height}
    _seed(body, req)
    if refs:
        body["image_urls"] = refs
    return body


def _frames(body: Dict[str, Any], first_key: str, last_key: str, first: Optional[str], last: Optional[str]) -> None:
    if first:
        body[first_key] = first
    if last:
        body[last_key] = last


def wan_3(req: RenderRequest, refs: List[str], first: Optional[str] = None, last: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"resolution": req.resolution, "aspect_ratio": req.aspect_ratio, "duration": req.seconds,
                            "audio": req.audio, "prompt_extend": False}
    if refs:
        body["reference_image_urls"] = refs
    _frames(body, "first_frame_url", "last_frame_url", first, last)
    return body                  # its seed is accepted but ignored, so none is sent or stored


def seedance_25(req: RenderRequest, refs: List[str], first: Optional[str] = None, last: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"resolution": req.resolution, "duration": req.seconds, "generate_audio": req.audio}
    if refs:
        body["reference_images"] = refs
    if not first:
        body["aspect_ratio"] = req.aspect_ratio
    _frames(body, "first_frame_url", "last_frame_url", first, last)
    return body


def seedance_2_reference(req: RenderRequest, refs: List[str], first: Optional[str] = None, last: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"quality": "fast", "resolution": req.resolution, "aspect_ratio": req.aspect_ratio,
                            "duration": req.seconds, "generate_audio": req.audio, "reference_images": refs}
    _seed(body, req)
    return body


def h3_turbo(req: RenderRequest, refs: List[str], first: Optional[str] = None, last: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"resolution": req.resolution, "duration": req.seconds}
    _seed(body, req)
    if refs:
        body["reference_images"] = refs
    if first:
        _frames(body, "image_url", "end_image_url", first, last)
    else:
        body["aspect_ratio"] = req.aspect_ratio     # a start photo sets the shape on its own
    return body


def h3(req: RenderRequest, refs: List[str], first: Optional[str] = None, last: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"duration": req.seconds}
    _seed(body, req)
    if refs:
        body["reference_images"] = refs
    if first:                                       # first-frame jobs ignore size and bill as 2K
        _frames(body, "image_url", "end_image_url", first, last)
    else:
        body.update(resolution=req.resolution, aspect_ratio=req.aspect_ratio)
    return body


BUILDERS = {
    "seedream_4": seedream_4, "seedream_5": seedream_5, "wan_3": wan_3, "seedance_25": seedance_25,
    "seedance_2_reference": seedance_2_reference, "h3_turbo": h3_turbo, "h3": h3,
}


def first_output_url(job: Dict[str, Any]) -> str:
    output = job.get("output") or {}
    url = output.get("video") or next(iter(output.get("images") or []), None)
    if not url:
        url = next((a.get("url") for a in job.get("assets") or [] if a.get("url")), None)
    if not url:
        raise RuntimeError("PixelDojo finished without a file")
    return url


class PixelDojo(Provider):
    name = "pixeldojo"
    poll_seconds = 4.0

    @property
    def configured(self) -> bool:
        return bool(MediaConfig.PIXELDOJO_API_KEY)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {MediaConfig.PIXELDOJO_API_KEY}"}

    async def ensure_asset(self, asset: Any) -> str:
        handle = db.remote_handle(asset.id, self.name)
        if handle:
            return handle
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await send(client, "POST", f"{MediaConfig.PIXELDOJO_BASE}/upload", headers=self._headers(),
                              files={"file": (f"{asset.id}.jpg", asset.read(), "image/jpeg")})
        if resp.status_code >= 300:
            raise RuntimeError(f"PixelDojo upload {resp.status_code}: {error_text(resp)}")
        url = resp.json()["url"]
        db.save_remote(asset.id, self.name, url, int(time.time()) + URL_LIFETIME)
        return url

    async def submit(self, req: RenderRequest) -> str:
        build = BUILDERS[req.option.extra["build"]]
        refs = [await self.ensure_asset(a) for a in req.refs]
        if req.option.kind == "image":
            body = build(req, refs)
        else:
            first = await self.ensure_asset(req.first) if req.first else None
            last = await self.ensure_asset(req.last) if req.last else None
            body = build(req, refs, first, last)
        body["prompt"] = req.prompt
        req.params = dict(body)
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await send(client, "POST", f"{MediaConfig.PIXELDOJO_BASE}/models/{req.option.id}/run",
                              json=body, headers=self._headers())
        if resp.status_code == 402:
            raise RuntimeError("PixelDojo: out of credits")
        if resp.status_code >= 300:
            raise RuntimeError(f"PixelDojo {resp.status_code}: {error_text(resp)}")
        job = resp.json()
        req.cost_credits = job.get("creditCost")
        return job["jobId"]

    async def poll(self, task_id: str, req: RenderRequest) -> Optional[RenderResult]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await send(client, "GET", f"{MediaConfig.PIXELDOJO_BASE}/jobs/{task_id}", headers=self._headers())
        if resp.status_code in (404, 410):
            raise RuntimeError("PixelDojo: the job expired before it could be collected")
        if resp.status_code >= 300:
            return None                     # a blip; the job's deadline ends the wait if it lasts
        job = resp.json()
        if job["status"] == "failed":
            refunded = " (credits refunded)" if job.get("refunded") else ""
            raise RuntimeError(f"PixelDojo: {job.get('error') or 'failed'}{refunded}")
        if job["status"] != "completed":
            return None
        data = await download(first_output_url(job))
        return RenderResult(data=data, mime=sniff_mime(data), seed=job.get("seed"),
                            cost_credits=job.get("creditCost"), task_id=task_id)

    async def balance(self) -> Optional[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{MediaConfig.PIXELDOJO_BASE}/credits", headers=self._headers())
        return {"amount": resp.json().get("credits"), "unit": "credits"} if resp.status_code == 200 else None
