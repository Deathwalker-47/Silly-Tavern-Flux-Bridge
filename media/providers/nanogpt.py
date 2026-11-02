"""NanoGPT: images on POST /v1/images, video on /generate-video plus /video/status.

Requests run on Vercel, which refuses anything over about 4.5 MB (413), so frames and
references are shrunk to fit first. NanoGPT ignores unknown fields, so each model sends
only the parameters in its catalog entry.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import httpx

from .. import assets
from ..config import MediaConfig
from .base import (Provider, RenderRequest, RenderResult, check_key, closest_ratio, download, error_text, image_size, send,
                   sniff_mime, to_data_uri)

MAX_REQUEST = 4_400_000
RATIOS = ["16:9", "9:16", "1:1", "3:4", "4:3", "21:9"]


class NanoGPT(Provider):
    name = "nanogpt"
    poll_seconds = 5.0

    def __init__(self) -> None:
        self._done: Dict[str, RenderResult] = {}

    @property
    def configured(self) -> bool:
        return bool(MediaConfig.NANOGPT_API_KEY)

    def _finished(self, task_id: str) -> RenderResult:
        """Images come back from submit, so a missing one was lost to a restart."""
        result = self._done.pop(task_id, None)
        if not result:
            raise RuntimeError("Interrupted by a bridge restart")
        return result

    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": MediaConfig.NANOGPT_API_KEY, "Content-Type": "application/json"}

    # ---- images ----

    @staticmethod
    def _refs_have_shape(refs: List[bytes], aspect: str, tolerance: float = 0.03) -> bool:
        """Seedream 5 Pro bills a named aspect ratio at the 2k price ($0.09, about 4 MP) and plain `1k`
        at half ($0.045, about 1 MP), where the picture takes its shape from the references. With
        mixed-shape references it picked 9:16 whichever came first, so the ratio is only left off
        when every reference already has the shape that was asked for, or when the user chose the
        1k price over the scene's shape. At 1k it ignores size, width/height and image_size."""
        if not refs:
            return False
        width, height = (float(n) for n in aspect.split(":"))
        wanted = width / height
        for data in refs:
            w, h = image_size(data)
            if abs(w / max(1, h) / wanted - 1) > tolerance:
                return False
        return True

    def build_image(self, req: RenderRequest, refs: List[bytes]) -> Dict[str, Any]:
        extra = req.option.extra
        body: Dict[str, Any] = {"model": req.option.id, "prompt": req.prompt, "n": 1}
        sizes = extra.get("sizes")
        if sizes:
            width, height = sizes[req.aspect_ratio]
            body["resolution"] = f"{width}x{height}"
        else:
            body["resolution"] = extra["resolution"]
        budget = req.budget and extra.get("cheaper_without_aspect") and refs
        if extra.get("send_aspect") and not budget and not self._refs_have_shape(refs, req.aspect_ratio):
            body["aspect_ratio"] = req.aspect_ratio
        if refs:
            body["input_references"] = [to_data_uri(r) for r in refs]
        return body

    async def _image(self, req: RenderRequest) -> str:
        if req.option.extra.get("needs_ref") and not req.refs:
            raise RuntimeError(f"{req.option.label} needs at least one reference image")
        body = self.build_image(req, assets.fit_refs(req.refs, MAX_REQUEST) if req.refs else [])
        req.params = {k: v for k, v in body.items() if k != "input_references"}
        req.params["references"] = len(req.refs)
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await send(client, "POST", f"{MediaConfig.NANOGPT_BASE}/v1/images", json=body, headers=self._headers())
        if resp.status_code >= 300:
            raise RuntimeError(f"NanoGPT {resp.status_code}: {error_text(resp)}")
        payload = resp.json()
        item = (payload.get("data") or [{}])[0]
        if item.get("url"):
            data = await download(item["url"])
        elif item.get("b64_json"):
            import base64
            data = base64.b64decode(item["b64_json"])
        else:
            raise RuntimeError("NanoGPT returned no image")
        task_id = uuid.uuid4().hex
        self._done[task_id] = RenderResult(data=data, mime=sniff_mime(data), cost_usd=payload.get("cost"),
                                           seed=item.get("seed"), task_id=task_id)
        return task_id

    # ---- video ----

    def build_video(self, req: RenderRequest, first: Optional[bytes], last: Optional[bytes], refs: List[bytes]) -> Dict[str, Any]:
        option = req.option
        allowed = set(option.extra.get("params") or ())
        body: Dict[str, Any] = {"model": option.id, "prompt": req.prompt}
        if first:
            body["imageDataUrl"] = to_data_uri(first)
        if refs:
            body["reference_images"] = [to_data_uri(r) for r in refs]
        if "duration" in allowed:
            body["duration"] = str(req.seconds)
        if "resolution" in allowed:
            body["resolution"] = req.resolution
        if "aspect_ratio" in allowed:
            if first:
                width, height = image_size(first)
                body["aspect_ratio"] = closest_ratio(width, height, RATIOS)
            else:
                body["aspect_ratio"] = req.aspect_ratio
        if "last_image" in allowed and last:
            body["last_image"] = to_data_uri(last)
        if "seed" in allowed and req.seed is not None:
            body["seed"] = int(req.seed)
        if "generateAudio" in allowed:
            body["generateAudio"] = bool(req.audio)
        return body

    async def _video(self, req: RenderRequest) -> str:
        images = assets.fit_refs([a for a in (req.first, req.last, *req.refs) if a], MAX_REQUEST)
        first = images.pop(0) if req.first else None
        last = images.pop(0) if req.last else None
        body = self.build_video(req, first, last, images)
        hidden = ("imageDataUrl", "last_image", "reference_images")
        req.params = {k: v for k, v in body.items() if k not in hidden}
        req.params["references"] = len(req.refs)
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await send(client, "POST", f"{MediaConfig.NANOGPT_BASE}/generate-video", json=body, headers=self._headers())
        if resp.status_code >= 300:
            raise RuntimeError(f"NanoGPT {resp.status_code}: {error_text(resp)}")
        payload = resp.json()
        run_id = payload.get("runId") or payload.get("id")
        if not run_id:
            raise RuntimeError(f"NanoGPT returned no runId: {resp.text[:200]}")
        req.cost_usd = payload.get("cost")
        return run_id

    async def submit(self, req: RenderRequest) -> str:
        return await (self._image(req) if req.option.kind == "image" else self._video(req))

    async def poll(self, task_id: str, req: RenderRequest) -> Optional[RenderResult]:
        if req.option.kind == "image":
            return self._finished(task_id)
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await send(client, "GET", f"{MediaConfig.NANOGPT_BASE}/video/status", params={"requestId": task_id},
                              headers=self._headers())
        if resp.status_code >= 300:
            return None
        data = (resp.json() or {}).get("data") or {}
        status = str(data.get("status") or "").upper()
        if status == "COMPLETED":
            url = ((data.get("output") or {}).get("video") or {}).get("url")
            if not url:
                raise RuntimeError("NanoGPT finished without a video URL")
            clip = await download(url)
            return RenderResult(data=clip, mime=sniff_mime(clip), seed=data.get("seed"), task_id=task_id)
        if status == "FAILED":
            raise RuntimeError(f"NanoGPT render failed: {data.get('error') or 'unknown error'}")
        return None

    async def balance(self) -> Optional[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(f"{MediaConfig.NANOGPT_BASE}/check-balance", headers=self._headers())
        check_key(resp)
        if resp.status_code != 200:
            return None
        payload = resp.json() or {}
        amount = payload.get("usd_balance", payload.get("balance", payload.get("usdBalance")))
        return {"amount": float(amount), "unit": "USD"} if amount is not None else None
