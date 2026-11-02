"""Sogni Supernet through the official SDK: the REST workflow API can't address exact
model ids such as the 10Eros fine-tune, the SDK can.

Jobs run over a socket, so they can't continue after a bridge restart. They fail as
interrupted, and the bridge cancels them on Sogni so they stop billing.

References travel three ways. Frames are bytes in referenceImage and referenceImageEnd.
MiniMax H3 references are bytes in referenceImage plus contextImages. Seedance and Wan 3
references are HTTPS links in referenceImageUrls, from Sogni's own upload endpoints, so
no reference ever needs a public URL on the bridge.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .. import db
from ..config import MediaConfig
from .base import Provider, RenderRequest, RenderResult, WH_SIZES, download, image_size, orientation, sniff_mime, wh_for

logger = logging.getLogger("media.sogni")

CONSENT_PAGE = "https://app.sogni.ai/account/model-consents/seedance-2-5"
LINK_LIFETIME = 50 * 60         # seconds a signed reference link is reused


def consent_message(error: Exception) -> Optional[str]:
    """Sogni's 4103 refusal, explained. It comes before any charge and a retry can't help."""
    try:
        from sogni_client.errors import is_model_consent_required_error
    except ImportError:
        return None
    if is_model_consent_required_error(error):
        return f"Sogni needs this model's one-time agreement. Accept it at {CONSENT_PAGE}, then try again."
    return None


class Sogni(Provider):
    name = "sogni"
    resumable = False
    poll_seconds = 0.2
    NETWORKS = ("fast", "relaxed")

    def __init__(self) -> None:
        self._live: Dict[str, Tuple[Any, Any]] = {}

    @property
    def configured(self) -> bool:
        if not MediaConfig.SOGNI_API_KEY:
            return False
        try:
            import sogni_client  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def network(cls, req: RenderRequest) -> str:
        """Fast = dedicated GPUs, billed; relaxed = free queue, can wait an hour or more."""
        for choice in (req.network, MediaConfig.SOGNI_NETWORK):
            if (choice or "").lower() in cls.NETWORKS:
                return choice.lower()
        return "fast"

    def _timeout(self, req: RenderRequest) -> int:
        if self.network(req) == "relaxed":
            return MediaConfig.SOGNI_RELAXED_TIMEOUT
        return MediaConfig.VIDEO_JOB_TIMEOUT if req.option.kind == "video" else MediaConfig.IMAGE_JOB_TIMEOUT

    async def _client(self, network: str, **kwargs: Any) -> Any:
        from sogni_client import SogniClient  # imported lazily: optional dependency

        # A second socket on one app id silently drops the first, and that job then hangs.
        return await SogniClient.create(api_key=MediaConfig.SOGNI_API_KEY, app_id=f"{MediaConfig.SOGNI_APP_ID}-{uuid.uuid4().hex[:8]}",
                                        network=network, **kwargs)

    async def _reference_link(self, client: Any, asset: Any) -> str:
        """An HTTPS link to this file on Sogni's storage, uploaded once."""
        handle = db.remote_handle(asset.id, self.name)
        if handle:
            return handle
        ids = {"imageId": uuid.uuid4().hex, "jobId": uuid.uuid4().hex, "type": "referenceImage"}
        upload = await client.projects.upload_url({**ids, "contentType": "image/jpeg"})
        await client.api_client.rest.put_bytes(upload, asset.read(), content_type="image/jpeg")
        link = await client.projects.download_url(ids)
        db.save_remote(asset.id, self.name, link, int(time.time()) + LINK_LIFETIME)
        return link

    def _h3_loras(self, req: RenderRequest) -> Dict[str, List[Any]]:
        names, strengths = [], []
        if MediaConfig.SOGNI_H3_LORA:
            names.append(MediaConfig.SOGNI_H3_LORA)
            strengths.append(MediaConfig.SOGNI_H3_LORA_STRENGTH)
        if req.adult and MediaConfig.SOGNI_H3_ADULT_LORA:
            names.append(MediaConfig.SOGNI_H3_ADULT_LORA)
            strengths.append(MediaConfig.SOGNI_H3_ADULT_LORA_STRENGTH)
        return {"loras": names, "loraStrengths": strengths} if names else {}

    def _size(self, req: RenderRequest, first: Optional[bytes]) -> Dict[str, int]:
        extra = req.option.extra
        sizes = extra.get("sizes")
        if sizes:
            width, height = sizes[req.aspect_ratio if req.aspect_ratio in sizes else next(iter(sizes))]
            return {"width": width, "height": height}
        tall = (extra.get("tall") or {}).get(req.resolution)
        shape = image_size(first) if first else None
        landscape = orientation(*shape) == "landscape" if shape else req.aspect_ratio in ("16:9", "4:3")
        if tall:
            width, height = tall
            return {"width": height, "height": width} if landscape else {"width": width, "height": height}
        if extra.get("sizing") == "wh" and req.resolution in WH_SIZES:
            width, height = wh_for(req.resolution, *(shape or (9, 16)))
            return {"width": width, "height": height}
        return {}

    async def build_video(self, req: RenderRequest, client: Any) -> Dict[str, Any]:
        option = req.option
        extra = option.extra
        params: Dict[str, Any] = {
            "type": "video",
            "network": self.network(req),
            "modelId": option.id,
            "positivePrompt": req.prompt,
            "numberOfMedia": 1,
            "duration": req.seconds,
            "disableNSFWFilter": True,
        }
        first = req.first.read() if req.first else None
        if req.mode == "reference":
            if extra.get("url_refs"):
                params["referenceImageUrls"] = [await self._reference_link(client, a) for a in req.refs]
                if extra.get("seedance_task"):
                    params["seedanceTaskType"] = "reference"
            else:
                images = [a.read() for a in req.refs]
                params["referenceImage"] = images[0]
                if images[1:]:
                    params["contextImages"] = images[1:]
        elif first:
            params["referenceImage"] = first
        if req.last and req.mode == "first_last":
            params["referenceImageEnd"] = req.last.read()
            if extra.get("last_model"):
                params["modelId"] = extra["last_model"]
        params.update(self._size(req, first))
        if extra.get("sends_ratio") and req.mode == "reference":
            params["ratio"] = req.aspect_ratio
        if extra.get("fps"):
            params["fps"] = extra["fps"]
        if req.seed is not None:
            params["seed"] = req.seed
        if extra.get("h3_lora"):
            params.update(self._h3_loras(req))
        return params

    def build_image(self, req: RenderRequest) -> Dict[str, Any]:
        option = req.option
        width, height = option.extra["sizes"][req.aspect_ratio]
        params: Dict[str, Any] = {
            "type": "image",
            "network": self.network(req),
            "modelId": option.id,
            "positivePrompt": req.prompt,
            "numberOfMedia": 1,
            "sizePreset": "custom",
            "width": width,
            "height": height,
            "disableNSFWFilter": True,
        }
        if req.seed is not None:
            params["seed"] = req.seed
        if req.refs:
            params["contextImages"] = [a.read() for a in req.refs]
        return params

    async def submit(self, req: RenderRequest) -> str:
        if req.option.extra.get("needs_ref") and not req.refs:
            raise RuntimeError(f"{req.option.label} needs at least one reference image")
        client = await self._client(self.network(req))
        await client.__aenter__()
        try:
            params = await self.build_video(req, client) if req.option.kind == "video" else self.build_image(req)
            req.params = {k: (f"<{len(v)} images>" if isinstance(v, list) and k in ("contextImages", "referenceImageUrls")
                              else "<image>" if isinstance(v, (bytes, bytearray)) else v) for k, v in params.items()}
            project = await client.projects.create(params)
        except Exception as e:
            await client.__aexit__(None, None, None)
            message = consent_message(e)
            if message:
                raise RuntimeError(message) from e
            raise
        self._live[project.id] = (client, project)
        return project.id

    async def poll(self, task_id: str, req: RenderRequest) -> Optional[RenderResult]:
        """Waits on the socket until the render ends, so this runs once per job."""
        live = self._live.pop(task_id, None)
        if not live:
            raise RuntimeError("Lost the connection to this Sogni job")
        client, project = live
        try:
            urls = await project.wait_for_completion(timeout=self._timeout(req))
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # Giving up doesn't stop the render on the Supernet: cancel it so a fast job
            # isn't billed and a relaxed one doesn't keep its queue slot.
            try:
                await asyncio.wait_for(project.cancel(), timeout=15)
            except Exception as e:  # best effort
                logger.warning(f"🎬 [Sogni] could not cancel project {task_id}: {e}")
            raise
        except Exception as e:
            message = consent_message(e)
            if message:
                raise RuntimeError(message) from e
            raise
        finally:
            await client.__aexit__(None, None, None)
        if not urls:
            raise RuntimeError("Sogni finished without a result")
        data = await download(urls[0])
        return RenderResult(data=data, mime=sniff_mime(data), seed=getattr(project, "seed", None) or req.seed, task_id=task_id)

    async def cancel(self, task_id: str) -> None:
        """Stops a render this process started, or one left behind by a restart."""
        live = self._live.get(task_id)
        if live:
            await asyncio.wait_for(live[1].cancel(), timeout=15)
            return
        client = await self._client("fast")
        async with client:
            await asyncio.wait_for(client.projects.cancel(task_id), timeout=15)

    async def balance(self) -> Optional[Dict[str, Any]]:
        client = await self._client("fast", disable_socket=True)
        async with client:
            data = await client.account.account_balance()
        spark = (data or {}).get("spark") or {}
        amount = spark.get("net", spark.get("settled")) if isinstance(spark, dict) else spark
        return {"amount": float(amount), "unit": "Spark", "partial": True} if amount is not None else None
