"""Replicate: Flux Dev with LoRAs by link, filter off. No balance API.

A private Hugging Face file goes in as a short-lived link on the bridge (see lora_files)."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import httpx

from .. import lora_files
from ..config import MediaConfig
from .base import Provider, RenderRequest, RenderResult, download, error_text, send, sniff_mime

API = "https://api.replicate.com/v1"

_versions: Dict[str, str] = {}


async def replicate_create(model: str, inp: Dict[str, Any], token: str, prefer_wait: Optional[int] = None) -> Dict:
    """Start a Replicate prediction.

    Official models run via /models/{owner}/{name}/predictions; community models
    (e.g. lucataco/flux-dev-multi-lora) 404 there and must be run by version id, so
    on 404 the latest version is looked up once and cached. "owner/name:version"
    pins a version explicitly.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if prefer_wait:
        headers["Prefer"] = f"wait={prefer_wait}"
    async with httpx.AsyncClient(timeout=120) as client:
        if ":" in model:
            version = model.split(":", 1)[1]
            resp = await send(client, "POST", f"{API}/predictions", json={"version": version, "input": inp}, headers=headers)
        else:
            resp = await send(client, "POST", f"{API}/models/{model}/predictions", json={"input": inp}, headers=headers)
            if resp.status_code == 404:
                version = _versions.get(model)
                if not version:
                    info = await client.get(f"{API}/models/{model}", headers=headers)
                    if info.status_code >= 300:
                        raise RuntimeError(f"Replicate model {model} not found ({info.status_code})")
                    version = ((info.json() or {}).get("latest_version") or {}).get("id")
                    if not version:
                        raise RuntimeError(f"Replicate model {model} has no published version")
                    _versions[model] = version
                resp = await send(client, "POST", f"{API}/predictions", json={"version": version, "input": inp}, headers=headers)
    if resp.status_code >= 300:
        raise RuntimeError(f"Replicate {resp.status_code}: {error_text(resp)}")
    return resp.json()


async def replicate_wait(prediction: Dict, headers: Dict[str, str], timeout: Optional[float] = None) -> str:
    """Poll a Replicate prediction and return its first output URL."""
    get_url = (prediction.get("urls") or {}).get("get") or f"{API}/predictions/{prediction.get('id')}"
    deadline = time.time() + (timeout or MediaConfig.VIDEO_JOB_TIMEOUT)
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            output = prediction_output(prediction)
            if output:
                return output
            if time.time() > deadline:
                raise TimeoutError("Replicate prediction timed out")
            await asyncio.sleep(MediaConfig.VIDEO_POLL_SECONDS)
            resp = await client.get(get_url, headers=headers)
            if resp.status_code < 300:
                prediction = resp.json()


def prediction_output(prediction: Dict) -> Optional[str]:
    """The output URL once finished, None while running, and an error when it failed."""
    status = prediction.get("status")
    if status == "succeeded":
        output = prediction.get("output")
        if isinstance(output, list):
            output = output[0] if output else None
        if not output:
            raise RuntimeError("Replicate finished without output")
        return output
    if status in ("failed", "canceled", "aborted"):
        raise RuntimeError(f"Replicate {status}: {prediction.get('error') or 'no details'}")
    return None


class Replicate(Provider):
    name = "replicate"

    @property
    def configured(self) -> bool:
        return bool(MediaConfig.REPLICATE_API_TOKEN)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {MediaConfig.REPLICATE_API_TOKEN}"}

    def build_input(self, req: RenderRequest) -> Dict[str, Any]:
        extra = req.option.extra
        inp: Dict[str, Any] = {
            "prompt": req.prompt,
            "aspect_ratio": req.aspect_ratio,
            "num_outputs": 1,
            "num_inference_steps": extra["steps"],
            "guidance_scale": extra["cfg"],
            "output_format": "jpg",
            "output_quality": 95,
            "disable_safety_checker": True,
        }
        if req.loras:
            inp["hf_loras"] = [l["url"] for l in req.loras]
            inp["lora_scales"] = [float(l["weight"]) for l in req.loras]
        if req.seed is not None:
            inp["seed"] = req.seed
        return inp

    async def submit(self, req: RenderRequest) -> str:
        inp = self.build_input(req)
        req.params = dict(inp)                 # the history keeps the Hugging Face links
        if "hf_loras" in inp:
            # The model has no token input and skips a LoRA it can't download without a word.
            inp["hf_loras"] = [lora_files.provider_url(url) for url in inp["hf_loras"]]
        prediction = await replicate_create(MediaConfig.REPLICATE_FLUX_MODEL, inp, MediaConfig.REPLICATE_API_TOKEN)
        return prediction["id"]

    async def poll(self, task_id: str, req: RenderRequest) -> Optional[RenderResult]:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await send(client, "GET", f"{API}/predictions/{task_id}", headers=self._headers())
        if resp.status_code >= 300:
            return None
        url = prediction_output(resp.json())
        if not url:
            return None
        data = await download(url)
        return RenderResult(data=data, mime=sniff_mime(data), task_id=task_id)

    async def cancel(self, task_id: str) -> None:
        async with httpx.AsyncClient(timeout=20) as client:
            await client.post(f"{API}/predictions/{task_id}/cancel", headers=self._headers())
