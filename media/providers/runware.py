"""Runware: Flux Dev and Juggernaut Z images with LoRAs. Direct: the result comes back at once."""

from __future__ import annotations

import asyncio
import base64
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .. import loras, telemetry
from ..config import MediaConfig
from .base import Provider, RenderRequest, RenderResult, check_key, download, send, sniff_mime


LORA_READY_WAIT = 300       # seconds: a LoRA uploaded moments ago is still processing
LORA_READY_POLL = 10


def describe_errors(errors: List[Dict[str, Any]]) -> str:
    first = errors[0] if errors else {}
    if first.get("code") == "unsupportedLoraModel":
        return (f"Runware refused a LoRA built for {first.get('loraModelArchitecture', 'another model')} "
                f"on a {first.get('baseModelArchitecture', 'different')} model. {first.get('message', '')}")
    return first.get("message") or str(first)[:300]


def lora_processing(errors: Optional[List[Dict[str, Any]]]) -> bool:
    return bool(errors) and "not ready for inference" in str(errors[0].get("message"))


class Runware(Provider):
    name = "runware"
    resumable = False

    def __init__(self) -> None:
        self._done: Dict[str, RenderResult] = {}

    @property
    def configured(self) -> bool:
        return bool(MediaConfig.RUNWARE_API_KEY)

    def build_task(self, req: RenderRequest, resolved: List[Dict[str, Any]], task_uuid: str) -> Dict[str, Any]:
        extra = req.option.extra
        width, height = extra["sizes"][req.aspect_ratio]
        task: Dict[str, Any] = {
            "taskType": "imageInference",
            "taskUUID": task_uuid,
            "positivePrompt": req.prompt,
            "model": req.option.id,
            "steps": extra["steps"],
            "width": width,
            "height": height,
            "numberResults": 1,
            "outputType": "URL",
            "outputFormat": "JPG",
            "includeCost": True,
            "lora": [{"model": item["lora"], "weight": item["weight"]} for item in resolved],
        }
        if extra.get("cfg"):
            task["CFGScale"] = extra["cfg"]
        if len(req.negative_prompt) >= 2:
            task["negativePrompt"] = req.negative_prompt
        if req.seed is not None:
            task["seed"] = req.seed
        return task

    async def render(self, task: Dict[str, Any], headers: Dict[str, str]) -> Tuple[httpx.Response, Dict[str, Any]]:
        """The first render after an upload is refused until Runware finishes processing the LoRA."""
        waited = 0
        async with httpx.AsyncClient(timeout=180) as client:
            while True:
                resp = await send(client, "POST", MediaConfig.RUNWARE_ENDPOINT, json=[task], headers=headers)
                try:
                    payload = resp.json()
                except ValueError:
                    raise RuntimeError(f"Runware HTTP {resp.status_code}: {resp.text[:200]}")
                if not lora_processing(payload.get("errors")) or waited >= LORA_READY_WAIT:
                    if waited:
                        telemetry.record("loras", "waited_ready", ok=not payload.get("errors"), provider=self.name, seconds=waited)
                    return resp, payload
                await asyncio.sleep(LORA_READY_POLL)
                waited += LORA_READY_POLL

    async def submit(self, req: RenderRequest) -> str:
        failures: List[Tuple[str, str]] = []
        with telemetry.step("loras", "resolve", provider=self.name, count=len(req.loras)) as facts:
            # In a thread: the model check is a blocking HTTP call, and a new LoRA's upload
            # waits minutes for Runware to finish processing it.
            resolved = await asyncio.to_thread(
                loras.resolve_runware, [{"lora": l["url"], "weight": l["weight"]} for l in req.loras],
                MediaConfig.RUNWARE_API_KEY, req.option.lora_arch, failures)
            facts["resolved"] = len(resolved)
            if failures:
                facts["failed"] = [reason[:300] for _src, reason in failures]
        if len(resolved) < len(req.loras):
            names = {l["url"]: l.get("id") or l["url"].rsplit("/", 1)[-1] for l in req.loras}
            why = "; ".join(f"{names.get(src, src)}: {reason}" for src, reason in failures)
            raise RuntimeError(f"Runware couldn't use {len(req.loras) - len(resolved)} of {len(req.loras)} LoRAs. {why}".strip())
        task_uuid = str(uuid.uuid4())
        task = self.build_task(req, resolved, task_uuid)
        req.params = {k: v for k, v in task.items() if k not in ("taskUUID", "taskType")}
        headers = {"Authorization": f"Bearer {MediaConfig.RUNWARE_API_KEY}", "Content-Type": "application/json"}
        resp, payload = await self.render(task, headers)
        if payload.get("errors"):           # no retry: a refused LoRA would only fail the same way
            raise RuntimeError(f"Runware: {describe_errors(payload['errors'])}")
        item = (payload.get("data") or [{}])[0]
        if item.get("imageURL"):
            data = await download(item["imageURL"])
        elif item.get("imageBase64Data"):
            data = base64.b64decode(item["imageBase64Data"])
        else:
            raise RuntimeError(f"Runware returned no image (HTTP {resp.status_code})")
        self._done[task_uuid] = RenderResult(data=data, mime=sniff_mime(data), seed=item.get("seed", req.seed),
                                             cost_usd=item.get("cost"), task_id=task_uuid)
        return task_uuid

    async def poll(self, task_id: str, req: RenderRequest) -> Optional[RenderResult]:
        return self._done.pop(task_id, None)

    async def balance(self) -> Optional[Dict[str, Any]]:
        task = {"taskType": "accountManagement", "taskUUID": str(uuid.uuid4()), "operation": "getDetails"}
        headers = {"Authorization": f"Bearer {MediaConfig.RUNWARE_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(MediaConfig.RUNWARE_ENDPOINT, json=[task], headers=headers)
        check_key(resp)
        if resp.status_code != 200:
            return None
        item = (resp.json().get("data") or [{}])[0]
        amount = item.get("balance", (item.get("organization") or {}).get("balance"))
        return {"amount": float(amount), "unit": "USD"} if amount is not None else None
