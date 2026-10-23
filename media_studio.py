"""
Media Studio: video generation and the bridge's helper LLM.

flux_lora_bridge.py owns the HTTP routes and the image pipeline; this module holds the
video and helper logic so it can be tested without FastAPI.

Video providers, all behind one async job API:
  runware    Runware videoInference (async + getResponse polling)
  nanogpt    NanoGPT /generate-video (+ /video/status polling)
  sogni      Sogni Supernet via the official sogni-client SDK (exact model ids)
  replicate  Replicate predictions (+ polling)
  mock       A local test clip; only when VIDEO_MOCK_PROVIDER=true

Every request turns the provider's own safety checker off where the API has that
switch. Closed vendor models (Seedance, MiniMax via partners) can still filter
upstream; no request flag reaches that.

Helper LLM (OpenAI-compatible; NanoGPT by default):
  decide()      should the character send a photo or a video right now, and the prompts
  voice_tags()  add Grok speech tags to a line for TavernVoice, without changing words
  models()      model list for the settings UI
"""

from __future__ import annotations  # keeps type hints lazy (the test suite stubs httpx)

import asyncio
import base64
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("media_studio")


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "true" if default else "false").strip().lower() in ("1", "true", "yes", "on")


class MediaConfig:
    """Environment settings for video providers and the helper LLM."""

    RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY", "")
    RUNWARE_ENDPOINT = os.getenv("RUNWARE_ENDPOINT", "https://api.runware.ai/v1")

    NANOGPT_API_KEY = os.getenv("NANOGPT_API_KEY", "")
    NANOGPT_BASE = os.getenv("NANOGPT_BASE", "https://nano-gpt.com/api").rstrip("/")

    SOGNI_API_KEY = os.getenv("SOGNI_API_KEY", "")
    # Sogni wants one stable app id per installation, never a fresh one per run.
    SOGNI_APP_ID = os.getenv("SOGNI_APP_ID", "") or str(uuid.uuid5(uuid.NAMESPACE_DNS, f"flux-lora-bridge.{socket.gethostname()}"))
    SOGNI_NETWORK = os.getenv("SOGNI_NETWORK", "fast")
    # Default LoRA for every MiniMax H3 job on Sogni: VBVR Video Reasoning (prompt adherence).
    SOGNI_H3_LORA = os.getenv("SOGNI_H3_LORA", "h3-vbvr-video-reasoning")
    SOGNI_H3_LORA_STRENGTH = float(os.getenv("SOGNI_H3_LORA_STRENGTH", 0.8))

    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

    VIDEO_MOCK_PROVIDER = _env_bool("VIDEO_MOCK_PROVIDER", False)
    VIDEO_JOB_TIMEOUT = int(os.getenv("VIDEO_JOB_TIMEOUT", 1200))   # seconds, keyframe + render + download
    VIDEO_POLL_SECONDS = float(os.getenv("VIDEO_POLL_SECONDS", 3))
    VIDEO_STORE_TTL = int(os.getenv("VIDEO_STORE_TTL", 3600))       # seconds a finished clip stays fetchable
    VIDEO_STORE_MAX = int(os.getenv("VIDEO_STORE_MAX", 40))

    # Helper LLM. Any OpenAI-compatible /chat/completions endpoint works; NanoGPT is the default.
    HELPER_API_URL = os.getenv("HELPER_API_URL", "https://nano-gpt.com/api/v1/chat/completions")
    HELPER_API_KEY = os.getenv("HELPER_API_KEY", "") or NANOGPT_API_KEY
    HELPER_MODEL = os.getenv("HELPER_MODEL", "deepseek/deepseek-v4.1-flash")
    # Prompt summarization for txt2img; empty = HELPER_MODEL.
    SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "")
    # Legacy fallback when no helper key is set (Together is otherwise deprecated).
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
    TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
    TOGETHER_MODEL = "deepseek-ai/DeepSeek-V3"


# ============================================================================
# VIDEO MODEL CATALOG
# ============================================================================

VIDEO_FAMILIES: Dict[str, str] = {
    "ltx-2.3": "LTX-2.3",
    "wan-2.2": "Wan 2.2",
    "wan-2.7": "Wan 2.7",
    "seedance-1.5": "Seedance 1.5",
    "seedance-2.0": "Seedance 2.0",
    "minimax-h3": "MiniMax H3",
}

PROVIDER_LABELS = {
    "runware": "Runware",
    "nanogpt": "NanoGPT",
    "sogni": "Sogni",
    "replicate": "Replicate",
    "mock": "Mock (local test clip)",
}


@dataclass
class VideoVariant:
    id: str
    label: str
    resolutions: List[str]
    default_resolution: str
    duration_range: Optional[Tuple[float, float]] = None   # continuous range, seconds
    duration_options: Optional[List[int]] = None           # or a fixed set of lengths
    last_frame: bool = False                               # accepts a last-frame image too
    hint: str = ""                                         # price / speed note for the UI
    extra: Dict[str, Any] = field(default_factory=dict)    # provider-specific details

    def to_public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "resolutions": self.resolutions,
            "default_resolution": self.default_resolution,
            "durations": self.duration_options or list(self.duration_range or (5, 5)),
            "duration_mode": "options" if self.duration_options else "range",
            "last_frame": self.last_frame,
            "hint": self.hint,
        }

    def pick_duration(self, requested: Optional[float]) -> int:
        want = float(requested or 5)
        if self.duration_options:
            return min(self.duration_options, key=lambda d: (abs(d - want), d))
        lo, hi = self.duration_range or (5, 5)
        return int(round(min(max(want, lo), hi)))

    def pick_resolution(self, requested: Optional[str]) -> str:
        if requested:
            for res in self.resolutions:
                if res.lower() == requested.lower():
                    return res
        return self.default_resolution


def V(id: str, label: str, res: List[str], default_res: Optional[str] = None, *, dur=None, options=None,
      last: bool = False, hint: str = "", **extra) -> VideoVariant:
    return VideoVariant(
        id=id, label=label, resolutions=res, default_resolution=default_res or res[0],
        duration_range=dur, duration_options=list(options) if options else None,
        last_frame=last, hint=hint, extra=extra,
    )


# provider -> family -> variants (the first variant is the default).
# A model a provider doesn't offer is simply absent, so that menu just has fewer options.
VIDEO_CATALOG: Dict[str, Dict[str, List[VideoVariant]]] = {
    "runware": {
        "ltx-2.3": [V("lightricks:ltx@2.3-fast", "LTX-2.3 Fast", ["480p", "720p", "1080p"], "720p",
                      dur=(1, 20), last=True, hint="$0.03/s · ~46s", sizing="wh")],
        "wan-2.7": [V("alibaba:wan@2.7", "Wan 2.7", ["720p", "1080p"], "720p",
                      dur=(2, 15), last=True, hint="$0.10/s · ~3 min", sizing="resolution")],
        "seedance-1.5": [V("bytedance:seedance@1.5-pro", "Seedance 1.5 Pro", ["480p", "720p", "1080p"], "480p",
                           dur=(4, 12), last=True, hint="$0.012/s · ~2 min", sizing="resolution")],
        "seedance-2.0": [
            V("bytedance:seedance@2.0-fast", "Seedance 2.0 Fast", ["480p", "720p"], "480p",
              dur=(4, 15), last=True, hint="$0.06/s · ~2 min", sizing="resolution"),
            V("bytedance:seedance@2.0", "Seedance 2.0", ["480p", "720p", "1080p"], "480p",
              dur=(4, 15), last=True, hint="$0.07/s · ~3 min", sizing="resolution"),
        ],
        "minimax-h3": [
            V("minimax:h3@max-turbo", "MiniMax H3 Max Turbo", ["480p", "768p"], "480p",
              dur=(5, 15), last=True, hint="$0.025/s · ~30s", sizing="resolution"),
            V("minimax:h3@0", "MiniMax H3", ["768p", "1440p"], "768p",
              dur=(4, 15), last=True, hint="$0.08/s · ~5 min", sizing="resolution"),
        ],
    },
    "nanogpt": {
        "ltx-2.3": [V("ltx-2.3-spicy/image-to-video", "LTX-2.3 Spicy", ["480p", "720p", "1080p"], "480p",
                      options=range(3, 21), hint="$0.02/s @480p", params=["resolution", "duration", "seed"])],
        "wan-2.2": [V("wan-video-22", "Wan 2.2 14B", ["480p", "720p"], "480p",
                      options=[5, 8], hint="$0.04/s @480p", params=["resolution", "duration"])],
        "wan-2.7": [V("alibaba/wan-2.7/image-to-video-spicy", "Wan 2.7 Spicy", ["720p", "1080p"], "720p",
                      options=[5, 10, 15], hint="$0.10/s @720p",
                      params=["negative_prompt", "resolution", "duration", "enable_prompt_expansion", "seed"])],
        "seedance-1.5": [V("bytedance-seedance-v1.5-pro", "Seedance 1.5 Pro", ["480p", "720p", "1080p"], "480p",
                           options=range(4, 13), hint="$0.024/s @480p",
                           params=["resolution", "duration", "aspect_ratio", "generateAudio"])],
        "seedance-2.0": [
            V("bytedance/seedance-2.0-fast/image-to-video-spicy", "Seedance 2.0 Fast Spicy", ["480p", "720p", "1080p"], "480p",
              options=range(4, 16), last=True, hint="$0.10/s @480p",
              params=["resolution", "duration", "aspect_ratio", "generateAudio", "last_image", "seed"]),
            V("bytedance/seedance-2.0/image-to-video-spicy", "Seedance 2.0 Spicy", ["480p", "720p", "1080p"], "480p",
              options=range(4, 16), last=True, hint="$0.12/s @480p",
              params=["resolution", "duration", "aspect_ratio", "generateAudio", "last_image", "seed"]),
        ],
        "minimax-h3": [V("minimax-h3/image-to-video-spicy", "MiniMax H3 Spicy", ["480p", "768p"], "480p",
                         options=range(3, 16), last=True, hint="$0.04/s @480p",
                         params=["resolution", "duration", "last_image", "seed"])],
    },
    "sogni": {
        "ltx-2.3": [
            V("ltx23-22b-10eros-v1.4-fp8mixed_i2v", "LTX-2.3 10Eros (uncensored)", ["480p", "720p", "1080p"], "720p",
              dur=(2, 20), hint="spicy fine-tune", sizing="wh"),
            V("ltx23-22b-fp8_i2v_distilled", "LTX-2.3 Distilled", ["480p", "720p", "1080p"], "720p",
              dur=(2, 20), hint="fast", sizing="wh"),
        ],
        "wan-2.2": [
            V("wan_v2.2-14b-fp8_i2v_lightx2v", "Wan 2.2 14B LightX2V", ["480p", "720p"], "480p",
              dur=(2, 8), hint="fast", sizing="wh"),
            V("wan_v2.2-14b-fp8_i2v", "Wan 2.2 14B", ["480p", "720p"], "480p",
              dur=(2, 8), hint="full quality", sizing="wh"),
        ],
        "seedance-2.0": [
            V("seedance-2-0-mini-uncensored", "Seedance 2.0 Mini Uncensored", ["default"], "default",
              dur=(4, 15), hint="premium Spark"),
            V("seedance-2-0-fast", "Seedance 2.0 Fast", ["default"], "default", dur=(4, 15), hint="premium Spark"),
            V("seedance-2-0", "Seedance 2.0", ["default"], "default", dur=(4, 15), hint="premium Spark"),
        ],
        "minimax-h3": [
            V("minimax-h3-fastvideo-int8_i2v_turbo", "MiniMax H3 FastH3 Turbo", ["default"], "default",
              dur=(5, 15), last=True, hint="fastest · VBVR LoRA",
              last_model="minimax-h3-fastvideo-int8_flf2v_turbo", h3_lora=True),
            V("minimax-h3-fl2va-fp8_i2v_turbo", "MiniMax H3 LightX2V Turbo", ["default"], "default",
              dur=(5, 15), last=True, hint="fast · VBVR LoRA",
              last_model="minimax-h3-fl2va-fp8_flf2v_turbo", h3_lora=True),
            V("minimax-h3-fl2va-fp8_i2v", "MiniMax H3 Standard", ["default"], "default",
              dur=(5, 15), last=True, hint="best quality, slow · VBVR LoRA",
              last_model="minimax-h3-fl2va-fp8_flf2v", h3_lora=True),
        ],
    },
    "replicate": {
        "ltx-2.3": [V("lightricks/ltx-2.3-fast", "LTX-2.3 Fast", ["1080p"], "1080p",
                      options=[6, 8, 10, 12, 14, 16, 18, 20], last=True, hint="1080p minimum")],
        "wan-2.2": [V("wan-video/wan-2.2-i2v-fast", "Wan 2.2 I2V Fast", ["480p", "720p"], "480p",
                      options=[5, 6, 7], last=True, hint="cheap, fast")],
        "wan-2.7": [V("wan-video/wan-2.7-i2v", "Wan 2.7 I2V", ["720p", "1080p"], "720p",
                      dur=(2, 15), last=True)],
        "seedance-1.5": [V("bytedance/seedance-1.5-pro", "Seedance 1.5 Pro", ["480p", "720p", "1080p"], "480p",
                           dur=(2, 12), last=True)],
        "seedance-2.0": [
            V("bytedance/seedance-2.0-fast", "Seedance 2.0 Fast", ["480p", "720p"], "480p", dur=(4, 15), last=True),
            V("bytedance/seedance-2.0", "Seedance 2.0", ["480p", "720p", "1080p"], "480p", dur=(4, 15), last=True),
        ],
        "minimax-h3": [V("minimax/h3", "MiniMax H3", ["768P", "2K"], "768P",
                         options=range(4, 16), last=True)],
    },
}

MOCK_CATALOG = {
    family: [V("mock", "Mock clip (no API call)", ["480p", "720p"], "480p", dur=(1, 20), last=True, hint="free, instant")]
    for family in VIDEO_FAMILIES
}


def provider_configured(provider: str) -> bool:
    if provider == "runware":
        return bool(MediaConfig.RUNWARE_API_KEY)
    if provider == "nanogpt":
        return bool(MediaConfig.NANOGPT_API_KEY)
    if provider == "replicate":
        return bool(MediaConfig.REPLICATE_API_TOKEN)
    if provider == "sogni":
        if not MediaConfig.SOGNI_API_KEY:
            return False
        try:
            import sogni_client  # noqa: F401
            return True
        except ImportError:
            return False
    if provider == "mock":
        return MediaConfig.VIDEO_MOCK_PROVIDER
    return False


def catalog_for(provider: str) -> Dict[str, List[VideoVariant]]:
    if provider == "mock":
        return MOCK_CATALOG
    return VIDEO_CATALOG.get(provider, {})


def public_catalog() -> Dict[str, Any]:
    """Provider -> families -> variants, for the settings dropdowns."""
    providers = []
    names = list(VIDEO_CATALOG) + (["mock"] if MediaConfig.VIDEO_MOCK_PROVIDER else [])
    for name in names:
        families = [
            {"id": fam, "label": VIDEO_FAMILIES[fam], "variants": [v.to_public() for v in variants]}
            for fam, variants in catalog_for(name).items()
        ]
        providers.append({
            "id": name,
            "label": PROVIDER_LABELS.get(name, name),
            "configured": provider_configured(name),
            "families": families,
        })
    return {"families": VIDEO_FAMILIES, "providers": providers}


def find_variant(provider: str, family: str, variant_id: Optional[str] = None) -> VideoVariant:
    variants = catalog_for(provider).get(family)
    if not variants:
        raise ValueError(f"{PROVIDER_LABELS.get(provider, provider)} has no {VIDEO_FAMILIES.get(family, family)} model")
    if variant_id:
        for v in variants:
            if v.id == variant_id:
                return v
        raise ValueError(f"Unknown {provider} model '{variant_id}' for {family}")
    return variants[0]


# ============================================================================
# SIZING HELPERS
# ============================================================================

# width x height per resolution tier and orientation, multiples of 32 (LTX/Wan friendly)
WH_SIZES = {
    "480p": {"landscape": (832, 480), "portrait": (480, 832), "square": (640, 640)},
    "720p": {"landscape": (1280, 704), "portrait": (704, 1280), "square": (960, 960)},
    "1080p": {"landscape": (1920, 1088), "portrait": (1088, 1920), "square": (1440, 1440)},
}


def image_size(data: bytes) -> Tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            return im.size
    except Exception:
        return (1024, 1024)


def orientation(width: int, height: int) -> str:
    ratio = width / max(1, height)
    if ratio > 1.15:
        return "landscape"
    if ratio < 0.87:
        return "portrait"
    return "square"


def closest_ratio(width: int, height: int, options: List[str]) -> str:
    target = width / max(1, height)

    def value(opt: str) -> float:
        a, b = opt.split(":")
        return float(a) / float(b)
    return min((o for o in options if ":" in o), key=lambda o: abs(value(o) - target))


def wh_for(resolution: str, width: int, height: int) -> Tuple[int, int]:
    table = WH_SIZES.get(resolution.lower(), WH_SIZES["720p"])
    return table[orientation(width, height)]


def sniff_image_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def to_data_uri(data: bytes) -> str:
    return f"data:{sniff_image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


async def download(url: str, timeout: float = 180) -> bytes:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"download failed ({resp.status_code}) for {url[:80]}")
    return resp.content


# ============================================================================
# VIDEO REQUEST + PROVIDERS
# ============================================================================

@dataclass
class VideoRequest:
    provider: str
    family: str
    variant: VideoVariant
    prompt: str
    image: bytes
    last_image: Optional[bytes] = None
    negative_prompt: str = ""
    duration: int = 5
    resolution: str = "480p"
    seed: int = -1
    audio: bool = False


Report = Callable[[str, str], None]  # (status, human message)


class VideoProvider:
    name = "base"

    async def generate(self, req: VideoRequest, report: Report) -> bytes:
        raise NotImplementedError


class RunwareVideo(VideoProvider):
    """videoInference with deliveryMethod=async, then getResponse polling."""
    name = "runware"

    async def _post(self, tasks: List[Dict], timeout: float = 120) -> Dict:
        headers = {"Authorization": f"Bearer {MediaConfig.RUNWARE_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(MediaConfig.RUNWARE_ENDPOINT, json=tasks, headers=headers)
        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError(f"Runware HTTP {resp.status_code}: {resp.text[:200]}")
        errors = payload.get("errors") or []
        if resp.status_code != 200 or errors:
            message = errors[0].get("message") if errors and isinstance(errors[0], dict) else resp.text[:300]
            raise RuntimeError(f"Runware error: {message}")
        return payload

    async def upload(self, data: bytes) -> str:
        task = {"taskType": "imageUpload", "taskUUID": str(uuid.uuid4()), "image": to_data_uri(data)}
        payload = await self._post([task])
        for item in payload.get("data") or []:
            if item.get("imageUUID"):
                return item["imageUUID"]
        raise RuntimeError("Runware imageUpload returned no imageUUID")

    def build_task(self, req: VideoRequest, first: str, last: Optional[str], task_uuid: str) -> Dict:
        v = req.variant
        frames = [{"image": first, "frame": "first"}]
        if last and v.last_frame:
            frames.append({"image": last, "frame": "last"})
        task: Dict[str, Any] = {
            "taskType": "videoInference",
            "taskUUID": task_uuid,
            "model": v.id,
            "positivePrompt": req.prompt,
            "duration": req.duration,
            "deliveryMethod": "async",
            "outputType": "URL",
            "outputFormat": "MP4",
            "numberResults": 1,
            "includeCost": True,
            "inputs": {"frameImages": frames},
            "safety": {"checkContent": False},
        }
        if v.extra.get("sizing") == "wh":
            w, h = image_size(req.image)
            task["width"], task["height"] = wh_for(req.resolution, w, h)
        else:
            task["resolution"] = req.resolution
        if req.negative_prompt:
            task["negativePrompt"] = req.negative_prompt
        if req.seed is not None and req.seed >= 0:
            task["seed"] = int(req.seed)
        return task

    async def generate(self, req: VideoRequest, report: Report) -> bytes:
        report("uploading", "Uploading keyframe to Runware")
        first = await self.upload(req.image)
        last = await self.upload(req.last_image) if (req.last_image and req.variant.last_frame) else None
        task_uuid = str(uuid.uuid4())
        await self._post([self.build_task(req, first, last, task_uuid)])
        report("rendering", f"Rendering on Runware ({req.variant.label})")

        deadline = time.time() + MediaConfig.VIDEO_JOB_TIMEOUT
        while time.time() < deadline:
            await asyncio.sleep(MediaConfig.VIDEO_POLL_SECONDS)
            payload = await self._post([{"taskType": "getResponse", "taskUUID": task_uuid}], timeout=60)
            for item in payload.get("data") or []:
                if item.get("taskUUID") not in (None, task_uuid):
                    continue
                status = (item.get("status") or "").lower()
                if item.get("videoURL") and status in ("", "success"):
                    report("downloading", "Downloading video")
                    return await download(item["videoURL"])
                if status == "error":
                    raise RuntimeError(f"Runware render failed: {item.get('error') or item}")
        raise TimeoutError("Runware render timed out")


class NanoGPTVideo(VideoProvider):
    """POST /generate-video, then poll /video/status until COMPLETED or FAILED."""
    name = "nanogpt"

    def build_body(self, req: VideoRequest) -> Dict:
        v = req.variant
        allowed = set(v.extra.get("params") or [])
        body: Dict[str, Any] = {"model": v.id, "prompt": req.prompt, "imageDataUrl": to_data_uri(req.image)}
        if "duration" in allowed:
            body["duration"] = str(req.duration)
        if "resolution" in allowed:
            body["resolution"] = req.resolution
        if "aspect_ratio" in allowed:
            w, h = image_size(req.image)
            body["aspect_ratio"] = closest_ratio(w, h, ["16:9", "9:16", "1:1", "3:4", "4:3", "21:9"])
        if "last_image" in allowed and req.last_image:
            body["last_image"] = to_data_uri(req.last_image)
        if "seed" in allowed and req.seed is not None and req.seed >= 0:
            body["seed"] = int(req.seed)
        if "negative_prompt" in allowed and req.negative_prompt:
            body["negative_prompt"] = req.negative_prompt
        if "generateAudio" in allowed:
            body["generateAudio"] = bool(req.audio)
        if "enable_prompt_expansion" in allowed:
            body["enable_prompt_expansion"] = False
        if "enable_safety_checker" in allowed:
            body["enable_safety_checker"] = False
        return body

    async def generate(self, req: VideoRequest, report: Report) -> bytes:
        headers = {"x-api-key": MediaConfig.NANOGPT_API_KEY, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{MediaConfig.NANOGPT_BASE}/generate-video", json=self.build_body(req), headers=headers)
        if resp.status_code >= 300:
            raise RuntimeError(f"NanoGPT {resp.status_code}: {_error_text(resp)}")
        run_id = resp.json().get("runId") or resp.json().get("id")
        if not run_id:
            raise RuntimeError(f"NanoGPT returned no runId: {resp.text[:200]}")
        report("rendering", f"Rendering on NanoGPT ({req.variant.label})")

        deadline = time.time() + MediaConfig.VIDEO_JOB_TIMEOUT
        async with httpx.AsyncClient(timeout=60) as client:
            while time.time() < deadline:
                await asyncio.sleep(MediaConfig.VIDEO_POLL_SECONDS + 2)
                st = await client.get(f"{MediaConfig.NANOGPT_BASE}/video/status", params={"requestId": run_id}, headers=headers)
                if st.status_code >= 300:
                    continue
                data = (st.json() or {}).get("data") or {}
                status = str(data.get("status") or "").upper()
                if status == "COMPLETED":
                    url = ((data.get("output") or {}).get("video") or {}).get("url")
                    if not url:
                        raise RuntimeError("NanoGPT finished without a video URL")
                    report("downloading", "Downloading video")
                    return await download(url)
                if status == "FAILED":
                    raise RuntimeError(f"NanoGPT render failed: {data.get('error') or 'unknown error'}")
        raise TimeoutError("NanoGPT render timed out")


class SogniVideo(VideoProvider):
    """Sogni Supernet through the official SDK: the REST workflow API can't address
    exact model ids such as the 10Eros fine-tune, the SDK can."""
    name = "sogni"

    def build_params(self, req: VideoRequest) -> Dict[str, Any]:
        v = req.variant
        model_id = v.id
        params: Dict[str, Any] = {
            "type": "video",
            "modelId": model_id,
            "positivePrompt": req.prompt,
            "numberOfMedia": 1,
            "duration": req.duration,
            "referenceImage": req.image,
            "disableNSFWFilter": True,
        }
        if req.last_image and v.last_frame and v.extra.get("last_model"):
            params["modelId"] = v.extra["last_model"]
            params["referenceImageEnd"] = req.last_image
        if v.extra.get("sizing") == "wh" and req.resolution in WH_SIZES:
            w, h = image_size(req.image)
            params["width"], params["height"] = wh_for(req.resolution, w, h)
        if req.negative_prompt:
            params["negativePrompt"] = req.negative_prompt
        if req.seed is not None and req.seed >= 0:
            params["seed"] = int(req.seed)
        if v.extra.get("h3_lora") and MediaConfig.SOGNI_H3_LORA:
            params["loras"] = [MediaConfig.SOGNI_H3_LORA]
            params["loraStrengths"] = [MediaConfig.SOGNI_H3_LORA_STRENGTH]
        return params

    async def generate(self, req: VideoRequest, report: Report) -> bytes:
        from sogni_client import SogniClient  # imported lazily: optional dependency

        params = self.build_params(req)
        report("rendering", f"Rendering on Sogni ({req.variant.label})")
        async with await SogniClient.create(
            api_key=MediaConfig.SOGNI_API_KEY,
            app_id=MediaConfig.SOGNI_APP_ID,
            network=MediaConfig.SOGNI_NETWORK,
        ) as sogni:
            project = await sogni.projects.create(params)
            urls = await asyncio.wait_for(project.wait_for_completion(), timeout=MediaConfig.VIDEO_JOB_TIMEOUT)
        if not urls:
            raise RuntimeError("Sogni finished without a result")
        report("downloading", "Downloading video")
        return await download(urls[0])


class ReplicateVideo(VideoProvider):
    """Replicate model predictions with per-model input names."""
    name = "replicate"
    API = "https://api.replicate.com/v1"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {MediaConfig.REPLICATE_API_TOKEN}"}

    async def upload(self, data: bytes) -> str:
        """Replicate's Files API; the returned URL is readable by the model."""
        mime = sniff_image_mime(data)
        ext = mime.split("/")[1]
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.API}/files", headers=self._headers(),
                files={"content": (f"frame.{ext}", data, mime)},
            )
        if resp.status_code >= 300:
            raise RuntimeError(f"Replicate file upload {resp.status_code}: {_error_text(resp)}")
        url = ((resp.json() or {}).get("urls") or {}).get("get")
        if not url:
            raise RuntimeError("Replicate file upload returned no URL")
        return url

    def build_input(self, req: VideoRequest, image_url: str, last_url: Optional[str]) -> Dict[str, Any]:
        model = req.variant.id
        w, h = image_size(req.image)
        seed = int(req.seed) if req.seed is not None and req.seed >= 0 else None
        inp: Dict[str, Any] = {"prompt": req.prompt}
        if model.startswith("wan-video/wan-2.2"):
            inp.update({
                "image": image_url,
                "resolution": req.resolution,
                "frames_per_second": 16,
                "num_frames": max(81, min(121, req.duration * 16 + 1)),
                "go_fast": True,
                "disable_safety_checker": True,
            })
            if last_url:
                inp["last_image"] = last_url
        elif model.startswith("wan-video/wan-2.7"):
            inp.update({"first_frame": image_url, "resolution": req.resolution, "duration": req.duration,
                        "enable_prompt_expansion": False})
            if req.negative_prompt:
                inp["negative_prompt"] = req.negative_prompt
            if last_url:
                inp["last_frame"] = last_url
        elif model.startswith("bytedance/seedance"):
            ratios = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "9:21"]
            inp.update({"image": image_url, "duration": req.duration, "resolution": req.resolution,
                        "aspect_ratio": closest_ratio(w, h, ratios), "generate_audio": bool(req.audio)})
            if last_url:
                inp["last_frame_image"] = last_url
        elif model.startswith("minimax/h3"):
            inp.update({"first_frame_image": image_url, "duration": req.duration,
                        "resolution": req.resolution, "ratio": "adaptive"})
            if last_url:
                inp["last_frame_image"] = last_url
        elif model.startswith("lightricks/ltx"):
            inp.update({"image": image_url, "duration": req.duration, "resolution": req.resolution,
                        "aspect_ratio": "9:16" if orientation(w, h) == "portrait" else "16:9",
                        "generate_audio": bool(req.audio)})
            if last_url:
                inp["last_frame_image"] = last_url
        else:
            inp["image"] = image_url
        if seed is not None and not model.startswith(("minimax/", "lightricks/")):
            inp["seed"] = seed
        return inp

    async def generate(self, req: VideoRequest, report: Report) -> bytes:
        report("uploading", "Uploading keyframe to Replicate")
        image_url = await self.upload(req.image)
        last_url = await self.upload(req.last_image) if (req.last_image and req.variant.last_frame) else None
        body = {"input": self.build_input(req, image_url, last_url)}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.API}/models/{req.variant.id}/predictions",
                json=body, headers={**self._headers(), "Content-Type": "application/json"},
            )
        if resp.status_code >= 300:
            raise RuntimeError(f"Replicate {resp.status_code}: {_error_text(resp)}")
        prediction = resp.json()
        report("rendering", f"Rendering on Replicate ({req.variant.label})")
        output = await replicate_wait(prediction, self._headers())
        report("downloading", "Downloading video")
        return await download(output)


async def replicate_wait(prediction: Dict, headers: Dict[str, str], timeout: Optional[float] = None) -> str:
    """Poll a Replicate prediction and return its first output URL."""
    get_url = (prediction.get("urls") or {}).get("get") or f"{ReplicateVideo.API}/predictions/{prediction.get('id')}"
    deadline = time.time() + (timeout or MediaConfig.VIDEO_JOB_TIMEOUT)
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
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
            if time.time() > deadline:
                raise TimeoutError("Replicate prediction timed out")
            await asyncio.sleep(MediaConfig.VIDEO_POLL_SECONDS)
            resp = await client.get(get_url, headers=headers)
            if resp.status_code < 300:
                prediction = resp.json()


class MockVideo(VideoProvider):
    """Returns a bundled 2-second clip; for testing the full flow without spending credits."""
    name = "mock"
    CLIP = Path(__file__).parent / "tests" / "fixtures" / "mock_clip.mp4"

    async def generate(self, req: VideoRequest, report: Report) -> bytes:
        report("rendering", "Rendering (mock)")
        await asyncio.sleep(2)
        return self.CLIP.read_bytes()


PROVIDERS: Dict[str, VideoProvider] = {
    "runware": RunwareVideo(),
    "nanogpt": NanoGPTVideo(),
    "sogni": SogniVideo(),
    "replicate": ReplicateVideo(),
    "mock": MockVideo(),
}


def _error_text(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        return resp.text[:300]
    if isinstance(payload, dict):
        err = payload.get("error") or payload.get("detail") or payload.get("message") or payload
        if isinstance(err, dict):
            err = err.get("message") or err
        return str(err)[:300]
    return str(payload)[:300]


# ============================================================================
# VIDEO STORE + JOBS
# ============================================================================

class VideoStore:
    """In-memory TTL cache of finished clips (single worker, same as the image store)."""

    def __init__(self):
        self._items: "OrderedDict[str, Tuple[bytes, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, data: bytes) -> str:
        vid = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            for key in [k for k, (_d, exp) in self._items.items() if exp <= now]:
                self._items.pop(key, None)
            while len(self._items) >= MediaConfig.VIDEO_STORE_MAX:
                self._items.popitem(last=False)
            self._items[vid] = (data, now + MediaConfig.VIDEO_STORE_TTL)
        return vid

    def get(self, vid: str) -> Optional[bytes]:
        with self._lock:
            entry = self._items.get(vid)
            if not entry or entry[1] <= time.time():
                self._items.pop(vid, None)
                return None
            return entry[0]


video_store = VideoStore()


@dataclass
class VideoJob:
    id: str
    provider: str
    family: str
    variant: str
    status: str = "queued"   # queued | keyframe | uploading | rendering | downloading | done | failed
    message: str = "Queued"
    error: str = ""
    video_id: str = ""
    keyframe_b64: str = ""
    keyframe_generated: bool = False
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    finished: float = 0.0

    def to_public(self) -> Dict[str, Any]:
        out = {
            "id": self.id,
            "provider": self.provider,
            "family": self.family,
            "variant": self.variant,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "elapsed": round((self.finished or time.time()) - self.created, 1),
        }
        if self.video_id:
            out["video_path"] = f"/videos/{self.video_id}.mp4"
        if self.keyframe_generated and self.keyframe_b64:
            out["keyframe_b64"] = self.keyframe_b64
        return out


class VideoJobs:
    MAX_JOBS = 200

    def __init__(self):
        self.jobs: "OrderedDict[str, VideoJob]" = OrderedDict()
        self._tasks: set = set()

    def create(self, provider: str, family: str, variant: str, run: Callable[[VideoJob], Awaitable[bytes]]) -> VideoJob:
        job = VideoJob(id=uuid.uuid4().hex[:16], provider=provider, family=family, variant=variant)
        self.jobs[job.id] = job
        while len(self.jobs) > self.MAX_JOBS:
            self.jobs.popitem(last=False)
        task = asyncio.get_running_loop().create_task(self._run(job, run))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: VideoJob, run: Callable[[VideoJob], Awaitable[bytes]]):
        started = time.time()
        try:
            data = await asyncio.wait_for(run(job), timeout=MediaConfig.VIDEO_JOB_TIMEOUT + 120)
            if not data or len(data) < 1000:
                raise RuntimeError("provider returned an empty video")
            job.video_id = video_store.add(data)
            job.status, job.message = "done", f"Done in {time.time() - started:.0f}s"
            logger.info(f"🎬 [Video {job.id}] {job.provider}/{job.variant} done in {time.time() - started:.1f}s ({len(data) // 1024} KB)")
        except Exception as e:
            job.status, job.error = "failed", str(e) or e.__class__.__name__
            job.message = "Failed"
            logger.error(f"🎬 [Video {job.id}] {job.provider}/{job.variant} failed: {job.error}")
        finally:
            job.updated = job.finished = time.time()

    def get(self, job_id: str) -> Optional[VideoJob]:
        return self.jobs.get(job_id)


video_jobs = VideoJobs()


def reporter(job: VideoJob) -> Report:
    def report(status: str, message: str):
        job.status, job.message, job.updated = status, message, time.time()
        logger.info(f"🎬 [Video {job.id}] {status}: {message}")
    return report


# ============================================================================
# HELPER LLM
# ============================================================================

INLINE_TAGS = ["pause", "long-pause", "hum-tune", "laugh", "chuckle", "giggle", "cry",
               "tsk", "tongue-click", "lip-smack", "breath", "inhale", "exhale", "sigh"]
WRAPPING_TAGS = ["soft", "whisper", "loud", "build-intensity", "decrease-intensity",
                 "higher-pitch", "lower-pitch", "slow", "fast", "sing-song", "singing", "emphasis"]
SUBTLE_INLINE = ["pause", "laugh", "chuckle", "sigh", "breath"]
SUBTLE_WRAPPING = ["whisper", "soft"]

_TAG_RE = re.compile(r"\[[^\]\n]{1,30}\]|</?[a-zA-Z][\w-]{0,30}>")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def strip_tags(text: str) -> str:
    return _TAG_RE.sub(" ", text or "")


def same_words(a: str, b: str) -> bool:
    """True when two texts have identical words in order, ignoring speech tags and punctuation."""
    norm = lambda s: [w.lower() for w in _WORD_RE.findall(strip_tags(s))]  # noqa: E731
    return norm(a) == norm(b)


def extract_json(text: str) -> Dict[str, Any]:
    """First JSON object in an LLM reply (tolerates code fences and chatter)."""
    if not text:
        raise ValueError("empty reply")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {text[:120]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unterminated JSON object in reply: {text[:120]}")


DECIDE_SYSTEM = """You are the media director for a roleplay chat between {user} and {char}.
After {char}'s latest message, decide whether {char} also sends {user} a photo or a short video, the way a person texting would.

Send media only when it clearly fits:
- image: {user} asks for a picture or selfie, {char} says they'll send one, shows off an outfit, a look or a place, or a strongly visual moment happens.
- video: {user} asks for a video or clip, or a moment where motion matters (dancing, a gesture, a reveal) and a still photo would lose it. Videos are expensive: choose image unless motion matters or a video was asked for.
- none: ordinary conversation. Most messages are "none".
{limits}{force}
Reply with ONLY a JSON object:
{{"action": "none|image|video", "reason": "<10 words>", "image_prompt": "...", "motion_prompt": "...", "caption": "..."}}
- image_prompt: one still frame of {char} for an image model. Use the exact name {char}. Include appearance and outfit as established in the chat, pose, expression, setting, lighting and framing (selfie, mirror shot, candid photo...). Only what is visible, under 80 words. Empty when action is none.
- motion_prompt: video only. What moves during about 5 seconds starting from that frame: body motion, expression change, camera move. Under 40 words.
- caption: a short line {char} might text along with it, or ""."""

VOICE_SYSTEM = """You add expressive speech tags to one line of dialogue that a text-to-speech voice will speak as {char}.
Rules:
- Return exactly the same words in the same order. Never add, drop or change a word. Only insert tags.
- Inline tags go where the sound happens: {inline}
- Wrapping tags go around a whole phrase, with an opening and a matching closing tag: {wrapping}
- Use 0 to 2 tags. A neutral line needs none.
- Match the mood from the context.
Output only the tagged line, nothing else."""


class HelperLLM:
    """OpenAI-compatible chat client used for decisions, voice tags and summaries."""

    def __init__(self):
        self._models_cache: Tuple[float, List[Dict]] = (0.0, [])

    def backend(self, model: Optional[str] = None) -> Tuple[str, str, str, str]:
        """(url, key, model, label). Helper first; Together only as a legacy fallback."""
        if MediaConfig.HELPER_API_KEY:
            return MediaConfig.HELPER_API_URL, MediaConfig.HELPER_API_KEY, model or MediaConfig.HELPER_MODEL, "helper"
        if MediaConfig.TOGETHER_API_KEY:
            return MediaConfig.TOGETHER_URL, MediaConfig.TOGETHER_API_KEY, MediaConfig.TOGETHER_MODEL, "together"
        return "", "", "", "none"

    @property
    def configured(self) -> bool:
        return self.backend()[3] != "none"

    async def chat(self, messages: List[Dict[str, str]], *, model: Optional[str] = None, max_tokens: int = 500,
                   temperature: float = 0.4, timeout: float = 45) -> str:
        url, key, model_id, label = self.backend(model)
        if not key:
            raise RuntimeError("No helper LLM configured (set NANOGPT_API_KEY or HELPER_API_KEY)")
        payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 300:
            raise RuntimeError(f"helper LLM ({label}/{model_id}) {resp.status_code}: {_error_text(resp)}")
        data = resp.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content.strip()

    async def decide(self, *, messages: List[Dict[str, Any]], char: str, user: str, allow_image: bool = True,
                     allow_video: bool = True, force: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        limits = ""
        if not allow_image or not allow_video:
            allowed = [a for a, ok in (("image", allow_image), ("video", allow_video)) if ok]
            limits = f"\nRight now only these are allowed: {', '.join(allowed) or 'none'}. Anything else must be \"none\".\n"
        force_text = ""
        if force in ("image", "video"):
            force_text = f"\n{user} pressed a button asking for this: you MUST answer with action \"{force}\" and fill its prompts.\n"
        system = DECIDE_SYSTEM.format(user=user or "User", char=char or "Character", limits=limits, force=force_text)

        transcript = []
        for m in messages[-12:]:
            name = m.get("name") or ("User" if m.get("is_user") else char)
            text = re.sub(r"\s+", " ", str(m.get("text") or "")).strip()
            if text:
                transcript.append(f"{name}: {text[:1500]}")
        convo = "\n".join(transcript)[-7000:]
        reply = await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": f"Chat so far:\n{convo}\n\nDecide now."}],
            model=model, max_tokens=450, temperature=0.3,
        )
        decision = extract_json(reply)
        action = str(decision.get("action") or "none").lower()
        if force in ("image", "video"):
            action = force
        if action not in ("none", "image", "video"):
            action = "none"
        if (action == "image" and not allow_image and not force) or (action == "video" and not allow_video and not force):
            action = "none"
        return {
            "action": action,
            "reason": str(decision.get("reason") or "")[:200],
            "image_prompt": str(decision.get("image_prompt") or "").strip(),
            "motion_prompt": str(decision.get("motion_prompt") or "").strip(),
            "caption": str(decision.get("caption") or "").strip(),
        }

    async def voice_tags(self, text: str, *, char: str = "", tag_level: str = "full", context: str = "",
                         model: Optional[str] = None) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text or tag_level == "none":
            return {"text": text, "changed": False}
        inline = SUBTLE_INLINE if tag_level == "subtle" else INLINE_TAGS
        wrapping = SUBTLE_WRAPPING if tag_level == "subtle" else WRAPPING_TAGS
        system = VOICE_SYSTEM.format(
            char=char or "the character",
            inline=" ".join(f"[{t}]" for t in inline),
            wrapping=" ".join(f"<{t}>...</{t}>" for t in wrapping),
        )
        user_msg = (f"Context (earlier lines, do not output): {context[-1200:]}\n\n" if context else "") + f"Line: {text}"
        reply = await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            model=model, max_tokens=max(60, len(text) // 2 + 60), temperature=0.3, timeout=20,
        )
        reply = reply.strip().strip('"').strip()
        if reply.lower().startswith("line:"):
            reply = reply[5:].strip()
        if not reply or not same_words(text, reply):
            return {"text": text, "changed": False, "rejected": bool(reply)}
        return {"text": reply, "changed": reply != text}

    async def models(self, force: bool = False) -> List[Dict[str, Any]]:
        """Text models from NanoGPT's public catalog (cached for an hour)."""
        cached_at, cached = self._models_cache
        if cached and not force and time.time() - cached_at < 3600:
            return cached
        url, key, _model, label = self.backend()
        out: List[Dict[str, Any]] = []
        try:
            if "nano-gpt.com" in url or label == "none":
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get("https://nano-gpt.com/api/v1/models", params={"detailed": "true"})
                items = (resp.json() or {}).get("data") or []
            else:
                base = url.rsplit("/chat/completions", 1)[0]
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
                items = (resp.json() or {}).get("data") or []
            for m in items:
                outputs = ((m.get("architecture") or {}).get("output_modalities")) or ["text"]
                if "text" not in outputs:
                    continue
                pricing = m.get("pricing") or {}
                out.append({
                    "id": m.get("id"),
                    "name": m.get("name") or m.get("id"),
                    "input_price": _float(pricing.get("prompt", pricing.get("input"))),
                    "output_price": _float(pricing.get("completion", pricing.get("output"))),
                    "context": m.get("context_length") or 0,
                })
            out.sort(key=lambda m: ((m["input_price"] or 0) + (m["output_price"] or 0), m["id"] or ""))
        except Exception as e:
            logger.warning(f"🧠 [Helper] model list failed: {e}")
        if out:
            self._models_cache = (time.time(), out)
        return out


def _float(value) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


helper = HelperLLM()
