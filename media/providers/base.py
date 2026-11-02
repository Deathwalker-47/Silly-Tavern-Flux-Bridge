"""The interface every provider implements, and the helpers they share."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .. import telemetry

MAX_RATE_RETRIES = 3


@dataclass
class RenderRequest:
    option: Any                     # ModelOption
    mode: str                       # lora | reference | text | first | first_last
    prompt: str
    negative_prompt: str = ""
    refs: List[Any] = field(default_factory=list)   # Asset
    first: Any = None
    last: Any = None
    loras: List[Dict[str, Any]] = field(default_factory=list)
    aspect_ratio: str = "9:16"
    seconds: int = 15
    resolution: str = ""
    seed: Optional[int] = None
    audio: bool = False
    adult: bool = False
    network: str = ""               # Sogni only: fast | relaxed
    params: Dict[str, Any] = field(default_factory=dict)   # filled in by submit: what was sent, no keys or image data
    cost_usd: Optional[float] = None
    cost_credits: Optional[float] = None


@dataclass
class RenderResult:
    data: bytes
    mime: str
    seed: Optional[int] = None
    cost_usd: Optional[float] = None
    cost_credits: Optional[float] = None
    task_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


class Provider:
    name = ""
    resumable = True        # polling can carry on after a bridge restart
    poll_seconds = 3.0

    @property
    def configured(self) -> bool:
        return False

    async def ensure_asset(self, asset: Any) -> str:
        """Upload once and return the provider's handle for this file."""
        raise NotImplementedError

    async def submit(self, req: RenderRequest) -> str:
        """Start a render and return its task id."""
        raise NotImplementedError

    async def poll(self, task_id: str, req: RenderRequest) -> Optional[RenderResult]:
        """The finished result, or None while it's still running."""
        raise NotImplementedError

    async def cancel(self, task_id: str) -> None:
        """Stop the render if the provider can. Most can't, and the job just stops waiting."""

    async def balance(self) -> Optional[Dict[str, Any]]:
        """{"amount": number, "unit": "USD" | "credits" | "Spark"} or None when there's no balance API."""
        return None


def check_key(resp: httpx.Response) -> None:
    """A balance call is how a new key gets checked, so a refusal must not read as "no balance"."""
    if resp.status_code in (401, 403):
        raise RuntimeError(f"The provider refused this key (HTTP {resp.status_code})")


def sniff_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return "application/octet-stream"


def sniff_image_mime(data: bytes) -> str:
    mime = sniff_mime(data)
    return mime if mime.startswith("image/") else "image/jpeg"


def to_data_uri(data: bytes) -> str:
    return f"data:{sniff_image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


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


# width x height per resolution tier and orientation. Multiples of 64: LTX and Wan accept these.
WH_SIZES = {
    "480p": {"landscape": (896, 512), "portrait": (512, 896), "square": (640, 640)},
    "720p": {"landscape": (1280, 704), "portrait": (704, 1280), "square": (960, 960)},
    "1080p": {"landscape": (1920, 1088), "portrait": (1088, 1920), "square": (1408, 1408)},
}


def wh_for(resolution: str, width: int, height: int) -> Tuple[int, int]:
    table = WH_SIZES.get(resolution.lower(), WH_SIZES["720p"])
    return table[orientation(width, height)]


def error_text(resp: "httpx.Response") -> str:
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


def retry_after(resp: "httpx.Response") -> float:
    """Seconds a 429 asks us to wait: the header, then the JSON field, else a short default."""
    raw: Any = (getattr(resp, "headers", None) or {}).get("retry-after")
    if raw is None:
        try:
            body = resp.json()
            raw = body.get("retry_after") if isinstance(body, dict) else None
        except ValueError:
            raw = None
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 5.0


async def send(client: Any, method: str, url: str, **kwargs: Any) -> "httpx.Response":
    """One request that waits out 429s, up to 3 times."""
    for attempt in range(MAX_RATE_RETRIES + 1):
        resp = await client.request(method, url, **kwargs)
        if resp.status_code != 429 or attempt == MAX_RATE_RETRIES:
            return resp
        wait = retry_after(resp)
        telemetry.record("provider", "rate_limited", url=url.split("?")[0], wait_seconds=wait, attempt=attempt + 1)
        await asyncio.sleep(wait)
    return resp


async def download(url: str, timeout: float = 180) -> bytes:
    with telemetry.step("provider", "download") as facts:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"download failed ({resp.status_code}) for {url[:80]}")
        facts["bytes"] = len(resp.content)
    return resp.content
