"""Private LoRA files on Hugging Face, handed to providers without the token.

Replicate and Runware fetch a LoRA by its URL, and a private Hugging Face file answers
them with 401. Giving them HF_TOKEN would let them read every private repo on the
account, so with HF_TOKEN set each Hugging Face URL becomes a random link on this bridge
that streams the file for LINK_TTL seconds. The random part is the only key. Providers
have to reach the link, so this works on the public bridge, not on localhost.
"""

from __future__ import annotations

import logging
import re
import secrets
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlsplit

import httpx

from .config import MediaConfig

logger = logging.getLogger(__name__)

HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
LINK_TTL = 30 * 60      # seconds: a provider may sit in its queue a while before downloading
PASSED_HEADERS = ("content-length", "content-range", "accept-ranges", "etag", "last-modified")

_links: Dict[str, Tuple[str, float]] = {}
_transport: Optional[httpx.AsyncBaseTransport] = None   # tests put a mock Hugging Face here


def is_hf(url: str) -> bool:
    return (urlsplit(url).hostname or "").lower() in HF_HOSTS


def auth_headers(url: str) -> Dict[str, str]:
    """The token, for Hugging Face URLs only. httpx drops it when Hugging Face redirects to its CDN."""
    return {"Authorization": f"Bearer {MediaConfig.HF_TOKEN}"} if MediaConfig.HF_TOKEN and is_hf(url) else {}


def file_name(url: str) -> str:
    """A plain name that still ends in .safetensors, since Replicate only downloads URLs that do."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", unquote(urlsplit(url).path.rsplit("/", 1)[-1])).strip("_") or "lora"
    return name if name.endswith(".safetensors") else f"{name}.safetensors"


def provider_url(url: str) -> str:
    """Where a provider should download this LoRA: a link on this bridge for a Hugging Face file."""
    if not (MediaConfig.HF_TOKEN and MediaConfig.PUBLIC_BASE_URL and is_hf(url)):
        return url
    _forget_expired()
    link = secrets.token_urlsafe(24)
    _links[link] = (url, time.time() + LINK_TTL)
    return f"{MediaConfig.PUBLIC_BASE_URL}/lora-files/{link}/{file_name(url)}"


def source(link: str) -> Optional[str]:
    """The Hugging Face URL behind a link, or None once it has expired."""
    _forget_expired()
    entry = _links.get(link)
    return entry[0] if entry else None


def _forget_expired() -> None:
    now = time.time()
    for link in [k for k, (_url, until) in _links.items() if until < now]:
        del _links[link]


async def stream(url: str, method: str = "GET", byte_range: Optional[str] = None) -> Any:
    """The file as it arrives from Hugging Face. Ranges pass through, because Replicate's
    downloader fetches a file in parallel pieces."""
    from fastapi import HTTPException
    from fastapi.responses import Response, StreamingResponse  # imported lazily; not part of the test stubs

    headers = {**auth_headers(url), "Accept-Encoding": "identity"}
    if byte_range:
        headers["Range"] = byte_range
    client = httpx.AsyncClient(transport=_transport, follow_redirects=True, timeout=httpx.Timeout(30, read=300))
    try:
        upstream = await client.send(client.build_request("GET", url, headers=headers), stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Hugging Face unreachable ({type(e).__name__})")

    async def close() -> None:
        await upstream.aclose()
        await client.aclose()

    if upstream.status_code >= 400:
        await close()
        logger.warning(f"📦 [LoRA link] Hugging Face answered {upstream.status_code} for {file_name(url)}")
        raise HTTPException(status_code=404 if upstream.status_code == 404 else 502,
                            detail=f"Hugging Face answered HTTP {upstream.status_code}")
    passed = {k: upstream.headers[k] for k in PASSED_HEADERS if k in upstream.headers}
    if upstream.headers.get("content-encoding", "identity") != "identity":
        passed.pop("content-length", None)      # it ignored identity; the bytes below come out decoded
    if method == "HEAD":
        await close()
        return Response(status_code=upstream.status_code, headers=passed, media_type="application/octet-stream")

    async def body():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await close()

    return StreamingResponse(body(), status_code=upstream.status_code, headers=passed, media_type="application/octet-stream")


async def check_token() -> Dict[str, Any]:
    """The key panel's check after saving. Hugging Face has no balance, so ask whose token it is."""
    try:
        async with httpx.AsyncClient(transport=_transport, timeout=20) as client:
            resp = await client.get("https://huggingface.co/api/whoami-v2",
                                    headers={"Authorization": f"Bearer {MediaConfig.HF_TOKEN}"})
    except httpx.HTTPError as e:
        return {"amount": None, "error": f"Hugging Face unreachable ({type(e).__name__})"}
    if resp.status_code != 200:
        return {"amount": None, "error": f"Hugging Face refused the token (HTTP {resp.status_code})"}
    return {"amount": None}
