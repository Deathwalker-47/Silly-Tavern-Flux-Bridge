"""Private LoRA files on Hugging Face, handed to providers without the token.

Replicate and Runware fetch a LoRA by its URL, and a private Hugging Face file answers
them with 401. Giving them HF_TOKEN would let them read every private repo on the
account, so with HF_TOKEN set each Hugging Face URL becomes a link on this bridge that
streams the file. Providers have to reach the link, so this works on the public bridge,
not on localhost.

A link is the Hugging Face URL plus a signature made with a secret only the bridge knows,
so it can't be forged or pointed at another file, and it needs no memory: it works after
a restart, and from any process that has the secret. It doesn't expire, because Runware
keeps the URL it was given and downloads the file again later (seen 7 hours after an
upload). Changing the secret retires every link at once.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlsplit

import httpx

from .config import MediaConfig

logger = logging.getLogger(__name__)

HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
SECRET_FILE = "lora_link_secret"    # in DATA_DIR, made on first use when LORA_LINK_SECRET is unset
PASSED_HEADERS = ("content-length", "content-range", "accept-ranges", "etag", "last-modified")

_secrets: Dict[str, bytes] = {}     # secret file path -> its contents, read once
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
    """Where a provider should download this LoRA: a link on this bridge for a Hugging Face file.
    The same file always gets the same link."""
    if not (MediaConfig.HF_TOKEN and MediaConfig.PUBLIC_BASE_URL and is_hf(url)):
        return url
    return f"{MediaConfig.PUBLIC_BASE_URL}/lora-files/{_b64(url.encode())}.{_sign(url)}/{file_name(url)}"


def source(link: str) -> Optional[str]:
    """The Hugging Face URL behind a link, or None if the bridge didn't make it."""
    encoded, _, signature = link.rpartition(".")
    if not encoded or not signature:
        return None
    try:
        url = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except ValueError:      # bad base64 (binascii.Error) or bad UTF-8 (UnicodeDecodeError) are both ValueErrors
        return None
    if not is_hf(url) or not hmac.compare_digest(signature, _sign(url)):
        return None
    return url


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _sign(url: str) -> str:
    return _b64(hmac.new(_secret(), url.encode(), hashlib.sha256).digest())[:32]


def _secret() -> bytes:
    """LORA_LINK_SECRET, or a random one made once and kept in DATA_DIR, so links outlive restarts."""
    if MediaConfig.LORA_LINK_SECRET:
        return MediaConfig.LORA_LINK_SECRET.encode()
    path = Path(MediaConfig.DATA_DIR) / SECRET_FILE
    if str(path) not in _secrets:
        _secrets[str(path)] = _read_or_make(path)
    return _secrets[str(path)]


def _read_or_make(path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:    # O_EXCL: if another process makes it first, use theirs
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            made = secrets.token_urlsafe(32).encode()
            with os.fdopen(fd, "wb") as f:
                f.write(made)
            logger.info(f"📦 [LoRA link] Made a new link secret in {path}")
            return made
    value = path.read_bytes().strip()
    if not value:
        raise RuntimeError(f"{path} is empty. Delete it and the bridge makes a new one.")
    return value


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
