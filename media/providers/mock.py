"""Local test provider: spends nothing, and its task ids survive a restart like a real one."""

from __future__ import annotations

import time
import uuid
from io import BytesIO
from typing import Optional

from ..config import MediaConfig, ROOT
from .base import Provider, RenderRequest, RenderResult, sniff_mime

CLIP = ROOT / "tests" / "fixtures" / "mock_clip.mp4"


def _picture(label: str, size: tuple) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (60 + abs(hash(label)) % 140, 70, 120))
    draw = ImageDraw.Draw(image)
    for row, start in enumerate(range(0, min(len(label), 800), 60)):
        draw.text((24, 24 + row * 18), label[start:start + 60], fill=(255, 255, 255))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class Mock(Provider):
    name = "mock"
    poll_seconds = 0.5

    @property
    def configured(self) -> bool:
        return MediaConfig.VIDEO_MOCK_PROVIDER

    async def ensure_asset(self, asset) -> str:
        return asset.id

    async def submit(self, req: RenderRequest) -> str:
        if req.prompt.startswith("MOCK_FAIL"):
            raise RuntimeError("Mock provider failed on request")
        req.params = {"prompt": req.prompt, "aspect_ratio": req.aspect_ratio, "references": len(req.refs),
                      "loras": [l.get("url") for l in req.loras], "seed": req.seed}
        return f"{time.time() + MediaConfig.MOCK_RENDER_SECONDS:.3f}:{uuid.uuid4().hex[:8]}"

    async def poll(self, task_id: str, req: RenderRequest) -> Optional[RenderResult]:
        if time.time() < float(task_id.split(":")[0]):
            return None
        if req.option.kind == "video":
            data = CLIP.read_bytes()
        else:
            sizes = req.option.extra.get("sizes") or {}
            width, height = sizes.get(req.aspect_ratio, (768, 1344))
            data = _picture(req.prompt, (width // 2, height // 2))
        return RenderResult(data=data, mime=sniff_mime(data), seed=req.seed, cost_usd=0.0, task_id=task_id)
