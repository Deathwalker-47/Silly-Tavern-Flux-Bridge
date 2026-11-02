"""Provider adapters. Every provider implements the same small interface (base.py)."""

from __future__ import annotations

from typing import Dict

from .base import (Provider, RenderRequest, RenderResult, closest_ratio, download, error_text, sniff_mime,
                   to_data_uri)
from .mock import Mock
from .nanogpt import NanoGPT
from .pixeldojo import PixelDojo
from .replicate import Replicate, replicate_create, replicate_wait
from .runware import Runware
from .sogni import Sogni

PROVIDERS: Dict[str, Provider] = {
    "runware": Runware(),
    "nanogpt": NanoGPT(),
    "sogni": Sogni(),
    "replicate": Replicate(),
    "pixeldojo": PixelDojo(),
    "mock": Mock(),
}


def provider_configured(name: str) -> bool:
    provider = PROVIDERS.get(name)
    return bool(provider and provider.configured)
