"""Environment settings for Media Studio."""

from __future__ import annotations

import os
import socket
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "true" if default else "false").strip().lower() in ("1", "true", "yes", "on")


class MediaConfig:
    RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY", "")
    RUNWARE_ENDPOINT = os.getenv("RUNWARE_ENDPOINT", "https://api.runware.ai/v1")

    NANOGPT_API_KEY = os.getenv("NANOGPT_API_KEY", "")
    NANOGPT_BASE = os.getenv("NANOGPT_BASE", "https://nano-gpt.com/api").rstrip("/")

    SOGNI_API_KEY = os.getenv("SOGNI_API_KEY", "")
    # Sogni wants one stable app id per installation, never a fresh one per run.
    SOGNI_APP_ID = os.getenv("SOGNI_APP_ID", "") or str(uuid.uuid5(uuid.NAMESPACE_DNS, f"flux-lora-bridge.{socket.gethostname()}"))
    # "fast" = dedicated GPUs, billed; "relaxed" = free queue that can wait a long time.
    SOGNI_NETWORK = os.getenv("SOGNI_NETWORK", "fast")
    SOGNI_RELAXED_TIMEOUT = int(os.getenv("SOGNI_RELAXED_TIMEOUT", 3600))  # seconds, relaxed jobs only
    # VBVR Video Reasoning goes on every MiniMax H3 clip; the second LoRA only on adult scenes.
    SOGNI_H3_LORA = os.getenv("SOGNI_H3_LORA", "h3-vbvr-video-reasoning")
    SOGNI_H3_LORA_STRENGTH = float(os.getenv("SOGNI_H3_LORA_STRENGTH", 0.8))
    SOGNI_H3_ADULT_LORA = os.getenv("SOGNI_H3_ADULT_LORA", "h3-mystic-xxx-v5")
    SOGNI_H3_ADULT_LORA_STRENGTH = float(os.getenv("SOGNI_H3_ADULT_LORA_STRENGTH", 0.5))

    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
    REPLICATE_FLUX_MODEL = os.getenv("REPLICATE_FLUX_MODEL", "lucataco/flux-dev-multi-lora")

    # Private LoRA files: the bridge reads them with this token and gives providers a signed
    # link of its own, so PUBLIC_BASE_URL must be this bridge's public address. The links are
    # signed with LORA_LINK_SECRET, or with a secret the bridge makes and keeps in DATA_DIR.
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://bridge.midnighttavern.online").rstrip("/")
    LORA_LINK_SECRET = os.getenv("LORA_LINK_SECRET", "")

    PIXELDOJO_API_KEY = os.getenv("PIXELDOJO_API_KEY", "")
    PIXELDOJO_BASE = os.getenv("PIXELDOJO_BASE", "https://pixeldojo.ai/api/v1").rstrip("/")

    # Pay as you go, no subscription. Venice's own uncensored flag is per model; SpicyAPI has no filter.
    VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")
    VENICE_BASE = os.getenv("VENICE_BASE", "https://api.venice.ai/api/v1").rstrip("/")
    SPICYAPI_API_KEY = os.getenv("SPICYAPI_API_KEY", "")
    SPICYAPI_BASE = os.getenv("SPICYAPI_BASE", "https://api.spicyapi.ai/api/v1").rstrip("/")

    VIDEO_MOCK_PROVIDER = _env_bool("VIDEO_MOCK_PROVIDER", False)
    MOCK_RENDER_SECONDS = float(os.getenv("MOCK_RENDER_SECONDS", 2))

    IMAGE_JOB_TIMEOUT = int(os.getenv("IMAGE_JOB_TIMEOUT", 900))    # seconds, render and download
    VIDEO_JOB_TIMEOUT = int(os.getenv("VIDEO_JOB_TIMEOUT", 1200))
    VIDEO_POLL_SECONDS = float(os.getenv("VIDEO_POLL_SECONDS", 3))
    PROVIDER_CONCURRENCY = int(os.getenv("PROVIDER_CONCURRENCY", 3))

    DATA_DIR = Path(os.getenv("DATA_DIR") or ROOT / "data")
    MEDIA_FILE_TTL_DAYS = int(os.getenv("MEDIA_FILE_TTL_DAYS", 7))        # after the extension saved the file
    MEDIA_UNSAVED_TTL_DAYS = int(os.getenv("MEDIA_UNSAVED_TTL_DAYS", 30))  # keyframes and files never saved

    # Director and helper. Any OpenAI-compatible /chat/completions endpoint works; NanoGPT is the default.
    DIRECTOR_MODEL = os.getenv("DIRECTOR_MODEL", "z-ai/glm-5.3-flash-uncensored")
    HELPER_API_URL = os.getenv("HELPER_API_URL", "https://nano-gpt.com/api/v1/chat/completions")
    HELPER_API_KEY = os.getenv("HELPER_API_KEY", "") or NANOGPT_API_KEY
    HELPER_MODEL = os.getenv("HELPER_MODEL", "") or DIRECTOR_MODEL
    # Prompt summarization for txt2img; empty = HELPER_MODEL.
    SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "")
