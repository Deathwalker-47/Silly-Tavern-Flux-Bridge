#!/usr/bin/env python3
"""
Run the bridge against the local mocks in dev/mock_providers.py (no real API calls).

    python dev/mock_providers.py   # terminal 1
    python dev/run_local.py        # terminal 2 -> http://localhost:7861

Real keys in your environment are ignored for the mocked services. The "mock"
video provider (bundled 2 s clip) is enabled alongside the mocked Runware.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOCK = os.getenv("MOCK_BASE", "http://localhost:8788")

os.environ.update({
    "RUNWARE_API_KEY": "mock-key",
    "RUNWARE_ENDPOINT": f"{MOCK}/runware",
    "HELPER_API_URL": f"{MOCK}/v1/chat/completions",
    "HELPER_API_KEY": "mock-key",
    "HELPER_MODEL": "mock/fast-helper",
    "VIDEO_MOCK_PROVIDER": "true",
    "VIDEO_POLL_SECONDS": "1",
    "REPLICATE_API_TOKEN": "",
    "NANOGPT_API_KEY": "",
    "SOGNI_API_KEY": "",
    "IMAGE_PROVIDER_ORDER": "runware",
    "PUBLIC_BASE_URL": os.getenv("PUBLIC_BASE_URL", "http://localhost:7861"),
    "KONTEXT_UPLOAD_IMAGES": "true",
})
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))  # run_path doesn't add the script's folder like `python file.py` does
runpy.run_path(str(ROOT / "flux_lora_bridge.py"), run_name="__main__")
