"""Shared by the media tests: a throwaway database, fake HTTP and small images.

The sibling test modules replace httpx with a stub, and which one loads first depends on
the run, so this adds what the media code needs to whichever httpx it finds.
"""
import asyncio
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import unittest.mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import httpx
except ImportError:
    httpx = types.ModuleType("httpx")
    sys.modules["httpx"] = httpx
if not hasattr(httpx, "HTTPError"):
    httpx.HTTPError = type("HTTPError", (Exception,), {})
if not hasattr(httpx, "AsyncClient"):
    httpx.AsyncClient = object

from media import db  # noqa: E402
from media.config import MediaConfig  # noqa: E402


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b"", headers=None):
        self.status_code = status
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeHTTP:
    """Stands in for httpx.AsyncClient. Answers from `routes` by URL part, in the order given,
    and keeps every request in `calls`."""

    def __init__(self, routes):
        self.routes = [(part, list(replies) if isinstance(replies, list) else replies) for part, replies in routes]
        self.calls = []
        fake = self

        class Client:
            def __init__(self, *_a, **_kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def request(self, method, url, **kw):
                return fake.answer(method, url, kw)

            async def post(self, url, **kw):
                return fake.answer("POST", url, kw)

            async def get(self, url, **kw):
                return fake.answer("GET", url, kw)

        self.client = Client

    def answer(self, method, url, kw):
        self.calls.append({"method": method, "url": url, **kw})
        for part, replies in self.routes:
            if part in url:
                reply = (replies.pop(0) if len(replies) > 1 else replies[0]) if isinstance(replies, list) else replies
                if isinstance(reply, Exception):
                    raise reply
                return reply
        raise AssertionError(f"no fake reply for {method} {url}")

    def call(self, part):
        return next(c for c in self.calls if part in c["url"])

    def patch(self):
        return unittest.mock.patch.object(httpx, "AsyncClient", self.client)


class MediaCase(unittest.TestCase):
    """A fresh database and data folder for each test."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.data = Path(self._dir.name)
        self._patches = [unittest.mock.patch.object(MediaConfig, "DATA_DIR", self.data),
                         unittest.mock.patch.object(MediaConfig, "VIDEO_MOCK_PROVIDER", True),
                         unittest.mock.patch.object(MediaConfig, "MOCK_RENDER_SECONDS", 0.05)]
        for patch in self._patches:
            patch.start()
        db.connect(self.data / "media.db")

    def tearDown(self):
        db.close()
        for patch in self._patches:
            patch.stop()
        self._dir.cleanup()


def run(coro):
    return asyncio.run(coro)


def noisy_jpeg(width=800, height=1200, quality=92):
    """A photo-like JPEG: noise keeps it well above the 20 KB floor."""
    from PIL import Image

    image = Image.effect_noise((width, height), 64).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def small_jpeg(width=64, height=64):
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), (200, 120, 90)).save(buf, format="JPEG")
    return buf.getvalue()
