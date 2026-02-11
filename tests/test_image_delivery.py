r"""
Tests verifying that all providers deliver valid images end-to-end
in the format SillyTavern's AutoImageGen plugin expects.

Plugin contract (silly-tavern-pluggin/index.js):
  - Line 482: ``return data.images[0]``          -> expects base64 string at images[0]
  - Line 511: ``img.src = `data:image/png;base64,${base64Image}` ``  -> data URI
  - Line 555: ``if (base64Image)`` displayImage   -> must be truthy (non-empty string)

So the bridge must return::

  {
    "images": ["<non-empty base64 string that decodes to valid image bytes>"],
    "parameters": { ... },
    "info": "..."
  }
"""
import base64
import json
import sys
import types
import unittest


# ---------------------------------------------------------------------------
# Stubs (reuse pattern from test_provider_response_parsing.py)
# ---------------------------------------------------------------------------

def install_test_stubs():
    fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class FastAPI:
        def add_middleware(self, *a, **kw): return None
        def get(self, *a, **kw): return lambda fn: fn
        def post(self, *a, **kw): return lambda fn: fn
        def on_event(self, *a, **kw): return lambda fn: fn

    class Request:
        async def json(self): return {}

    class APIRouter: pass

    fastapi.FastAPI = FastAPI
    fastapi.HTTPException = HTTPException
    fastapi.Request = Request
    fastapi.APIRouter = APIRouter

    cors_mod = types.ModuleType("fastapi.middleware.cors")
    class CORSMiddleware: pass
    cors_mod.CORSMiddleware = CORSMiddleware

    responses_mod = types.ModuleType("fastapi.responses")
    class JSONResponse(dict): pass
    responses_mod.JSONResponse = JSONResponse

    pydantic_mod = types.ModuleType("pydantic")
    class BaseModel: pass
    def Field(default=None, **kwargs): return default
    pydantic_mod.BaseModel = BaseModel
    pydantic_mod.Field = Field

    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.run = lambda *a, **k: None

    httpx_mod = types.ModuleType("httpx")
    class AsyncClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): raise NotImplementedError
        async def get(self, *a, **kw): raise NotImplementedError
    httpx_mod.AsyncClient = AsyncClient

    sse_root = types.ModuleType("sse_starlette")
    sse_mod = types.ModuleType("sse_starlette.sse")
    class EventSourceResponse:
        def __init__(self, gen): self.generator = gen
    sse_mod.EventSourceResponse = EventSourceResponse

    pil_mod = types.ModuleType("PIL")
    pil_mod.Image = object

    sys.modules.setdefault("fastapi", fastapi)
    sys.modules.setdefault("fastapi.middleware.cors", cors_mod)
    sys.modules.setdefault("fastapi.responses", responses_mod)
    sys.modules.setdefault("pydantic", pydantic_mod)
    sys.modules.setdefault("uvicorn", uvicorn_mod)
    sys.modules.setdefault("httpx", httpx_mod)
    sys.modules.setdefault("sse_starlette", sse_root)
    sys.modules.setdefault("sse_starlette.sse", sse_mod)
    sys.modules.setdefault("PIL", pil_mod)


install_test_stubs()
import flux_lora_bridge as bridge


# ---------------------------------------------------------------------------
# Realistic image byte fixtures (valid magic bytes)
# ---------------------------------------------------------------------------

# Minimal valid JPEG: FF D8 FF E0 ... (JFIF header + padding)
FAKE_JPEG = b'\xff\xd8\xff\xe0' + b'\x00' * 100

# Minimal valid PNG: 89 50 4E 47 0D 0A 1A 0A (PNG signature + padding)
FAKE_PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100

# Minimal valid WEBP: RIFF....WEBP (RIFF header + padding)
FAKE_WEBP = b'RIFF' + b'\x00\x00\x00\x00' + b'WEBP' + b'\x00' * 100

# Non-image data (HTML error page — should NOT pass as an image)
FAKE_HTML_ERROR = b'<!DOCTYPE html><html><body><h1>403 Forbidden</h1></body></html>'


# ---------------------------------------------------------------------------
# HTTP Fakes
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.content = content
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


class FakeAsyncClient:
    routes = {"POST": {}, "GET": {}}

    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    @classmethod
    def set_route(cls, method, url, response):
        cls.routes[method][url] = response

    async def post(self, url, json=None, headers=None):
        return self.routes["POST"][url]

    async def get(self, url, headers=None):
        return self.routes["GET"][url]


# ---------------------------------------------------------------------------
# Helper: Together SDK mock objects (simulates real SDK, not plain dict)
# ---------------------------------------------------------------------------

class FakeImageChoice:
    """Mimics together.types.ImageChoice from SDK v2"""
    def __init__(self, url=None, b64_json=None):
        self.url = url
        self.b64_json = b64_json
        self.index = 0
        self.revised_prompt = None


class FakeImageResponse:
    """Mimics together.types.ImageResponse from SDK v2"""
    def __init__(self, data):
        self.data = data


# ---------------------------------------------------------------------------
# Shared params for all provider generate() calls
# ---------------------------------------------------------------------------

GEN_PARAMS = {"steps": 1, "cfg_scale": 1, "width": 512, "height": 512, "seed": 42}


def assert_valid_image_bytes(test_case, image_bytes, expected_bytes=None):
    """Assert that image_bytes is non-empty bytes suitable for SillyTavern delivery."""
    test_case.assertIsInstance(image_bytes, bytes, "Provider must return bytes")
    test_case.assertGreater(len(image_bytes), 0, "Image bytes must be non-empty")

    # Verify base64 encoding produces a truthy string (plugin line 555: `if (base64Image)`)
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    test_case.assertIsInstance(b64, str)
    test_case.assertGreater(len(b64), 0, "Base64 string must be non-empty (truthy in JS)")

    # Verify round-trip: base64 decodes back to original bytes
    test_case.assertEqual(base64.b64decode(b64), image_bytes)

    if expected_bytes is not None:
        test_case.assertEqual(image_bytes, expected_bytes)


def assert_is_real_image(test_case, image_bytes):
    """Assert that bytes start with known image magic numbers."""
    test_case.assertGreaterEqual(len(image_bytes), 4, "Image data too small")
    is_jpeg = image_bytes[:3] == b'\xff\xd8\xff'
    is_png = image_bytes[:4] == b'\x89PNG'
    is_webp = image_bytes[:4] == b'RIFF'
    is_gif = image_bytes[:4] == b'GIF8'
    test_case.assertTrue(
        is_jpeg or is_png or is_webp or is_gif,
        f"Bytes do not start with known image magic (first 8 bytes: {image_bytes[:8].hex()})"
    )


# ===========================================================================
# TEST: Each provider returns valid image bytes
# ===========================================================================

class TestProviderImageDelivery(unittest.IsolatedAsyncioTestCase):
    """Verify every provider's generate() returns valid image bytes."""

    def setUp(self):
        self.original_async_client = bridge.httpx.AsyncClient
        bridge.httpx.AsyncClient = FakeAsyncClient
        FakeAsyncClient.routes = {"POST": {}, "GET": {}}

    def tearDown(self):
        bridge.httpx.AsyncClient = self.original_async_client

    # --- Runware ---

    async def test_runware_returns_valid_jpeg(self):
        """Runware returns JPEG bytes that survive base64 round-trip."""
        FakeAsyncClient.set_route(
            "POST", bridge.Config.RUNWARE_ENDPOINT,
            FakeResponse(200, {"data": [{"imageURL": "http://cdn/runware.jpg"}]})
        )
        FakeAsyncClient.set_route(
            "GET", "http://cdn/runware.jpg",
            FakeResponse(200, content=FAKE_JPEG)
        )
        client = bridge.RunwareClient("test-key")
        image = await client.generate("test prompt", "", [], GEN_PARAMS)
        assert_valid_image_bytes(self, image, FAKE_JPEG)
        assert_is_real_image(self, image)

    # --- Pixel Dojo ---

    async def test_pixeldojo_returns_valid_png(self):
        """Pixel Dojo returns PNG bytes that survive base64 round-trip."""
        FakeAsyncClient.set_route(
            "POST", bridge.Config.PIXELDOJO_ENDPOINT,
            FakeResponse(200, {"images": ["http://cdn/pixeldojo.png"]})
        )
        FakeAsyncClient.set_route(
            "GET", "http://cdn/pixeldojo.png",
            FakeResponse(200, content=FAKE_PNG)
        )
        client = bridge.PixelDojoClient("test-key")
        image = await client.generate("test prompt", "", [], GEN_PARAMS)
        assert_valid_image_bytes(self, image, FAKE_PNG)
        assert_is_real_image(self, image)

    # --- Wavespeed ---

    async def test_wavespeed_returns_valid_jpeg(self):
        """Wavespeed returns JPEG bytes from immediate outputs."""
        bridge.Config.WAVESPEED_API_KEY = "test-key"
        FakeAsyncClient.set_route(
            "POST", bridge.Config.WAVESPEED_ENDPOINT,
            FakeResponse(200, {"data": {"outputs": ["http://cdn/wavespeed.jpg"]}})
        )
        FakeAsyncClient.set_route(
            "GET", "http://cdn/wavespeed.jpg",
            FakeResponse(200, content=FAKE_JPEG)
        )
        client = bridge.WavespeedClient()
        image = await client.generate("test prompt", "", [], GEN_PARAMS)
        assert_valid_image_bytes(self, image, FAKE_JPEG)
        assert_is_real_image(self, image)

    # --- FAL ---

    async def test_fal_returns_valid_jpeg(self):
        """FAL returns JPEG bytes from direct images response."""
        FakeAsyncClient.set_route(
            "POST", bridge.Config.FAL_ENDPOINT,
            FakeResponse(200, {"images": [{"url": "http://cdn/fal.jpg"}]})
        )
        FakeAsyncClient.set_route(
            "GET", "http://cdn/fal.jpg",
            FakeResponse(200, content=FAKE_JPEG)
        )
        client = bridge.FALClient("test-key")
        image = await client.generate("test prompt", "", [], GEN_PARAMS)
        assert_valid_image_bytes(self, image, FAKE_JPEG)
        assert_is_real_image(self, image)

    # --- Together AI (dict fallback path) ---

    async def test_together_dict_returns_valid_jpeg(self):
        """Together AI dict response returns JPEG bytes."""
        class FakeTogetherSDK:
            class images:
                @staticmethod
                def generate(**kwargs):
                    return {"result": {"images": [{"url": "http://cdn/together.jpg"}]}}

        FakeAsyncClient.set_route(
            "GET", "http://cdn/together.jpg",
            FakeResponse(200, content=FAKE_JPEG)
        )
        client = bridge.TogetherAIClient.__new__(bridge.TogetherAIClient)
        client.client = FakeTogetherSDK()
        image = await client.generate("test prompt", "", [], GEN_PARAMS)
        assert_valid_image_bytes(self, image, FAKE_JPEG)
        assert_is_real_image(self, image)

    # --- Together AI (SDK object path — the REAL production path) ---

    async def test_together_sdk_object_url_returns_valid_jpeg(self):
        """Together AI SDK response object with .data[0].url returns JPEG bytes.

        This tests the REAL production path. The SDK returns an ImageResponse
        object (not a dict), which has .data = [ImageChoice(url=..., b64_json=...)].
        """
        class FakeTogetherSDK:
            class images:
                @staticmethod
                def generate(**kwargs):
                    return FakeImageResponse([
                        FakeImageChoice(url="http://cdn/together_sdk.jpg", b64_json=None)
                    ])

        FakeAsyncClient.set_route(
            "GET", "http://cdn/together_sdk.jpg",
            FakeResponse(200, content=FAKE_JPEG)
        )
        client = bridge.TogetherAIClient.__new__(bridge.TogetherAIClient)
        client.client = FakeTogetherSDK()
        image = await client.generate("test prompt", "", [], GEN_PARAMS)
        assert_valid_image_bytes(self, image, FAKE_JPEG)
        assert_is_real_image(self, image)

    async def test_together_sdk_object_b64_returns_valid_png(self):
        """Together AI SDK response with url=None and b64_json returns PNG bytes.

        When Together returns base64 instead of a URL, the code should use
        b64_json instead of trying to download from a None URL.
        """
        encoded = base64.b64encode(FAKE_PNG).decode('utf-8')

        class FakeTogetherSDK:
            class images:
                @staticmethod
                def generate(**kwargs):
                    return FakeImageResponse([
                        FakeImageChoice(url=None, b64_json=encoded)
                    ])

        client = bridge.TogetherAIClient.__new__(bridge.TogetherAIClient)
        client.client = FakeTogetherSDK()
        image = await client.generate("test prompt", "", [], GEN_PARAMS)
        assert_valid_image_bytes(self, image, FAKE_PNG)
        assert_is_real_image(self, image)

    # --- HF ZeroGPU (base64 path) ---

    async def test_hf_base64_returns_valid_webp(self):
        """HF ZeroGPU base64 response returns WEBP bytes."""
        encoded = base64.b64encode(FAKE_WEBP).decode('utf-8')

        class FakeGradio:
            @staticmethod
            def predict(**kwargs):
                return {"data": [{"b64_json": encoded}]}

        client = bridge.HFZeroGPUClient.__new__(bridge.HFZeroGPUClient)
        client.client = FakeGradio()
        image = await client.generate("test prompt", "", [], GEN_PARAMS)
        assert_valid_image_bytes(self, image, FAKE_WEBP)
        assert_is_real_image(self, image)

    # --- HF ZeroGPU (URL path) ---

    async def test_hf_url_returns_valid_jpeg(self):
        """HF ZeroGPU URL response returns JPEG bytes."""
        class FakeGradio:
            @staticmethod
            def predict(**kwargs):
                return ({"url": "http://cdn/hf.jpg"}, '{}')

        FakeAsyncClient.set_route(
            "GET", "http://cdn/hf.jpg",
            FakeResponse(200, content=FAKE_JPEG)
        )
        client = bridge.HFZeroGPUClient.__new__(bridge.HFZeroGPUClient)
        client.client = FakeGradio()
        image = await client.generate("test prompt", "", [], GEN_PARAMS)
        assert_valid_image_bytes(self, image, FAKE_JPEG)
        assert_is_real_image(self, image)


# ===========================================================================
# TEST: Together AI error handling (the bug)
# ===========================================================================

class TestTogetherErrorHandling(unittest.IsolatedAsyncioTestCase):
    """Tests that expose the Together AI bugs."""

    def setUp(self):
        self.original_async_client = bridge.httpx.AsyncClient
        bridge.httpx.AsyncClient = FakeAsyncClient
        FakeAsyncClient.routes = {"POST": {}, "GET": {}}

    def tearDown(self):
        bridge.httpx.AsyncClient = self.original_async_client

    async def test_together_cdn_403_should_raise(self):
        """BUG: When Together CDN returns 403, bridge should raise — not return HTML as image.

        Currently (before fix), the code downloads the error page body and
        returns it as 'image bytes' because raise_for_status() is missing.
        """
        class FakeTogetherSDK:
            class images:
                @staticmethod
                def generate(**kwargs):
                    return FakeImageResponse([
                        FakeImageChoice(url="http://cdn/expired.jpg", b64_json=None)
                    ])

        FakeAsyncClient.set_route(
            "GET", "http://cdn/expired.jpg",
            FakeResponse(403, content=FAKE_HTML_ERROR, text="Forbidden")
        )
        client = bridge.TogetherAIClient.__new__(bridge.TogetherAIClient)
        client.client = FakeTogetherSDK()

        # This SHOULD raise an exception so the fallback chain tries the next provider.
        # Before the fix, this silently returns the HTML error body as "image bytes".
        with self.assertRaises(Exception, msg="Together should raise on HTTP 403, not return HTML as image"):
            await client.generate("test prompt", "", [], GEN_PARAMS)

    async def test_together_sdk_url_none_b64_present_should_use_b64(self):
        """BUG: When SDK returns url=None + b64_json=<data>, should decode b64.

        The hasattr check `hasattr(response.data[0], 'url')` returns True even
        when url IS None (attribute exists, value is None). So the code sets
        image_url = None and never checks b64_json. Then it raises RuntimeError.
        """
        encoded = base64.b64encode(FAKE_JPEG).decode('utf-8')

        class FakeTogetherSDK:
            class images:
                @staticmethod
                def generate(**kwargs):
                    return FakeImageResponse([
                        FakeImageChoice(url=None, b64_json=encoded)
                    ])

        client = bridge.TogetherAIClient.__new__(bridge.TogetherAIClient)
        client.client = FakeTogetherSDK()

        # This SHOULD return valid JPEG bytes decoded from base64.
        # Before the fix, this raises RuntimeError("Together returned no image URL")
        # because the hasattr check enters the wrong branch.
        try:
            image = await client.generate("test prompt", "", [], GEN_PARAMS)
            assert_valid_image_bytes(self, image, FAKE_JPEG)
        except RuntimeError as e:
            if "no image URL" in str(e):
                self.fail(
                    "Together raised 'no image URL' when b64_json was available. "
                    "The hasattr(response.data[0], 'url') check returns True even "
                    "when url is None, preventing the b64_json branch from executing."
                )
            raise


# ===========================================================================
# TEST: Base64 encoding matches plugin expectations
# ===========================================================================

class TestBase64PluginContract(unittest.TestCase):
    """Verify that base64 encoding produces output the JS plugin can consume."""

    def test_jpeg_base64_is_truthy(self):
        """Plugin line 555: `if (base64Image)` — must be truthy string."""
        b64 = base64.b64encode(FAKE_JPEG).decode('utf-8')
        self.assertTrue(len(b64) > 0, "Base64 JPEG must be non-empty (truthy in JS)")

    def test_png_base64_is_truthy(self):
        b64 = base64.b64encode(FAKE_PNG).decode('utf-8')
        self.assertTrue(len(b64) > 0, "Base64 PNG must be non-empty (truthy in JS)")

    def test_base64_is_pure_ascii(self):
        """Plugin uses template literal: `data:image/png;base64,${base64Image}`
        — no special chars that would break the data URI."""
        b64 = base64.b64encode(FAKE_JPEG).decode('utf-8')
        self.assertTrue(b64.isascii(), "Base64 must be pure ASCII for data URI")
        # No whitespace, newlines, or special chars
        for char in b64:
            self.assertIn(char, 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')

    def test_empty_bytes_produces_empty_b64(self):
        """If a provider returns empty bytes, the base64 would be empty string.
        JS: `if ("")` → false → displayImage never called → silent failure."""
        b64 = base64.b64encode(b"").decode('utf-8')
        self.assertEqual(b64, "", "Empty bytes → empty string → falsy in JS → no image displayed")

    def test_html_error_base64_roundtrip(self):
        """HTML error page base64-encodes fine but is NOT a valid image.
        This is what happens with the Together CDN 403 bug."""
        b64 = base64.b64encode(FAKE_HTML_ERROR).decode('utf-8')
        # It IS a truthy string, so the plugin WOULD try to display it
        self.assertTrue(len(b64) > 0)
        # But the decoded content is HTML, not an image
        decoded = base64.b64decode(b64)
        self.assertTrue(decoded.startswith(b'<!DOCTYPE'), "This is HTML, not image data")
        # Browser would create <img src="data:image/png;base64,...HTML...">
        # which fails to render — the user sees nothing


if __name__ == "__main__":
    unittest.main()
