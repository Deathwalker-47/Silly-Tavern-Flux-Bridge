r"""
Tests for the Immich integration and the auth gate on the credit-spending endpoints.

Two things matter here beyond the happy path:

* ``/edit`` and friends bill Runware per call, so the token gate must actually
  reject, and must reject *before* any provider call is made.
* The bridge is not guaranteed to keep its route to Immich. Every Immich-backed
  endpoint has to fail as a clean 503 the client can fall back from, and an edit
  that succeeds must never be thrown away just because the upload afterwards failed.
"""
import asyncio
import base64
import json
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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

    class JSONResponse:
        """Holds either an object or an array, so ``dict(r)`` and ``list(r)`` both work."""
        def __init__(self, content=None, **kw):
            self.content = content if content is not None else {}

        def __iter__(self):
            return iter(self.content)

        def __getitem__(self, key):
            return self.content[key]

        def __setitem__(self, key, value):
            self.content[key] = value

        def keys(self):
            return self.content.keys()

    responses_mod.JSONResponse = JSONResponse

    pydantic_mod = types.ModuleType("pydantic")
    class BaseModel:
        def __init__(self, **kw):
            for key, value in kw.items():
                setattr(self, key, value)
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
        async def request(self, *a, **kw): raise NotImplementedError
    httpx_mod.AsyncClient = AsyncClient

    sse_root = types.ModuleType("sse_starlette")
    sse_mod = types.ModuleType("sse_starlette.sse")
    class EventSourceResponse:
        def __init__(self, gen): self.generator = gen
    sse_mod.EventSourceResponse = EventSourceResponse

    try:
        import PIL  # noqa: F401
    except ImportError:
        pil_mod = types.ModuleType("PIL")
        pil_mod.Image = object
        pil_mod.ImageDraw = object
        pil_mod.ImageFilter = object
        pil_mod.ImageOps = object
        sys.modules.setdefault("PIL", pil_mod)

    sys.modules.setdefault("fastapi", fastapi)
    sys.modules.setdefault("fastapi.middleware.cors", cors_mod)
    sys.modules.setdefault("fastapi.responses", responses_mod)
    sys.modules.setdefault("pydantic", pydantic_mod)
    sys.modules.setdefault("uvicorn", uvicorn_mod)
    sys.modules.setdefault("httpx", httpx_mod)
    sys.modules.setdefault("sse_starlette", sse_root)
    sys.modules.setdefault("sse_starlette.sse", sse_mod)


install_test_stubs()
import flux_lora_bridge as bridge

FAKE_JPEG = b'\xff\xd8\xff\xe0' + b'\x00' * 100


def run(coro):
    return asyncio.run(coro)


class FakeHeaders(dict):
    """Case-insensitive header mapping, like Starlette's."""
    def get(self, key, default=None):
        return super().get(key.lower(), default)


class FakeRequest:
    def __init__(self, headers=None, body=None):
        self.headers = FakeHeaders({k.lower(): v for k, v in (headers or {}).items()})
        self._body = body or {}

    async def json(self):
        return self._body


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", text="", headers=None):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.text = text or (json.dumps(json_data) if json_data is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeImmichHTTP:
    """Routes Immich API calls by (method, path suffix) and records the calls."""

    def __init__(self, routes, runware=None):
        self.routes = routes
        self.runware = list(runware or [])
        self.calls = []
        outer = self

        class AsyncClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

            async def request(self, method, url, headers=None, **kw):
                outer.calls.append({"method": method, "url": url, "headers": headers, **kw})
                for (route_method, suffix), response in outer.routes.items():
                    if method == route_method and url.endswith(suffix):
                        if isinstance(response, Exception):
                            raise response
                        return response
                return FakeResponse(404, text="no route")

            async def post(self, url, json=None, headers=None):
                outer.calls.append({"method": "POST", "url": url, "json": json})
                return outer.runware.pop(0)

            async def get(self, url, headers=None):
                outer.calls.append({"method": "GET", "url": url})
                return outer.runware.pop(0)

        self.AsyncClient = AsyncClient


class HTTPPatch:
    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        self.original = bridge.httpx
        bridge.httpx = self.fake
        return self.fake

    def __exit__(self, *exc):
        bridge.httpx = self.original
        return False


class ConfigPatch:
    """Temporarily override Config attributes."""
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: getattr(bridge.Config, k) for k in self.values}
        for key, value in self.values.items():
            setattr(bridge.Config, key, value)
        return bridge.Config

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            setattr(bridge.Config, key, value)
        return False


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------

class EditTokenTests(unittest.TestCase):
    def test_no_token_configured_leaves_endpoint_open(self):
        with ConfigPatch(EDIT_API_TOKEN=""):
            bridge._check_edit_token(FakeRequest())  # must not raise

    def test_bearer_token_accepted(self):
        with ConfigPatch(EDIT_API_TOKEN="s3cret"):
            bridge._check_edit_token(FakeRequest({"Authorization": "Bearer s3cret"}))

    def test_x_edit_token_header_accepted(self):
        with ConfigPatch(EDIT_API_TOKEN="s3cret"):
            bridge._check_edit_token(FakeRequest({"X-Edit-Token": "s3cret"}))

    def test_missing_token_rejected(self):
        with ConfigPatch(EDIT_API_TOKEN="s3cret"):
            with self.assertRaises(bridge.HTTPException) as ctx:
                bridge._check_edit_token(FakeRequest())
        self.assertEqual(ctx.exception.status_code, 401)

    def test_wrong_token_rejected(self):
        with ConfigPatch(EDIT_API_TOKEN="s3cret"):
            with self.assertRaises(bridge.HTTPException) as ctx:
                bridge._check_edit_token(FakeRequest({"Authorization": "Bearer nope"}))
        self.assertEqual(ctx.exception.status_code, 401)

    def test_internal_call_without_request_is_allowed(self):
        # /edit/immich checks the token itself, then reuses the /edit path.
        with ConfigPatch(EDIT_API_TOKEN="s3cret"):
            bridge._check_edit_token(None)

    def test_rejection_happens_before_any_provider_call(self):
        fake = FakeImmichHTTP({})
        with ConfigPatch(EDIT_API_TOKEN="s3cret"), HTTPPatch(fake):
            with self.assertRaises(bridge.HTTPException):
                run(bridge.edit_image(
                    bridge.EditRequest(image="data:image/jpeg;base64,AAAA", instruction="x"),
                    FakeRequest(),
                ))
        self.assertEqual(fake.calls, [], "must not spend credits on an unauthorised call")


# ---------------------------------------------------------------------------
# Immich client
# ---------------------------------------------------------------------------

class ImmichClientTests(unittest.TestCase):
    def setUp(self):
        self.client = bridge.ImmichClient("https://photos.example.com", "immich-key")

    def test_unconfigured_client_raises_immich_unavailable(self):
        client = bridge.ImmichClient("", "")
        self.assertFalse(client.configured)
        with self.assertRaises(bridge.ImmichUnavailable):
            run(client.list_albums())

    def test_api_key_is_sent_as_x_api_key(self):
        fake = FakeImmichHTTP({("GET", "/api/albums"): FakeResponse(200, [])})
        with HTTPPatch(fake):
            run(self.client.list_albums())
        self.assertEqual(fake.calls[0]["headers"]["x-api-key"], "immich-key")

    def test_network_failure_becomes_immich_unavailable(self):
        fake = FakeImmichHTTP({("GET", "/api/albums"): RuntimeError("connection refused")})
        with HTTPPatch(fake):
            with self.assertRaises(bridge.ImmichUnavailable) as ctx:
                run(self.client.list_albums())
        self.assertIn("Cannot reach Immich", str(ctx.exception))

    def test_http_error_becomes_immich_unavailable(self):
        fake = FakeImmichHTTP({("GET", "/api/albums"): FakeResponse(401, text="unauthorized")})
        with HTTPPatch(fake):
            with self.assertRaises(bridge.ImmichUnavailable) as ctx:
                run(self.client.list_albums())
        self.assertIn("401", str(ctx.exception))

    def test_upload_sends_required_device_fields(self):
        fake = FakeImmichHTTP({("POST", "/api/assets"): FakeResponse(200, {"id": "asset-1"})})
        with HTTPPatch(fake):
            asset_id = run(self.client.upload_asset(FAKE_JPEG, "edit.jpg"))
        self.assertEqual(asset_id, "asset-1")
        data = fake.calls[0]["data"]
        # Immich rejects the upload without both of these.
        self.assertTrue(data["deviceAssetId"])
        self.assertTrue(data["deviceId"])
        self.assertIn("fileCreatedAt", data)
        self.assertIn("fileModifiedAt", data)
        self.assertEqual(fake.calls[0]["files"]["assetData"][0], "edit.jpg")

    def test_upload_without_id_in_response_raises(self):
        fake = FakeImmichHTTP({("POST", "/api/assets"): FakeResponse(200, {})})
        with HTTPPatch(fake):
            with self.assertRaises(bridge.ImmichUnavailable):
                run(self.client.upload_asset(FAKE_JPEG, "edit.jpg"))

    def test_each_upload_gets_a_distinct_device_asset_id(self):
        fake = FakeImmichHTTP({("POST", "/api/assets"): FakeResponse(200, {"id": "a"})})
        with HTTPPatch(fake):
            run(self.client.upload_asset(FAKE_JPEG, "one.jpg"))
        fake2 = FakeImmichHTTP({("POST", "/api/assets"): FakeResponse(200, {"id": "b"})})
        with HTTPPatch(fake2):
            run(self.client.upload_asset(FAKE_JPEG, "two.jpg"))
        self.assertNotEqual(fake.calls[0]["data"]["deviceAssetId"],
                            fake2.calls[0]["data"]["deviceAssetId"])

    def test_ensure_album_reuses_existing_album(self):
        fake = FakeImmichHTTP({
            ("GET", "/api/albums"): FakeResponse(200, [{"id": "album-9", "albumName": "AI Edits"}]),
        })
        with HTTPPatch(fake):
            album_id = run(self.client.ensure_album("AI Edits"))
        self.assertEqual(album_id, "album-9")
        self.assertEqual(len(fake.calls), 1, "should not create an album that already exists")

    def test_ensure_album_creates_when_missing(self):
        fake = FakeImmichHTTP({
            ("GET", "/api/albums"): FakeResponse(200, [{"id": "x", "albumName": "Holidays"}]),
            ("POST", "/api/albums"): FakeResponse(200, {"id": "album-new"}),
        })
        with HTTPPatch(fake):
            album_id = run(self.client.ensure_album("AI Edits"))
        self.assertEqual(album_id, "album-new")
        self.assertEqual(fake.calls[1]["json"]["albumName"], "AI Edits")

    def test_add_to_album_uses_ids_body(self):
        fake = FakeImmichHTTP({("PUT", "/api/albums/album-1/assets"): FakeResponse(200, [])})
        with HTTPPatch(fake):
            run(self.client.add_to_album("album-1", ["asset-1"]))
        self.assertEqual(fake.calls[0]["json"], {"ids": ["asset-1"]})

    def test_save_edit_uploads_then_files_in_album(self):
        fake = FakeImmichHTTP({
            ("POST", "/api/assets"): FakeResponse(200, {"id": "asset-7"}),
            ("GET", "/api/albums"): FakeResponse(200, [{"id": "album-3", "albumName": "AI Edits"}]),
            ("PUT", "/api/albums/album-3/assets"): FakeResponse(200, []),
        })
        with ConfigPatch(IMMICH_EDIT_ALBUM="AI Edits"), HTTPPatch(fake):
            result = run(self.client.save_edit(FAKE_JPEG, "edit.jpg"))
        self.assertEqual(result["asset_id"], "asset-7")
        self.assertEqual(result["album_id"], "album-3")

    def test_album_failure_still_returns_the_uploaded_asset(self):
        # The image is already in the library; losing the album is not worth an error.
        fake = FakeImmichHTTP({
            ("POST", "/api/assets"): FakeResponse(200, {"id": "asset-7"}),
            ("GET", "/api/albums"): FakeResponse(500, text="boom"),
        })
        with ConfigPatch(IMMICH_EDIT_ALBUM="AI Edits"), HTTPPatch(fake):
            result = run(self.client.save_edit(FAKE_JPEG, "edit.jpg"))
        self.assertEqual(result["asset_id"], "asset-7")
        self.assertIsNone(result["album_id"])

    def test_thumbnail_request_passes_size(self):
        fake = FakeImmichHTTP({
            ("GET", "/api/assets/a1/thumbnail"): FakeResponse(200, content=FAKE_JPEG,
                                                             headers={"content-type": "image/jpeg"}),
        })
        with HTTPPatch(fake):
            data, content_type = run(self.client.fetch_media("a1", "thumbnail", "preview"))
        self.assertEqual(data, FAKE_JPEG)
        self.assertEqual(content_type, "image/jpeg")
        self.assertEqual(fake.calls[0]["params"], {"size": "preview"})

    def test_original_request_sends_no_size(self):
        fake = FakeImmichHTTP({
            ("GET", "/api/assets/a1/original"): FakeResponse(200, content=FAKE_JPEG, headers={}),
        })
        with HTTPPatch(fake):
            run(self.client.fetch_media("a1", "original"))
        self.assertIsNone(fake.calls[0]["params"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class ImmichEndpointTests(unittest.TestCase):
    def test_status_reports_unconfigured_without_raising(self):
        original = bridge.immich_client
        bridge.immich_client = bridge.ImmichClient("", "")
        try:
            result = run(bridge.immich_status())
        finally:
            bridge.immich_client = original
        self.assertFalse(dict(result)["configured"])

    def test_albums_returns_503_when_immich_is_down(self):
        original = bridge.immich_client
        bridge.immich_client = bridge.ImmichClient("https://photos.example.com", "k")
        fake = FakeImmichHTTP({("GET", "/api/albums"): RuntimeError("no route to host")})
        try:
            with HTTPPatch(fake):
                with self.assertRaises(bridge.HTTPException) as ctx:
                    run(bridge.immich_albums())
        finally:
            bridge.immich_client = original
        self.assertEqual(ctx.exception.status_code, 503)

    def test_albums_are_trimmed_to_picker_fields(self):
        original = bridge.immich_client
        bridge.immich_client = bridge.ImmichClient("https://photos.example.com", "k")
        fake = FakeImmichHTTP({("GET", "/api/albums"): FakeResponse(200, [
            {"id": "a1", "albumName": "Trip", "assetCount": 12, "albumThumbnailAssetId": "t1"},
        ])})
        try:
            with HTTPPatch(fake):
                result = run(bridge.immich_albums())
        finally:
            bridge.immich_client = original
        self.assertEqual(list(result)[0], {"id": "a1", "name": "Trip", "count": 12,
                                           "thumbnail_asset_id": "t1"})

    def test_album_assets_exclude_videos(self):
        original = bridge.immich_client
        bridge.immich_client = bridge.ImmichClient("https://photos.example.com", "k")
        fake = FakeImmichHTTP({("GET", "/api/albums/a1"): FakeResponse(200, {
            "id": "a1", "albumName": "Trip",
            "assets": [{"id": "i1", "type": "IMAGE", "originalFileName": "a.jpg"},
                       {"id": "v1", "type": "VIDEO", "originalFileName": "b.mp4"}],
        })})
        try:
            with HTTPPatch(fake):
                result = run(bridge.immich_album_assets("a1"))
        finally:
            bridge.immich_client = original
        assets = dict(result)["assets"]
        self.assertEqual([a["id"] for a in assets], ["i1"])


if __name__ == "__main__":
    unittest.main()
