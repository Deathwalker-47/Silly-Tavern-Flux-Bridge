import base64
import sys
import types
import unittest


def install_test_stubs():
    # fastapi + middleware
    fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class FastAPI:
        def add_middleware(self, *args, **kwargs):
            return None
        def get(self, *args, **kwargs):
            return lambda fn: fn
        def post(self, *args, **kwargs):
            return lambda fn: fn
        def on_event(self, *args, **kwargs):
            return lambda fn: fn

    class Request:
        async def json(self):
            return {}

    class APIRouter:
        pass

    fastapi.FastAPI = FastAPI
    fastapi.HTTPException = HTTPException
    fastapi.Request = Request
    fastapi.APIRouter = APIRouter

    cors_mod = types.ModuleType("fastapi.middleware.cors")
    class CORSMiddleware:
        pass
    cors_mod.CORSMiddleware = CORSMiddleware

    responses_mod = types.ModuleType("fastapi.responses")
    class JSONResponse(dict):
        pass
    responses_mod.JSONResponse = JSONResponse

    pydantic_mod = types.ModuleType("pydantic")
    class BaseModel:
        pass
    def Field(default=None, **kwargs):
        return default
    pydantic_mod.BaseModel = BaseModel
    pydantic_mod.Field = Field

    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.run = lambda *a, **k: None

    httpx_mod = types.ModuleType("httpx")
    class AsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        async def post(self, *args, **kwargs):
            raise NotImplementedError
        async def get(self, *args, **kwargs):
            raise NotImplementedError
    httpx_mod.AsyncClient = AsyncClient

    sse_root = types.ModuleType("sse_starlette")
    sse_mod = types.ModuleType("sse_starlette.sse")
    class EventSourceResponse:
        def __init__(self, generator):
            self.generator = generator
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

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @classmethod
    def set_route(cls, method, url, response):
        cls.routes[method][url] = response

    async def post(self, url, json=None, headers=None):
        return self.routes["POST"][url]

    async def get(self, url):
        return self.routes["GET"][url]


class ProviderResponseParsingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_async_client = bridge.httpx.AsyncClient
        bridge.httpx.AsyncClient = FakeAsyncClient
        FakeAsyncClient.routes = {"POST": {}, "GET": {}}

    def tearDown(self):
        bridge.httpx.AsyncClient = self.original_async_client

    async def test_runware_alternate_url_key(self):
        FakeAsyncClient.set_route("POST", bridge.Config.RUNWARE_ENDPOINT, FakeResponse(200, {"data": [{"url": "http://img/runware.jpg"}]}))
        FakeAsyncClient.set_route("GET", "http://img/runware.jpg", FakeResponse(200, content=b"runware"))
        client = bridge.RunwareClient("rk")
        image = await client.generate("p", "", [], {"steps": 1, "cfg_scale": 1, "width": 512, "height": 512})
        self.assertEqual(image, b"runware")

    async def test_pixeldojo_nested_result(self):
        FakeAsyncClient.set_route("POST", bridge.Config.PIXELDOJO_ENDPOINT, FakeResponse(200, {"result": {"image_url": "http://img/pd.jpg"}}))
        FakeAsyncClient.set_route("GET", "http://img/pd.jpg", FakeResponse(200, content=b"pixeldojo"))
        client = bridge.PixelDojoClient("pk")
        image = await client.generate("p", "", [], {"steps": 1, "cfg_scale": 1, "width": 512, "height": 512})
        self.assertEqual(image, b"pixeldojo")

    async def test_wavespeed_outputs_top_level(self):
        bridge.Config.WAVESPEED_API_KEY = "wk"
        FakeAsyncClient.set_route("POST", bridge.Config.WAVESPEED_ENDPOINT, FakeResponse(200, {"outputs": ["http://img/wave.jpg"]}))
        FakeAsyncClient.set_route("GET", "http://img/wave.jpg", FakeResponse(200, content=b"wavespeed"))
        client = bridge.WavespeedClient()
        image = await client.generate("p", "", [], {"steps": 1, "cfg_scale": 1, "width": 512, "height": 512})
        self.assertEqual(image, b"wavespeed")

    async def test_fal_nested_data_images(self):
        FakeAsyncClient.set_route("POST", bridge.Config.FAL_ENDPOINT, FakeResponse(200, {"data": {"images": [{"url": "http://img/fal.jpg"}]}}))
        FakeAsyncClient.set_route("GET", "http://img/fal.jpg", FakeResponse(200, content=b"fal"))
        client = bridge.FALClient("fk")
        image = await client.generate("p", "", [], {"steps": 1, "cfg_scale": 1, "width": 512, "height": 512})
        self.assertEqual(image, b"fal")

    async def test_together_dict_response(self):
        class FakeTogether:
            class images:
                @staticmethod
                def generate(**kwargs):
                    return {"result": {"images": [{"url": "http://img/together.jpg"}]}}

        FakeAsyncClient.set_route("GET", "http://img/together.jpg", FakeResponse(200, content=b"together"))
        client = bridge.TogetherAIClient.__new__(bridge.TogetherAIClient)
        client.client = FakeTogether()
        image = await client.generate("p", "", [], {"steps": 1, "cfg_scale": 1, "width": 512, "height": 512})
        self.assertEqual(image, b"together")

    async def test_hf_dict_base64(self):
        payload = base64.b64encode(b"hf").decode("utf-8")

        class FakeGradio:
            @staticmethod
            def predict(**kwargs):
                return {"data": [{"b64_json": payload}]}

        client = bridge.HFZeroGPUClient.__new__(bridge.HFZeroGPUClient)
        client.client = FakeGradio()
        image = await client.generate("p", "", [], {"steps": 1, "cfg_scale": 1, "width": 512, "height": 512, "seed": 1})
        self.assertEqual(image, b"hf")


if __name__ == "__main__":
    unittest.main()
