r"""
Tests for the FLUX.1 Kontext instruction-edit path (POST /edit).

The payload shape is the risky part: Runware's model docs put the input images at
``inputs.referenceImages`` and the BFL-hosted variants (bfl:3@1 / bfl:4@1) reject
``steps``/``CFGScale`` entirely, while the open-weight dev model (runware:106@1)
accepts both. Kontext also only renders a fixed set of dimensions, so arbitrary
phone photos have to be snapped to the nearest one. These tests pin all of that
down without spending Runware credits.
"""
import asyncio
import base64
from io import BytesIO
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Stubs (same pattern as the sibling test modules)
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

    # Only stub PIL when Pillow is genuinely missing (see sibling test modules).
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


try:
    from PIL import Image as _RealImage
    HAS_PILLOW = hasattr(_RealImage, "open")
except Exception:
    HAS_PILLOW = False


FAKE_JPEG = b'\xff\xd8\xff\xe0' + b'\x00' * 100


# ---------------------------------------------------------------------------
# HTTP fake that records what was actually posted
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
            raise RuntimeError(f"HTTP {self.status_code}")


class RecordingHTTP:
    """Stands in for the httpx module and records every posted task payload."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []
        self.gets = []
        outer = self

        class AsyncClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

            async def post(self, url, json=None, headers=None):
                outer.posts.append(json)
                return outer.responses.pop(0)

            async def get(self, url, headers=None):
                outer.gets.append(url)
                return outer.responses.pop(0)

        self.AsyncClient = AsyncClient

    @property
    def tasks(self):
        """The single task object out of each posted array."""
        return [payload[0] for payload in self.posts]


class HTTPPatch:
    """Swap bridge.httpx for a recorder regardless of whether real httpx is installed."""

    def __init__(self, testcase, responses):
        self.testcase = testcase
        self.recorder = RecordingHTTP(responses)

    def __enter__(self):
        self.original = bridge.httpx
        bridge.httpx = self.recorder
        return self.recorder

    def __exit__(self, *exc):
        bridge.httpx = self.original
        return False


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Dimension snapping
# ---------------------------------------------------------------------------

class KontextSizeTests(unittest.TestCase):
    def test_every_bucket_is_supported_by_the_api(self):
        # Guards against a typo creeping into the table Runware validates against.
        for width, height in bridge.KONTEXT_DIMENSIONS:
            self.assertTrue(672 <= width <= 1568, f"{width}x{height}")
            self.assertTrue(672 <= height <= 1568, f"{width}x{height}")

    def test_landscape_phone_photo_snaps_to_4_3(self):
        self.assertEqual(bridge._closest_kontext_size(4032, 3024), (1184, 880))

    def test_portrait_phone_photo_snaps_to_3_4(self):
        self.assertEqual(bridge._closest_kontext_size(3024, 4032), (880, 1184))

    def test_widescreen_snaps_to_16_9(self):
        self.assertEqual(bridge._closest_kontext_size(1920, 1080), (1392, 752))

    def test_square_stays_square(self):
        self.assertEqual(bridge._closest_kontext_size(2000, 2000), (1024, 1024))

    def test_snapped_result_is_always_a_supported_pair(self):
        for source in [(4032, 3024), (3024, 4032), (1920, 1080), (1080, 1920), (2000, 2000), (5000, 1000)]:
            self.assertIn(bridge._closest_kontext_size(*source), bridge.KONTEXT_DIMENSIONS)

    def test_missing_dimensions_fall_back_to_square(self):
        self.assertEqual(bridge._closest_kontext_size(None, None), (1024, 1024))
        self.assertEqual(bridge._closest_kontext_size(0, 0), (1024, 1024))


# ---------------------------------------------------------------------------
# Model capability detection
# ---------------------------------------------------------------------------

class KontextModelTests(unittest.TestCase):
    def test_bfl_hosted_models_do_not_take_sampling_params(self):
        self.assertFalse(bridge._kontext_supports_sampling("bfl:3@1"))  # pro
        self.assertFalse(bridge._kontext_supports_sampling("bfl:4@1"))  # max

    def test_open_weight_dev_model_takes_sampling_params(self):
        self.assertTrue(bridge._kontext_supports_sampling("runware:106@1"))

    def test_default_model_is_a_known_kontext_air_id(self):
        self.assertIn(bridge.Config.KONTEXT_MODEL, ("bfl:3@1", "bfl:4@1", "runware:106@1"))


class RunwareUUIDTests(unittest.TestCase):
    def test_uuid_is_recognised_and_passed_through(self):
        self.assertTrue(bridge._is_runware_image_uuid("b7f4e1a2-3c5d-4e6f-8a9b-0c1d2e3f4a5b"))

    def test_data_uri_and_url_are_not_uuids(self):
        self.assertFalse(bridge._is_runware_image_uuid("data:image/jpeg;base64,AAAA"))
        self.assertFalse(bridge._is_runware_image_uuid("https://example.com/a.jpg"))
        self.assertFalse(bridge._is_runware_image_uuid(""))


# ---------------------------------------------------------------------------
# Request payload shape — the part most likely to be rejected by the API
# ---------------------------------------------------------------------------

class KontextPayloadTests(unittest.TestCase):
    def setUp(self):
        self.client = bridge.RunwareClient("test-key")

    def _edit(self, responses, model="bfl:3@1", params=None, images=None):
        merged = {"width": 1184, "height": 880, "model": model}
        merged.update(params or {})
        with HTTPPatch(self, responses) as http:
            result = run(self.client.edit("make the sky orange", images or ["uuid-1"], merged))
        return result, http

    def test_reference_images_are_nested_under_inputs(self):
        _result, http = self._edit([FakeResponse(200, {"data": [{"imageBase64Data": base64.b64encode(FAKE_JPEG).decode()}]})])
        task = http.tasks[0]
        self.assertEqual(task["inputs"], {"referenceImages": ["uuid-1"]})
        self.assertNotIn("referenceImages", task, "must not sit at the root by default")

    def test_core_fields_match_the_documented_shape(self):
        _result, http = self._edit([FakeResponse(200, {"data": [{"imageBase64Data": base64.b64encode(FAKE_JPEG).decode()}]})])
        task = http.tasks[0]
        self.assertEqual(task["taskType"], "imageInference")
        self.assertEqual(task["model"], "bfl:3@1")
        self.assertEqual(task["positivePrompt"], "make the sky orange")
        self.assertEqual((task["width"], task["height"]), (1184, 880))
        self.assertEqual(task["numberResults"], 1)
        self.assertEqual(task["outputFormat"], "JPG")
        self.assertIn("taskUUID", task)

    def test_hosted_model_omits_steps_and_cfg(self):
        _result, http = self._edit(
            [FakeResponse(200, {"data": [{"imageBase64Data": base64.b64encode(FAKE_JPEG).decode()}]})],
            model="bfl:3@1",
            params={"steps": 30, "cfg_scale": 3.0},
        )
        task = http.tasks[0]
        self.assertNotIn("steps", task)
        self.assertNotIn("CFGScale", task)

    def test_dev_model_sends_steps_and_cfg(self):
        _result, http = self._edit(
            [FakeResponse(200, {"data": [{"imageBase64Data": base64.b64encode(FAKE_JPEG).decode()}]})],
            model="runware:106@1",
            params={"steps": 30, "cfg_scale": 3.0},
        )
        task = http.tasks[0]
        self.assertEqual(task["steps"], 30)
        self.assertEqual(task["CFGScale"], 3.0)

    def test_more_than_two_references_are_trimmed(self):
        # pro/max cap out at 2 reference images; sending 3 would be a hard 400.
        _result, http = self._edit(
            [FakeResponse(200, {"data": [{"imageBase64Data": base64.b64encode(FAKE_JPEG).decode()}]})],
            images=["a", "b", "c"],
        )
        self.assertEqual(http.tasks[0]["inputs"]["referenceImages"], ["a", "b"])

    def test_arbitrary_size_is_snapped_before_sending(self):
        _result, http = self._edit(
            [FakeResponse(200, {"data": [{"imageBase64Data": base64.b64encode(FAKE_JPEG).decode()}]})],
            params={"width": 4032, "height": 3024},
        )
        task = http.tasks[0]
        self.assertIn((task["width"], task["height"]), bridge.KONTEXT_DIMENSIONS)

    def test_instruction_is_passed_verbatim(self):
        instruction = "Remove the car, keep everything else identical"
        with HTTPPatch(self, [FakeResponse(200, {"data": [{"imageBase64Data": base64.b64encode(FAKE_JPEG).decode()}]})]) as http:
            run(self.client.edit(instruction, ["uuid-1"], {"width": 1024, "height": 1024}))
        self.assertEqual(http.tasks[0]["positivePrompt"], instruction)


# ---------------------------------------------------------------------------
# Failure handling and the reference-placement fallback
# ---------------------------------------------------------------------------

class KontextFallbackTests(unittest.TestCase):
    def setUp(self):
        self.client = bridge.RunwareClient("test-key")
        self.params = {"width": 1024, "height": 1024}

    def test_retries_at_root_level_when_inputs_shape_is_rejected(self):
        responses = [
            FakeResponse(400, text='{"errors":[{"message":"unknown parameter inputs.referenceImages"}]}'),
            FakeResponse(200, {"data": [{"imageBase64Data": base64.b64encode(FAKE_JPEG).decode()}]}),
        ]
        with HTTPPatch(self, responses) as http:
            result = run(self.client.edit("edit it", ["uuid-1"], self.params))
        self.assertEqual(result, FAKE_JPEG)
        self.assertEqual(len(http.tasks), 2)
        self.assertIn("inputs", http.tasks[0])
        self.assertEqual(http.tasks[1]["referenceImages"], ["uuid-1"])
        self.assertNotIn("inputs", http.tasks[1])

    def test_unrelated_error_is_not_retried(self):
        responses = [FakeResponse(400, text='{"errors":[{"message":"insufficient credits"}]}')]
        with HTTPPatch(self, responses) as http:
            with self.assertRaises(Exception) as ctx:
                run(self.client.edit("edit it", ["uuid-1"], self.params))
        self.assertEqual(len(http.tasks), 1, "a credit error must not burn a second call")
        self.assertIn("insufficient credits", str(ctx.exception))

    def test_task_errors_inside_a_200_body_are_surfaced(self):
        responses = [FakeResponse(200, {"errors": [{"message": "content moderated", "code": "moderation"}]})]
        with HTTPPatch(self, responses):
            with self.assertRaises(Exception) as ctx:
                run(self.client.edit("edit it", ["uuid-1"], self.params))
        self.assertIn("content moderated", str(ctx.exception))

    def test_missing_instruction_is_rejected_before_any_call(self):
        with HTTPPatch(self, []) as http:
            with self.assertRaises(ValueError):
                run(self.client.edit("   ", ["uuid-1"], self.params))
        self.assertEqual(http.posts, [])

    def test_missing_image_is_rejected_before_any_call(self):
        with HTTPPatch(self, []) as http:
            with self.assertRaises(ValueError):
                run(self.client.edit("edit it", [], self.params))
        self.assertEqual(http.posts, [])

    def test_missing_api_key_is_rejected(self):
        client = bridge.RunwareClient("")
        with self.assertRaises(ValueError):
            run(client.edit("edit it", ["uuid-1"], self.params))


# ---------------------------------------------------------------------------
# imageUpload
# ---------------------------------------------------------------------------

class ImageUploadTests(unittest.TestCase):
    def setUp(self):
        self.client = bridge.RunwareClient("test-key")

    def test_upload_returns_image_uuid(self):
        response = FakeResponse(200, {"data": [{"taskType": "imageUpload", "imageUUID": "abc-123"}]})
        with HTTPPatch(self, [response]) as http:
            image_uuid = run(self.client.upload_image(FAKE_JPEG, "image"))
        self.assertEqual(image_uuid, "abc-123")
        task = http.tasks[0]
        self.assertEqual(task["taskType"], "imageUpload")
        self.assertTrue(task["image"].startswith("data:image/jpeg;base64,"))

    def test_upload_without_uuid_raises(self):
        with HTTPPatch(self, [FakeResponse(200, {"data": [{}]})]):
            with self.assertRaises(Exception):
                run(self.client.upload_image(FAKE_JPEG, "image"))

    def test_extract_image_uuid_walks_nested_payloads(self):
        self.assertEqual(bridge._extract_image_uuid({"data": [{"imageUUID": "x-1"}]}), "x-1")
        self.assertIsNone(bridge._extract_image_uuid({"data": [{"imageURL": "http://x"}]}))


class ResponseParsingTests(unittest.TestCase):
    """Runware answers with imageURL by default, but base64/dataURI outputs are also
    valid — and a task object always carries a taskType string next to the image."""

    def test_base64_output_is_preferred_over_neighbouring_strings(self):
        payload = {"data": [{"taskType": "imageInference",
                             "imageBase64Data": base64.b64encode(FAKE_JPEG).decode()}]}
        result = run(bridge._resolve_image_bytes_from_payload(payload, "RunwareKontext"))
        self.assertEqual(result, FAKE_JPEG)

    def test_data_uri_output_is_decoded(self):
        payload = {"data": [{"taskType": "imageInference",
                             "imageDataURI": "data:image/jpeg;base64," + base64.b64encode(FAKE_JPEG).decode()}]}
        result = run(bridge._resolve_image_bytes_from_payload(payload, "RunwareKontext"))
        self.assertEqual(result, FAKE_JPEG)

    def test_url_output_is_fetched(self):
        payload = {"data": [{"taskType": "imageInference", "imageURL": "https://cdn.example/x.jpg"}]}
        with HTTPPatch(self, [FakeResponse(200, content=FAKE_JPEG)]) as http:
            result = run(bridge._resolve_image_bytes_from_payload(payload, "RunwareKontext"))
        self.assertEqual(result, FAKE_JPEG)
        self.assertEqual(http.gets, ["https://cdn.example/x.jpg"])


class RunwareErrorTextTests(unittest.TestCase):
    def test_clean_payload_reports_no_error(self):
        self.assertIsNone(bridge._runware_error_text({"data": [{"imageURL": "http://x"}]}))

    def test_error_list_is_joined(self):
        text = bridge._runware_error_text({"errors": [{"message": "bad", "parameter": "width"}]})
        self.assertIn("bad", text)
        self.assertIn("width", text)


# ---------------------------------------------------------------------------
# Input normalization (needs real Pillow)
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_PILLOW, "Pillow not installed")
class NormalizeImageTests(unittest.TestCase):
    def _png(self, size, mode="RGBA"):
        buffer = BytesIO()
        _RealImage.new(mode, size, (120, 60, 30, 255)[: 4 if mode == "RGBA" else 3]).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_oversized_photo_is_downscaled_to_max_edge(self):
        data, width, height = bridge._normalize_edit_image(self._png((4032, 3024)), "image")
        self.assertLessEqual(max(width, height), bridge.Config.KONTEXT_MAX_EDGE)
        self.assertEqual(data[:3], b'\xff\xd8\xff', "should be re-encoded as JPEG")

    def test_downscaling_preserves_aspect_ratio(self):
        _data, width, height = bridge._normalize_edit_image(self._png((4032, 3024)), "image")
        self.assertAlmostEqual(width / height, 4032 / 3024, places=2)

    def test_small_image_keeps_its_dimensions(self):
        _data, width, height = bridge._normalize_edit_image(self._png((800, 600)), "image")
        self.assertEqual((width, height), (800, 600))

    def test_alpha_channel_is_flattened(self):
        data, _w, _h = bridge._normalize_edit_image(self._png((400, 400), mode="RGBA"), "image")
        with _RealImage.open(BytesIO(data)) as img:
            self.assertEqual(img.mode, "RGB")

    def test_garbage_input_raises_a_readable_error(self):
        with self.assertRaises(ValueError):
            bridge._normalize_edit_image(b"not an image at all", "image")


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_PILLOW, "Pillow not installed")
class PrepareInputsTests(unittest.TestCase):
    def setUp(self):
        buffer = BytesIO()
        _RealImage.new("RGB", (1600, 1200), (10, 20, 30)).save(buffer, format="JPEG")
        self.data_uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        self.original_upload = bridge.Config.KONTEXT_UPLOAD_IMAGES

    def tearDown(self):
        bridge.Config.KONTEXT_UPLOAD_IMAGES = self.original_upload

    def test_data_uri_is_inlined_when_uploads_are_disabled(self):
        bridge.Config.KONTEXT_UPLOAD_IMAGES = False
        references, width, height = run(bridge._prepare_kontext_inputs(self.data_uri, None, "test"))
        self.assertEqual(len(references), 1)
        self.assertTrue(references[0].startswith("data:image/jpeg;base64,"))
        self.assertEqual((width, height), (1184, 880))  # 4:3 source

    def test_reference_image_becomes_a_second_entry(self):
        bridge.Config.KONTEXT_UPLOAD_IMAGES = False
        references, _w, _h = run(bridge._prepare_kontext_inputs(self.data_uri, self.data_uri, "test"))
        self.assertEqual(len(references), 2)

    def test_existing_uuid_is_passed_through_untouched(self):
        bridge.Config.KONTEXT_UPLOAD_IMAGES = False
        uuid_ref = "b7f4e1a2-3c5d-4e6f-8a9b-0c1d2e3f4a5b"
        references, width, height = run(bridge._prepare_kontext_inputs(uuid_ref, None, "test"))
        self.assertEqual(references, [uuid_ref])
        self.assertEqual((width, height), (1024, 1024))  # dimensions unknown

    def test_upload_failure_falls_back_to_inlining(self):
        bridge.Config.KONTEXT_UPLOAD_IMAGES = True

        async def failing_upload(image_bytes, label="image"):
            raise RuntimeError("upload service down")

        original = bridge.clients["runware"].upload_image
        bridge.clients["runware"].upload_image = failing_upload
        try:
            references, _w, _h = run(bridge._prepare_kontext_inputs(self.data_uri, None, "test"))
        finally:
            bridge.clients["runware"].upload_image = original
        self.assertTrue(references[0].startswith("data:"))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            run(bridge._prepare_kontext_inputs("", None, "test"))

    def test_non_image_input_raises(self):
        with self.assertRaises(ValueError):
            run(bridge._prepare_kontext_inputs("::: not base64 :::", None, "test"))

    def test_oversized_input_is_rejected(self):
        original_limit = bridge.Config.KONTEXT_MAX_INPUT_MB
        bridge.Config.KONTEXT_MAX_INPUT_MB = 0.0001
        try:
            with self.assertRaises(ValueError) as ctx:
                run(bridge._prepare_kontext_inputs(self.data_uri, None, "test"))
            self.assertIn("limit", str(ctx.exception))
        finally:
            bridge.Config.KONTEXT_MAX_INPUT_MB = original_limit


if __name__ == "__main__":
    unittest.main()
