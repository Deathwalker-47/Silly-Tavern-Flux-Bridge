"""What each provider is sent and how its replies are read. A wrong field name either fails
the request or, on PixelDojo and NanoGPT, is ignored and billed anyway."""
import asyncio
import base64
import sys
import types
import unittest
import unittest.mock

from media_support import FakeHTTP, FakeResponse, MediaCase, httpx, noisy_jpeg, run, small_jpeg
from media import assets, catalog, db, loras
from media.config import MediaConfig
from media.providers import base
from media.providers.base import RenderRequest
from media.providers.mock import Mock
from media.providers.nanogpt import MAX_REQUEST, NanoGPT
from media.providers.pixeldojo import BUILDERS, PixelDojo, first_output_url
from media.providers.replicate import Replicate, prediction_output
from media.providers import runware as runware_module
from media.providers.runware import Runware, describe_errors
from media.providers.sogni import CONSENT_PAGE, Sogni


def req_for(kind, provider, family, variant="", **kw):
    option = catalog.find(kind, provider, family, variant)
    fields = dict(option=option, mode=option.modes[0], prompt="she turns and smiles", resolution=option.lowest)
    if kind == "video":
        fields["seconds"] = option.longest()
    fields.update(kw)
    return RenderRequest(**fields)


class ProviderCase(MediaCase):
    def asset(self, width=800, height=1200):
        return assets.store(noisy_jpeg(width, height))[0]


class PixelDojoBodyTests(ProviderCase):
    def build(self, *route, refs=(), first=None, last=None, **kw):
        req = req_for(*route, **kw)
        build = BUILDERS[req.option.extra["build"]]
        return build(req, list(refs)) if req.option.kind == "image" else build(req, list(refs), first, last)

    def test_seedream_5_needs_custom_size_for_a_set_shape(self):
        body = self.build("image", "pixeldojo", "seedream-5-pro", aspect_ratio="16:9", seed=5, refs=["u"])
        self.assertEqual(body, {"image_size": "custom", "customWidth": 1344, "customHeight": 768, "seed": 5, "image_urls": ["u"]})

    def test_seedream_4_uses_named_sizes(self):
        self.assertEqual(self.build("image", "pixeldojo", "seedream-4.5", aspect_ratio="3:4")["image_size"], "portrait_3_4")

    def test_wan_uppercase_resolution_and_no_seed(self):
        body = self.build("video", "pixeldojo", "wan-3.0", seed=7, mode="reference", refs=["u"], aspect_ratio="9:16", seconds=15)
        self.assertEqual(body["resolution"], "480P")
        self.assertEqual(body["reference_image_urls"], ["u"])
        self.assertIs(body["prompt_extend"], False)
        self.assertNotIn("seed", body)

    def test_seedance_25_never_sends_a_seed_and_a_first_frame_sets_the_shape(self):
        body = self.build("video", "pixeldojo", "seedance-2.5", mode="first", first="f", seed=7)
        self.assertEqual(body["first_frame_url"], "f")
        self.assertNotIn("aspect_ratio", body)
        self.assertNotIn("seed", body)
        self.assertIs(body["generate_audio"], False)

    def test_h3_first_frame_omits_size_because_it_ignores_it(self):
        body = self.build("video", "pixeldojo", "minimax-h3", "minimax-h3", mode="first_last", first="f", last="l")
        self.assertEqual((body["image_url"], body["end_image_url"]), ("f", "l"))
        self.assertNotIn("resolution", body)

    def test_h3_reference_sets_size_and_shape(self):
        body = self.build("video", "pixeldojo", "minimax-h3", "minimax-h3", mode="reference", refs=["u"], aspect_ratio="3:4")
        self.assertEqual((body["resolution"], body["aspect_ratio"], body["reference_images"]), ("768p", "3:4", ["u"]))

    def test_h3_turbo_with_a_photo_lets_it_set_the_shape(self):
        self.assertNotIn("aspect_ratio", self.build("video", "pixeldojo", "minimax-h3", mode="first", first="f"))
        self.assertIn("resolution", self.build("video", "pixeldojo", "minimax-h3", mode="first", first="f"))


class PixelDojoFlowTests(ProviderCase):
    def http(self, **routes):
        return FakeHTTP([("/upload", FakeResponse(200, {"url": "https://u/1"})),
                         ("/models/", routes.get("run", FakeResponse(200, {"jobId": "j1", "creditCost": 1.25}))),
                         ("/jobs/", routes.get("job", FakeResponse(200, {"status": "processing"}))),
                         ("https://out", FakeResponse(200, content=small_jpeg()))])

    def test_uploads_once_per_day_and_sends_the_prompt(self):
        http = self.http()
        asset = self.asset()
        req = req_for("image", "pixeldojo", "seedream-4.5", refs=[asset, asset], aspect_ratio="9:16")
        with http.patch():
            task = run(PixelDojo().submit(req))
        self.assertEqual(task, "j1")
        self.assertEqual(sum("/upload" in c["url"] for c in http.calls), 1)
        self.assertEqual(req.cost_credits, 1.25)
        sent = http.call("/models/seedream-4/run")["json"]
        self.assertEqual(sent["image_urls"], ["https://u/1", "https://u/1"])
        self.assertEqual(sent["prompt"], req.prompt)

    def test_out_of_credits_says_so(self):
        http = self.http(run=FakeResponse(402, {"error": "no credits"}))
        with http.patch(), self.assertRaisesRegex(RuntimeError, "out of credits"):
            run(PixelDojo().submit(req_for("image", "pixeldojo", "seedream-4.5", aspect_ratio="9:16")))

    def test_rate_limit_waits_as_told_then_succeeds(self):
        http = self.http(run=[FakeResponse(429, {"error": "slow"}, headers={"retry-after": "0"}), FakeResponse(200, {"jobId": "j2"})])
        with http.patch(), unittest.mock.patch.object(base, "retry_after", return_value=0):
            self.assertEqual(run(PixelDojo().submit(req_for("image", "pixeldojo", "seedream-4.5", aspect_ratio="9:16"))), "j2")

    def test_polling_states(self):
        req = req_for("image", "pixeldojo", "seedream-4.5")
        with self.http().patch():
            self.assertIsNone(run(PixelDojo().poll("j1", req)))
        done = FakeResponse(200, {"status": "completed", "creditCost": 1, "output": {"images": ["https://out/1.jpg"]}})
        with self.http(job=done).patch():
            result = run(PixelDojo().poll("j1", req))
        self.assertEqual((result.mime, result.cost_credits), ("image/jpeg", 1))

    def test_a_failed_job_reports_the_refund(self):
        job = FakeResponse(200, {"status": "failed", "error": "rejected", "refunded": True})
        with self.http(job=job).patch(), self.assertRaisesRegex(RuntimeError, "rejected .credits refunded"):
            run(PixelDojo().poll("j1", req_for("image", "pixeldojo", "seedream-4.5")))

    def test_an_expired_job_fails_instead_of_waiting(self):
        with self.http(job=FakeResponse(410)).patch(), self.assertRaisesRegex(RuntimeError, "expired"):
            run(PixelDojo().poll("j1", req_for("image", "pixeldojo", "seedream-4.5")))

    def test_output_can_be_video_images_or_assets(self):
        self.assertEqual(first_output_url({"output": {"video": "v"}}), "v")
        self.assertEqual(first_output_url({"output": {"images": ["i"]}}), "i")
        self.assertEqual(first_output_url({"output": {}, "assets": [{"url": "a"}]}), "a")


class NanoGPTTests(ProviderCase):
    def test_only_the_models_own_video_fields_are_sent(self):
        req = req_for("video", "nanogpt", "wan-2.2", mode="first", seed=3, seconds=8)
        body = NanoGPT().build_video(req, noisy_jpeg(), None, [])
        self.assertEqual(body["model"], "wan-video-22")
        self.assertEqual(body["duration"], "8")
        self.assertTrue(body["imageDataUrl"].startswith("data:image/jpeg"))
        for field in ("seed", "aspect_ratio", "generateAudio", "last_image"):
            self.assertNotIn(field, body)

    def test_last_frame_seed_and_shape_from_the_first_frame(self):
        req = req_for("video", "nanogpt", "seedance-2", mode="first_last", seed=9)
        body = NanoGPT().build_video(req, noisy_jpeg(1200, 800), noisy_jpeg(), [])
        self.assertEqual(body["aspect_ratio"], "4:3")
        self.assertEqual(body["seed"], 9)
        self.assertTrue(body["last_image"].startswith("data:"))
        self.assertIs(body["generateAudio"], False)

    def test_seedream_5_pro_is_half_price_only_when_every_reference_has_the_shape(self):
        tall, wide, square = noisy_jpeg(864, 1536), noisy_jpeg(1536, 864), noisy_jpeg(1000, 1000)
        req = req_for("image", "nanogpt", "seedream-5-pro", aspect_ratio="9:16")
        provider = NanoGPT()
        cheap = provider.build_image(req, [tall, tall])
        self.assertEqual((cheap["resolution"], "aspect_ratio" in cheap), ("1k", False))
        for refs in ([tall, wide], [square], [wide]):
            self.assertEqual(provider.build_image(req, refs)["aspect_ratio"], "9:16", refs)
        self.assertEqual(provider.build_image(req, [])["aspect_ratio"], "9:16", "with no reference nothing sets the shape")
        near = provider.build_image(req, [noisy_jpeg(880, 1536)])        # 2% off: still takes the references' shape
        self.assertNotIn("aspect_ratio", near)

    def test_seedream_5_pro_stays_at_1k_when_the_user_picks_price_over_shape(self):
        tall, wide = noisy_jpeg(864, 1536), noisy_jpeg(1536, 864)
        req = req_for("image", "nanogpt", "seedream-5-pro", aspect_ratio="16:9")
        req.budget = True
        body = NanoGPT().build_image(req, [tall, wide])
        self.assertEqual((body["resolution"], "aspect_ratio" in body), ("1k", False))
        self.assertEqual(NanoGPT().build_image(req, [])["aspect_ratio"], "16:9", "with no reference something must set the shape")
        qwen = req_for("image", "nanogpt", "qwen-edit", aspect_ratio="16:9")
        qwen.budget = True
        self.assertEqual(NanoGPT().build_image(qwen, [tall])["aspect_ratio"], "16:9", "only models that bill the ratio higher drop it")

    def test_seedream_45_image_body(self):
        req = req_for("image", "nanogpt", "seedream-4.5", aspect_ratio="16:9")
        body = NanoGPT().build_image(req, [noisy_jpeg()])
        self.assertEqual(body["resolution"], "4096x2304")
        self.assertEqual(len(body["input_references"]), 1)
        self.assertNotIn("aspect_ratio", body)

    def test_qwen_needs_a_reference(self):
        with self.assertRaisesRegex(RuntimeError, "needs at least one reference"):
            run(NanoGPT().submit(req_for("image", "nanogpt", "qwen-edit", aspect_ratio="9:16")))

    def test_ten_references_stay_under_the_request_limit(self):
        picture = base64.b64encode(small_jpeg()).decode()
        http = FakeHTTP([("/v1/images", FakeResponse(200, {"data": [{"b64_json": picture}], "cost": 0.04}))])
        nano = NanoGPT()
        asset = assets.store(noisy_jpeg(1800, 2400, quality=85))[0]
        req = req_for("image", "nanogpt", "seedream-4.5", refs=[asset] * 10, aspect_ratio="9:16")
        with http.patch():
            task = run(nano.submit(req))
            result = run(nano.poll(task, req))
        self.assertLess(len(str(http.calls[0]["json"])), MAX_REQUEST * 1.1)
        self.assertEqual(result.cost_usd, 0.04)
        self.assertEqual(req.params["references"], 10)
        self.assertNotIn("input_references", req.params)

    def test_an_image_lost_to_a_restart_fails_at_once(self):
        with self.assertRaisesRegex(RuntimeError, "restart"):
            run(NanoGPT().poll("gone", req_for("image", "nanogpt", "seedream-4.5")))

    def test_video_status_flow(self):
        req = req_for("video", "nanogpt", "wan-3.0", mode="first")
        running = FakeResponse(200, {"data": {"status": "IN_QUEUE"}})
        done = FakeResponse(200, {"data": {"status": "COMPLETED", "output": {"video": {"url": "https://out/v.mp4"}}}})
        http = FakeHTTP([("/video/status", [running, done]), ("https://out", FakeResponse(200, content=b"\x00\x00\x00\x18ftypmp42"))])
        with http.patch():
            self.assertIsNone(run(NanoGPT().poll("r1", req)))
            result = run(NanoGPT().poll("r1", req))
        self.assertEqual(result.mime, "video/mp4")
        self.assertEqual(http.calls[0]["params"], {"requestId": "r1"})

    def test_balance(self):
        with FakeHTTP([("/check-balance", FakeResponse(200, {"usd_balance": 12.5}))]).patch():
            self.assertEqual(run(NanoGPT().balance()), {"amount": 12.5, "unit": "USD"})


class RunwareTests(ProviderCase):
    def test_task_shape(self):
        req = req_for("image", "runware", "flux-lora", aspect_ratio="16:9", seed=4, negative_prompt="blurry")
        task = Runware().build_task(req, [{"lora": "air:1@1", "weight": 0.8}], "t1")
        self.assertEqual((task["width"], task["height"], task["steps"]), (1344, 768, 28))
        self.assertEqual(task["lora"], [{"model": "air:1@1", "weight": 0.8}])
        self.assertEqual(task["outputType"], "URL")
        self.assertIs(task["includeCost"], True)
        self.assertEqual(task["CFGScale"], 3.5)
        self.assertEqual(task["negativePrompt"], "blurry")

    def test_flux_loras_are_never_sent_to_the_zimage_architecture(self):
        seen = []
        lora_item = {"url": "https://hf/z.safetensors", "weight": 1.0}
        http = FakeHTTP([("runware", FakeResponse(200, {"data": [{"imageURL": "https://out/1.jpg", "cost": 0.0129}]})),
                         ("https://out", FakeResponse(200, content=small_jpeg()))])
        with unittest.mock.patch.object(MediaConfig, "RUNWARE_ENDPOINT", "https://runware/v1"), \
             unittest.mock.patch.object(loras, "_resolve", lambda items, key, arch: seen.append(arch) or [{"lora": "a@1", "weight": 1.0}]), \
             http.patch():
            runware = Runware()
            req = req_for("image", "runware", "zimage", loras=[lora_item], aspect_ratio="9:16")
            task = run(runware.submit(req))
            result = run(runware.poll(task, req))
        self.assertEqual(seen, ["zimage"])
        self.assertEqual(result.cost_usd, 0.0129)
        self.assertEqual(req.params["model"], "rundiffusion:200@100")

    PROCESSING = FakeResponse(400, {"errors": [{"message": "The model specified in the 'lora.model' parameter is not ready for inference yet."}]})
    RENDERED = FakeResponse(200, {"data": [{"imageURL": "https://out/1.jpg", "cost": 0.01}]})

    def render_with(self, replies, wait):
        http = FakeHTTP([("runware", replies), ("https://out", FakeResponse(200, content=small_jpeg()))])
        lora_item = {"url": "https://hf/a.safetensors", "weight": 1.0}
        with unittest.mock.patch.object(MediaConfig, "RUNWARE_ENDPOINT", "https://runware/v1"), \
             unittest.mock.patch.object(loras, "_resolve", lambda items, key, arch: [{"lora": "a@1", "weight": 1.0}]), \
             unittest.mock.patch.object(runware_module, "LORA_READY_POLL", 0), \
             unittest.mock.patch.object(runware_module, "LORA_READY_WAIT", wait), \
             http.patch():
            try:
                run(Runware().submit(req_for("image", "runware", "flux-lora", loras=[lora_item], aspect_ratio="9:16")))
                return None, sum("runware" in c["url"] for c in http.calls)
            except RuntimeError as e:
                return str(e), sum("runware" in c["url"] for c in http.calls)

    def test_a_lora_still_processing_is_waited_for(self):
        self.assertEqual(self.render_with([self.PROCESSING, self.PROCESSING, self.RENDERED], 300), (None, 3))

    def test_a_lora_that_never_finishes_processing_fails(self):
        error, calls = self.render_with([self.PROCESSING, self.RENDERED], 0)
        self.assertIn("not ready for inference", error)
        self.assertEqual(calls, 1)

    def test_a_refused_lora_is_not_retried(self):
        refused = FakeResponse(400, {"errors": [{"message": "Invalid value for 'lora.model'"}]})
        error, calls = self.render_with([refused, self.RENDERED], 300)
        self.assertIn("Invalid value", error)
        self.assertEqual(calls, 1)

    def test_a_lora_for_another_architecture_explains_itself(self):
        text = describe_errors([{"code": "unsupportedLoraModel", "loraModelArchitecture": "flux1d", "baseModelArchitecture": "z_image"}])
        self.assertIn("flux1d", text)
        self.assertIn("z_image", text)

    def test_balance_reads_either_shape(self):
        with unittest.mock.patch.object(MediaConfig, "RUNWARE_ENDPOINT", "https://runware/v1"):
            for payload in ({"balance": 3.5}, {"organization": {"balance": 3.5}}):
                with FakeHTTP([("runware", FakeResponse(200, {"data": [payload]}))]).patch():
                    self.assertEqual(run(Runware().balance()), {"amount": 3.5, "unit": "USD"})


class ReplicateTests(ProviderCase):
    def test_loras_go_by_their_huggingface_links_with_the_filter_off(self):
        req = req_for("image", "replicate", "flux-lora", aspect_ratio="9:16", seed=2,
                      loras=[{"url": "https://hf/a.safetensors", "weight": 0.9}])
        inp = Replicate().build_input(req)
        self.assertEqual(inp["hf_loras"], ["https://hf/a.safetensors"])
        self.assertEqual(inp["lora_scales"], [0.9])
        self.assertIs(inp["disable_safety_checker"], True)

    def test_prediction_states(self):
        self.assertIsNone(prediction_output({"status": "processing"}))
        self.assertEqual(prediction_output({"status": "succeeded", "output": ["u"]}), "u")
        with self.assertRaisesRegex(RuntimeError, "failed: nope"):
            prediction_output({"status": "failed", "error": "nope"})


class FakeProject:
    id = "p1"
    seed = 11

    def __init__(self, urls=("https://out/v.mp4",), error=None):
        self.urls, self.error, self.cancelled = list(urls), error, False

    async def wait_for_completion(self, timeout=None):
        if self.error:
            raise self.error
        return self.urls

    async def cancel(self):
        self.cancelled = True


def fake_sdk(project):
    made = []

    class Projects:
        async def create(self, params):
            made.append(params)
            return project

        async def upload_url(self, ids):
            return f"https://up/{ids['imageId']}"

        async def download_url(self, ids):
            return f"https://down/{ids['imageId']}"

        async def cancel(self, project_id):
            made.append({"cancelled": project_id})

    class Client:
        def __init__(self):
            self.projects = Projects()
            self.api_client = types.SimpleNamespace(rest=types.SimpleNamespace(put_bytes=self._put))
            self.account = types.SimpleNamespace(account_balance=self._balance)

        async def _put(self, *_a, **_kw):
            return None

        async def _balance(self):
            return {"spark": {"net": 42.0}}

        @classmethod
        async def create(cls, **kw):
            made.append({"app_id": kw.get("app_id")})
            return cls()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    module = types.ModuleType("sogni_client")
    module.SogniClient = Client
    errors = types.ModuleType("sogni_client.errors")
    errors.is_model_consent_required_error = lambda e: "consent" in str(e)
    return made, unittest.mock.patch.dict(sys.modules, {"sogni_client": module, "sogni_client.errors": errors})


class SogniTests(ProviderCase):
    def params(self, *route, **kw):
        made, sdk = fake_sdk(FakeProject())
        sogni = Sogni()
        req = req_for(*route, **kw)
        with sdk:
            run(sogni.submit(req))
        return next(m for m in made if "app_id" not in m), req

    def test_every_connection_gets_its_own_app_id(self):
        made, sdk = fake_sdk(FakeProject())
        sogni = Sogni()
        with sdk:
            run(sogni._client("fast"))
            run(sogni._client("fast"))
        self.assertEqual(len({m["app_id"] for m in made}), 2)

    def test_h3_reference_sends_bytes_and_context_images(self):
        one, two = self.asset(), self.asset(900, 900)
        params, req = self.params("video", "sogni", "minimax-h3", mode="reference", refs=[one, two], aspect_ratio="16:9")
        self.assertIsInstance(params["referenceImage"], bytes)
        self.assertEqual(len(params["contextImages"]), 1)
        self.assertEqual((params["width"], params["height"]), (1344, 768))
        self.assertEqual(params["loras"], ["h3-vbvr-video-reasoning"])
        self.assertIs(params["disableNSFWFilter"], True)
        self.assertEqual(req.params["contextImages"], "<1 images>")
        self.assertEqual(req.params["referenceImage"], "<image>")

    def test_adult_scenes_add_the_second_h3_lora(self):
        params, _ = self.params("video", "sogni", "minimax-h3", mode="reference", refs=[self.asset()], adult=True)
        self.assertEqual(params["loras"], ["h3-vbvr-video-reasoning", "h3-mystic-xxx-v5"])
        self.assertEqual(params["loraStrengths"], [0.8, 0.5])

    def test_seedance_and_wan_send_links_never_the_bridge_url(self):
        params, _ = self.params("video", "sogni", "seedance-2.5", mode="reference", refs=[self.asset()], aspect_ratio="9:16")
        self.assertTrue(params["referenceImageUrls"][0].startswith("https://down/"))
        self.assertEqual(params["seedanceTaskType"], "reference")
        self.assertEqual((params["width"], params["height"], params["fps"]), (480, 854, 24))
        wan, _ = self.params("video", "sogni", "wan-3.0", mode="reference", refs=[self.asset()], aspect_ratio="16:9")
        self.assertEqual(wan["ratio"], "16:9")
        self.assertEqual(wan["fps"], 30)
        self.assertEqual((wan["width"], wan["height"]), (854, 480))

    def test_a_reference_link_is_reused(self):
        asset = self.asset()
        self.params("video", "sogni", "seedance-2.5", mode="reference", refs=[asset])
        link = db.remote_handle(asset.id, "sogni")
        params, _ = self.params("video", "sogni", "seedance-2.5", mode="reference", refs=[asset])
        self.assertEqual(params["referenceImageUrls"], [link])

    def test_last_frame_moves_h3_to_its_flf_model(self):
        params, _ = self.params("video", "sogni", "minimax-h3", "minimax-h3-fl2va-fp8_i2v", mode="first_last",
                                first=self.asset(), last=self.asset(900, 900))
        self.assertEqual(params["modelId"], "minimax-h3-fl2va-fp8_flf2v")
        self.assertIsInstance(params["referenceImageEnd"], bytes)

    def test_image_sizes_and_context_images(self):
        params, _ = self.params("image", "sogni", "qwen-edit", refs=[self.asset()], aspect_ratio="9:16")
        self.assertEqual(params["sizePreset"], "custom")
        self.assertEqual((params["width"], params["height"]), (768, 1344))
        self.assertEqual(len(params["contextImages"]), 1)

    def test_network_follows_the_request_then_the_env(self):
        with unittest.mock.patch.object(MediaConfig, "SOGNI_NETWORK", "relaxed"):
            self.assertEqual(Sogni.network(req_for("video", "sogni", "ltx-2.3")), "relaxed")
            self.assertEqual(Sogni.network(req_for("video", "sogni", "ltx-2.3", network="fast")), "fast")
        with unittest.mock.patch.object(MediaConfig, "SOGNI_NETWORK", "bogus"):
            self.assertEqual(Sogni.network(req_for("video", "sogni", "ltx-2.3", network="nope")), "fast")

    def finish(self, project):
        made, sdk = fake_sdk(project)
        sogni = Sogni()
        req = req_for("video", "sogni", "minimax-h3", mode="reference", refs=[self.asset()])
        with sdk:
            task = run(sogni.submit(req))
            return run(sogni.poll(task, req))

    def test_a_finished_render_is_downloaded(self):
        http = FakeHTTP([("https://out", FakeResponse(200, content=b"\x00\x00\x00\x18ftypmp42"))])
        with http.patch():
            result = self.finish(FakeProject())
        self.assertEqual((result.mime, result.seed), ("video/mp4", 11))

    def test_a_timeout_cancels_the_render_so_it_stops_billing(self):
        project = FakeProject(error=asyncio.TimeoutError())
        with self.assertRaises(asyncio.TimeoutError):
            self.finish(project)
        self.assertTrue(project.cancelled)

    def test_the_consent_error_points_at_the_page(self):
        with self.assertRaisesRegex(RuntimeError, CONSENT_PAGE):
            self.finish(FakeProject(error=RuntimeError("consent required")))

    def test_an_orphan_from_a_restart_is_cancelled_by_id(self):
        made, sdk = fake_sdk(FakeProject())
        with sdk:
            run(Sogni().cancel("old-project"))
        self.assertIn({"cancelled": "old-project"}, made)

    def test_balance(self):
        _made, sdk = fake_sdk(FakeProject())
        with sdk:
            self.assertEqual(run(Sogni().balance()), {"amount": 42.0, "unit": "Spark", "partial": True})


class MockProviderTests(ProviderCase):
    def test_task_ids_finish_even_in_a_new_process(self):
        req = req_for("image", "mock", "seedream-4.5", aspect_ratio="9:16")
        task = run(Mock().submit(req))
        self.assertIsNone(run(Mock().poll(f"{2**31}:x", req)))
        self.assertTrue(task.split(":")[1])
        self.assertEqual(run(Mock().poll("1.0:x", req)).mime, "image/jpeg")

    def test_a_prompt_can_ask_it_to_fail(self):
        with self.assertRaisesRegex(RuntimeError, "failed on request"):
            run(Mock().submit(req_for("image", "mock", "seedream-4.5", prompt="MOCK_FAIL")))


class SharedTests(unittest.TestCase):
    def test_retry_after_reads_the_header_then_the_body(self):
        self.assertEqual(base.retry_after(FakeResponse(429, {}, headers={"retry-after": "7"})), 7.0)
        self.assertEqual(base.retry_after(FakeResponse(429, {"retry_after": 3})), 3.0)
        self.assertEqual(base.retry_after(FakeResponse(429)), 5.0)

    def test_sniffing(self):
        self.assertEqual(base.sniff_mime(b"\xff\xd8\xff\xe0"), "image/jpeg")
        self.assertEqual(base.sniff_mime(b"\x00\x00\x00\x18ftypmp42"), "video/mp4")
        self.assertEqual(base.sniff_mime(b"RIFF0000WEBP"), "image/webp")

    def test_clips_from_a_frame_keep_the_frames_shape_and_the_models_limits(self):
        """Sogni's LTX needs both sides at 640+, so the old fixed 512x896 was widened to 640x896 and reshaped a 9:16 frame."""
        frames = {"9:16": (9, 16), "16:9": (16, 9), "3:4": (3, 4), "4:3": (4, 3), "1:1": (1, 1)}
        for option in catalog.options("video"):
            if option.extra.get("sizing") != "wh":
                continue
            step, lowest = option.extra["size_step"], option.extra["min_side"]
            for resolution in option.resolutions:
                for name, (w, h) in frames.items():
                    width, height = base.wh_exact(resolution, w, h, step=step, min_side=lowest)
                    label = f"{option.label} {resolution} {name} -> {width}x{height}"
                    self.assertEqual((width % step, height % step), (0, 0), label)
                    self.assertGreaterEqual(min(width, height), lowest, label)
                    self.assertAlmostEqual(width / height, w / h, delta=0.03 * w / h, msg=label)

    def test_sogni_sizes_a_clip_from_the_frame_it_was_given(self):
        for shape, frame, expect in (("9:16", noisy_jpeg(900, 1600), (640, 1152)), ("16:9", noisy_jpeg(1600, 900), (1152, 640))):
            req = req_for("video", "sogni", "ltx-2.5", aspect_ratio=shape, resolution="480p")
            size = Sogni()._size(req, frame)
            self.assertEqual((size["width"], size["height"]), expect, shape)


if __name__ == "__main__":
    unittest.main()
