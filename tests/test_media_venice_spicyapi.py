"""Venice and SpicyAPI: each task goes to its own model id, only that model's fields are sent,
and both reject anything else with a 400."""
import unittest
import unittest.mock

from media_support import FakeHTTP, FakeResponse, MediaCase, noisy_jpeg, run, small_jpeg
from media import assets, catalog, keys
from media.config import MediaConfig
from media.providers import spicyapi, venice
from media.providers.base import RenderRequest
from media.providers.spicyapi import SpicyAPI
from media.providers.venice import Venice

CLIP = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
HF = "https://huggingface.co/u/r/resolve/main/a.safetensors"


def req_for(kind, provider, family, variant="", **kw):
    option = catalog.find(kind, provider, family, variant)
    fields = dict(option=option, mode=option.modes[0], prompt="she turns and smiles", resolution=option.lowest)
    if kind == "video":
        fields["seconds"] = option.longest()
    fields.update(kw)
    return RenderRequest(**fields)


def envelope(data, code=200):
    return {"code": code, "msg": "success" if code == 200 else "nope", "data": data}


class CatalogTests(unittest.TestCase):
    def test_every_mode_has_a_model_id(self):
        for table in (catalog.IMAGE_CATALOG, catalog.VIDEO_CATALOG):
            for provider in ("venice", "spicyapi"):
                for options in table[provider].values():
                    for option in options:
                        if option.family == "best":
                            continue
                        tasks = option.extra.get("tasks") or {}
                        name = f"{provider}/{option.id}"
                        if provider == "venice" and option.kind == "image":
                            self.assertTrue(option.extra.get("edit"), name)
                            continue
                        for mode in option.modes:
                            wanted = "first" if mode == "first_last" else mode
                            self.assertIn(wanted, tasks, f"{name} {mode}")

    def test_the_best_option_has_both_steps_on_spicyapi(self):
        self.assertTrue(catalog.find("image", "spicyapi", "seedream-5-pro"))
        self.assertTrue(catalog.find("image", "spicyapi", "seedream-4.5"))
        self.assertIsNone(catalog.find("image", "venice", "best"))

    def test_venice_seedance_and_ltx_wait_on_a_test(self):
        for family in ("seedance-2", "seedance-2.5", "ltx-2.3", "ltx-2.5"):
            for option in catalog.VIDEO_CATALOG["venice"][family]:
                self.assertEqual(option.nsfw, "check", option.id)
        self.assertEqual(catalog.find("video", "venice", "wan-3.0").nsfw, "allowed")

    def test_keys_can_be_saved_for_both(self):
        self.assertEqual(keys.FIELDS["venice"], "VENICE_API_KEY")
        self.assertEqual(keys.FIELDS["spicyapi"], "SPICYAPI_API_KEY")


class SpicyBodyTests(unittest.TestCase):
    def test_each_task_has_its_own_model(self):
        self.assertEqual(spicyapi.task_model(req_for("video", "spicyapi", "wan-3.0", mode="reference")), "alibaba/wan-3.0/reference-to-video")
        self.assertEqual(spicyapi.task_model(req_for("video", "spicyapi", "wan-3.0", mode="first_last")), "alibaba/wan-3.0/image-to-video")
        self.assertEqual(spicyapi.task_model(req_for("image", "spicyapi", "seedream-4.5", refs=["a"])), "bytedance/seedream-4.5/edit")
        self.assertEqual(spicyapi.task_model(req_for("image", "spicyapi", "seedream-4.5")), "bytedance/seedream-4.5/text-to-image")
        with self.assertRaisesRegex(RuntimeError, "needs at least one reference"):
            spicyapi.task_model(req_for("image", "spicyapi", "qwen-edit", "alibaba/qwen-image-spicy-edit"))

    def test_wan_reference_sends_refs_shape_and_silence(self):
        req = req_for("video", "spicyapi", "wan-3.0", mode="reference", aspect_ratio="9:16", seed=4)
        inp = spicyapi.video(req, ["spicy://f/1"], None, None)
        self.assertEqual(inp, {"prompt": req.prompt, "duration_seconds": 15, "resolution": "480p", "generate_audio": False,
                               "seed": 4, "reference_image_urls": ["spicy://f/1"], "aspect_ratio": "9:16"})

    def test_wan_from_a_frame_is_adaptive_and_spicy_wan_names_the_frames_shape(self):
        inp = spicyapi.video(req_for("video", "spicyapi", "wan-3.0", mode="first_last", aspect_ratio="3:4"), [], "f", "l")
        self.assertEqual((inp["image_url"], inp["last_image_url"], inp["aspect_ratio"]), ("f", "l", "adaptive"))
        inp = spicyapi.video(req_for("video", "spicyapi", "wan-3.0", "alibaba/wan-3.0-spicy", mode="first", aspect_ratio="3:4"), [], "f", None)
        self.assertEqual(inp["aspect_ratio"], "3:4")
        self.assertNotIn("last_image_url", inp)

    def test_seedance_and_h3_frames_take_their_own_shape(self):
        inp = spicyapi.video(req_for("video", "spicyapi", "seedance-2.5", mode="first", aspect_ratio="9:16"), [], "f", None)
        self.assertNotIn("aspect_ratio", inp)
        inp = spicyapi.video(req_for("video", "spicyapi", "minimax-h3", mode="first", seed=3), [], "f", None)
        self.assertNotIn("seed", inp)              # base H3 has no seed field
        self.assertNotIn("generate_audio", inp)

    def test_seedream_sends_its_resolution_and_shape(self):
        req = req_for("image", "spicyapi", "seedream-5-pro", aspect_ratio="16:9")
        self.assertEqual(spicyapi.seedream(req, ["r"]), {"prompt": req.prompt, "aspect_ratio": "16:9", "resolution": "1k", "image_urls": ["r"]})

    def test_loras_are_capped_at_three(self):
        stack = [{"url": f"https://hf/{n}.safetensors", "weight": 0.5 + n / 10} for n in range(5)] + [{"id": "no-url"}]
        req = req_for("image", "spicyapi", "flux-lora", aspect_ratio="9:16", seed=1, loras=stack)
        inp = spicyapi.lora_image(req, [])
        self.assertEqual(len(inp["loras"]), 3)
        self.assertEqual(inp["loras"][0], {"path": "https://hf/0.safetensors", "scale": 0.5})
        self.assertEqual((inp["steps"], inp["guidance_scale"], inp["seed"]), (28, 3.5, 1))

    def test_output_waits_for_a_pending_link(self):
        self.assertIsNone(spicyapi.output_url({"output": {"assets": [{"pending": True}]}}, "video"))
        record = {"output": {"assets": [{"url": "https://o/last.jpg", "role": "image"}, {"url": "https://o/c.mp4", "role": "video"}]}}
        self.assertEqual(spicyapi.output_url(record, "video"), "https://o/c.mp4")
        with self.assertRaisesRegex(RuntimeError, "without a file"):
            spicyapi.output_url({"output": {"assets": []}}, "video")


class SpicyFlowTests(MediaCase):
    def setUp(self):
        super().setUp()
        patch = unittest.mock.patch.object(MediaConfig, "SPICYAPI_API_KEY", "sk-spicy-test")
        patch.start()
        self.addCleanup(patch.stop)

    def http(self, **routes):
        return FakeHTTP([("/common/upload-url", FakeResponse(200, envelope({"fileId": "f1", "uploadUrl": "https://store/put", "method": "PUT",
                                                                           "headers": {"Content-Type": "image/jpeg"}, "maxBytes": 10_000_000}))),
                         ("https://store/put", FakeResponse(200)),
                         ("/files/f1/commit", FakeResponse(200, envelope({"uri": "spicy://f/abc"}))),
                         ("/jobs/createTask", routes.get("create", FakeResponse(202, envelope({"taskId": "t1", "state": "queued", "estimatedCost": "0.6750"})))),
                         ("/jobs/recordInfo", routes.get("record", FakeResponse(200, envelope({"taskId": "t1", "state": "running"})))),
                         ("/chat/credit", routes.get("credit", FakeResponse(200, envelope({"available": "9.25", "held": "0.75", "total": "10"})))),
                         ("https://o/", FakeResponse(200, content=CLIP))])

    def test_frames_upload_once_and_the_task_holds_only_its_fields(self):
        http = self.http()
        frame = assets.store(noisy_jpeg(720, 1280))[0]
        req = req_for("video", "spicyapi", "wan-3.0", mode="first_last", first=frame, last=frame, aspect_ratio="9:16")
        with http.patch():
            self.assertEqual(run(SpicyAPI().submit(req)), "t1")
        self.assertEqual(sum("upload-url" in c["url"] for c in http.calls), 1)
        sent = http.call("/jobs/createTask")
        self.assertEqual(sent["json"]["model"], "alibaba/wan-3.0/image-to-video")
        self.assertEqual((sent["json"]["input"]["image_url"], sent["json"]["input"]["last_image_url"]), ("spicy://f/abc", "spicy://f/abc"))
        self.assertIn("Idempotency-Key", sent["headers"])
        self.assertEqual(req.cost_usd, 0.675)
        self.assertNotIn("image_url", req.params)
        self.assertEqual(req.params["model"], "alibaba/wan-3.0/image-to-video")

    def test_private_loras_go_as_bridge_links_and_the_history_keeps_the_hugging_face_url(self):
        http = self.http()
        req = req_for("image", "spicyapi", "zimage", aspect_ratio="9:16", loras=[{"url": HF, "weight": 0.8}])
        with unittest.mock.patch.multiple(MediaConfig, HF_TOKEN="hf_test", PUBLIC_BASE_URL="https://bridge.test",
                                          LORA_LINK_SECRET="test-secret"), http.patch():
            run(SpicyAPI().submit(req))
        path = http.call("/jobs/createTask")["json"]["input"]["loras"][0]["path"]
        self.assertTrue(path.startswith("https://bridge.test/lora-files/"), path)
        self.assertEqual(req.params["loras"], [HF])

    def test_out_of_credit_says_so(self):
        for reply in (FakeResponse(402, {"code": 40201, "msg": "insufficient balance"}),
                      FakeResponse(200, {"code": 40201, "msg": "insufficient balance"})):
            with self.http(create=reply).patch(), self.assertRaisesRegex(RuntimeError, "out of credit"):
                run(SpicyAPI().submit(req_for("image", "spicyapi", "seedream-4.5", aspect_ratio="9:16")))

    def test_polling_states_and_the_settled_cost(self):
        req = req_for("video", "spicyapi", "wan-3.0", mode="first")
        with self.http().patch():
            self.assertIsNone(run(SpicyAPI().poll("t1", req)))
        done = FakeResponse(200, envelope({"state": "succeeded", "cost": "0.6750", "settled": True, "input": {"seed": 9},
                                           "output": {"assets": [{"url": "https://o/c.mp4", "role": "video"}]}}))
        with self.http(record=done).patch():
            result = run(SpicyAPI().poll("t1", req))
        self.assertEqual((result.mime, result.cost_usd, result.seed), ("video/mp4", 0.675, 9))

    def test_a_failed_task_reports_its_reason(self):
        failed = FakeResponse(200, envelope({"state": "failed", "errorCode": "content_rejected", "errorMessage": "rejected upstream"}))
        with self.http(record=failed).patch(), self.assertRaisesRegex(RuntimeError, "failed: rejected upstream"):
            run(SpicyAPI().poll("t1", req_for("video", "spicyapi", "wan-3.0", mode="first")))

    def test_balance_and_a_refused_key(self):
        with self.http().patch():
            self.assertEqual(run(SpicyAPI().balance()), {"amount": 9.25, "unit": "USD"})
        with self.http(credit=FakeResponse(401, {"code": 401, "msg": "bad key"})).patch(), self.assertRaisesRegex(RuntimeError, "refused this key"):
            run(SpicyAPI().balance())


class VeniceBodyTests(unittest.TestCase):
    def test_wan_reference(self):
        req = req_for("video", "venice", "wan-3.0", mode="reference", aspect_ratio="3:4")
        body = venice.build_video(req, None, None, ["data:r"])
        self.assertEqual(body, {"model": "wan-3-0-reference-to-video", "prompt": req.prompt, "duration": "15s", "resolution": "480p",
                                "audio": False, "reference_image_urls": ["data:r"], "aspect_ratio": "3:4"})

    def test_h3_max_turbo_from_a_frame_sends_no_audio_switch_or_shape(self):
        req = req_for("video", "venice", "minimax-h3", "minimax-h3-max-turbo", mode="first", aspect_ratio="9:16")
        body = venice.build_video(req, "data:f", None, [])
        self.assertEqual(body, {"model": "minimax-h3-max-turbo-image-to-video", "prompt": req.prompt, "duration": "15s",
                                "resolution": "480P", "image_url": "data:f"})

    def test_ltx_names_the_frames_shape(self):
        req = req_for("video", "venice", "ltx-2.3", mode="first", aspect_ratio="16:9")
        self.assertEqual(venice.build_video(req, "data:f", None, [])["aspect_ratio"], "16:9")

    def test_images_edit_with_references_and_generate_without(self):
        req = req_for("image", "venice", "seedream-5-pro", aspect_ratio="9:16", seed=5)
        path, body = venice.build_image(req, ["data:a", "data:b"])
        self.assertEqual(path, "/image/multi-edit")
        self.assertEqual(body, {"modelId": "seedream-v5-pro-edit", "prompt": req.prompt, "images": ["data:a", "data:b"],
                                "aspect_ratio": "9:16", "safe_mode": False, "resolution": "1K"})
        path, body = venice.build_image(req, [])
        self.assertEqual(path, "/image/generate")
        self.assertEqual((body["model"], body["safe_mode"], body["return_binary"], body["seed"]), ("seedream-v5-pro", False, True, 5))
        with self.assertRaisesRegex(RuntimeError, "needs at least one reference"):
            venice.build_image(req_for("image", "venice", "qwen-edit"), [])


class VeniceFlowTests(MediaCase):
    def setUp(self):
        super().setUp()
        patch = unittest.mock.patch.object(MediaConfig, "VENICE_API_KEY", "venice-test-key")
        patch.start()
        self.addCleanup(patch.stop)

    def http(self, **routes):
        return FakeHTTP([("/video/quote", FakeResponse(200, {"quote": 0.98})),
                         ("/video/queue", routes.get("queue", FakeResponse(200, {"model": "m", "queue_id": "q1"}))),
                         ("/video/retrieve", routes.get("retrieve", FakeResponse(200, {"status": "PROCESSING", "average_execution_time": 1,
                                                                                         "execution_duration": 1},
                                                                                     headers={"content-type": "application/json"}))),
                         ("/video/complete", FakeResponse(200, {"success": True})),
                         ("/image/", routes.get("image", FakeResponse(200, content=small_jpeg(), headers={"content-type": "image/jpeg"}))),
                         ("/api_keys/rate_limits", FakeResponse(200, {"data": {"balances": {"USD": 7.5, "DIEM": 0}}})),
                         ("https://dl/", FakeResponse(200, content=CLIP))])

    def test_a_clip_is_quoted_then_queued_with_data_urls(self):
        http = self.http()
        frame = assets.store(noisy_jpeg(720, 1280))[0]
        req = req_for("video", "venice", "wan-3.0", mode="first", first=frame)
        with http.patch():
            self.assertEqual(run(Venice().submit(req)), "q1")
        self.assertEqual(req.cost_usd, 0.98)
        sent = http.call("/video/queue")["json"]
        self.assertTrue(sent["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("image_url", req.params)
        self.assertEqual(http.call("/video/quote")["json"], {"model": "wan-3-0-image-to-video", "duration": "15s", "resolution": "480p",
                                                              "aspect_ratio": "adaptive", "audio": False})

    def test_a_queue_with_its_own_download_link_keeps_it_in_the_task_id(self):
        queue = FakeResponse(200, {"model": "m", "queue_id": "q2", "download_url": "https://dl/q2.mp4"})
        frame = assets.store(noisy_jpeg(720, 1280))[0]
        with self.http(queue=queue).patch():
            task = run(Venice().submit(req_for("video", "venice", "wan-3.0", mode="first", first=frame)))
        self.assertEqual(task, "q2|https://dl/q2.mp4")
        done = FakeResponse(200, {"status": "COMPLETED"}, headers={"content-type": "application/json"})
        with self.http(retrieve=done).patch():
            result = run(Venice().poll(task, req_for("video", "venice", "wan-3.0", mode="first")))
        self.assertEqual(result.mime, "video/mp4")

    def test_polling_states(self):
        req = req_for("video", "venice", "wan-3.0", mode="first")
        with self.http().patch():
            self.assertIsNone(run(Venice().poll("q1", req)))
        http = self.http(retrieve=FakeResponse(200, content=CLIP, headers={"content-type": "video/mp4"}))
        with http.patch():
            self.assertEqual(run(Venice().poll("q1", req)).mime, "video/mp4")
        self.assertEqual(http.call("/video/complete")["json"], {"queue_id": "q1"})
        with self.http(retrieve=FakeResponse(404, {"error": "gone"})).patch(), self.assertRaisesRegex(RuntimeError, "expired"):
            run(Venice().poll("q1", req))
        with self.http(retrieve=FakeResponse(422, {"error": "policy"})).patch(), self.assertRaisesRegex(RuntimeError, "content policy"):
            run(Venice().poll("q1", req))

    def test_images_come_back_at_once_and_a_blur_is_an_error(self):
        venice_provider = Venice()
        ref = assets.store(noisy_jpeg(800, 1200))[0]
        req = req_for("image", "venice", "qwen-edit", refs=[ref], aspect_ratio="9:16")
        with self.http().patch():
            task = run(venice_provider.submit(req))
            self.assertEqual(run(venice_provider.poll(task, req)).mime, "image/jpeg")
        blurred = FakeResponse(200, content=small_jpeg(), headers={"x-venice-is-blurred": "true"})
        with self.http(image=blurred).patch(), self.assertRaisesRegex(RuntimeError, "blurred"):
            run(venice_provider.submit(req))

    def test_balance_comes_from_the_keys_own_limits(self):
        with self.http().patch():
            self.assertEqual(run(Venice().balance()), {"amount": 7.5, "unit": "USD"})


if __name__ == "__main__":
    unittest.main()
