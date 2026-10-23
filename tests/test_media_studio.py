r"""
Tests for Media Studio: video provider payloads, the model catalog, video jobs,
the helper LLM guards and the avatar/LoRA image-mode choice.

The provider payloads are the risky part: every provider names the first frame,
the last frame and the safety switch differently, and a wrong name either fails
the request or silently drops the frame. These tests pin them down without
spending credits.
"""
import asyncio
import base64
from io import BytesIO
from pathlib import Path
import sys
import types
import unittest
import unittest.mock

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

    fastapi.FastAPI = FastAPI
    fastapi.HTTPException = HTTPException
    fastapi.Request = Request

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

    sys.modules.setdefault("fastapi", fastapi)
    sys.modules.setdefault("fastapi.middleware.cors", cors_mod)
    sys.modules.setdefault("fastapi.responses", responses_mod)
    sys.modules.setdefault("pydantic", pydantic_mod)
    sys.modules.setdefault("uvicorn", uvicorn_mod)
    sys.modules.setdefault("httpx", httpx_mod)
    sys.modules.setdefault("sse_starlette", sse_root)
    sys.modules.setdefault("sse_starlette.sse", sse_mod)


install_test_stubs()
import flux_lora_bridge as bridge  # noqa: E402
import media_studio as ms  # noqa: E402

try:
    from PIL import Image as _Image
    HAS_PILLOW = hasattr(_Image, "new")
except Exception:
    HAS_PILLOW = False


def make_jpeg(width: int, height: int) -> bytes:
    if not HAS_PILLOW:
        return b"\xff\xd8\xff\xe0" + b"\x00" * 100
    buf = BytesIO()
    _Image.new("RGB", (width, height), (200, 120, 90)).save(buf, format="JPEG")
    return buf.getvalue()


PORTRAIT = make_jpeg(768, 1152)
LANDSCAPE = make_jpeg(1152, 768)


def run(coro):
    return asyncio.run(coro)


def request(provider, family, variant=None, *, image=PORTRAIT, last=None, duration=5, resolution=None, **kw):
    v = ms.find_variant(provider, family, variant)
    return ms.VideoRequest(
        provider=provider, family=family, variant=v, prompt="she turns and smiles", image=image,
        last_image=last, duration=v.pick_duration(duration), resolution=v.pick_resolution(resolution), **kw,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class CatalogTests(unittest.TestCase):
    def test_every_variant_is_well_formed(self):
        for provider, families in ms.VIDEO_CATALOG.items():
            for family, variants in families.items():
                self.assertIn(family, ms.VIDEO_FAMILIES, f"{provider}: unknown family {family}")
                self.assertTrue(variants, f"{provider}/{family} has no variants")
                for v in variants:
                    self.assertIn(v.default_resolution, v.resolutions, f"{provider}/{v.id}")
                    self.assertTrue(v.duration_range or v.duration_options, f"{provider}/{v.id} has no durations")

    def test_missing_models_are_simply_absent(self):
        # Sogni has no Wan 2.7 or Seedance 1.5; Runware has no Wan 2.2.
        self.assertNotIn("wan-2.7", ms.VIDEO_CATALOG["sogni"])
        self.assertNotIn("seedance-1.5", ms.VIDEO_CATALOG["sogni"])
        self.assertNotIn("wan-2.2", ms.VIDEO_CATALOG["runware"])
        with self.assertRaises(ValueError):
            ms.find_variant("runware", "wan-2.2")

    def test_every_requested_family_is_offered_somewhere(self):
        offered = {fam for fams in ms.VIDEO_CATALOG.values() for fam in fams}
        self.assertEqual(offered, set(ms.VIDEO_FAMILIES))

    def test_duration_snaps_to_options_and_clamps_to_range(self):
        wan22 = ms.find_variant("nanogpt", "wan-2.2")       # options 5 / 8
        self.assertEqual(wan22.pick_duration(6), 5)
        self.assertEqual(wan22.pick_duration(7.5), 8)
        ltx = ms.find_variant("runware", "ltx-2.3")         # range 1..20
        self.assertEqual(ltx.pick_duration(30), 20)
        self.assertEqual(ltx.pick_duration(None), 5)

    def test_unknown_resolution_falls_back_to_default(self):
        h3 = ms.find_variant("runware", "minimax-h3")
        self.assertEqual(h3.pick_resolution("4k"), h3.default_resolution)
        self.assertEqual(h3.pick_resolution("768P"), "768p")

    def test_public_catalog_marks_unconfigured_providers(self):
        old = ms.MediaConfig.NANOGPT_API_KEY
        ms.MediaConfig.NANOGPT_API_KEY = ""
        try:
            cat = ms.public_catalog()
            nanogpt = next(p for p in cat["providers"] if p["id"] == "nanogpt")
            self.assertFalse(nanogpt["configured"])
            self.assertTrue(nanogpt["families"])
        finally:
            ms.MediaConfig.NANOGPT_API_KEY = old


# ---------------------------------------------------------------------------
# Provider payloads
# ---------------------------------------------------------------------------

class RunwarePayloadTests(unittest.TestCase):
    def test_first_and_last_frames_async_mp4_and_safety_off(self):
        req = request("runware", "seedance-2.0", last=LANDSCAPE, resolution="720p")
        task = ms.RunwareVideo().build_task(req, "uuid-first", "uuid-last", "task-1")
        self.assertEqual(task["taskType"], "videoInference")
        self.assertEqual(task["model"], "bytedance:seedance@2.0-fast")
        self.assertEqual(task["inputs"]["frameImages"], [
            {"image": "uuid-first", "frame": "first"}, {"image": "uuid-last", "frame": "last"},
        ])
        self.assertEqual(task["deliveryMethod"], "async")
        self.assertEqual(task["outputFormat"], "MP4")
        self.assertNotIn("safety", task)  # opt-in on Runware; the live API rejects checkContent
        self.assertEqual(task["resolution"], "720p")
        self.assertNotIn("width", task)

    @unittest.skipUnless(HAS_PILLOW, "needs Pillow to read the keyframe size")
    def test_ltx_uses_width_height_matching_the_keyframe_orientation(self):
        req = request("runware", "ltx-2.3", resolution="480p")
        task = ms.RunwareVideo().build_task(req, "u1", None, "t")
        self.assertEqual((task["width"], task["height"]), (512, 896))  # portrait keyframe
        self.assertNotIn("resolution", task)
        self.assertEqual(len(task["inputs"]["frameImages"]), 1)

    def test_every_preset_size_is_a_multiple_of_64(self):
        # Runware's real API rejected 480 px: "Video width must be ... in multiples of 64".
        for tier, sizes in ms.WH_SIZES.items():
            for orient, (w, h) in sizes.items():
                self.assertEqual((w % 64, h % 64), (0, 0), f"{tier}/{orient} = {w}x{h}")


class NanoGPTPayloadTests(unittest.TestCase):
    def test_only_the_models_own_parameters_are_sent(self):
        body = ms.NanoGPTVideo().build_body(request("nanogpt", "wan-2.2", duration=8))
        self.assertEqual(body["model"], "wan-video-22")
        self.assertTrue(body["imageDataUrl"].startswith("data:image/"))
        self.assertEqual(body["duration"], "8")
        self.assertNotIn("seed", body)          # wan-video-22 has no seed parameter
        self.assertNotIn("aspect_ratio", body)

    def test_last_frame_and_audio_off_where_supported(self):
        body = ms.NanoGPTVideo().build_body(request("nanogpt", "seedance-2.0", last=PORTRAIT, seed=7))
        self.assertTrue(body["last_image"].startswith("data:image/"))
        self.assertIs(body["generateAudio"], False)
        self.assertEqual(body["seed"], 7)
        self.assertIn(body["aspect_ratio"], ["16:9", "9:16", "1:1", "3:4", "4:3", "21:9"])

    def test_spicy_variants_are_the_defaults(self):
        for family in ("ltx-2.3", "wan-2.7", "seedance-2.0", "minimax-h3"):
            self.assertIn("spicy", ms.find_variant("nanogpt", family).id)


class SogniParamTests(unittest.TestCase):
    def test_filter_off_and_raw_bytes_for_the_frame(self):
        params = ms.SogniVideo().build_params(request("sogni", "ltx-2.3"))
        self.assertEqual(params["modelId"], "ltx23-22b-10eros-v1.4-fp8mixed_i2v")
        self.assertIs(params["disableNSFWFilter"], True)
        self.assertIsInstance(params["referenceImage"], bytes)
        self.assertEqual(params["numberOfMedia"], 1)

    def test_minimax_h3_duration_respects_the_24fps_floor(self):
        # Sogni's real API: "Video duration must be between 5.1666 and 15.0833".
        self.assertEqual(ms.find_variant("sogni", "minimax-h3").pick_duration(5), 6)

    def test_minimax_h3_gets_the_vbvr_lora_by_default(self):
        params = ms.SogniVideo().build_params(request("sogni", "minimax-h3"))
        self.assertEqual(params["loras"], ["h3-vbvr-video-reasoning"])
        self.assertEqual(params["loraStrengths"], [0.8])
        self.assertNotIn("width", params)

    def test_last_frame_switches_h3_to_its_first_last_frame_model(self):
        params = ms.SogniVideo().build_params(request("sogni", "minimax-h3", last=LANDSCAPE))
        self.assertEqual(params["modelId"], "minimax-h3-fastvideo-int8_flf2v_turbo")
        self.assertIsInstance(params["referenceImageEnd"], bytes)

    def test_non_h3_models_get_no_lora(self):
        self.assertNotIn("loras", ms.SogniVideo().build_params(request("sogni", "wan-2.2")))

    def test_network_follows_request_then_env(self):
        build = ms.SogniVideo().build_params
        with unittest.mock.patch.object(ms.MediaConfig, "SOGNI_NETWORK", "relaxed"):
            self.assertEqual(build(request("sogni", "ltx-2.3"))["network"], "relaxed")
            self.assertEqual(build(request("sogni", "ltx-2.3", sogni_network="fast"))["network"], "fast")
        with unittest.mock.patch.object(ms.MediaConfig, "SOGNI_NETWORK", "bogus"):
            self.assertEqual(build(request("sogni", "ltx-2.3", sogni_network="nope"))["network"], "fast")


class ReplicatePayloadTests(unittest.TestCase):
    def build(self, family, variant=None, **kw):
        req = request("replicate", family, variant, **kw)
        return ms.ReplicateVideo().build_input(req, "https://f/first", "https://f/last" if kw.get("last") else None)

    def test_wan_22_names_and_safety_off(self):
        inp = self.build("wan-2.2", last=PORTRAIT, duration=6)
        self.assertEqual(inp["image"], "https://f/first")
        self.assertEqual(inp["last_image"], "https://f/last")
        self.assertIs(inp["disable_safety_checker"], True)
        self.assertEqual(inp["num_frames"], 97)  # 6 s at 16 fps (+1)

    def test_wan_27_uses_first_frame_and_last_frame(self):
        inp = self.build("wan-2.7", last=PORTRAIT)
        self.assertEqual(inp["first_frame"], "https://f/first")
        self.assertEqual(inp["last_frame"], "https://f/last")
        self.assertNotIn("image", inp)

    def test_seedance_and_minimax_field_names(self):
        sd = self.build("seedance-1.5", last=PORTRAIT)
        self.assertEqual(sd["last_frame_image"], "https://f/last")
        self.assertIs(sd["generate_audio"], False)
        h3 = self.build("minimax-h3")
        self.assertEqual(h3["first_frame_image"], "https://f/first")
        self.assertEqual(h3["ratio"], "adaptive")

    @unittest.skipUnless(HAS_PILLOW, "needs Pillow to read the keyframe size")
    def test_ltx_portrait_keyframe_requests_9_16(self):
        self.assertEqual(self.build("ltx-2.3")["aspect_ratio"], "9:16")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class VideoJobTests(unittest.TestCase):
    def test_mock_job_runs_to_done_and_stores_the_clip(self):
        async def scenario():
            req = ms.VideoRequest(provider="mock", family="ltx-2.3", variant=ms.MOCK_CATALOG["ltx-2.3"][0],
                                  prompt="p", image=PORTRAIT)

            async def run_job(job):
                return await ms.MockVideo().generate(req, ms.reporter(job))

            job = ms.video_jobs.create("mock", "ltx-2.3", "mock", run_job)
            self.assertEqual(job.status, "queued")
            for _ in range(100):
                await asyncio.sleep(0.05)
                if job.status in ("done", "failed"):
                    break
            return job

        job = run(scenario())
        self.assertEqual(job.status, "done", job.error)
        public = job.to_public()
        self.assertTrue(public["video_path"].endswith(".mp4"))
        self.assertEqual(ms.video_store.get(job.video_id), ms.MockVideo.CLIP.read_bytes())

    def test_failing_job_reports_the_error(self):
        async def scenario():
            async def boom(job):
                raise RuntimeError("provider exploded")
            job = ms.video_jobs.create("mock", "ltx-2.3", "mock", boom)
            for _ in range(40):
                await asyncio.sleep(0.02)
                if job.status == "failed":
                    break
            return job

        job = run(scenario())
        self.assertEqual(job.status, "failed")
        self.assertIn("provider exploded", job.error)


# ---------------------------------------------------------------------------
# Helper LLM
# ---------------------------------------------------------------------------

class HelperTests(unittest.TestCase):
    def with_reply(self, reply):
        helper = ms.HelperLLM()

        async def fake_chat(messages, **kw):
            self.last_messages = messages
            return reply
        helper.chat = fake_chat
        return helper

    def test_same_words_ignores_tags_and_punctuation(self):
        self.assertTrue(ms.same_words("No way, really?", "No way, [laugh] <whisper>really?</whisper>"))
        self.assertFalse(ms.same_words("No way, really?", "No way, honestly? [laugh]"))

    def test_voice_tags_accepts_a_tag_only_edit(self):
        out = run(self.with_reply("Oh! [laugh] You called.").voice_tags("Oh! You called.", char="Mia"))
        self.assertEqual(out, {"text": "Oh! [laugh] You called.", "changed": True})

    def test_voice_tags_rejects_changed_words(self):
        out = run(self.with_reply("Oh hey! [laugh] You finally called.").voice_tags("Oh! You called."))
        self.assertEqual(out["text"], "Oh! You called.")
        self.assertFalse(out["changed"])
        self.assertTrue(out["rejected"])

    def test_sanitizer_fixes_small_model_tag_mistakes(self):
        inline, wrap = ms.INLINE_TAGS, ms.WRAPPING_TAGS
        # Llama 3.1 8B really answered like this: inline tags in wrapping syntax.
        self.assertEqual(ms.sanitize_speech_tags("Wait, <pause> seriously? <laugh> You drove here?", inline, wrap),
                         "Wait, [pause] seriously? [laugh] You drove here?")
        self.assertEqual(ms.sanitize_speech_tags("[whisper] hi <Whisper>quiet</Whisper> [Long Pause] <dance>x</dance>", inline, wrap),
                         "hi <whisper>quiet</whisper> [long-pause] x")
        self.assertEqual(ms.sanitize_speech_tags("<soft>goodnight", inline, wrap), "<soft>goodnight</soft>")
        self.assertEqual(ms.sanitize_speech_tags("stray</slow> end", inline, wrap), "stray end")

    def test_voice_tags_are_sanitized_after_the_word_check(self):
        out = run(self.with_reply("Oh! <laugh> You called.").voice_tags("Oh! You called."))
        self.assertEqual(out["text"], "Oh! [laugh] You called.")

    def test_voice_tags_level_none_skips_the_llm(self):
        out = run(self.with_reply("SHOULD NOT BE USED").voice_tags("Hi.", tag_level="none"))
        self.assertEqual(out["text"], "Hi.")

    def test_decide_parses_fenced_json(self):
        reply = '```json\n{"action": "image", "reason": "asked for a selfie", "image_prompt": "Mia, mirror selfie", "motion_prompt": "", "caption": "here"}\n```'
        out = run(self.with_reply(reply).decide(messages=[{"name": "You", "is_user": True, "text": "send a selfie"}], char="Mia", user="You"))
        self.assertEqual(out["action"], "image")
        self.assertEqual(out["image_prompt"], "Mia, mirror selfie")

    def test_decide_respects_disallowed_actions_and_force(self):
        reply = '{"action": "video", "image_prompt": "x", "motion_prompt": "y"}'
        blocked = run(self.with_reply(reply).decide(messages=[], char="Mia", user="You", allow_video=False))
        self.assertEqual(blocked["action"], "none")
        forced = run(self.with_reply('{"action": "none"}').decide(messages=[], char="Mia", user="You", force="video"))
        self.assertEqual(forced["action"], "video")

    def test_extract_json_handles_chatter(self):
        self.assertEqual(ms.extract_json('Sure! {"a": {"b": 1}} hope that helps')["a"]["b"], 1)
        with self.assertRaises(ValueError):
            ms.extract_json("no json here")


# ---------------------------------------------------------------------------
# Bridge: Replicate Flux client + avatar/LoRA mode choice
# ---------------------------------------------------------------------------

class ReplicateFluxTests(unittest.TestCase):
    def test_loras_go_in_as_huggingface_urls_with_safety_off(self):
        client = bridge.ReplicateClient("tok")
        inp = client.build_input("nimya33 on a beach", [
            {"url": "https://huggingface.co/a/b/resolve/main/x.safetensors", "weight": 0.8},
            {"url": "https://huggingface.co/a/b/resolve/main/y.safetensors", "weight": 0.5},
        ], {"width": 832, "height": 1216, "steps": 40, "cfg_scale": 3.5, "seed": 42})
        self.assertEqual(len(inp["hf_loras"]), 2)
        self.assertEqual(inp["lora_scales"], [0.8, 0.5])
        self.assertIs(inp["disable_safety_checker"], True)
        self.assertEqual(inp["aspect_ratio"], "2:3")
        self.assertEqual(inp["seed"], 42)

    def test_default_image_chain_is_runware_then_replicate(self):
        self.assertEqual(bridge.Config.IMAGE_PROVIDER_ORDER[:2], ["runware", "replicate"])
        for deprecated in ("wavespeed", "fal", "together"):
            self.assertNotIn(deprecated, bridge.Config.IMAGE_PROVIDER_ORDER)


class AvatarModeTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self._orig = (bridge.txt2img, bridge._prepare_kontext_inputs, bridge.clients["runware"].edit,
                      bridge.lora_manager.match_loras_by_keywords, bridge.Config.ENABLE_SUMMARIZATION)
        bridge.Config.ENABLE_SUMMARIZATION = False

        async def fake_txt2img(req):
            self.calls.append(("txt2img", req.prompt))
            return types.SimpleNamespace(images=[base64.b64encode(b"lora-image").decode()], image_urls=[],
                                         parameters={"final_prompt": req.prompt})

        async def fake_prepare(image, reference_image, req_id):
            self.calls.append(("prepare", image, reference_image))
            return ["uuid-avatar"], 1024, 1024

        async def fake_edit(instruction, refs, params):
            self.calls.append(("edit", instruction, refs, params))
            return PORTRAIT

        bridge.txt2img = fake_txt2img
        bridge._prepare_kontext_inputs = fake_prepare
        bridge.clients["runware"].edit = fake_edit

    def tearDown(self):
        (bridge.txt2img, bridge._prepare_kontext_inputs, bridge.clients["runware"].edit,
         bridge.lora_manager.match_loras_by_keywords, bridge.Config.ENABLE_SUMMARIZATION) = self._orig

    def set_lora(self, has_lora):
        data = [{"id": "char", "data": {"category": "character"}, "reason": "keyword:mia"}] if has_lora else []
        bridge.lora_manager.match_loras_by_keywords = lambda prompt, neg: data

    def media_request(self, mode, avatars=("data:image/png;base64,AAAA",)):
        return types.SimpleNamespace(prompt="Mia at the beach", negative_prompt="", mode=mode, character_name="Mia",
                                     character_prompts={}, visible_characters=["Mia"], avatars=list(avatars),
                                     width=832, height=1216, steps=30, cfg_scale=3.5, seed=-1)

    def test_auto_uses_lora_when_the_character_has_one(self):
        self.set_lora(True)
        _img, meta = run(bridge._media_image(self.media_request("auto")))
        self.assertEqual(meta["mode"], "lora")
        self.assertEqual(self.calls[0][0], "txt2img")

    def test_auto_uses_avatar_kontext_without_a_lora(self):
        self.set_lora(False)
        _img, meta = run(bridge._media_image(self.media_request("auto")))
        self.assertEqual(meta["mode"], "avatar")
        edit = next(c for c in self.calls if c[0] == "edit")
        self.assertEqual(edit[3]["model"], bridge.Config.AVATAR_KONTEXT_MODEL)
        self.assertIn("Mia", edit[1])

    def test_avatar_mode_without_avatars_falls_back_to_txt2img(self):
        self.set_lora(False)
        _img, meta = run(bridge._media_image(self.media_request("avatar", avatars=())))
        self.assertEqual(meta["mode"], "text")

    def test_lora_mode_ignores_avatars(self):
        self.set_lora(False)
        _img, meta = run(bridge._media_image(self.media_request("lora")))
        self.assertEqual(self.calls[0][0], "txt2img")
        self.assertEqual(meta["mode"], "text")

    def test_avatar_model_is_not_the_immich_editor_model(self):
        # /edit (Immich editor) keeps KONTEXT_MODEL; avatars have their own setting.
        self.assertEqual(bridge.Config.AVATAR_KONTEXT_MODEL, "runware:106@1")
        self.assertEqual(bridge.Config.KONTEXT_MODEL, "bfl:3@1")


if __name__ == "__main__":
    unittest.main()
