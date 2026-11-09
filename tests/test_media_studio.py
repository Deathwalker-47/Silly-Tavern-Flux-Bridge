r"""
Tests for what stayed around the media package: the helper LLM, voice tags, the Replicate
Flux client and Runware's LoRA mapping. The rest lives in tests/test_media_*.py.
"""
import asyncio
import importlib
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
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
        def include_router(self, *a, **kw): return None

    class Request:
        async def json(self): return {}

    class APIRouter:
        def _route(self, *a, **kw): return lambda fn: fn
        get = post = put = patch = delete = api_route = _route

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
import media as ms  # noqa: E402
helper_module = importlib.import_module("media.helper")  # media.helper the name is the client object

def run(coro):
    return asyncio.run(coro)


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
        self.assertTrue(helper_module.same_words("No way, really?", "No way, [laugh] <whisper>really?</whisper>"))
        self.assertFalse(helper_module.same_words("No way, really?", "No way, honestly? [laugh]"))

    def test_voice_tags_accepts_a_tag_only_edit(self):
        out = run(self.with_reply("Oh! [laugh] You called.").voice_tags("Oh! You called.", char="Mia"))
        self.assertEqual(out, {"text": "Oh! [laugh] You called.", "changed": True})

    def test_voice_tags_rejects_changed_words(self):
        out = run(self.with_reply("Oh hey! [laugh] You finally called.").voice_tags("Oh! You called."))
        self.assertEqual(out["text"], "Oh! You called.")
        self.assertFalse(out["changed"])
        self.assertTrue(out["rejected"])

    def test_sanitizer_fixes_small_model_tag_mistakes(self):
        inline, wrap = helper_module.INLINE_TAGS, helper_module.WRAPPING_TAGS
        # Llama 3.1 8B really answered like this: inline tags in wrapping syntax.
        self.assertEqual(helper_module.sanitize_speech_tags("Wait, <pause> seriously? <laugh> You drove here?", inline, wrap),
                         "Wait, [pause] seriously? [laugh] You drove here?")
        self.assertEqual(helper_module.sanitize_speech_tags("[whisper] hi <Whisper>quiet</Whisper> [Long Pause] <dance>x</dance>", inline, wrap),
                         "hi <whisper>quiet</whisper> [long-pause] x")
        self.assertEqual(helper_module.sanitize_speech_tags("<soft>goodnight", inline, wrap), "<soft>goodnight</soft>")
        self.assertEqual(helper_module.sanitize_speech_tags("stray</slow> end", inline, wrap), "stray end")

    def test_voice_tags_are_sanitized_after_the_word_check(self):
        out = run(self.with_reply("Oh! <laugh> You called.").voice_tags("Oh! You called."))
        self.assertEqual(out["text"], "Oh! [laugh] You called.")

    def test_cast_lists_the_people_who_can_answer_with_known_voices_only(self):
        reply = json.dumps({"people": [
            {"name": "Nadia", "gender": "Female", "about": "The innkeeper.", "sound": "warm, low", "voice": "ara"},
            {"name": "Sam", "gender": "male", "about": "The caller.", "voice": "rex"},
            {"name": "nadia", "gender": "female", "about": "Duplicate."},
            {"name": "Bram", "gender": "dwarf", "about": "The cook.", "voice": "made-up"},
            {"name": "  ", "about": "No name."},
        ]})
        helper = self.with_reply(f"Here you go:\n```json\n{reply}\n```")
        voices = [{"voice_id": "ara", "name": "Ara", "description": "Warm"}, {"voice_id": "rex", "name": "Rex"}]
        card = {"name": "Seaside Inn", "description": "An inn run by Nadia. Bram cooks.", "scenario": "", "lore": "Bram is gruff."}
        out = run(helper.cast(card, user="Sam", voices=voices))
        self.assertEqual([p["name"] for p in out["people"]], ["Nadia", "Bram"])
        self.assertEqual(out["people"][0], {"name": "Nadia", "gender": "female", "about": "The innkeeper.",
                                            "sound": "warm, low", "appearance": "", "age": None, "voice": "ara"})
        self.assertEqual((out["people"][1]["gender"], out["people"][1]["voice"]), ("other", ""))
        system, user_msg = self.last_messages[0]["content"], self.last_messages[1]["content"]
        self.assertIn("phone call from Sam", system)
        self.assertIn("- ara: Ara, Warm", system)
        self.assertIn("lore:\nBram is gruff.", user_msg)
        self.assertNotIn("scenario:", user_msg, "empty fields are left out")

    def test_cast_sees_each_voice_language_so_a_malayalam_speaker_gets_a_malayalam_voice(self):
        reply = json.dumps({"people": [{"name": "Meera", "gender": "female", "about": "Talks in Malayalam.", "voice": "heygen:m1"}]})
        helper = self.with_reply(reply)
        voices = [{"voice_id": "ara", "name": "Ara", "gender": "female"},
                  {"voice_id": "heygen:m1", "name": "Meera", "gender": "female", "language": "Malayalam", "description": "Malayalam"},
                  {"voice_id": "elevenlabs:a", "name": "Asha", "gender": "female", "language": "Malayalam, English", "description": "warm"}]
        out = run(helper.cast({"name": "Kochi", "description": "Meera only speaks Malayalam."}, user="Sam", voices=voices))
        self.assertEqual(out["people"][0]["voice"], "heygen:m1")
        system = self.last_messages[0]["content"]
        self.assertIn("- heygen:m1: Meera, female, Malayalam, Malayalam", system)
        self.assertIn("- elevenlabs:a: Asha, female, Malayalam, English, warm", system)
        self.assertIn("pick a voice listed for that language", system)

    def test_cast_reports_a_look_and_only_an_age_the_card_states(self):
        reply = json.dumps({"people": [
            {"name": "Nadia", "about": "The innkeeper.", "appearance": "Tall, red hair, green apron.", "age": "34"},
            {"name": "Bram", "about": "The cook.", "appearance": "", "age": "a grown man"},
            {"name": "Tam", "about": "The stable kid.", "age": "9"},
        ]})
        out = run(self.with_reply(reply).cast({"name": "Seaside Inn", "description": "An inn."}, user="Sam"))
        looks = {p["name"]: (p["appearance"], p["age"]) for p in out["people"]}
        self.assertEqual(looks["Nadia"], ("Tall, red hair, green apron.", 34))
        self.assertEqual(looks["Bram"], ("", None), "an age that isn't a number is left unset")
        self.assertEqual(looks["Tam"][1], 9, "reported as written; the panel and the guard judge it")
        self.assertIn("Never guess an age", self.last_messages[0]["content"])

    def test_voice_tags_level_none_skips_the_llm(self):
        out = run(self.with_reply("SHOULD NOT BE USED").voice_tags("Hi.", tag_level="none"))
        self.assertEqual(out["text"], "Hi.")

    def test_extract_json_handles_chatter(self):
        self.assertEqual(helper_module.extract_json('Sure! {"a": {"b": 1}} hope that helps')["a"]["b"], 1)
        with self.assertRaises(ValueError):
            helper_module.extract_json("no json here")


# ---------------------------------------------------------------------------
# Bridge: Replicate Flux client
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


class HelperModelTests(unittest.TestCase):
    """GLM reasons before answering, and the reasoning spends max_tokens first, so the answer
    needs its budget on top of the reasoning. Without that, long chats came back empty or cut off."""

    def send(self, model, max_tokens=100):
        sent = []

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def post(self, _url, json=None, headers=None):
                sent.append(json)
                return types.SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "ok"}}]})
        with unittest.mock.patch.object(helper_module, "httpx", types.SimpleNamespace(AsyncClient=FakeClient)), \
             unittest.mock.patch.object(ms.MediaConfig, "HELPER_API_KEY", "key"):
            asyncio.run(ms.HelperLLM().chat([{"role": "user", "content": "hi"}], model=model, max_tokens=max_tokens))
        return sent[0]

    def test_glm_gets_low_reasoning_and_room_to_answer(self):
        payload = self.send("z-ai/glm-5.3-flash-uncensored")
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_tokens"], 100 + helper_module.REASONING_ROOM)

    def test_the_director_keeps_its_whole_answer_budget_after_reasoning(self):
        # On 2026-11-08 the plan's reasoning used 2,560 to 3,000 of a 3,000 budget, leaving no answer.
        self.assertGreaterEqual(helper_module.REASONING_ROOM, 3000)
        self.assertEqual(self.send("z-ai/glm-5.3-flash-uncensored", max_tokens=3000)["max_tokens"],
                         3000 + helper_module.REASONING_ROOM)

    def test_other_uncensored_models_are_left_alone(self):
        payload = self.send("some/other-uncensored")
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["max_tokens"], 100)

    def test_a_sfw_model_cannot_run_in_the_chat_pipeline(self):
        with self.assertRaisesRegex(RuntimeError, "uncensored"):
            self.send("openai/gpt-4o-mini")

    def test_default_model_is_glm_uncensored(self):
        self.assertTrue(ms.MediaConfig.HELPER_MODEL.startswith("z-ai/glm"))

    def test_no_together_fallback(self):
        with unittest.mock.patch.object(ms.MediaConfig, "HELPER_API_KEY", ""):
            self.assertEqual(ms.HelperLLM().backend()[3], "none")

    def test_summarizer_keeps_the_prompt_on_an_empty_reply(self):
        async def empty(*_a, **_kw):
            return ""
        with unittest.mock.patch.object(ms.helper, "chat", empty), \
             unittest.mock.patch.object(ms.MediaConfig, "HELPER_API_KEY", "key"):
            result = asyncio.run(bridge.DeepSeekSummarizer().summarize_prompt("a long scene"))
        self.assertEqual(result, "a long scene")


class FakeRunwareSocket:
    """Runware's WebSocket API: authentication, then one message per upload status.
    A reply is a status item, {"error": ...}, or {"other_task": True} to come from another task."""

    def __init__(self, replies, refuse_key=False):
        self.replies, self.refuse_key, self.sent, self.was_opened = list(replies), refuse_key, [], False

    def opened(self):
        self.was_opened = True
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def send(self, text):
        self.sent.extend(json.loads(text))

    def recv(self, timeout=None):
        if len(self.sent) == 1:
            if self.refuse_key:
                return json.dumps({"errors": [{"code": "invalidApiKey", "message": "Invalid API key"}]})
            return json.dumps({"data": [{"taskType": "authentication", "connectionSessionUUID": "s1"}]})
        if not self.replies:
            raise TimeoutError()
        reply = dict(self.replies.pop(0))
        task = self.sent[1]["taskUUID"]
        if "error" in reply:
            return json.dumps({"errors": [{"message": reply["error"], "taskUUID": task}]})
        task = "another-task" if reply.pop("other_task", False) else task
        return json.dumps({"data": [{"taskType": "modelUpload", "taskUUID": task, "air": self.sent[1]["air"], **reply}]})


class RunwareLoraMappingTests(unittest.TestCase):
    """Two of 26 mapped LoRAs pointed at models Runware no longer had (uploads that timed out)."""
    URL = "https://huggingface.co/u/r/resolve/main/a.safetensors"

    def resolve(self, entry, visible, upload_id="deathwalker:new@1", arch="flux"):
        saved, uploads = {}, []

        def upload(url, _key, _arch):
            uploads.append(url)
            return upload_id
        with unittest.mock.patch.object(bridge, "load_runware_lora_mapping", return_value={self.URL: dict(entry)}), \
             unittest.mock.patch.object(bridge, "save_runware_lora_mapping", side_effect=saved.update), \
             unittest.mock.patch.object(bridge, "runware_private_loras", return_value=visible), \
             unittest.mock.patch.dict(bridge._RUNWARE_FAILED, clear=True), \
             unittest.mock.patch.object(bridge, "upload_lora_to_runware", side_effect=upload):
            resolved = bridge.resolve_runware_loras([{"lora": self.URL, "weight": 0.8}], "key", arch)
        return resolved, uploads, saved

    def test_missing_model_is_uploaded_again(self):
        resolved, uploads, saved = self.resolve({"runware_id": "deathwalker:old@1", "uploaded_at": "2026-01-03T00:02:24"}, {})
        self.assertEqual(resolved, [{"lora": "deathwalker:new@1", "weight": 0.8}])
        self.assertEqual(uploads, [self.URL])
        self.assertEqual(saved[self.URL]["runware_id"], "deathwalker:new@1")
        self.assertEqual(saved[self.URL]["architecture"], "flux1d")

    def test_listed_fresh_or_unknown_mappings_are_trusted(self):
        old = {"runware_id": "deathwalker:old@1", "uploaded_at": "2026-01-03T00:02:24"}
        fresh = {"runware_id": "deathwalker:old@1", "uploaded_at": bridge.datetime.utcnow().isoformat()}
        for entry, visible in ((old, {"deathwalker:old@1": "flux1d"}), (fresh, {}), (old, None)):
            resolved, uploads, _ = self.resolve(entry, visible)
            self.assertEqual(resolved, [{"lora": "deathwalker:old@1", "weight": 0.8}])
            self.assertEqual(uploads, [])

    def test_private_model_listing_pages_and_caches(self):
        pages = [{"data": [{"results": [{"air": "a@1", "architecture": "flux1d"}, {"air": "b@1", "architecture": "flux1d"}], "totalResults": 3}]},
                 {"data": [{"results": [{"air": "c@1", "architecture": "z_image"}], "totalResults": 3}]}]
        calls = []

        def post(*_a, **kw):
            calls.append(kw["json"][0]["offset"])
            return types.SimpleNamespace(status_code=200, json=lambda page=pages[len(calls) - 1]: page)
        fake = types.SimpleNamespace(post=post, TimeoutException=TimeoutError)
        with unittest.mock.patch.object(bridge, "httpx", fake), \
             unittest.mock.patch.dict(bridge._RUNWARE_MODELS_CACHE, {"at": 0.0, "airs": None}):
            expected = {"a@1": "flux1d", "b@1": "flux1d", "c@1": "z_image"}
            self.assertEqual(bridge.runware_private_loras("key"), expected)
            self.assertEqual(bridge.runware_private_loras("key"), expected)
        self.assertEqual(calls, [0, 2])  # second call served from the cache

    # What Runware's WebSocket sent for a real upload on 2026-11-09, 143 s from first to last.
    PROCESSED = [{"status": "model re-processing"}, {"status": "validated"}, {"status": "downloaded"},
                 {"status": "ready", "message": "Model successfully deployed and ready for use."}]

    def upload(self, replies, arch="flux", source_status=200):
        """upload_lora_to_runware against a fake Runware socket. Returns (result or exception, socket, HEAD headers)."""
        socket, heads = FakeRunwareSocket(replies), []
        fake = types.SimpleNamespace(head=lambda _url, **kw: heads.append(kw["headers"]) or types.SimpleNamespace(status_code=source_status))
        with unittest.mock.patch.object(bridge, "httpx", fake), \
             unittest.mock.patch("websockets.sync.client.connect", lambda *_a, **_kw: socket.opened()):
            try:
                return bridge.upload_lora_to_runware(self.URL, "key", arch), socket, heads
            except Exception as e:
                return e, socket, heads

    def test_an_upload_is_followed_until_runware_can_use_it(self):
        # Over HTTP Runware answered "validated" and nothing more, so uploads it went on to
        # fail looked fine until the LoRA's first job died with "Invalid value for 'lora.model'".
        result, socket, _ = self.upload(self.PROCESSED)
        self.assertEqual(result, bridge.generate_air_id_from_url(self.URL)[0])
        self.assertEqual(socket.sent[0], {"taskType": "authentication", "apiKey": "key"})
        self.assertEqual(socket.sent[1]["taskType"], "modelUpload")
        self.assertEqual(socket.replies, [], "it read every status up to ready")

    def test_a_failed_upload_raises_runwares_reason(self):
        result, _, _ = self.upload([{"status": "validated"}, {"status": "downloaded"},
                                    {"status": "failed", "message": "Unsupported LoRA keys."}])
        self.assertIsInstance(result, bridge.RunwareProcessingFailed)
        self.assertIn("Unsupported LoRA keys.", str(result))

    def test_runware_errors_and_a_refused_key_surface(self):
        result, _, _ = self.upload([{"error": "bad file"}])
        self.assertIn("bad file", str(result))
        self.assertNotIsInstance(result, bridge.RunwareProcessingFailed)
        socket = FakeRunwareSocket([], refuse_key=True)
        with unittest.mock.patch.object(bridge, "httpx", types.SimpleNamespace(head=lambda *a, **kw: types.SimpleNamespace(status_code=200))), \
             unittest.mock.patch("websockets.sync.client.connect", lambda *_a, **_kw: socket.opened()), \
             self.assertRaisesRegex(Exception, "refused the key: Invalid API key"):
            bridge.upload_lora_to_runware(self.URL, "key", "flux")

    def test_other_tasks_messages_are_ignored(self):
        result, socket, _ = self.upload([{"status": "failed", "other_task": True}, *self.PROCESSED])
        self.assertTrue(str(result).startswith("deathwalker:"), result)

    def test_an_upload_still_processing_when_the_wait_ends_keeps_its_air(self):
        # The render waits for a LoRA that's still processing, and the mapping is re-checked later.
        result, _, _ = self.upload([{"status": "validated"}])     # then silence: recv times out
        self.assertTrue(str(result).startswith("deathwalker:"), result)
        with unittest.mock.patch.object(bridge, "RUNWARE_UPLOAD_WAIT", 0):
            result, socket, _ = self.upload(self.PROCESSED)
        self.assertTrue(str(result).startswith("deathwalker:"), result)
        self.assertEqual(len(socket.replies), 4, "no wait left, so nothing was read")

    def test_upload_refuses_a_deleted_source_file(self):
        # Runware answers "model re-processing" for a dead link and the LoRA never works.
        result, socket, _ = self.upload(self.PROCESSED, source_status=404)
        self.assertIn("HTTP 404", str(result))
        self.assertFalse(socket.was_opened)

    def test_upload_refuses_a_private_source_file(self):
        result, socket, _ = self.upload(self.PROCESSED, source_status=403)
        self.assertRegex(str(result), "isn't public.*Set HF_TOKEN")
        with unittest.mock.patch.object(bridge.ms.MediaConfig, "HF_TOKEN", "hf_test"):
            result, socket, _ = self.upload(self.PROCESSED, source_status=403)
        self.assertIn("HF_TOKEN can't read", str(result))
        self.assertFalse(socket.was_opened)

    def test_a_private_file_reaches_runware_as_a_bridge_link_and_the_check_sends_the_token(self):
        with unittest.mock.patch.multiple(bridge.ms.MediaConfig, HF_TOKEN="hf_test", PUBLIC_BASE_URL="https://bridge.test",
                                          LORA_LINK_SECRET="test-secret"):
            _, socket, heads = self.upload(self.PROCESSED)
        self.assertEqual(heads, [{"Authorization": "Bearer hf_test"}])
        self.assertRegex(socket.sent[1]["downloadURL"], r"^https://bridge\.test/lora-files/[\w-]+\.[\w-]{32}/a\.safetensors$")

    def test_flux_air_is_the_plain_url_hash_so_uploads_keep_their_id(self):
        expected = hashlib.sha256(self.URL.lower().encode()).hexdigest()[:12]
        self.assertEqual(bridge.generate_air_id_from_url(self.URL)[0], f"deathwalker:{expected}@1")
        self.assertNotEqual(bridge.generate_air_id_from_url(self.URL, "zimage")[0], f"deathwalker:{expected}@1")

    def test_zimage_upload_sends_its_architecture(self):
        air, socket, _ = self.upload(self.PROCESSED, arch="zimage")
        self.assertEqual(socket.sent[1]["architecture"], "z_image")
        self.assertEqual(socket.sent[1]["air"], air)

    def test_a_lora_runware_failed_to_process_is_reported_and_not_uploaded_again_at_once(self):
        calls = []

        def failing(url, _key, _arch):
            calls.append(url)
            raise bridge.RunwareProcessingFailed("Runware couldn't process this LoRA: Unsupported LoRA keys.")
        with unittest.mock.patch.object(bridge, "upload_lora_to_runware", side_effect=failing), \
             unittest.mock.patch.object(bridge, "load_runware_lora_mapping", return_value={}), \
             unittest.mock.patch.dict(bridge._RUNWARE_FAILED, clear=True):
            for _ in range(2):
                failures = []
                self.assertEqual(bridge.resolve_runware_loras([{"lora": self.URL, "weight": 1}], "key", "flux", failures), [])
                self.assertEqual(failures, [(self.URL, "Runware couldn't process this LoRA: Unsupported LoRA keys.")])
            self.assertEqual(len(calls), 1, "the second job got the reason without another upload")
            with unittest.mock.patch.object(bridge, "RUNWARE_FAILED_RETRY", 0):
                bridge.resolve_runware_loras([{"lora": self.URL, "weight": 1}], "key", "flux", [])
            self.assertEqual(len(calls), 2, "and it's tried again later")

    def test_a_network_error_is_not_remembered(self):
        calls = []

        def flaky(url, _key, _arch):
            calls.append(url)
            raise ConnectionError("reset")
        with unittest.mock.patch.object(bridge, "upload_lora_to_runware", side_effect=flaky), \
             unittest.mock.patch.object(bridge, "load_runware_lora_mapping", return_value={}), \
             unittest.mock.patch.dict(bridge._RUNWARE_FAILED, clear=True):
            for _ in range(2):
                failures = []
                bridge.resolve_runware_loras([{"lora": self.URL, "weight": 1}], "key", "flux", failures)
                self.assertEqual(failures, [(self.URL, "reset")])
        self.assertEqual(len(calls), 2)

    def test_two_jobs_that_need_the_same_new_lora_upload_it_once(self):
        stored, uploads, release = {}, [], threading.Event()

        def upload(url, _key, _arch):
            uploads.append(url)
            release.wait(5)             # the second job arrives while this upload is processing
            return "deathwalker:new@1"
        results = []
        with unittest.mock.patch.object(bridge, "upload_lora_to_runware", side_effect=upload), \
             unittest.mock.patch.object(bridge, "load_runware_lora_mapping", side_effect=lambda: json.loads(json.dumps(stored))), \
             unittest.mock.patch.object(bridge, "save_runware_lora_mapping", side_effect=stored.update), \
             unittest.mock.patch.object(bridge, "runware_private_loras", return_value={}):
            jobs = [threading.Thread(target=lambda: results.append(
                bridge.resolve_runware_loras([{"lora": self.URL, "weight": 1}], "key", "flux"))) for _ in range(2)]
            for job in jobs:
                job.start()
            time.sleep(0.2)
            release.set()
            for job in jobs:
                job.join(5)
        self.assertEqual(uploads, [self.URL])
        self.assertEqual(results, [[{"lora": "deathwalker:new@1", "weight": 1}]] * 2)
        self.assertEqual(stored[self.URL]["runware_id"], "deathwalker:new@1")

    def test_old_mapping_entries_count_as_flux(self):
        entry = {"runware_id": "deathwalker:old@1", "uploaded_at": "2026-01-03T00:02:24"}
        resolved, uploads, _ = self.resolve(entry, {"deathwalker:old@1": "flux1d"})
        self.assertEqual(resolved, [{"lora": "deathwalker:old@1", "weight": 0.8}])
        self.assertEqual(uploads, [])

    def test_file_moved_to_zimage_uploads_under_that_architecture(self):
        entry = {"runware_id": "deathwalker:old@1", "uploaded_at": "2026-01-03T00:02:24"}
        resolved, uploads, saved = self.resolve(entry, {"deathwalker:old@1": "flux1d"}, arch="zimage")
        self.assertEqual(resolved, [{"lora": "deathwalker:new@1", "weight": 0.8}])
        self.assertEqual(uploads, [self.URL])
        self.assertEqual(saved[self.URL]["architecture"], "z_image")

    def test_runware_listing_another_architecture_counts_as_stale(self):
        entry = {"runware_id": "deathwalker:old@1", "architecture": "z_image", "uploaded_at": "2026-01-03T00:02:24"}
        _, uploads, _ = self.resolve(entry, {"deathwalker:old@1": "flux1d"}, arch="zimage")
        self.assertEqual(uploads, [self.URL])


if __name__ == "__main__":
    unittest.main()
