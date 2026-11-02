r"""
Tests for what stayed around the media package: the helper LLM, voice tags, the Replicate
Flux client and Runware's LoRA mapping. The rest lives in tests/test_media_*.py.
"""
import asyncio
import importlib
import hashlib
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
        def include_router(self, *a, **kw): return None

    class Request:
        async def json(self): return {}

    class APIRouter:
        def _route(self, *a, **kw): return lambda fn: fn
        get = post = put = patch = delete = _route

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
    """GLM reasons before answering, so a small max_tokens would come back empty."""

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
        self.assertEqual(payload["max_tokens"], helper_module.REASONING_MIN_TOKENS)

    def test_a_larger_budget_is_kept(self):
        self.assertEqual(self.send("z-ai/glm-5.3-flash-uncensored", max_tokens=3000)["max_tokens"], 3000)

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

    def test_upload_uses_httpx_and_surfaces_errors(self):
        ok = lambda *a, **kw: types.SimpleNamespace(status_code=200)  # noqa: E731  (HEAD on the source file)
        reply = types.SimpleNamespace(status_code=200, text="", json=lambda: {"errors": [{"message": "bad file"}]})
        fake = types.SimpleNamespace(head=ok, post=lambda *a, **kw: reply, TimeoutException=TimeoutError)
        with unittest.mock.patch.object(bridge, "httpx", fake):
            with self.assertRaisesRegex(Exception, "bad file"):
                bridge.upload_lora_to_runware(self.URL, "key", "flux")

        def slow(*_a, **_kw):
            raise TimeoutError()
        with unittest.mock.patch.object(bridge, "httpx", types.SimpleNamespace(head=ok, post=slow, TimeoutException=TimeoutError)):
            self.assertTrue(bridge.upload_lora_to_runware(self.URL, "key", "flux").startswith("deathwalker:"))

    def test_upload_refuses_a_deleted_source_file(self):
        # Runware answers "model re-processing" for a dead link and the LoRA never works.
        posted = []
        fake = types.SimpleNamespace(head=lambda *a, **kw: types.SimpleNamespace(status_code=404),
                                     post=lambda *a, **kw: posted.append(1), TimeoutException=TimeoutError)
        with unittest.mock.patch.object(bridge, "httpx", fake):
            with self.assertRaisesRegex(Exception, "HTTP 404"):
                bridge.upload_lora_to_runware(self.URL, "key", "flux")
        self.assertEqual(posted, [])

    def test_upload_refuses_a_private_source_file(self):
        posted = []
        fake = types.SimpleNamespace(head=lambda *a, **kw: types.SimpleNamespace(status_code=403),
                                     post=lambda *a, **kw: posted.append(1), TimeoutException=TimeoutError)
        with unittest.mock.patch.object(bridge, "httpx", fake):
            with self.assertRaisesRegex(Exception, "isn't public"):
                bridge.upload_lora_to_runware(self.URL, "key", "flux")
        self.assertEqual(posted, [])

    def test_flux_air_is_the_plain_url_hash_so_uploads_keep_their_id(self):
        expected = hashlib.sha256(self.URL.lower().encode()).hexdigest()[:12]
        self.assertEqual(bridge.generate_air_id_from_url(self.URL)[0], f"deathwalker:{expected}@1")
        self.assertNotEqual(bridge.generate_air_id_from_url(self.URL, "zimage")[0], f"deathwalker:{expected}@1")

    def test_zimage_upload_sends_its_architecture(self):
        sent = []
        reply = types.SimpleNamespace(status_code=200, text="", json=lambda: {"data": [{"status": "ready"}]})

        def post(*_a, **kw):
            sent.append(kw["json"][0])
            return reply
        fake = types.SimpleNamespace(head=lambda *a, **kw: types.SimpleNamespace(status_code=200), post=post,
                                     TimeoutException=TimeoutError)
        with unittest.mock.patch.object(bridge, "httpx", fake):
            air = bridge.upload_lora_to_runware(self.URL, "key", "zimage")
        self.assertEqual(sent[0]["architecture"], "z_image")
        self.assertEqual(sent[0]["air"], air)

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
