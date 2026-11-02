"""The pipeline record: safe to keep, tied to its generation, and useful once it's read back."""
import base64
import importlib
import json
import time
import unittest
import unittest.mock

from media_support import MediaCase, noisy_jpeg, run
from media import assets, db, director, insights, telemetry
from media.config import MediaConfig
from media.jobs import JobRunner, TERMINAL


helper = importlib.import_module("media.helper")


class ScrubTests(unittest.TestCase):
    def test_image_data_becomes_a_size_note(self):
        data = "data:image/jpeg;base64," + base64.b64encode(b"x" * 3000).decode()
        self.assertIn("image data", telemetry.scrub({"a": data})["a"])
        self.assertEqual(telemetry.scrub(b"123"), "<3 bytes>")
        self.assertIn("image data", telemetry.scrub("Q" * 600))

    def test_secret_fields_are_hidden_at_any_depth(self):
        out = telemetry.scrub({"headers": {"Authorization": "Bearer abc", "x-api-key": "k"}, "items": [{"token": "t"}], "HELPER_API_KEY": "s"})
        self.assertNotIn("abc", json.dumps(out))
        self.assertEqual(out["items"][0]["token"], "<hidden>")
        self.assertEqual(out["HELPER_API_KEY"], "<hidden>")

    def test_ordinary_prompts_stay(self):
        self.assertEqual(telemetry.scrub({"prompt": "she smiles"}), {"prompt": "she smiles"})


class RecordTests(MediaCase):
    def test_a_failed_write_never_breaks_the_pipeline(self):
        with unittest.mock.patch.object(db, "add_event", side_effect=RuntimeError("disk full")):
            telemetry.record("job", "done")

    def test_step_times_and_records_failures_with_their_type(self):
        with self.assertRaises(ValueError):
            with telemetry.step("provider", "submit", provider="x") as facts:
                facts["extra"] = 1
                raise ValueError("bad reply")
        row = db.events_since(0)[0]
        data = json.loads(row["data_json"])
        self.assertEqual(row["ok"], 0)
        self.assertEqual((data["error_type"], data["extra"]), ("ValueError", 1))
        self.assertIsNotNone(row["ms"])

    def test_raw_text_goes_to_its_own_column_and_clears_on_schedule(self):
        with telemetry.step("llm", "plan") as facts:
            facts["raw"] = {"reply": "the whole reply"}
        row = db.events_since(0)[0]
        self.assertIn("whole reply", row["raw_json"])
        self.assertNotIn("whole reply", row["data_json"] or "")
        db.clear_raw_events(time.time() + 1)
        self.assertIsNone(db.events_since(0)[0]["raw_json"])

    def test_steps_inside_a_generation_carry_its_id(self):
        token = telemetry.current_generation.set("g1")
        try:
            telemetry.record("refs", "picked", sent=3)
        finally:
            telemetry.current_generation.reset(token)
        self.assertEqual(db.events_for("g1")[0]["stage"], "refs")


class JobTraceTests(MediaCase):
    KEY = "sk-very-secret-value"

    def finish_job(self, reply, **body):
        async def chat(messages, **_kw):
            return reply

        async def scenario():
            runner = JobRunner()
            row = runner.create(body)
            while db.generation(row["id"])["status"] not in TERMINAL or row["id"] in runner.tasks:
                await asyncio_sleep()
            return db.generation(row["id"])
        with unittest.mock.patch.object(director.helper, "chat", chat):
            return run(scenario())

    def test_a_whole_run_can_be_replayed_and_holds_no_secret_or_image(self):
        photo = assets.store(noisy_jpeg())[0]
        person = {"key": "c:maya", "name": "Maya", "age": 24, "refs": {"face_1": photo.id}}
        reply = json.dumps({"prompt": "{{ref:Maya:face_1}} smiles", "main_character": "c:maya", "people": [{"key": "c:maya", "role": "main"}]})
        row = self.finish_job(reply, kind="image", selection={"provider": "mock", "family": "seedream-4.5"}, chat_id="chat",
                              context={"characters": [person], "reply": "hi"}, origin="auto")
        self.assertEqual(row["status"], "done", row["error"])
        trace = telemetry.trace(row["id"])
        seen = [(s["stage"], s["event"]) for s in trace["steps"]]
        for expected in (("job", "created"), ("guard", "check"), ("refs", "picked"), ("pipeline", "aspect"), ("pipeline", "prompt"),
                         ("provider", "submit"), ("provider", "render"), ("pipeline", "stored"), ("job", "done")):
            self.assertIn(expected, seen)
        dump = json.dumps([dict(r) for r in db.events_for(row["id"])])
        self.assertNotIn(self.KEY, dump)
        self.assertNotIn("base64,", dump)


async def asyncio_sleep():
    import asyncio
    await asyncio.sleep(0.01)


class HelperTelemetryTests(MediaCase):
    def test_llm_calls_record_purpose_size_and_never_the_key(self):
        class Client:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def post(self, _url, json=None, headers=None):
                import types
                return types.SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"total_tokens": 9}})
        with unittest.mock.patch.object(helper.httpx, "AsyncClient", Client), \
             unittest.mock.patch.object(MediaConfig, "HELPER_API_KEY", "sk-secret-helper"):
            run(helper.HelperLLM().chat([{"role": "user", "content": "hi"}], model="mock/x", purpose="plan"))
        row = db.events_since(0)[0]
        data = json.loads(row["data_json"])
        self.assertEqual((row["stage"], row["event"], data["finish_reason"], data["usage"]["total_tokens"]), ("llm", "plan", "stop", 9))
        self.assertNotIn("sk-secret-helper", json.dumps(dict(row)))


class ReportTests(MediaCase):
    def add(self, id, rating=None, status="done", provider="mock", model="m1", kind="image", **kw):
        fields = dict(id=id, kind=kind, status=status, provider=provider, model_id=model, mode="reference", chat_id="c",
                      aspect_ratio="9:16", prompt="x" * 100, finished_at=int(time.time()) + 10, rating=rating)
        db.insert_generation(**{**fields, **kw})

    def test_routes_show_success_cost_and_rating(self):
        self.add("a", rating=5, cost_usd=0.04)
        self.add("b", rating=4, cost_usd=0.04)
        self.add("c", status="failed", error="provider said no")
        route = insights.report(1)["routes"][0]
        self.assertEqual((route["runs"], route["done"], route["failed"]), (3, 2, 1))
        self.assertEqual(route["rating"], 4.5)
        self.assertEqual(route["total_usd"], 0.08)
        self.assertEqual(route["top_errors"][0][0], "provider said no")

    def test_keyframes_stay_out_of_the_route_table_but_their_cost_counts(self):
        self.add("a", rating=5, cost_usd=0.5, kind="video")
        self.add("k", cost_usd=0.04, chat_id=None, model="key")
        report = insights.report(1)
        self.assertEqual([r["model"] for r in report["routes"]], ["m1"])
        self.assertEqual(report["totals"]["usd"], 0.54)

    def test_a_failing_route_is_called_out(self):
        for i in range(3):
            self.add(f"f{i}", status="failed", error="safety filter")
        self.add("ok")
        notes = insights.report(1)["suggestions"]
        self.assertTrue(any("safety filter" in n for n in notes), notes)

    def test_rating_gaps_between_models_are_called_out(self):
        for i in range(3):
            self.add(f"a{i}", rating=5, model="good")
            self.add(f"b{i}", rating=2, model="bad")
        notes = insights.report(1)["suggestions"]
        self.assertTrue(any("good rates 5" in n for n in notes), notes)
        self.assertTrue(any("bad on mock gets low ratings" in n for n in notes), notes)

    def test_ratings_split_by_person_and_prompt_length(self):
        self.add("a", rating=5)
        db.add_people("a", [{"key": "c:maya", "role": "main"}])
        tables = insights.report(1)["tables"]
        self.assertEqual(tables["person"][0]["value"], "c:maya")
        self.assertEqual(tables["prompt length"][0]["value"], "under 300")

    def test_slow_steps_and_voice_rejections_are_called_out(self):
        telemetry.record("director", "plan", ok=True, ms=120_000, attempt=1)
        for _ in range(10):
            telemetry.record("llm", "voice_tags", ok=True, ms=4000)
        for _ in range(3):
            telemetry.record("voice", "tags", ok=False, rejected=True)
        notes = insights.report(1)["suggestions"]
        self.assertTrue(any("director/plan is slow" in n for n in notes), notes)
        self.assertTrue(any("Voice tags changed the words" in n for n in notes), notes)
        self.assertTrue(any("Voice tags are slow" in n for n in notes), notes)

    def test_client_and_group_events_are_counted(self):
        telemetry.record("client", "auto_skip", reason="cooldown")
        telemetry.record("client", "group_round", stopped_by="user", messages=4)
        telemetry.record("group", "next_speaker", ok=True, speaker="Maya", long=True)
        report = insights.report(1)
        self.assertEqual(report["client"]["skips"][0], ("cooldown", 1))
        self.assertEqual(report["groups"]["round_stops"], {"user": 1})
        self.assertEqual(report["groups"]["long_replies"], 1)

    def test_export_is_one_json_object_per_line(self):
        self.add("a")
        telemetry.record("job", "done")
        lines = [json.loads(line) for line in insights.export_lines(1)]
        self.assertEqual({line["type"] for line in lines}, {"generation", "event"})


if __name__ == "__main__":
    unittest.main()
