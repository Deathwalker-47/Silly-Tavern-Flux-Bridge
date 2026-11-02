"""Jobs from start to finish on the mock provider: states, keyframes, cancel, restart, retention."""
import asyncio
import json
from pathlib import Path
import time
import unittest
import unittest.mock

from media_support import MediaCase, noisy_jpeg, run
from media import assets, catalog, db, director, pipelines, telemetry
from media.config import MediaConfig
from media.jobs import JobRunner, TERMINAL, public_job
from media.providers import PROVIDERS
from media.providers.base import RenderResult
from media.providers.mock import Mock

IMAGE = {"provider": "mock", "family": "seedream-4.5"}
VIDEO = {"provider": "mock", "family": "minimax-h3"}


def plan(**kw):
    base = {"prompt": "Maya at the window", "main_character": "c:maya", "people": [{"key": "c:maya", "role": "main"}],
            "motion_prompt": "Maya turns", "last_frame_prompt": "Maya sits", "aspect_ratio": "9:16", "caption": "hi"}
    return json.dumps({**base, **kw})


class JobCase(MediaCase):
    def setUp(self):
        super().setUp()
        self.runner = JobRunner()
        self.replies = []
        self.calls = []

        async def chat(messages, **_kw):
            self.calls.append(messages)
            return self.replies[0] if len(self.replies) == 1 else self.replies.pop(0)
        patch = unittest.mock.patch.object(director.helper, "chat", chat)
        patch.start()
        self.addCleanup(patch.stop)

    def person(self, age=24, slots=("face_1",)):
        refs = {slot: assets.store(noisy_jpeg(700 + i * 10, 900))[0].id for i, slot in enumerate(slots)}
        return {"key": "c:maya", "name": "Maya", "age": age, "appearance": "short dark hair", "refs": refs}

    def context(self, **kw):
        return {"context": {"characters": [self.person(**kw)], "reply": "Maya smiles", "history": []}}

    async def wait(self, job_id, seconds=5):
        deadline = time.time() + seconds
        while time.time() < deadline:
            row = db.generation(job_id)
            if row["status"] in TERMINAL and job_id not in self.runner.tasks:
                return row
            await asyncio.sleep(0.01)
        self.fail(f"job stuck at {db.generation(job_id)['status']}")

    def finished(self, body):
        async def scenario():
            return await self.wait(self.runner.create(body)["id"])
        return run(scenario())


class ImageJobTests(JobCase):
    def test_a_directed_image_runs_to_done_and_records_everything(self):
        self.replies = [plan()]
        row = self.finished({"kind": "image", "selection": IMAGE, "chat_id": "chat", "message_key": "m1", "folder": "Maya", **self.context()})
        self.assertEqual(row["status"], "done", row["error"])
        self.assertTrue(Path(row["file_path"]).exists())
        self.assertEqual(row["mime"], "image/jpeg")
        self.assertEqual(json.loads(row["plan_json"])["caption"], "hi")
        self.assertEqual(json.loads(row["refs_json"])[0]["slot"], "face_1")
        self.assertEqual(json.loads(row["params_json"])["references"], 1)
        self.assertEqual(db.list_generations("c:maya")[0]["id"], row["id"])

    def test_the_prompt_reaches_the_provider_with_numbered_tokens(self):
        self.replies = [plan(prompt="{{ref:Maya:face_1}} smiles")]
        row = self.finished({"kind": "image", "selection": IMAGE, **self.context()})
        self.assertEqual(row["prompt"], "image 1 smiles")

    def test_minor_guard_stops_the_job_before_any_provider_call(self):
        self.replies = [plan()]
        with unittest.mock.patch.object(Mock, "submit", side_effect=AssertionError("must not be called")):
            row = self.finished({"kind": "image", "selection": IMAGE, **self.context(age=17)})
        self.assertEqual(row["status"], "failed")
        self.assertIn("18", row["error"])
        self.assertIsNone(row["provider_task_id"])

    def test_the_director_flag_stops_the_job_too(self):
        self.replies = [plan(blocked=True, block_reason="sexualizes a minor")]
        row = self.finished({"kind": "image", "selection": IMAGE, **self.context()})
        self.assertEqual((row["status"], row["error"]), ("failed", "sexualizes a minor"))

    def test_a_failed_director_fails_the_job(self):
        self.replies = ["no json at all"]
        row = self.finished({"kind": "image", "selection": IMAGE, **self.context()})
        self.assertIn("director failed", row["error"])
        self.assertEqual(len(self.calls), 2)

    def test_a_provider_error_fails_the_job_with_its_text(self):
        row = self.finished({"kind": "image", "selection": IMAGE, "prompt": "MOCK_FAIL now"})
        self.assertEqual((row["status"], row["error"]), ("failed", "Mock provider failed on request"))

    def test_direct_jobs_skip_the_director(self):
        row = self.finished({"kind": "image", "selection": IMAGE, "prompt": "a red chair", "aspect_ratio": "16:9"})
        self.assertEqual(row["status"], "done")
        self.assertEqual(self.calls, [])
        self.assertEqual(row["aspect_ratio"], "16:9")

    def test_direct_prompts_still_pass_the_minor_guard(self):
        row = self.finished({"kind": "image", "selection": IMAGE, "prompt": "a schoolgirl"})
        self.assertEqual(row["status"], "failed")

    def test_a_fixed_aspect_beats_the_director(self):
        self.replies = [plan(aspect_ratio="16:9")]
        row = self.finished({"kind": "image", "selection": IMAGE, "aspect": "3:4", **self.context()})
        self.assertEqual(row["aspect_ratio"], "3:4")

    def test_no_aspect_from_the_director_follows_the_screen(self):
        self.replies = [plan(aspect_ratio="")]
        body = {"kind": "image", "selection": IMAGE, **self.context()}
        body["context"]["device"] = "landscape"
        self.assertEqual(self.finished(body)["aspect_ratio"], "16:9")

    def test_the_best_option_runs_two_steps_and_records_both(self):
        self.replies = [plan()]
        row = self.finished({"kind": "image", "selection": {"provider": "mock", "family": "best"}, **self.context(slots=("face_1", "full_body"))})
        self.assertEqual(row["status"], "done", row["error"])
        self.assertEqual(row["family"], "seedream-4.5")
        step = db.generation(row["parent_id"])
        self.assertEqual((step["status"], step["family"]), ("done", "seedream-5-pro"))
        self.assertIn("finished scene", row["prompt"])

    def test_a_route_test_of_the_best_option_sends_its_references_to_both_steps(self):
        asset = assets.store(noisy_jpeg(700, 900))[0]
        row = self.finished({"kind": "image", "selection": {"provider": "mock", "family": "best"}, "prompt": "a red chair", "test": True, "refs": [asset.id]})
        self.assertEqual(row["status"], "done", row["error"])
        for generation in (row, db.generation(row["parent_id"])):
            self.assertEqual(json.loads(generation["params_json"])["references"], 2 if generation is row else 1)

    def test_the_gallery_counts_what_the_earlier_steps_cost(self):
        self.replies = [plan()]
        row = self.finished({"kind": "image", "selection": {"provider": "mock", "family": "best"}, "message_key": "m1", **self.context()})
        db.update_generation(row["parent_id"], cost_usd=0.045, cost_credits=1.25)
        listed = db.list_generations("c:maya")[0]
        self.assertEqual((listed["steps_usd"], listed["steps_credits"]), (0.045, 1.25))

    def test_a_panel_image_lands_in_that_persons_gallery(self):
        row = self.finished({"kind": "image", "selection": IMAGE, "prompt": "a red chair", "person_key": "c:maya"})
        self.assertEqual(row["status"], "done", row["error"])
        self.assertEqual(db.list_generations("c:maya")[0]["id"], row["id"])


class LoraSwitchTests(JobCase):
    FLUX = {"provider": "mock", "family": "flux-lora"}

    def test_flux_with_no_lora_moves_to_seedream_with_references(self):
        self.replies = [plan()]
        row = self.finished({"kind": "image", "selection": self.FLUX, **self.context()})
        self.assertEqual((row["status"], row["family"]), ("done", "seedream-4.5"), row["error"])
        self.assertEqual(len(self.calls), 2)        # planned again in the new model's format

    def test_flux_jobs_describe_one_character_in_pov(self):
        self.replies = [plan()]
        with unittest.mock.patch("media.loras.character_lora", return_value={"id": "maya", "prepend_prompt": "maya33"}), \
             unittest.mock.patch("media.loras.lora_stack", return_value=[{"id": "maya", "prepend_prompt": "maya33", "url": "u", "weight": 1.0}]), \
             unittest.mock.patch("media.loras.compose", return_value=("maya33 Maya at the window", "")):
            row = self.finished({"kind": "image", "selection": self.FLUX, **self.context()})
        self.assertEqual(row["family"], "flux-lora")
        self.assertEqual(row["prompt"], "maya33 Maya at the window")
        self.assertIn('"force_pov": true', self.calls[0][-1]["content"])
        self.assertEqual(json.loads(row["loras_json"])[0]["id"], "maya")


class VideoJobTests(JobCase):
    def test_reference_mode_makes_no_still(self):
        self.replies = [plan()]
        row = self.finished({"kind": "video", "selection": {**VIDEO, "mode": "reference"}, **self.context()})
        self.assertEqual(row["status"], "done", row["error"])
        self.assertEqual(row["mode"], "reference")
        self.assertIsNone(row["parent_id"])
        self.assertEqual(row["mime"], "video/mp4")

    def test_reference_mode_with_empty_slots_makes_a_still_first(self):
        self.replies = [plan()]
        row = self.finished({"kind": "video", "selection": {**VIDEO, "mode": "reference"}, "image_selection": IMAGE,
                             **self.context(slots=())})
        self.assertEqual(row["status"], "done", row["error"])
        self.assertTrue(row["parent_id"])
        self.assertIn("<Picture 1> sets the people", row["prompt"])

    def test_first_frame_uses_this_messages_image(self):
        self.replies = [plan()]
        frame = assets.store(noisy_jpeg(), "frame")[0]
        row = self.finished({"kind": "video", "selection": {**VIDEO, "mode": "first"}, "message_image_asset": frame.id, **self.context()})
        self.assertEqual(row["status"], "done", row["error"])
        self.assertIsNone(row["parent_id"])

    def test_first_and_last_make_two_stills_with_the_same_shape(self):
        self.replies = [plan()]
        row = self.finished({"kind": "video", "selection": {**VIDEO, "mode": "first_last"}, "image_selection": IMAGE, **self.context()})
        self.assertEqual(row["status"], "done", row["error"])
        first = db.generation(row["parent_id"])
        last = next(r for r in db.generations_since(0) if r["kind"] == "image" and r["parent_id"] == row["id"] and r["id"] != first["id"])
        self.assertEqual(first["aspect_ratio"], last["aspect_ratio"])
        self.assertIn("opening frame", last["prompt"])

    def test_the_test_flag_picks_the_shortest_clip(self):
        row = self.finished({"kind": "video", "selection": {**VIDEO, "mode": "first"}, "test": True, "prompt": "a walk",
                             "message_image_asset": assets.store(noisy_jpeg(), "frame")[0].id})
        self.assertEqual(row["seconds"], 2)

    def test_normal_jobs_use_the_longest_clip(self):
        self.replies = [plan()]
        row = self.finished({"kind": "video", "selection": {**VIDEO, "mode": "reference"}, **self.context()})
        self.assertEqual(row["seconds"], 15)


class CreateTests(JobCase):
    def test_a_missing_reference_leaves_no_row(self):
        with self.assertRaises(KeyError):
            async def scenario():
                self.runner.create({"kind": "video", "selection": {**VIDEO, "mode": "first"}, "message_image_asset": "0" * 32})
            run(scenario())
        self.assertEqual(db.open_generations(), [])

    def test_an_unconfigured_provider_is_refused(self):
        with unittest.mock.patch.object(MediaConfig, "NANOGPT_API_KEY", ""), self.assertRaisesRegex(ValueError, "not configured"):
            self.runner.create({"kind": "image", "selection": {"provider": "nanogpt", "family": "seedream-4.5"}})

    def test_a_mode_the_model_lacks_is_refused(self):
        with unittest.mock.patch.object(MediaConfig, "NANOGPT_API_KEY", "k"), self.assertRaisesRegex(ValueError, "no reference mode"):
            self.runner.create({"kind": "video", "selection": {"provider": "nanogpt", "family": "wan-3.0", "mode": "reference"}})

    def test_a_banned_route_is_refused_unless_it_is_a_test(self):
        db.set_route("mock", "mock-image", "banned")
        with self.assertRaises(catalog.RouteNotAllowed):
            self.runner.create({"kind": "image", "selection": IMAGE, "prompt": "x"})
        row = self.finished({"kind": "image", "selection": IMAGE, "prompt": "x", "test": True})
        self.assertEqual(row["status"], "done")

    def test_a_double_tap_returns_the_running_job(self):
        async def scenario():
            first = self.runner.create({"kind": "image", "selection": IMAGE, "prompt": "x", "message_key": "m1"})
            second = self.runner.create({"kind": "image", "selection": IMAGE, "prompt": "x", "message_key": "m1"})
            self.assertEqual(first["id"], second["id"])
            other = self.runner.create({"kind": "video", "selection": {**VIDEO, "mode": "first"}, "prompt": "x", "message_key": "m1",
                                        "message_image_asset": assets.store(noisy_jpeg(), "frame")[0].id})
            self.assertNotEqual(first["id"], other["id"])
            await self.wait(first["id"])
            await self.wait(other["id"])
        run(scenario())


class CancelTests(JobCase):
    def test_cancel_stops_the_render_and_its_keyframe(self):
        async def scenario():
            with unittest.mock.patch.object(MediaConfig, "MOCK_RENDER_SECONDS", 30):
                self.replies = [plan()]
                row = self.runner.create({"kind": "video", "selection": {**VIDEO, "mode": "first"}, "image_selection": IMAGE, **self.context()})
                for _ in range(500):
                    if self.runner.children.get(row["id"]):
                        break
                    await asyncio.sleep(0.01)
                child = self.runner.children[row["id"]][0]
                self.runner.cancel(row["id"])
                await asyncio.sleep(0.05)
                return db.generation(row["id"]), db.generation(child)
        job, child = run(scenario())
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(child["status"], "cancelled")

    def test_cancel_tells_the_provider(self):
        stopped = []

        async def stop(task_id):
            stopped.append(task_id)

        async def scenario():
            with unittest.mock.patch.object(MediaConfig, "MOCK_RENDER_SECONDS", 30), \
                 unittest.mock.patch.object(PROVIDERS["mock"], "cancel", stop):
                row = self.runner.create({"kind": "image", "selection": IMAGE, "prompt": "x"})
                for _ in range(500):
                    if db.generation(row["id"])["provider_task_id"]:
                        break
                    await asyncio.sleep(0.01)
                self.runner.cancel(row["id"])
                await asyncio.sleep(0.05)
        run(scenario())
        self.assertEqual(len(stopped), 1)

    def test_cancelling_a_finished_job_changes_nothing(self):
        row = self.finished({"kind": "image", "selection": IMAGE, "prompt": "x"})

        async def scenario():
            return self.runner.cancel(row["id"])
        self.assertEqual(run(scenario())["status"], "done")


class RestartTests(JobCase):
    def open_row(self, **kw):
        fields = dict(id="r1", kind="image", status="rendering", provider="mock", model_id="mock-image", family="seedream-4.5",
                      mode="reference", prompt="x", aspect_ratio="9:16", chat_id="chat")
        db.insert_generation(**{**fields, **kw})

    def restart(self, runner=None):
        runner = runner or JobRunner()

        async def scenario():
            await runner.startup()
            for task in list(runner.tasks.values()):
                await task
            await runner.shutdown()
        run(scenario())
        return runner

    def test_a_polling_job_carries_on_from_its_task_id(self):
        self.open_row(provider_task_id="1.0:abc")
        self.restart()
        row = db.generation("r1")
        self.assertEqual(row["status"], "done", row["error"])
        self.assertTrue(Path(row["file_path"]).exists())

    def test_shutdown_leaves_the_row_open_for_the_next_start(self):
        async def scenario():
            with unittest.mock.patch.object(MediaConfig, "MOCK_RENDER_SECONDS", 0.3):
                runner = JobRunner()
                row = runner.create({"kind": "image", "selection": IMAGE, "prompt": "x"})
                for _ in range(500):
                    if db.generation(row["id"])["provider_task_id"]:
                        break
                    await asyncio.sleep(0.01)
                await runner.shutdown()
                return row["id"]
        job_id = run(scenario())
        self.assertEqual(db.generation(job_id)["status"], "rendering")
        self.restart()
        self.assertEqual(db.generation(job_id)["status"], "done")

    def test_a_job_with_no_task_id_fails_as_interrupted(self):
        self.open_row(status="planning")
        self.restart()
        row = db.generation("r1")
        self.assertEqual((row["status"], row["error"]), ("failed", "Interrupted by a bridge restart"))

    def test_a_socket_job_fails_and_is_cancelled_so_it_stops_billing(self):
        self.open_row(provider="sogni", model_id="minimax-h3-fastvideo-omni-int8_r2v_turbo", kind="video", provider_task_id="p9")
        stopped = []

        async def stop(task_id):
            stopped.append(task_id)

        async def scenario():
            with unittest.mock.patch.object(PROVIDERS["sogni"], "cancel", stop):
                runner = JobRunner()
                await runner.startup()
                await asyncio.sleep(0.02)
                await runner.shutdown()
        run(scenario())
        self.assertEqual(db.generation("r1")["status"], "failed")
        self.assertEqual(stopped, ["p9"])

    def test_a_keyframe_under_an_open_job_ends_with_it(self):
        self.open_row(id="video", kind="video", status="keyframe", provider="mock", model_id="mock-video", family="minimax-h3")
        self.open_row(id="frame", provider_task_id="1.0:abc", parent_id="video")
        self.restart()
        self.assertEqual(db.generation("video")["status"], "failed")
        self.assertEqual(db.generation("frame")["status"], "failed")

    def test_the_final_render_resumes_after_its_keyframe_finished(self):
        self.open_row(id="frame", status="done", parent_id="video")
        self.open_row(id="video", kind="video", provider="mock", model_id="mock-video", family="minimax-h3",
                      mode="first", provider_task_id="1.0:abc", parent_id="frame", seconds=5)
        self.restart()
        self.assertEqual(db.generation("video")["status"], "done")


class RetentionTests(JobCase):
    def finished_file(self):
        return self.finished({"kind": "image", "selection": IMAGE, "prompt": "x", "chat_id": "chat"})

    def test_saved_files_go_after_7_days_and_keep_their_row(self):
        row = self.finished_file()
        db.update_generation(row["id"], st_url="/user/x.jpg", acked_at=1000, rating=5)
        self.assertEqual(self.runner.sweep(now=1000 + 8 * 86400), 1)
        kept = db.generation(row["id"])
        self.assertIsNone(kept["file_path"])
        self.assertEqual(kept["rating"], 5)
        self.assertFalse(Path(row["file_path"]).exists())

    def test_unsaved_files_stay_for_30_days(self):
        row = self.finished_file()
        now = row["finished_at"]
        self.assertEqual(self.runner.sweep(now=now + 8 * 86400), 0)
        self.assertEqual(self.runner.sweep(now=now + 31 * 86400), 1)

    def test_only_unsaved_jobs_come_back_to_a_chat(self):
        row = self.finished_file()
        self.assertEqual([r["id"] for r in db.open_for_chat("chat")], [row["id"]])
        db.update_generation(row["id"], st_url="/user/x.jpg")
        self.assertEqual(db.open_for_chat("chat"), [])


class PublicJobTests(JobCase):
    def test_the_browser_gets_a_url_and_the_caption_never_a_path(self):
        self.replies = [plan()]
        row = self.finished({"kind": "image", "selection": IMAGE, "folder": "Maya", **self.context()})
        public = public_job(row)
        self.assertEqual(public["file_url"], f"/media/files/{row['id']}")
        self.assertEqual(public["caption"], "hi")
        self.assertNotIn(str(self.data), json.dumps(public))


class CollectTests(MediaCase):
    def test_a_network_blip_never_ends_the_wait(self):
        import httpx
        replies = [httpx.HTTPError("reset"), None, RenderResult(data=b"x" * 200, mime="image/jpeg")]

        class Provider:
            name = "fake"
            poll_seconds = 0

            async def poll(self, _task, _req):
                reply = replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply
        req = pipelines.RenderRequest(option=catalog.find("image", "mock", "seedream-4.5"), mode="reference", prompt="x")
        self.assertEqual(run(pipelines.collect(Provider(), "t", req, time.time())).mime, "image/jpeg")

    def test_the_deadline_ends_the_wait(self):
        class Provider:
            name = "fake"
            poll_seconds = 0

            async def poll(self, _task, _req):
                return None
        req = pipelines.RenderRequest(option=catalog.find("image", "mock", "seedream-4.5"), mode="reference", prompt="x")
        with self.assertRaises(TimeoutError):
            run(pipelines.collect(Provider(), "t", req, time.time() - 10_000))
        steps = [e for e in db.events_since(0) if e["event"] == "render"]
        self.assertEqual(steps[0]["ok"], 0)


if __name__ == "__main__":
    unittest.main()
