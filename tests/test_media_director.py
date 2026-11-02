"""The director's contract, the minor guard, references and placeholder numbering."""
import json
import unittest
import unittest.mock

from io import BytesIO

from PIL import Image

from media_support import MediaCase, noisy_jpeg, run
from media import assets, catalog, director, pipelines, references, telemetry
from media.director import DirectorInput, Plan
from media.references import Person, Ref, pick_references, render_tokens

MAYA = {"key": "c:maya", "name": "Maya", "age": 24, "appearance": "short dark hair",
        "refs": {"face_1": "a1", "face_2": "a2", "full_body": "a3", "portrait": "a4", "naked": "a5"}}
LENA = {"key": "c:lena", "name": "Lena", "age": 30, "appearance": "tall, red hair", "refs": {"face_1": "b1", "full_body": "b2"}}


def plan_json(**kw):
    return json.dumps({"prompt": "Maya at the window", **kw})


class PlanTests(unittest.TestCase):
    def test_fills_defaults_and_ignores_unknown_fields(self):
        plan = Plan.parse(plan_json(extra="ignored"))
        self.assertEqual(plan.ref_order, references.DEFAULT_ORDER)
        self.assertFalse(hasattr(plan, "extra"))

    def test_a_plan_with_no_prompt_is_malformed(self):
        with self.assertRaisesRegex(ValueError, "no prompt"):
            Plan.parse('{"prompt": "  "}')

    def test_a_blocked_plan_needs_no_prompt(self):
        self.assertTrue(Plan.parse('{"blocked": true, "block_reason": "minor"}').blocked)

    def test_people_given_as_strings_are_accepted(self):
        self.assertEqual(Plan.parse(plan_json(people=["c:maya"])).people, [{"key": "c:maya", "role": "other"}])


class GuardTests(unittest.TestCase):
    def adult(self):
        return Person.from_context(MAYA)

    def test_the_model_flag_stops_the_job(self):
        self.assertIn("minor", director.minor_block(Plan(prompt="x", blocked=True, block_reason="a minor"), []))

    def test_the_age_message_names_who_needs_one(self):
        people = [Person.from_context({"key": "c:x", "name": "Xena"}), self.adult()]
        self.assertTrue(director.minor_block(Plan(prompt="x"), people).startswith("Xena needs an age"))

    def test_a_person_with_no_age_stops_the_job(self):
        person = Person.from_context({"key": "c:x", "name": "X"})
        self.assertIn("age", director.minor_block(Plan(prompt="x"), [person]))

    def test_a_person_under_18_stops_the_job(self):
        person = Person.from_context({"key": "c:x", "name": "X", "age": 17})
        self.assertIn("age", director.minor_block(Plan(prompt="x"), [person]))

    def test_words_stop_the_job_even_when_the_model_missed_them(self):
        for text in ("a schoolgirl in class", "the teen smiles", "a little kid", "she is 16 years old", "a 15yo"):
            self.assertIsNotNone(director.minor_block(Plan(prompt=text), [self.adult()]), text)

    def test_motion_and_last_frame_text_are_checked_too(self):
        self.assertIsNotNone(director.minor_block(Plan(prompt="ok", motion_prompt="a child runs in"), [self.adult()]))
        self.assertIsNotNone(director.minor_block(Plan(prompt="ok", last_frame_prompt="underage girl"), [self.adult()]))

    def test_adult_scenes_pass_untouched(self):
        for text in ("she undresses by the bed", "a 25 years old woman", "twenty-four and laughing", "kidding around at the bar"):
            self.assertIsNone(director.minor_block(Plan(prompt=text, adult=True), [self.adult()]), text)


class ShapeTests(MediaCase):
    """The director picks the shape from the scene and the references, and no reference is ever reframed."""

    def test_the_director_is_told_each_references_real_shape(self):
        tall = assets.store(noisy_jpeg(864, 1536))[0].id
        square = assets.store(noisy_jpeg(1000, 1000))[0].id
        person = {**MAYA, "refs": {"face_1": tall, "full_body": square, "portrait": "not-on-the-bridge"}}
        public = DirectorInput({"characters": [person]}).public(catalog.find("image", "nanogpt", "seedream-4.5"))
        self.assertEqual(public["characters"][0]["reference_shapes"],
                         {"face_1": "864x1536, closest 9:16", "full_body": "1000x1000, closest 1:1"})
        self.assertIn("never cropped, padded or stretched", director.DIRECTOR_RULES)
        self.assertIn("let the scene decide", director.DIRECTOR_RULES)
        self.assertIn("only to break a tie", director.DIRECTOR_RULES)

    def test_the_scene_decides_the_shape_whatever_the_references_look_like(self):
        from types import SimpleNamespace
        job = SimpleNamespace(device="portrait", aspect_setting="")
        for kind, provider, family in (("image", "nanogpt", "seedream-4.5"), ("video", "sogni", "minimax-h3")):
            option = catalog.find(kind, provider, family)
            wide = Plan(prompt="x", aspect_ratio="16:9")
            tall = Plan(prompt="x", aspect_ratio="9:16")
            self.assertEqual(pipelines.choose_aspect(job, option, wide), "16:9", kind)
            self.assertEqual(pipelines.choose_aspect(job, option, tall), "9:16", kind)
        fixed = SimpleNamespace(device="portrait", aspect_setting="3:4")      # your own setting still wins over the scene
        self.assertEqual(pipelines.choose_aspect(fixed, catalog.find("image", "nanogpt", "seedream-4.5"), Plan(prompt="x", aspect_ratio="16:9")), "3:4")

    def test_stored_and_fitted_references_keep_their_shape(self):
        for width, height in ((700, 1300), (1300, 700), (1000, 1000), (900, 1200)):
            stored = assets.store(noisy_jpeg(width, height))[0]
            self.assertEqual(assets.dimensions(stored.id), (width, height))
            for sent in assets.fit_refs([stored], 4_400_000):
                w, h = Image.open(BytesIO(sent)).size
                self.assertAlmostEqual(w / h, width / height, delta=0.005)

    def test_an_existing_opening_image_sets_the_clips_shape(self):
        option = catalog.find("video", "sogni", "minimax-h3")
        wide = assets.store(noisy_jpeg(1600, 900), "frame")[0]
        tall = assets.store(noisy_jpeg(900, 1600), "frame")[0]
        self.assertEqual(pipelines.follow_opening_frame(option, wide, "9:16"), "16:9")
        self.assertEqual(pipelines.follow_opening_frame(option, tall, "16:9"), "9:16")
        self.assertEqual(pipelines.follow_opening_frame(option, None, "16:9"), "16:9")


class InputTests(unittest.TestCase):
    def test_people_resolve_by_key_or_name_once_each(self):
        ctx = DirectorInput({"characters": [MAYA, LENA], "user": {"key": "p:sam", "name": "Sam", "age": 30}})
        chosen = ctx.people_by_key(["c:maya", "Maya", "lena", "nobody", "p:sam"])
        self.assertEqual([p.name for p in chosen], ["Maya", "Lena", "Sam"])

    def test_context_is_passed_through_whole_with_no_cuts(self):
        long_reply = "word " * 5000
        option = catalog.find("image", "nanogpt", "seedream-4.5")
        public = DirectorInput({"characters": [MAYA], "reply": long_reply, "history": [{"text": "x"}]}).public(option)
        self.assertEqual(public["reply"], long_reply)
        self.assertEqual(public["characters"][0]["slots"], list(MAYA["refs"]))
        self.assertEqual(public["people_cap"], 3)

    def test_lora_models_cap_at_one_person(self):
        option = catalog.find("image", "runware", "flux-lora")
        self.assertEqual(DirectorInput({"characters": [MAYA, LENA]}).public(option)["people_cap"], 1)

    def test_messages_carry_the_rules_the_guide_and_the_mode(self):
        option = catalog.find("video", "sogni", "minimax-h3")
        messages = director.director_messages(DirectorInput({"characters": [MAYA]}), option, "reference")
        text = "\n".join(m["content"] for m in messages)
        self.assertIn("media director", text)
        self.assertIn("Target model: MiniMax H3 Reference, FastH3", text)
        self.assertIn("Mode: reference video", text)

    def test_every_family_has_a_guide(self):
        for kind in ("image", "video"):
            for option in catalog.options(kind):
                if option.provider != "mock":
                    self.assertTrue(director.load_guide(option.family), option.family)


class PlanCallTests(MediaCase):
    option = property(lambda self: catalog.find("image", "nanogpt", "seedream-4.5"))

    def with_replies(self, *replies):
        queue = list(replies)

        async def chat(*_a, **kw):
            self.sent = kw
            reply = queue.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        return unittest.mock.patch.object(director.helper, "chat", chat)

    def test_one_retry_then_success(self):
        with self.with_replies("not json", plan_json()):
            plan = run(director.make_plan(DirectorInput({}), self.option, "reference"))
        self.assertEqual(plan.prompt, "Maya at the window")
        self.assertEqual(self.sent["response_format"], director.PLAN_SCHEMA)

    def test_two_failures_end_the_job_with_no_raw_fallback(self):
        with self.with_replies(RuntimeError("500"), "still not json"):
            with self.assertRaisesRegex(RuntimeError, "director failed"):
                run(director.make_plan(DirectorInput({}), self.option, "reference"))

    def test_decide_respects_what_is_allowed(self):
        with self.with_replies('{"action": "video", "reason": "x"}'):
            out = run(director.decide({"allow_video": False, "history": [], "reply": "hi"}))
        self.assertEqual(out["action"], "none")

    def test_decide_gets_the_whole_message(self):
        long_text = "word " * 2000
        seen = []

        async def chat(messages, **_kw):
            seen.append(messages[1]["content"])
            return '{"action": "image"}'
        with unittest.mock.patch.object(director.helper, "chat", chat):
            out = run(director.decide({"reply": long_text, "char": "Maya", "user": "Sam"}))
        self.assertIn(long_text.strip(), seen[0])
        self.assertEqual(out["action"], "image")

    def test_next_speaker_must_be_a_member(self):
        members = [{"name": "Maya"}, {"name": "Lena"}]
        with self.with_replies('{"speaker": "lena", "beat": "shrugs", "long": true}'):
            self.assertEqual(run(director.next_speaker({"members": members})), {"speaker": "Lena", "beat": "shrugs", "long": True})
        with self.with_replies('{"speaker": "Someone else"}'):
            self.assertEqual(run(director.next_speaker({"members": members}))["speaker"], "user")


class ReferenceTests(unittest.TestCase):
    def people(self):
        return [Person.from_context(MAYA), Person.from_context(LENA)]

    def test_everyone_gets_a_face_before_a_second_image(self):
        order = ["face_1", "full_body", "portrait", "face_2", "naked"]
        picked = pick_references(self.people(), order, 3)
        self.assertEqual([(r.person.name, r.slot) for r in picked], [("Maya", "face_1"), ("Lena", "face_1"), ("Maya", "full_body")])

    def test_the_cap_is_exact_for_each_model_size(self):
        for cap, expected in ((1, 1), (3, 3), (7, 7), (9, 7), (30, 7)):     # only 7 images exist
            self.assertEqual(len(pick_references(self.people(), references.DEFAULT_ORDER, cap)), expected)

    def test_three_people_on_a_three_reference_model(self):
        trio = self.people() + [Person.from_context({"key": "p", "name": "Sam", "refs": {"face_1": "c1", "portrait": "c2"}})]
        picked = pick_references(trio, references.DEFAULT_ORDER, 3)
        self.assertEqual({r.person.name for r in picked}, {"Maya", "Lena", "Sam"})

    def test_a_scene_order_changes_who_gets_what_first(self):
        picked = pick_references(self.people()[:1], ["naked", "face_1"], 2)
        self.assertEqual([r.slot for r in picked], ["naked", "face_1"])

    def test_placeholders_become_this_models_numbered_tokens(self):
        people = self.people()
        refs = pick_references(people, references.DEFAULT_ORDER, 3)
        h3 = catalog.find("video", "sogni", "minimax-h3")
        text = "{{ref:Maya:face_1}} kisses {{ref:Lena:face_1}} in {{ref:Maya:full_body}}"
        self.assertEqual(render_tokens(text, h3, refs), "<Picture 1> kisses <Picture 2> in <Picture 3>")

    def test_token_formats_differ_by_model(self):
        refs = [Ref(self.people()[0], "face_1", "a1")]
        cases = {("video", "sogni", "seedance-2.5"): "@image1", ("video", "pixeldojo", "seedance-2.5"): "@Image1",
                 ("video", "sogni", "wan-3.0"): "@Image 1", ("video", "sogni", "seedance-2"): "Image1",
                 ("image", "nanogpt", "seedream-4.5"): "image 1"}
        for route, expected in cases.items():
            self.assertEqual(render_tokens("{{ref:Maya:face_1}}", catalog.find(*route), refs), expected)

    def test_a_dropped_slot_uses_that_persons_first_image(self):
        refs = [Ref(self.people()[0], "face_1", "a1")]
        option = catalog.find("image", "nanogpt", "seedream-4.5")
        self.assertEqual(render_tokens("{{ref:Maya:naked}}", option, refs), "image 1")

    def test_someone_with_no_reference_is_simply_named(self):
        option = catalog.find("image", "nanogpt", "seedream-4.5")
        self.assertEqual(render_tokens("{{ref:Lena:face_1}} waves", option, []), "Lena waves")

    def test_offset_leaves_room_for_an_opening_frame(self):
        refs = [Ref(self.people()[0], "face_1", "a1")]
        option = catalog.find("image", "nanogpt", "seedream-4.5")
        self.assertEqual(render_tokens("{{ref:Maya:face_1}}", option, refs, offset=1), "image 2")


if __name__ == "__main__":
    unittest.main()
