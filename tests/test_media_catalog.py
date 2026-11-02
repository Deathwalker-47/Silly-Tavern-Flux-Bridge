"""The catalog is the guard rail: what's listed is all that can run, each with its own limits."""
import unittest
import unittest.mock

from media_support import MediaCase
from media import catalog, db
from media.config import MediaConfig


def every(kind):
    table = catalog.IMAGE_CATALOG if kind == "image" else catalog.VIDEO_CATALOG
    return [o for families in table.values() for variants in families.values() for o in variants]


class ShapeTests(unittest.TestCase):
    def test_every_option_is_well_formed(self):
        for option in every("image") + every("video"):
            name = f"{option.provider}/{option.id}"
            self.assertIn(option.provider, catalog.PROVIDER_LABELS, name)
            self.assertTrue(option.modes, name)
            self.assertTrue(option.lowest, name)
            if "reference" in option.modes and option.family != "best":
                self.assertGreater(option.max_refs, 0, name)
                self.assertTrue(option.ref_field, name)
            if option.kind == "video":
                self.assertIn(option.lowest, option.resolutions, name)
                self.assertTrue(option.seconds, name)
                self.assertIn(option.family, catalog.VIDEO_FAMILIES, name)
            else:
                self.assertIn(option.family, catalog.IMAGE_FAMILIES, name)

    def test_no_image_is_under_1k(self):
        for option in every("image"):
            for width, height in (option.extra.get("sizes") or {}).values():
                self.assertGreaterEqual(width * height, catalog.IMAGE_FLOOR_PIXELS, option.id)

    def test_every_size_table_covers_its_aspects(self):
        for option in every("image") + every("video"):
            sizes = option.extra.get("sizes")
            if sizes and option.provider in ("runware", "replicate"):
                self.assertTrue(set(option.aspects) <= set(sizes), option.id)
            if sizes and option.extra.get("build") == "seedream_5":
                self.assertTrue(set(sizes) >= {"9:16", "16:9"}, option.id)

    def test_h3_reference_comes_first_on_sogni_and_is_the_cheapest_tier(self):
        first = catalog.VIDEO_CATALOG["sogni"]["minimax-h3"][0]
        self.assertEqual(first.id, "minimax-h3-fastvideo-omni-int8_r2v_turbo")
        self.assertEqual(first.modes, ("reference",))
        self.assertEqual(first.max_refs, 9)

    def test_lora_routes_declare_their_architecture(self):
        self.assertEqual(catalog.IMAGE_CATALOG["runware"]["flux-lora"][0].lora_arch, "flux")
        self.assertEqual(catalog.IMAGE_CATALOG["runware"]["zimage"][0].lora_arch, "zimage")
        self.assertIsNone(catalog.IMAGE_CATALOG["sogni"]["zimage"][0].lora_arch)

    def test_longest_stays_within_15_seconds(self):
        wan = catalog.VIDEO_CATALOG["nanogpt"]["wan-3.0"][0]
        self.assertEqual(wan.longest(), 15)
        short = catalog.VIDEO_CATALOG["nanogpt"]["wan-2.2"][0]
        self.assertEqual(short.longest(), 8)

    def test_people_cap_follows_the_reference_limit(self):
        self.assertEqual(catalog.VIDEO_CATALOG["sogni"]["seedance-2.5"][0].people_cap, 6)
        self.assertEqual(catalog.VIDEO_CATALOG["sogni"]["minimax-h3"][0].people_cap, 3)

    def test_cost_per_image_and_per_second(self):
        self.assertEqual(catalog.IMAGE_CATALOG["nanogpt"]["seedream-4.5"][0].cost(), 0.04)
        self.assertEqual(catalog.VIDEO_CATALOG["nanogpt"]["wan-3.0"][0].cost(15), 0.75)
        self.assertIsNone(catalog.VIDEO_CATALOG["sogni"]["ltx-2.5"][0].cost(10))


class AspectTests(unittest.TestCase):
    def option(self, aspects):
        return catalog.ModelOption(id="x", label="x", kind="image", family="f", provider="p", modes=("text",), aspects=aspects)

    def test_portrait_falls_to_3_4_then_landscape(self):
        self.assertEqual(catalog.pick_aspect(self.option(("9:16", "16:9")), "9:16"), "9:16")
        self.assertEqual(catalog.pick_aspect(self.option(("3:4", "4:3")), "9:16"), "3:4")
        self.assertEqual(catalog.pick_aspect(self.option(("16:9",)), "9:16"), "16:9")

    def test_landscape_prefers_landscape(self):
        self.assertEqual(catalog.pick_aspect(self.option(("9:16", "4:3")), "16:9"), "4:3")

    def test_no_aspects_means_the_first_frame_sets_it(self):
        self.assertEqual(catalog.pick_aspect(self.option(()), "9:16"), "9:16")


class RouteTests(MediaCase):
    def test_banned_route_cannot_be_picked(self):
        option = catalog.video("nanogpt", "wan-3.0")
        db.set_route(option.provider, option.id, "banned", "refused")
        self.assertIs(catalog.route_status(option), catalog.Route.BANNED)
        with self.assertRaises(catalog.RouteNotAllowed):
            catalog.video("nanogpt", "wan-3.0")

    def test_your_result_wins_over_the_catalog(self):
        option = catalog.find("video", "nanogpt", "wan-3.0")
        with unittest.mock.patch.object(option, "nsfw", "check"):
            with self.assertRaises(catalog.RouteNotAllowed):
                catalog.video("nanogpt", "wan-3.0")
            db.set_route(option.provider, option.id, "allowed")
            self.assertEqual(catalog.video("nanogpt", "wan-3.0").id, option.id)

    def test_unlisted_model_is_refused(self):
        with self.assertRaisesRegex(ValueError, "has no"):
            catalog.video("nanogpt", "seedance-2", "bytedance/nsfw-free")
        with self.assertRaisesRegex(ValueError, "has no"):
            catalog.image("runware", "wan-3.0")

    def test_mock_routes_appear_only_when_switched_on(self):
        self.assertTrue(catalog.find("video", "mock", "minimax-h3"))
        with unittest.mock.patch.object(MediaConfig, "VIDEO_MOCK_PROVIDER", False):
            self.assertIsNone(catalog.find("video", "mock", "minimax-h3"))


if __name__ == "__main__":
    unittest.main()
