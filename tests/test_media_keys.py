"""Keys saved from settings: they win over .env, stay out of every status line, and clear cleanly."""
import json
import os
import unittest.mock

from media_support import FakeHTTP, FakeResponse, MediaCase, run
from media import keys
from media.config import MediaConfig
from media.providers.nanogpt import NanoGPT
from media.providers.pixeldojo import PixelDojo
from media.providers.runware import Runware

SECRET = "settings-key-abcd"


class KeysTests(MediaCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(keys.apply)
        self.addCleanup(keys.clear, "nanogpt")
        self.addCleanup(keys.clear, "runware")

    def row(self, provider, hints=True):
        return next(r for r in keys.status(hints) if r["id"] == provider)

    def test_a_saved_key_wins_over_env_and_clearing_goes_back(self):
        with unittest.mock.patch.dict(os.environ, {"RUNWARE_API_KEY": "from-env-key-1234"}):
            keys.apply()
            self.assertEqual(self.row("runware")["source"], "env")
            keys.save("runware", SECRET)
            self.assertEqual(MediaConfig.RUNWARE_API_KEY, SECRET)
            self.assertEqual(self.row("runware")["source"], "settings")
            keys.clear("runware")
            self.assertEqual(MediaConfig.RUNWARE_API_KEY, "from-env-key-1234")

    def test_status_never_shows_more_than_the_last_four(self):
        keys.save("runware", SECRET)
        self.assertEqual(self.row("runware")["hint"], "abcd")
        self.assertEqual(self.row("runware", hints=False)["hint"], "")
        self.assertNotIn(SECRET, json.dumps(keys.status()))

    def test_a_short_key_shows_no_hint(self):
        keys.save("runware", "shortkey")
        self.assertEqual(self.row("runware")["hint"], "")

    def test_the_helper_follows_nanogpt(self):
        with unittest.mock.patch.dict(os.environ, {"HELPER_API_KEY": ""}):
            keys.save("nanogpt", SECRET)
            self.assertEqual(MediaConfig.HELPER_API_KEY, SECRET)

    def test_the_old_clients_hear_about_a_change(self):
        seen = []
        self.addCleanup(keys.on_change, lambda: None)
        keys.on_change(lambda: seen.append(MediaConfig.RUNWARE_API_KEY))
        keys.save("runware", SECRET)
        self.assertEqual(seen, [SECRET])

    def test_a_damaged_file_counts_as_no_saved_keys(self):
        keys._path().parent.mkdir(parents=True, exist_ok=True)
        keys._path().write_text("{broken", encoding="utf-8")
        self.addCleanup(keys._write, {})
        self.assertEqual(self.row("runware")["source"] in ("env", "none"), True)


class RefusedKeyTests(MediaCase):
    def test_a_refused_key_is_an_error_not_an_empty_balance(self):
        with unittest.mock.patch.object(MediaConfig, "RUNWARE_ENDPOINT", "https://runware/v1"):
            for provider, route in ((NanoGPT(), "/check-balance"), (PixelDojo(), "/credits"), (Runware(), "runware")):
                with FakeHTTP([(route, FakeResponse(401, {"error": "bad"}))]).patch(), self.assertRaisesRegex(RuntimeError, "refused this key"):
                    run(provider.balance())
