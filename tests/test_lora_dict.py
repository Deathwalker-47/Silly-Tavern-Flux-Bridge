"""Tests for the per-architecture LoRA dictionary and its migration."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lora_dict  # noqa: E402

FLAT = {
    "name": "Maya", "url": "https://hf.co/a.safetensors", "weight": 0.8, "rank": 3, "category": "character",
    "trigger_words": ["maya77"], "keywords": ["maya"], "prepend_prompt": "maya77", "append_prompt": "",
    "negative_prompt": "", "notes": "", "permanent": False,
}


class MigrationTests(unittest.TestCase):
    def test_file_fields_move_under_flux_and_shared_fields_stay(self):
        entry = lora_dict.migrate_entry(FLAT)
        self.assertEqual(entry["models"]["flux"]["url"], FLAT["url"])
        self.assertEqual(entry["models"]["flux"]["trigger_words"], ["maya77"])
        self.assertIsNone(entry["models"]["zimage"])
        self.assertEqual(entry["keywords"], ["maya"])
        self.assertNotIn("url", entry)

    def test_every_field_survives(self):
        self.assertEqual(lora_dict.flat_entry(lora_dict.migrate_entry(FLAT), "flux"), FLAT)

    def test_a_converted_entry_is_left_alone(self):
        entry = lora_dict.migrate_entry(FLAT)
        self.assertIs(lora_dict.migrate_entry(entry), entry)

    def test_file_is_backed_up_and_converted_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dict.json"
            path.write_text(json.dumps({"config": {"permanent_loras": []}, "loras": {"maya": FLAT}}), encoding="utf-8")
            self.assertTrue(lora_dict.migrate_file(path))
            self.assertEqual(json.loads(Path(f"{path}.bak").read_text(encoding="utf-8"))["loras"]["maya"]["url"], FLAT["url"])
            self.assertIn("models", json.loads(path.read_text(encoding="utf-8"))["loras"]["maya"])
            self.assertFalse(lora_dict.migrate_file(path))


class LookupTests(unittest.TestCase):
    def test_lora_for_is_none_until_populated(self):
        entry = lora_dict.migrate_entry(FLAT)
        self.assertIsNone(lora_dict.lora_for(entry, "zimage"))
        self.assertIsNone(lora_dict.flat_entry(entry, "zimage"))

    def test_one_url_under_both_architectures_is_refused(self):
        file = {"url": "https://hf.co/same.safetensors", "weight": 1.0}
        data = {"loras": {"a": {"models": {"flux": file, "zimage": None}}, "b": {"models": {"flux": None, "zimage": file}}}}
        with self.assertRaisesRegex(ValueError, "'b'.*both flux and zimage"):
            lora_dict.check_unique_urls(data)

    def test_the_same_url_twice_in_one_architecture_is_allowed(self):
        file = {"url": "https://hf.co/same.safetensors"}
        lora_dict.check_unique_urls({"loras": {"a": {"models": {"flux": file}}, "b": {"models": {"flux": file}}}})


class RealDictionaryTests(unittest.TestCase):
    """The dictionary in the repo, as the bridge loads it."""

    def setUp(self):
        self.data = json.loads((ROOT / "master_lora_dict.json").read_text(encoding="utf-8"))

    def test_every_entry_is_converted_and_has_a_flux_file(self):
        for lora_id, entry in self.data["loras"].items():
            self.assertNotIn("url", entry, lora_id)
            self.assertTrue(lora_dict.lora_for(entry, "flux")["url"], lora_id)
            self.assertIsNone(lora_dict.lora_for(entry, "zimage"), lora_id)

    def test_urls_are_unique_across_architectures(self):
        lora_dict.check_unique_urls(self.data)


if __name__ == "__main__":
    unittest.main()
