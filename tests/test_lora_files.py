"""Private LoRA links: which URLs get one, what the file is called, and that links expire.

The streaming itself runs over real FastAPI in media_http_check.py.
"""
import unittest
import unittest.mock

import media_support  # noqa: F401  (sets the path and a usable httpx)
from media import lora_files
from media.config import MediaConfig

HF = "https://huggingface.co/u/r/resolve/main/Dynamic_Poses(dynamic%20pose).safetensors"


class LoraLinkTests(unittest.TestCase):
    def setUp(self):
        patch = unittest.mock.patch.multiple(MediaConfig, HF_TOKEN="hf_test", PUBLIC_BASE_URL="https://bridge.test")
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(lora_files._links.clear)

    def test_without_a_token_providers_get_the_url_as_before(self):
        with unittest.mock.patch.object(MediaConfig, "HF_TOKEN", ""):
            self.assertEqual(lora_files.provider_url(HF), HF)
            self.assertEqual(lora_files.auth_headers(HF), {})

    def test_only_hugging_face_files_get_a_link_or_the_token(self):
        other = "https://example.com/a.safetensors"
        self.assertEqual(lora_files.provider_url(other), other)
        self.assertEqual(lora_files.auth_headers(other), {})
        self.assertEqual(lora_files.auth_headers("https://hf.co/u/r/resolve/main/a.safetensors"),
                         {"Authorization": "Bearer hf_test"})

    def test_each_link_is_new_ends_in_safetensors_and_leads_back_to_the_file(self):
        first, second = lora_files.provider_url(HF), lora_files.provider_url(HF)
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^https://bridge\.test/lora-files/[\w-]{32}/Dynamic_Poses_dynamic_pose_\.safetensors$")
        self.assertEqual(lora_files.source(first.split("/")[-2]), HF)
        self.assertIsNone(lora_files.source("made-up"))

    def test_links_expire(self):
        link = lora_files.provider_url(HF).split("/")[-2]
        later = lora_files.time.time() + lora_files.LINK_TTL + 1
        with unittest.mock.patch("media.lora_files.time.time", return_value=later):
            self.assertIsNone(lora_files.source(link))
        self.assertEqual(lora_files._links, {})

    def test_file_names_stay_plain(self):
        self.assertEqual(lora_files.file_name("https://hf.co/u/r/resolve/main/Realism_LoRA%20.safetensors"),
                         "Realism_LoRA_.safetensors")
        self.assertEqual(lora_files.file_name("https://hf.co/u/r/resolve/main/weights"), "weights.safetensors")


if __name__ == "__main__":
    unittest.main()
