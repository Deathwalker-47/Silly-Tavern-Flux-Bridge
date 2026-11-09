"""Private LoRA links: which URLs get one, what the file is called, and that only the
bridge can make one.

The streaming itself runs over real FastAPI in media_http_check.py.
"""
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import media_support  # noqa: F401  (sets the path and a usable httpx)
from media import lora_files
from media.config import MediaConfig

HF = "https://huggingface.co/u/r/resolve/main/Dynamic_Poses(dynamic%20pose).safetensors"
LINK = r"[\w-]+\.[\w-]{32}"


def link_of(url):
    return url.split("/")[-2]


class LoraLinkTests(unittest.TestCase):
    def setUp(self):
        patch = unittest.mock.patch.multiple(MediaConfig, HF_TOKEN="hf_test", PUBLIC_BASE_URL="https://bridge.test",
                                             LORA_LINK_SECRET="test-secret")
        patch.start()
        self.addCleanup(patch.stop)

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

    def test_a_file_always_gets_the_same_link_which_ends_in_safetensors_and_leads_back_to_it(self):
        # Runware keeps the link it was given and downloads the file again hours later.
        first, second = lora_files.provider_url(HF), lora_files.provider_url(HF)
        self.assertEqual(first, second)
        self.assertRegex(first, rf"^https://bridge\.test/lora-files/{LINK}/Dynamic_Poses_dynamic_pose_\.safetensors$")
        self.assertEqual(lora_files.source(link_of(first)), HF)

    def test_only_links_the_bridge_signed_lead_anywhere(self):
        link = link_of(lora_files.provider_url(HF))
        encoded, signature = link.split(".")
        other = link_of(lora_files.provider_url("https://huggingface.co/u/other/resolve/main/b.safetensors"))
        changed = signature[:-1] + ("A" if signature[-1] != "A" else "B")
        for bad in ("made-up", "", ".", f"{encoded}.", f".{signature}", f"{encoded}.{changed}",
                    f"{encoded}.{other.split('.')[1]}", f"{encoded[:-2]}.{signature}", f"!!!.{signature}"):
            self.assertIsNone(lora_files.source(bad), bad)
        with unittest.mock.patch.object(MediaConfig, "LORA_LINK_SECRET", "another-secret"):
            self.assertIsNone(lora_files.source(link), "a new secret retires the old links")
        # signed, but not Hugging Face: the link can never become a way to fetch other URLs
        elsewhere = "https://example.com/a.safetensors"
        self.assertIsNone(lora_files.source(f"{lora_files._b64(elsewhere.encode())}.{lora_files._sign(elsewhere)}"))

    def test_without_a_set_secret_the_bridge_makes_one_and_keeps_it_across_restarts(self):
        folder = Path(tempfile.mkdtemp())
        with unittest.mock.patch.multiple(MediaConfig, LORA_LINK_SECRET="", DATA_DIR=folder), \
             unittest.mock.patch.dict(lora_files._secrets, clear=True):
            link = lora_files.provider_url(HF)
            kept = (folder / lora_files.SECRET_FILE).read_bytes()
            self.assertGreaterEqual(len(kept), 40)
            lora_files._secrets.clear()                     # a restart
            self.assertEqual(lora_files.provider_url(HF), link)
            self.assertEqual((folder / lora_files.SECRET_FILE).read_bytes(), kept)
            self.assertEqual(lora_files.source(link_of(link)), HF)
        with unittest.mock.patch.multiple(MediaConfig, LORA_LINK_SECRET="", DATA_DIR=Path(tempfile.mkdtemp())), \
             unittest.mock.patch.dict(lora_files._secrets, clear=True):
            self.assertIsNone(lora_files.source(link_of(link)), "another install can't read this one's links")

    def test_file_names_stay_plain(self):
        self.assertEqual(lora_files.file_name("https://hf.co/u/r/resolve/main/Realism_LoRA%20.safetensors"),
                         "Realism_LoRA_.safetensors")
        self.assertEqual(lora_files.file_name("https://hf.co/u/r/resolve/main/weights"), "weights.safetensors")


if __name__ == "__main__":
    unittest.main()
