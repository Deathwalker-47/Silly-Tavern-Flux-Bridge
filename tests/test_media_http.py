"""Runs tests/media_http_check.py in its own process, away from the FastAPI stubs here."""
from pathlib import Path
import subprocess
import sys
import unittest

CHECK = Path(__file__).with_name("media_http_check.py")


class HttpTests(unittest.TestCase):
    def test_the_real_app_end_to_end(self):
        probe = subprocess.run([sys.executable, "-c", "import fastapi, httpx, sse_starlette"], capture_output=True)
        if probe.returncode:
            self.skipTest("needs the real fastapi, httpx and sse_starlette")
        done = subprocess.run([sys.executable, str(CHECK)], capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace")
        self.assertEqual(done.returncode, 0, done.stdout[-1500:] + done.stderr[-3000:])
        self.assertIn("PASS", done.stdout)


if __name__ == "__main__":
    unittest.main()
