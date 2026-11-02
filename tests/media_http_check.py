"""Drives the real bridge app over real FastAPI routing, with the mock provider.

The other tests stub FastAPI, so they can't see a route that was never wired, a bad path
pattern or a token check that isn't applied. This runs in its own process
(tests/test_media_http.py) with the real packages. Prints PASS and exits 0, or fails.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

folder = tempfile.mkdtemp()
os.environ.update({
    "DATA_DIR": folder, "VIDEO_MOCK_PROVIDER": "true", "MOCK_RENDER_SECONDS": "0.05", "EDIT_API_TOKEN": "test-token",
    "HELPER_API_KEY": "fake", "DIRECTOR_MODEL": "mock/fast", "NANOGPT_API_KEY": "", "RUNWARE_API_KEY": "", "REPLICATE_API_TOKEN": "",
    "SOGNI_API_KEY": "", "PIXELDOJO_API_KEY": "", "LORA_DICT_PATH": os.path.join(folder, "none.json"),
})
Path(folder, "none.json").write_text(json.dumps({"version": 3, "loras": {}}), encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.chdir(ROOT)

from fastapi.testclient import TestClient  # noqa: E402

import flux_lora_bridge as bridge  # noqa: E402
from media import director  # noqa: E402
from media.routes import public_generation  # noqa: E402
from media_support import noisy_jpeg, small_jpeg  # noqa: E402

AUTH = {"Authorization": "Bearer test-token"}
PLAN = json.dumps({"prompt": "{{ref:Maya:face_1}} by the window", "motion_prompt": "{{ref:Maya:face_1}} turns", "main_character": "c:maya",
                   "people": [{"key": "c:maya", "role": "main"}], "aspect_ratio": "9:16", "caption": "hello"})


async def fake_chat(messages, **kw):
    purpose = kw.get("purpose")
    if purpose == "decide":
        return '{"action": "image", "reason": "asked"}'
    if purpose == "next_speaker":
        return '{"speaker": "Maya", "beat": "smiles"}'
    return PLAN


director.helper.chat = fake_chat


def wait(client, job_id):
    for _ in range(400):
        job = client.get(f"/media/jobs/{job_id}", headers=AUTH).json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job stuck: {job}")


def main():
    with TestClient(bridge.app) as client:
        # the token guards everything except files
        assert client.get("/media/capabilities").status_code == 401
        assert client.get("/media/insights").status_code == 401

        caps = client.get("/media/capabilities", headers=AUTH).json()
        mock_image = next(p for p in caps["image"]["providers"] if p["id"] == "mock")
        assert mock_image["configured"] and mock_image["families"]
        assert [f for f in mock_image["families"] if f["id"] == "seedream-5-pro"] == [], "hidden steps stay out of the pickers"
        assert caps["reference_window"]["max_bytes"] == 10 * 1024 * 1024
        assert list(caps["modes"]) == ["reference", "first", "first_last"]

        # references: the window, and uploading once
        photo = noisy_jpeg()
        upload = client.post("/assets", content=photo, headers=AUTH)
        assert upload.status_code == 200, upload.text
        asset = upload.json()["asset_id"]
        assert client.post("/assets", content=photo, headers=AUTH).json()["existed"] is True
        assert client.get(f"/assets/{asset}/exists", headers=AUTH).json() == {"exists": True}
        refused = client.post("/assets", content=small_jpeg(), headers=AUTH)
        assert refused.status_code == 422 and "20 KB" in refused.json()["detail"]
        assert client.post("/assets?purpose=frame", content=small_jpeg(), headers=AUTH).status_code == 200
        assert client.post("/assets?purpose=nope", content=photo, headers=AUTH).status_code == 400

        # the director's two quick questions
        assert client.post("/director/decide", json={"reply": "x"}, headers=AUTH).json()["action"] == "image"
        speaker = client.post("/director/next-speaker", json={"members": [{"name": "Maya"}]}, headers=AUTH).json()
        assert speaker["speaker"] == "Maya"

        # a whole job
        person = {"key": "c:maya", "name": "Maya", "age": 24, "refs": {"face_1": asset}}
        body = {"kind": "video", "selection": {"provider": "mock", "family": "minimax-h3", "mode": "reference"}, "chat_id": "chat", "message_id": 3,
                "message_key": "k1", "folder": "Maya", "context": {"characters": [person], "reply": "hi"}}
        started = client.post("/media/jobs", json=body, headers=AUTH)
        assert started.status_code == 200, started.text
        job = wait(client, started.json()["job_id"])
        assert job["status"] == "done", job
        assert job["caption"] == "hello"
        assert client.get(job["file_url"]).status_code == 200, "files work with no token, for <video> tags"
        assert client.get("/media/files/nothing").status_code == 404
        listed = client.get("/media/jobs", params={"chat_id": "chat"}, headers=AUTH).json()["jobs"]
        assert [j["id"] for j in listed] == [job["id"]]
        assert client.post(f"/media/jobs/{job['id']}/ack", json={"st_url": "/user/images/Maya/x.mp4"}, headers=AUTH).json() == {"ok": True}
        assert client.get("/media/jobs", params={"chat_id": "chat"}, headers=AUTH).json()["jobs"] == []

        # gallery, rating, trace
        gallery = client.get("/media/generations", params={"person": "c:maya"}, headers=AUTH).json()
        assert gallery["unrated"] == 1 and gallery["generations"][0]["url"] == "/user/images/Maya/x.mp4"
        assert client.patch(f"/media/generations/{job['id']}", json={"rating": 9}, headers=AUTH).status_code == 400
        assert client.patch(f"/media/generations/{job['id']}", json={"rating": 5, "comment": "great"}, headers=AUTH).json() == {"ok": True}
        assert client.get("/media/generations", params={"person": "c:maya"}, headers=AUTH).json()["unrated"] == 0
        trace = client.get(f"/media/generations/{job['id']}/trace", headers=AUTH).json()
        assert any(s["stage"] == "provider" and s["event"] == "render" for s in trace["steps"])

        # bad requests are clear refusals, not 500s
        bad = client.post("/media/jobs", json={"kind": "image", "selection": {"provider": "mock", "family": "wan-3.0"}}, headers=AUTH)
        assert bad.status_code == 400 and "has no" in bad.json()["detail"]
        gone = client.post("/media/jobs", json={"kind": "video", "selection": {"provider": "mock", "family": "minimax-h3", "mode": "first"},
                                                "message_image_asset": "0" * 32}, headers=AUTH)
        assert gone.status_code == 400 and "Upload it again" in gone.json()["detail"], gone.text

        # test this route, then fail it: it leaves the pickers, model ids with slashes included
        assert client.post("/media/routes/test", json={"prompt": " "}, headers=AUTH).status_code == 400
        test = client.post("/media/routes/test", json={"kind": "image", "prompt": "a chair", "selection": {"provider": "mock", "family": "best"}}, headers=AUTH)
        assert wait(client, test.json()["job_id"])["status"] == "done"
        assert client.put("/media/routes/nanogpt/alibaba/wan-3.0/image-to-video-spicy", json={"status": "banned"}, headers=AUTH).json() == {"ok": True}
        caps = client.get("/media/capabilities", headers=AUTH).json()
        nano = next(p for p in caps["video"]["providers"] if p["id"] == "nanogpt")
        assert "wan-3.0" not in [f["id"] for f in nano["families"]]
        assert [r["model"] for r in caps["failed_routes"]] == ["alibaba/wan-3.0/image-to-video-spicy"]

        # restore puts it back
        client.put("/media/routes/nanogpt/alibaba/wan-3.0/image-to-video-spicy", json={"status": "allowed"}, headers=AUTH)
        caps = client.get("/media/capabilities", headers=AUTH).json()
        nano = next(p for p in caps["video"]["providers"] if p["id"] == "nanogpt")
        assert caps["failed_routes"] == [] and "wan-3.0" in [f["id"] for f in nano["families"]]

        # a gallery row counts every step of its generation
        row = {"id": "g", "kind": "video", "created_at": 1, "cost_credits": 11.25, "steps_usd": 0.04, "steps_credits": None}
        assert (public_generation(row)["cost_usd"], public_generation(row)["cost_credits"]) == (0.04, 11.25)
        assert public_generation({"id": "g", "kind": "image", "created_at": 1})["cost_usd"] is None

        # the record reads back
        assert client.post("/media/events", json={"events": [{"event": "attached", "generation_id": job["id"], "ms": 12}]}, headers=AUTH).json() == {"ok": True}
        report = client.get("/media/insights", params={"days": 1}, headers=AUTH).json()
        assert report["totals"]["done"] >= 1 and report["routes"] and report["client"]["events"]["attached"] == 1
        lines = client.get("/media/insights/export", params={"days": 1}, headers=AUTH).text.splitlines()
        assert json.loads(lines[0])["type"] == "generation"
        assert client.get("/media/balances", headers=AUTH).json() == {"balances": []}

        # cancel
        slow = bridge.app  # the app object is kept alive by the client; the job below finishes or cancels inside this block
        started = client.post("/media/jobs", json={"kind": "image", "selection": {"provider": "mock", "family": "seedream-4.5"}, "prompt": "x"}, headers=AUTH)
        cancelled = client.post(f"/media/jobs/{started.json()['job_id']}/cancel", headers=AUTH).json()
        assert cancelled["status"] in ("cancelled", "done"), cancelled
        assert slow

    print("PASS")


if __name__ == "__main__":
    main()
